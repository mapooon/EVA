from functools import partial

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        prefix_len,
        causal,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_len = prefix_len
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        rotary_dim = self.head_dim if self.head_dim % 2 == 0 else self.head_dim - 1
        self.rotary_dim = rotary_dim
        if rotary_dim > 0:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
            self.register_buffer('rope_inv_freq', inv_freq, persistent=False)
        else:
            self.register_buffer('rope_inv_freq', torch.empty(0), persistent=False)

    def _build_attention_allow_mask(self, seq_len, device):
        prefix_len = min(self.prefix_len, seq_len)
        idx = torch.arange(seq_len, device=device)
        q_idx = idx[:, None]
        k_idx = idx[None, :]

        allow_prefix_prefix = (q_idx < prefix_len) & (k_idx < prefix_len)
        allow_image_prefix = (q_idx >= prefix_len) & (k_idx < prefix_len)
        allow_image_image_causal = (q_idx >= prefix_len) & (k_idx >= prefix_len) & (k_idx <= q_idx)

        return allow_prefix_prefix | allow_image_prefix | allow_image_image_causal

    def _apply_rope(self, q, k):
        if self.rotary_dim <= 0:
            return q, k

        seq_len = q.size(2)
        positions = torch.arange(seq_len, device=q.device, dtype=self.rope_inv_freq.dtype)
        return self._apply_rope_with_positions(q, k, positions)

    def _apply_rope_with_positions(self, q, k, positions):
        if self.rotary_dim <= 0:
            return q, k

        if not torch.is_tensor(positions):
            positions = torch.tensor(positions, device=q.device, dtype=self.rope_inv_freq.dtype)
        else:
            positions = positions.to(device=q.device, dtype=self.rope_inv_freq.dtype)
        if positions.dim() == 0:
            positions = positions.unsqueeze(0)

        freqs = torch.einsum('l,d->ld', positions, self.rope_inv_freq)
        cos = freqs.cos()[None, None, :, :, None]
        sin = freqs.sin()[None, None, :, :, None]

        def rotate(x):
            x_rot = x[..., :self.rotary_dim]
            x_pass = x[..., self.rotary_dim:]

            x_rot = x_rot.reshape(*x_rot.shape[:-1], self.rotary_dim // 2, 2).float()
            x0 = x_rot[..., 0:1]
            x1 = x_rot[..., 1:2]
            out0 = x0 * cos - x1 * sin
            out1 = x0 * sin + x1 * cos
            out = torch.cat([out0, out1], dim=-1).reshape(*x_rot.shape[:-2], self.rotary_dim)
            out = out.to(dtype=x.dtype)
            return torch.cat([out, x_pass], dim=-1)

        return rotate(q), rotate(k)

    def _project_qkv(self, x):
        bsz, seq_len, _dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_rope(q, k)

        allow_mask = None
        if self.causal:
            allow_mask = self._build_attention_allow_mask(seq_len, x.device)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allow_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def prefill(self, x):
        bsz, seq_len, dim = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_rope(q, k)

        allow_mask = None
        if self.causal:
            allow_mask = self._build_attention_allow_mask(seq_len, x.device)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allow_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, k, v

    def forward_cached(self, x, past_k, past_v, position):
        bsz, seq_len, dim = x.shape
        if seq_len != 1:
            raise ValueError(f'forward_cached expects a single-step token, got seq_len={seq_len}')

        q, k, v = self._project_qkv(x)
        q, k = self._apply_rope_with_positions(q, k, [position])

        if past_k is None:
            k_all = k
            v_all = v
        else:
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)

        out = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, k_all, v_all


class CausalBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        prefix_len,
        causal,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            prefix_len=prefix_len,
            causal=causal,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def prefill(self, x):
        attn_in = self.norm1(x)
        attn_out, k, v = self.attn.prefill(attn_in)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)

    def forward_cached(self, x, layer_cache, position):
        past_k, past_v = layer_cache
        attn_in = self.norm1(x)
        attn_out, new_k, new_v = self.attn.forward_cached(attn_in, past_k, past_v, position)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, (new_k, new_v)


class EVA(nn.Module):
    """Class-conditional autoregressive variational generator over cached latents."""

    def __init__(
        self,
        latent_shape: Sequence[int],
        class_num: int,
        reparam_dim: int,
        transformer_latent_dim=256,
        encoder_depth=8,
        encoder_num_heads=8,
        decoder_depth=8,
        decoder_num_heads=8,
        norm_layer=nn.LayerNorm,
        grad_checkpointing=False,
    ):
        super().__init__()

        latent_shape = tuple(int(dim) for dim in latent_shape)
        if len(latent_shape) < 2 or any(dim <= 0 for dim in latent_shape):
            raise ValueError(f"latent_shape must be [channels, ...] with positive dimensions, got {latent_shape}")
        if reparam_dim <= 0:
            raise ValueError(f"reparam_dim must be positive, got {reparam_dim}")

        self.latent_shape = latent_shape
        self.vae_embed_dim = latent_shape[0]
        self.spatial_shape = latent_shape[1:]
        self.seq_len = math.prod(self.spatial_shape)
        self.token_embed_dim = self.vae_embed_dim
        self.transformer_latent_dim = transformer_latent_dim
        # Keep this attribute name for checkpoint-era tooling. It does not affect state-dict keys.
        self.reparam_latent_dim = reparam_dim
        self.register_token_size = 4
        self.class_token_size = 1
        self.prefix_len = self.register_token_size + self.class_token_size
        self.grad_checkpointing = grad_checkpointing

        # class conditioning
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, transformer_latent_dim)
        self.label_drop_prob = 0.1
        self.fake_latent = nn.Parameter(torch.zeros(1, transformer_latent_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.register_token_size, transformer_latent_dim)
        )

        # encoder
        self.encoder_token_proj = nn.Linear(self.token_embed_dim, transformer_latent_dim, bias=True)
        self.encoder_in_norm = nn.LayerNorm(transformer_latent_dim, eps=1e-6)
        self.encoder_blocks = nn.ModuleList([
            CausalBlock(
                transformer_latent_dim,
                encoder_num_heads,
                prefix_len=self.prefix_len,
                causal=False,
                qkv_bias=True,
                norm_layer=norm_layer,
                attn_drop=0.1,
                proj_drop=0.1,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(transformer_latent_dim)

        self.posterior_mu = nn.Linear(transformer_latent_dim, reparam_dim)
        self.posterior_logvar = nn.Linear(transformer_latent_dim, reparam_dim)

        # decoder
        self.decoder_input_proj = nn.Linear(reparam_dim, transformer_latent_dim)
        self.decoder_in_norm = nn.LayerNorm(transformer_latent_dim, eps=1e-6)
        self.decoder_blocks = nn.ModuleList([
            CausalBlock(
                transformer_latent_dim,
                decoder_num_heads,
                prefix_len=self.prefix_len,
                causal=True,
                qkv_bias=True,
                norm_layer=norm_layer,
                attn_drop=0.1,
                proj_drop=0.1,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(transformer_latent_dim)
        self.output_head = nn.Linear(transformer_latent_dim, self.token_embed_dim)

        self.decoder_prior_mu = nn.Linear(transformer_latent_dim, reparam_dim)
        self.decoder_prior_logvar = nn.Linear(transformer_latent_dim, reparam_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.register_tokens, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        if tuple(x.shape[1:]) != self.latent_shape:
            raise ValueError(f"expected latent shape [B, {self.latent_shape}], got {tuple(x.shape)}")
        return x.flatten(2).transpose(1, 2).contiguous()

    def unpatchify(self, x):
        expected = (self.seq_len, self.token_embed_dim)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected tokens [B, {expected[0]}, {expected[1]}], got {tuple(x.shape)}")
        return x.transpose(1, 2).reshape(x.shape[0], *self.latent_shape).contiguous()

    @staticmethod
    def _kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
        inv_var_p = torch.exp(-logvar_p)
        var_ratio = torch.exp(logvar_q - logvar_p)
        diff = mu_q - mu_p
        kl = 0.5 * (logvar_p - logvar_q + var_ratio + diff.pow(2) * inv_var_p - 1.0)
        return kl.mean()

    def _label_condition(self, labels: Optional[torch.Tensor], bsz: Optional[int] = None, training_drop=False):
        if labels is None:
            if bsz is None:
                raise ValueError('bsz must be provided when labels is None.')
            class_embedding = self.fake_latent.repeat(bsz, 1)
            return class_embedding
        class_embedding = self.class_emb(labels)
        if training_drop and self.training and self.label_drop_prob > 0:
            drop_mask = (torch.rand(labels.size(0), device=labels.device) < self.label_drop_prob).unsqueeze(-1)
            class_embedding = torch.where(drop_mask, self.fake_latent.to(class_embedding.dtype), class_embedding)
        return class_embedding

    def _prepend_prefix(self, x, class_embedding):
        bsz = x.size(0)
        prefix = torch.cat(
            [self.register_tokens.expand(bsz, -1, -1), class_embedding.unsqueeze(1)],
            dim=1,
        ).to(dtype=x.dtype, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        return x

    def _run_blocks(self, x, blocks):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in blocks:
                x = checkpoint(block, x)
        else:
            for block in blocks:
                x = block(x)
        return x

    def encode(self, tokens, class_embedding):
        x = self.encoder_token_proj(tokens)
        x = self._prepend_prefix(x, class_embedding)
        x = self.encoder_in_norm(x)
        x = self._run_blocks(x, self.encoder_blocks)
        x = self.encoder_norm(x)
        x = x[:, self.prefix_len:]

        mu_q = self.posterior_mu(x)
        logvar_q = self.posterior_logvar(x).clamp(-10.0, 10.0)
        std_q = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std_q)
        z = mu_q + eps * std_q

        return z, mu_q, logvar_q

    def decode(self, z, class_embedding, return_prior_params=False):
        x = self.decoder_input_proj(z)
        x = self._prepend_prefix(x, class_embedding)
        x = self.decoder_in_norm(x)
        x = self._run_blocks(x, self.decoder_blocks)
        x = self.decoder_norm(x)
        hidden_full = x
        hidden = hidden_full[:, self.prefix_len:]
        preds = self.output_head(hidden)

        if return_prior_params:
            # Raw image-position priors: index t predicts next latent z_{t+1}.
            mu_img = self.decoder_prior_mu(hidden)
            logvar_img = self.decoder_prior_logvar(hidden).clamp(-10.0, 10.0)
            # class token is treated as z_{-1}; its prior predicts z_0.
            class_idx = self.register_token_size
            class_hidden = hidden_full[:, class_idx:class_idx + 1, :]
            mu_class = self.decoder_prior_mu(class_hidden)
            logvar_class = self.decoder_prior_logvar(class_hidden).clamp(-10.0, 10.0)
            # Align to q(z_t): [z_0 from class] + [z_1.. from image 0..T-2]
            mu_aligned = torch.cat([mu_class, mu_img[:, :-1, :]], dim=1)
            logvar_aligned = torch.cat([logvar_class, logvar_img[:, :-1, :]], dim=1)
            mu_next = mu_img[:, -1:, :]
            logvar_next = logvar_img[:, -1:, :]
            return preds, hidden, mu_aligned, logvar_aligned, mu_next, logvar_next
        return preds, hidden, None, None, None, None

    def _decoder_prefill(self, class_embedding):
        bsz = class_embedding.size(0)
        x = torch.zeros(
            bsz,
            0,
            self.transformer_latent_dim,
            device=class_embedding.device,
            dtype=class_embedding.dtype,
        )
        x = self._prepend_prefix(x, class_embedding)
        x = self.decoder_in_norm(x)

        layer_caches = []
        for block in self.decoder_blocks:
            x, layer_cache = block.prefill(x)
            layer_caches.append(layer_cache)
        x = self.decoder_norm(x)
        return x, layer_caches

    def _decoder_step(self, z, layer_caches, step_idx, return_prior_params=False):
        if z.size(1) != 1:
            raise ValueError(f'_decoder_step expects a single latent token, got seq_len={z.size(1)}')

        x = self.decoder_input_proj(z)
        x = self.decoder_in_norm(x)

        position = self.prefix_len + step_idx
        new_caches = []
        for block, layer_cache in zip(self.decoder_blocks, layer_caches):
            x, layer_cache = block.forward_cached(x, layer_cache, position)
            new_caches.append(layer_cache)
        x = self.decoder_norm(x)
        preds = self.output_head(x)

        if return_prior_params:
            mu_next = self.decoder_prior_mu(x)
            logvar_next = self.decoder_prior_logvar(x).clamp(-10.0, 10.0)
            return preds, mu_next, logvar_next, new_caches
        return preds, None, None, new_caches

    def forward(self, imgs, labels):
        class_embedding = self._label_condition(labels, training_drop=True)

        tokens = self.patchify(imgs)
        z, mu_q, logvar_q = self.encode(tokens, class_embedding)

        preds, _decoder_hidden, mu_p, logvar_p, _mu_next, _logvar_next = self.decode(
            z,
            class_embedding,
            return_prior_params=True,
        )

        recon_loss = F.mse_loss(preds, tokens)
        kl_loss = self._kl_gaussian(mu_q, logvar_q, mu_p, logvar_p)
        total_loss = recon_loss + kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

    @staticmethod
    def _guidance_scale_for_step(step, total_steps, guidance_scale):
        return 1.0 + (guidance_scale - 1.0) * float(step + 1) / float(total_steps)

    @staticmethod
    def _guided_prior_params(mu_cond, logvar_cond, mu_uncond, logvar_uncond, scale):
        mu = mu_uncond + scale * (mu_cond - mu_uncond)
        alpha = float(max(0.0, min(1.0, scale)))
        precision = (1.0 - alpha) * torch.exp(-logvar_uncond) + alpha * torch.exp(-logvar_cond)
        logvar = -torch.log(precision.clamp_min(1e-8))
        return mu, logvar.clamp(-10.0, 10.0)

    @torch.no_grad()
    def sample_tokens(
        self,
        bsz,
        labels=None,
        guidance_scale=1.0,
        temperature=1.0,
        progress=False,
    ):
        num_iter = self.seq_len
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if labels is None:
            cond_class_embedding = self.fake_latent.repeat(bsz, 1)
        else:
            cond_class_embedding = self.class_emb(labels)
        uncond_class_embedding = self.fake_latent.repeat(bsz, 1)
        need_uncond_decode = guidance_scale != 1.0

        generated_tokens = torch.zeros(bsz, 0, self.token_embed_dim, device=cond_class_embedding.device)

        step_iter = range(num_iter)
        if progress:
            from tqdm import tqdm
            step_iter = tqdm(step_iter)

        if need_uncond_decode:
            class_embedding_decode = torch.cat([cond_class_embedding, uncond_class_embedding], dim=0)
        else:
            class_embedding_decode = cond_class_embedding
        prefix_hidden, layer_caches = self._decoder_prefill(class_embedding_decode)

        class_idx = self.register_token_size
        class_hidden = prefix_hidden[:, class_idx:class_idx + 1, :]
        mu_boot = self.decoder_prior_mu(class_hidden)
        logvar_boot = self.decoder_prior_logvar(class_hidden).clamp(-10.0, 10.0)
        if need_uncond_decode:
            mu_cond, mu_uncond = mu_boot.chunk(2, dim=0)
            logvar_cond, logvar_uncond = logvar_boot.chunk(2, dim=0)
            scale = self._guidance_scale_for_step(0, num_iter, guidance_scale)
            next_mu_p, next_logvar_p = self._guided_prior_params(
                mu_cond, logvar_cond, mu_uncond, logvar_uncond, scale
            )
        else:
            next_mu_p, next_logvar_p = mu_boot, logvar_boot

        for step in step_iter:
            std_next = torch.exp(0.5 * next_logvar_p) * temperature
            z_t = next_mu_p + torch.randn_like(next_mu_p) * std_next

            if need_uncond_decode:
                z_step = torch.cat([z_t, z_t], dim=0)
            else:
                z_step = z_t

            preds, mu_next_all, logvar_next_all, layer_caches = self._decoder_step(
                z_step,
                layer_caches,
                step,
                return_prior_params=True,
            )

            if need_uncond_decode:
                next_token, _ = preds.chunk(2, dim=0)
                mu_cond, mu_uncond = mu_next_all.chunk(2, dim=0)
                logvar_cond, logvar_uncond = logvar_next_all.chunk(2, dim=0)
                scale = self._guidance_scale_for_step(step, num_iter, guidance_scale)
                next_mu_p, next_logvar_p = self._guided_prior_params(
                    mu_cond, logvar_cond, mu_uncond, logvar_uncond, scale
                )
            else:
                next_token = preds
                next_mu_p = mu_next_all
                next_logvar_p = logvar_next_all

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        tokens_2d = self.unpatchify(generated_tokens)
        return tokens_2d


def eva_small(**kwargs):
    defaults = dict(
        transformer_latent_dim=512,
        encoder_depth=8,
        encoder_num_heads=8,
        decoder_depth=8,
        decoder_num_heads=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    defaults.update(kwargs)
    return EVA(**defaults)


def eva_base(**kwargs):
    defaults = dict(
        transformer_latent_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_depth=12,
        decoder_num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    defaults.update(kwargs)
    return EVA(**defaults)


def eva_large(**kwargs):
    defaults = dict(
        transformer_latent_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_depth=16,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    defaults.update(kwargs)
    return EVA(**defaults)


def eva_huge(**kwargs):
    defaults = dict(
        transformer_latent_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_depth=20,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    defaults.update(kwargs)
    return EVA(**defaults)


EVA_MODELS = {
    'small': eva_small,
    'base': eva_base,
    'large': eva_large,
    'huge': eva_huge,
}


def build_eva(size, *, latent_shape, class_num, reparam_dim, grad_checkpointing=False):
    try:
        factory = EVA_MODELS[size]
    except KeyError as exc:
        choices = ', '.join(EVA_MODELS)
        raise ValueError(f"unknown EVA size {size!r}; expected one of: {choices}") from exc
    return factory(
        latent_shape=latent_shape,
        class_num=class_num,
        reparam_dim=reparam_dim,
        grad_checkpointing=grad_checkpointing,
    )

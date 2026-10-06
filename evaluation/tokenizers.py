from __future__ import annotations

from typing import Any

import torch

from tasks import TaskSpec


def load_tokenizer(task: TaskSpec, config: dict[str, Any], device: torch.device):
    if task.name == 'imagenet':
        from models.image_vae import AutoencoderKL

        tokenizer = AutoencoderKL(
            embed_dim=task.latent_shape[0],
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=config['paths']['tokenizer_path'],
        )
    else:
        try:
            from diffusers import AutoencoderOobleck
        except ImportError as exc:
            raise RuntimeError('diffusers is required for VGGSound tokenization') from exc
        tokenizer = AutoencoderOobleck.from_pretrained(
            config['tokenizer']['model_id'],
            subfolder=config['tokenizer'].get('subfolder', 'vae'),
            revision=config['tokenizer'].get('revision'),
            torch_dtype=torch.float32,
        )
    tokenizer = tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad = False
    return tokenizer

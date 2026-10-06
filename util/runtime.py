from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(runtime: dict[str, Any]) -> DistributedContext:
    requested = runtime.get('device', 'cuda')
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('runtime.device=cuda but CUDA is not available')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
    else:
        device = torch.device(requested)
        backend = 'gloo'

    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=runtime.get('dist_url', 'env://'),
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
    return DistributedContext(distributed, rank, world_size, local_rank, device)


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != 'cuda' or dtype_name == 'fp32':
        return contextlib.nullcontext()
    if dtype_name == 'bf16':
        dtype = torch.bfloat16
    elif dtype_name == 'fp16':
        dtype = torch.float16
    else:
        raise ValueError(f'unsupported amp dtype: {dtype_name}')
    return torch.autocast(device_type='cuda', dtype=dtype)


def parameter_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith('.bias'):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, rate: float) -> None:
    current = model.state_dict()
    for name, ema_value in ema_state.items():
        source = current[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(rate).add_(source, alpha=1.0 - rate)
        else:
            ema_value.copy_(source)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format_version': 1,
            'model': model.state_dict(),
            'model_ema': ema_state,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'config': config,
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    expected_config: dict[str, Any] | None = None,
) -> tuple[int, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = checkpoint.get('config')
    if expected_config is not None and saved_config is not None:
        expected_identity = (
            expected_config['task'],
            expected_config['model']['size'],
            expected_config['model']['reparam_dim'],
        )
        saved_identity = (
            saved_config['task'],
            saved_config['model']['size'],
            saved_config['model']['reparam_dim'],
        )
        if saved_identity != expected_identity:
            raise ValueError(
                f'checkpoint identity {saved_identity} does not match current config {expected_identity}'
            )
    model.load_state_dict(checkpoint['model'], strict=True)
    ema_state = checkpoint.get('model_ema')
    if ema_state is None:
        ema_state = clone_state_dict(model)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    return int(checkpoint.get('epoch', -1)) + 1, ema_state


def load_inference_weights(path: str | Path, model: torch.nn.Module, use_ema: bool = True) -> None:
    path = Path(path)
    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError('install safetensors to load this checkpoint') from exc
        state = load_file(str(path), device='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and ('model' in checkpoint or 'model_ema' in checkpoint):
            key = 'model_ema' if use_ema and checkpoint.get('model_ema') is not None else 'model'
            state = checkpoint[key]
        else:
            state = checkpoint
    model.load_state_dict(state, strict=True)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

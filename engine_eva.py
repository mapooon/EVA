from __future__ import annotations

import math
import time
from typing import Any

import torch

from datasets import DiagonalGaussianDistribution
from tasks import TaskSpec
from util.runtime import all_reduce_mean, autocast_context, update_ema


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    progress: float,
    *,
    learning_rate: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and progress < warmup_epochs:
        lr = learning_rate * progress / warmup_epochs
    else:
        lr = learning_rate
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def train_one_epoch(
    *,
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    data_loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    task: TaskSpec,
    learning_rate: float,
    log_writer=None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train = config['train']
    totals = {'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    start = time.time()

    for step, (moments, labels) in enumerate(data_loader):
        progress = epoch + step / len(data_loader)
        lr = adjust_learning_rate(
            optimizer,
            progress,
            learning_rate=learning_rate,
            warmup_epochs=train['warmup_epochs'],
        )
        moments = moments.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            latents = DiagonalGaussianDistribution(moments).sample()
            latents.mul_(task.latent_scale)

        with autocast_context(device, train['amp_dtype']):
            losses = model(latents, labels)
            loss = losses['loss']
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(f'non-finite loss at epoch={epoch}, step={step}: {loss.item()}')

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema_state, model_without_ddp, config['checkpoint']['ema_rate'])

        for name in totals:
            totals[name] += float(losses[name].item())
        if step % 20 == 0:
            averaged = {name: totals[name] / (step + 1) for name in totals}
            elapsed = time.time() - start
            print(
                f"epoch {epoch} [{step}/{len(data_loader)}] "
                f"loss={averaged['loss']:.4f} recon={averaged['recon_loss']:.4f} "
                f"kl={averaged['kl_loss']:.4f} lr={lr:.3e} time={elapsed:.1f}s"
            )
        if log_writer is not None:
            global_step = epoch * len(data_loader) + step
            for name in totals:
                # Only rank zero owns a writer.  Do not start a distributed
                # collective here, because the other ranks do not enter this
                # branch.
                log_writer.add_scalar(f'train/{name}', float(losses[name].item()), global_step)
            log_writer.add_scalar('train/lr', lr, global_step)

    return {
        name: all_reduce_mean(total / max(1, len(data_loader)), device)
        for name, total in totals.items()
    }

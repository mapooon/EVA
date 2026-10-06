from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets import CachedLatentDataset
from engine_eva import train_one_epoch
from models import build_eva
from tasks import get_task_spec
from util.config import load_config, save_resolved_config
from util.runtime import (
    barrier,
    clone_state_dict,
    init_distributed,
    load_training_checkpoint,
    move_optimizer_state,
    parameter_groups,
    save_training_checkpoint,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser('Train EVA from cached tokenizer latents')
    parser.add_argument('--config', required=True, help='YAML training config')
    return parser.parse_args()


def build_model(config, task):
    return build_eva(
        config['model']['size'],
        latent_shape=task.latent_shape,
        class_num=task.class_num,
        reparam_dim=config['model']['reparam_dim'],
        grad_checkpointing=config['train']['grad_checkpointing'],
    )


def run_train(config, task, context):
    output_dir = Path(config['paths']['output_dir'])
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir)
    barrier()

    dataset = CachedLatentDataset(config['paths']['cached_path'], task)
    if len(dataset.classes) != task.class_num:
        raise ValueError(
            f'{task.name} expects {task.class_num} class directories, found {len(dataset.classes)} '
            f"under {config['paths']['cached_path']}"
        )
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config['runtime']['num_workers'],
        pin_memory=config['runtime']['pin_memory'],
        drop_last=True,
    )

    model_without_ddp = build_model(config, task).to(context.device)
    parameter_count = sum(parameter.numel() for parameter in model_without_ddp.parameters())
    if context.is_main:
        print(f'EVA-{config["model"]["size"]}: {parameter_count / 1e6:.2f}M parameters')

    effective_batch_size = config['train']['batch_size'] * context.world_size
    learning_rate = config['train']['base_lr'] * effective_batch_size / 256.0
    optimizer = torch.optim.AdamW(
        parameter_groups(model_without_ddp, config['train']['weight_decay']),
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=context.device.type == 'cuda' and config['train']['amp_dtype'] == 'fp16',
    )
    ema_state = clone_state_dict(model_without_ddp)
    start_epoch = 0
    resume = config['paths'].get('resume')
    if resume:
        start_epoch, ema_state = load_training_checkpoint(
            resume,
            model=model_without_ddp,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=config,
        )
        ema_state = {name: value.to(context.device) for name, value in ema_state.items()}
        move_optimizer_state(optimizer, context.device)
        if context.is_main:
            print(f'Resumed {resume} at epoch {start_epoch}')

    model = model_without_ddp
    if context.distributed:
        model = DistributedDataParallel(model_without_ddp, device_ids=[context.local_rank])

    log_writer = None
    if context.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_writer = SummaryWriter(output_dir / 'tensorboard')
        except Exception as exc:
            print(f'TensorBoard is unavailable ({exc}); continuing without event logs.')

    for epoch in range(start_epoch, config['train']['epochs']):
        if sampler is not None:
            sampler.set_epoch(epoch)
        stats = train_one_epoch(
            model=model,
            model_without_ddp=model_without_ddp,
            ema_state=ema_state,
            data_loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=context.device,
            epoch=epoch,
            config=config,
            task=task,
            learning_rate=learning_rate,
            log_writer=log_writer,
        )
        if context.is_main:
            print(f'epoch {epoch} complete: {stats}')
            frequency = config['checkpoint']['save_frequency']
            if epoch % frequency == 0 or epoch + 1 == config['train']['epochs']:
                save_training_checkpoint(
                    output_dir / f'checkpoint-{epoch}.pth',
                    model=model_without_ddp,
                    ema_state=ema_state,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    config=config,
                )
                save_training_checkpoint(
                    output_dir / 'checkpoint-last.pth',
                    model=model_without_ddp,
                    ema_state=ema_state,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    config=config,
                )
        barrier()

    if log_writer is not None:
        log_writer.close()


def main():
    args = parse_args()
    config = load_config(args.config, expected_mode='train')
    context = init_distributed(config['runtime'])
    seed_everything(config['runtime']['seed'], context.rank)
    if context.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    task = get_task_spec(config['task'])
    run_train(config, task, context)


if __name__ == '__main__':
    main()

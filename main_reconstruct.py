from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.reconstruct import reconstruct_samples
from models import build_eva
from tasks import get_task_spec
from util.config import load_config, save_resolved_config
from util.runtime import barrier, init_distributed, load_inference_weights, seed_everything


def parse_args():
    parser = argparse.ArgumentParser('Reconstruct samples with EVA for rFID')
    parser.add_argument('--config', required=True, help='YAML reconstruction config')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, expected_mode='reconstruct')
    context = init_distributed(config['runtime'])
    seed_everything(config['runtime']['seed'], context.rank)
    if context.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config['paths']['output_dir'])
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir)
    barrier()

    task = get_task_spec(config['task'])
    model = build_eva(
        config['model']['size'],
        latent_shape=task.latent_shape,
        class_num=task.class_num,
        reparam_dim=config['model']['reparam_dim'],
        grad_checkpointing=False,
    )
    load_inference_weights(
        config['paths']['checkpoint'],
        model,
        use_ema=config['checkpoint']['use_ema'],
    )
    model.to(context.device).eval()
    reconstruct_samples(model, task=task, config=config, context=context)


if __name__ == '__main__':
    main()

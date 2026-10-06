from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.metrics import evaluate_samples
from tasks import get_task_spec
from util.config import load_config, save_resolved_config
from util.runtime import init_distributed, seed_everything


def parse_args():
    parser = argparse.ArgumentParser('Evaluate existing EVA sample artifacts')
    parser.add_argument('--config', required=True, help='YAML metrics config')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, expected_mode='evaluate')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError(
            'main_evaluate.py is intentionally single-process; run it with python, not torchrun'
        )
    context = init_distributed(config['runtime'])
    seed_everything(config['runtime']['seed'])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir)
    evaluate_samples(
        task=get_task_spec(config['task']),
        config=config,
        device=context.device,
    )


if __name__ == '__main__':
    main()

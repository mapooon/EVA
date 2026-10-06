from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from tasks import TaskSpec

from .artifacts import count_samples, load_and_validate_manifest


def _imagenet_metrics(
    samples_path: Path,
    *,
    kind: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    try:
        import torch_fidelity
    except ImportError as exc:
        raise RuntimeError('torch-fidelity is required for ImageNet metrics') from exc

    common = dict(
        input1=str(samples_path),
        cuda=device.type == 'cuda',
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    if kind == 'generation':
        raw = torch_fidelity.calculate_metrics(
            input2=None,
            fid_statistics_file=config['paths']['fid_stats_path'],
            isc=True,
            **common,
        )
        return {
            'fid': float(raw['frechet_inception_distance']),
            'is_mean': float(raw['inception_score_mean']),
            'is_std': float(raw['inception_score_std']),
        }

    reference_path = Path(config['paths']['reference_path'])
    if not reference_path.is_dir():
        raise FileNotFoundError(f'reconstruction reference directory not found: {reference_path}')
    sample_count = count_samples(samples_path, 'imagenet')
    reference_count = count_samples(reference_path, 'imagenet')
    if sample_count != reference_count:
        raise RuntimeError(
            f'reconstruction/reference count mismatch: {sample_count} vs {reference_count}'
        )
    raw = torch_fidelity.calculate_metrics(
        input2=str(reference_path),
        isc=False,
        **common,
    )
    return {'rfid': float(raw['frechet_inception_distance'])}


def _vggsound_metrics(
    samples_path: Path,
    *,
    kind: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    from .vggsound_metrics import compute_vggsound_metrics

    raw = compute_vggsound_metrics(
        samples_path,
        fid_stats_path=config['paths']['fid_stats_path'],
        classifier_path=(config['paths'].get('is_model_path') if kind == 'generation' else None),
        device=device,
    )
    if kind == 'reconstruction':
        return {'rfid': float(raw['fid'])}
    return {key: float(value) for key, value in raw.items()}


@torch.no_grad()
def evaluate_samples(
    *,
    task: TaskSpec,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    samples_path = Path(config['paths']['samples_path'])
    if not samples_path.is_dir():
        raise FileNotFoundError(f'sample directory not found: {samples_path}')
    kind = config['metrics']['kind']
    artifact_type = 'generation' if kind == 'generation' else 'reconstruction'
    manifest = load_and_validate_manifest(
        samples_path,
        task_name=task.name,
        artifact_type=artifact_type,
    )
    if manifest is None:
        print(f'Warning: no manifest found for {samples_path}; evaluating files directly.')

    if task.name == 'imagenet':
        metrics = _imagenet_metrics(samples_path, kind=kind, config=config, device=device)
    else:
        metrics = _vggsound_metrics(samples_path, kind=kind, config=config, device=device)

    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'metrics.json'
    payload = {
        'format_version': 1,
        'task': task.name,
        'kind': kind,
        'samples_path': str(samples_path),
        'metrics': metrics,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')

    message = ', '.join(f'{name}: {value:.4f}' for name, value in metrics.items())
    print(message)
    print(f'Saved metrics to {output_path}')
    return metrics

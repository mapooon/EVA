from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = 'manifest.json'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def count_samples(root: str | Path, task_name: str) -> int:
    extension = '*.png' if task_name == 'imagenet' else '*.wav'
    return sum(1 for _ in Path(root).rglob(extension))


def write_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / MANIFEST_NAME
    temporary = root / f'.{MANIFEST_NAME}.tmp'
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    temporary.replace(destination)
    return destination


def load_and_validate_manifest(
    samples_path: str | Path,
    *,
    task_name: str,
    artifact_type: str,
) -> dict[str, Any] | None:
    samples_path = Path(samples_path)
    candidates = [samples_path / MANIFEST_NAME, samples_path.parent / MANIFEST_NAME]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return None

    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('status') != 'complete':
        raise RuntimeError(f'artifact is not complete according to {manifest_path}')
    if manifest.get('task') != task_name:
        raise ValueError(
            f"manifest task {manifest.get('task')!r} does not match requested task {task_name!r}"
        )
    if manifest.get('artifact_type') != artifact_type:
        raise ValueError(
            f"manifest artifact_type {manifest.get('artifact_type')!r} does not match "
            f'{artifact_type!r}'
        )
    expected = manifest.get('sample_count')
    actual = count_samples(samples_path, task_name)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f'artifact contains {actual} samples, but {manifest_path} records {expected}'
        )
    return manifest


def model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(config['paths']['checkpoint'])
    return {
        'model': {
            'size': config['model']['size'],
            'reparam_dim': config['model']['reparam_dim'],
        },
        'checkpoint': {
            'path': str(checkpoint),
            'sha256': sha256_file(checkpoint),
            'use_ema': bool(config['checkpoint']['use_ema']),
        },
    }

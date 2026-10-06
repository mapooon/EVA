from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    class_num: int
    latent_shape: tuple[int, ...]
    latent_scale: float
    cache_extensions: tuple[str, ...]
    class_mapping_path: Path | None = None


ASSET_ROOT = Path(__file__).resolve().parents[1] / 'assets'


TASKS = {
    'imagenet': TaskSpec(
        name='imagenet',
        class_num=1000,
        latent_shape=(16, 16, 16),
        latent_scale=0.2325,
        cache_extensions=('.npz',),
    ),
    'vggsound': TaskSpec(
        name='vggsound',
        class_num=309,
        latent_shape=(64, 215),
        latent_scale=1.0,
        cache_extensions=('.npy',),
        class_mapping_path=ASSET_ROOT / 'vggsound_class_mapping.json',
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ', '.join(TASKS)
        raise ValueError(f"unknown task {name!r}; expected one of: {choices}") from exc


@lru_cache(maxsize=None)
def load_class_mapping(task: TaskSpec) -> dict[int, dict[str, str]]:
    if task.class_mapping_path is None:
        return {}
    with task.class_mapping_path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    expected_keys = {str(index) for index in range(task.class_num)}
    if set(raw) != expected_keys:
        raise ValueError(
            f'{task.class_mapping_path} must contain exactly indices 0..{task.class_num - 1}'
        )
    mapping = {int(index): value for index, value in raw.items()}
    class_ids = [mapping[index]['class_id'] for index in range(task.class_num)]
    if len(set(class_ids)) != task.class_num:
        raise ValueError(f'{task.class_mapping_path} contains duplicate class IDs')
    return mapping

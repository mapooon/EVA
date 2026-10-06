from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from models.eva import EVA_MODELS
from tasks import get_task_spec


ROOT_KEYS = {
    'defaults', 'task', 'run', 'model', 'paths', 'train', 'checkpoint',
    'generation', 'reconstruction', 'metrics', 'runtime', 'cache', 'tokenizer',
}
SECTION_KEYS = {
    'run': {'mode'},
    'model': {'size', 'reparam_dim'},
    'paths': {
        'data_path', 'cached_path', 'output_dir', 'resume', 'checkpoint',
        'tokenizer_path', 'fid_stats_path', 'is_model_path', 'samples_path',
        'reference_path',
    },
    'train': {
        'epochs', 'batch_size', 'base_lr', 'warmup_epochs', 'weight_decay',
        'grad_clip', 'amp_dtype', 'grad_checkpointing',
    },
    'checkpoint': {'save_frequency', 'use_ema', 'ema_rate'},
    'generation': {
        'batch_size', 'num_samples', 'guidance_scale', 'temperature', 'amp_dtype',
    },
    'reconstruction': {
        'batch_size', 'num_samples', 'amp_dtype',
    },
    'metrics': {'kind'},
    'runtime': {'device', 'seed', 'num_workers', 'pin_memory', 'dist_url'},
    'cache': {
        'split', 'batch_size', 'num_workers', 'skip_existing', 'sample_rate',
        'audio_channels', 'amp_dtype',
    },
    'tokenizer': {'model_id', 'subfolder', 'revision'},
}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = ' -> '.join(str(item) for item in (*stack, path))
        raise ConfigError(f'cyclic config inheritance: {chain}')
    if not path.is_file():
        raise ConfigError(f'config file not found: {path}')

    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f'config root must be a mapping: {path}')

    merged: dict[str, Any] = {}
    defaults = data.pop('defaults', [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list) or not all(isinstance(item, str) for item in defaults):
        raise ConfigError(f'defaults must be a list of YAML paths: {path}')
    for default in defaults:
        merged = _deep_merge(merged, _load_recursive(path.parent / default, (*stack, path)))
    return _deep_merge(merged, data)


def _validate_keys(config: dict[str, Any]) -> None:
    unknown_root = set(config) - ROOT_KEYS
    if unknown_root:
        raise ConfigError(f"unknown root config keys: {', '.join(sorted(unknown_root))}")
    for section, allowed in SECTION_KEYS.items():
        value = config.get(section, {})
        if not isinstance(value, dict):
            raise ConfigError(f'{section} must be a mapping')
        unknown = set(value) - allowed
        if unknown:
            raise ConfigError(f"unknown keys in {section}: {', '.join(sorted(unknown))}")


def _require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f'missing required config value: {dotted_key}')
        value = value[key]
    if value is None or value == '':
        raise ConfigError(f'config value must not be empty: {dotted_key}')
    return value


def validate_config(config: dict[str, Any], expected_mode: str | None = None) -> None:
    _validate_keys(config)
    task = _require(config, 'task')
    get_task_spec(task)
    mode = _require(config, 'run.mode')
    modes = {'train', 'generate', 'reconstruct', 'evaluate', 'cache'}
    if mode not in modes:
        raise ConfigError(f"run.mode must be one of {', '.join(sorted(modes))}; got {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ConfigError(f"this entry point requires run.mode={expected_mode!r}, got {mode!r}")

    runtime = config.get('runtime', {})
    if runtime.get('num_workers', 0) < 0:
        raise ConfigError('runtime.num_workers must be non-negative')

    amp_section = {
        'train': 'train',
        'generate': 'generation',
        'reconstruct': 'reconstruction',
        'cache': 'cache',
    }.get(mode)
    if amp_section is not None:
        amp_dtype = _require(config, f'{amp_section}.amp_dtype')
        if amp_dtype not in {'fp32', 'fp16', 'bf16'}:
            raise ConfigError(f'{amp_section}.amp_dtype must be fp32, fp16, or bf16')

    if mode in {'train', 'generate', 'reconstruct'}:
        size = _require(config, 'model.size')
        if size not in EVA_MODELS:
            raise ConfigError(f"model.size must be one of {', '.join(EVA_MODELS)}, got {size!r}")
        reparam_dim = _require(config, 'model.reparam_dim')
        if not isinstance(reparam_dim, int) or reparam_dim <= 0:
            raise ConfigError('model.reparam_dim must be a positive integer')
        _require(config, 'paths.output_dir')
        if mode == 'train':
            _require(config, 'paths.cached_path')
            if _require(config, 'train.epochs') <= 0:
                raise ConfigError('train.epochs must be positive')
            if _require(config, 'train.batch_size') <= 0:
                raise ConfigError('train.batch_size must be positive')
        elif mode == 'generate':
            _require(config, 'paths.checkpoint')
            if _require(config, 'generation.batch_size') <= 0:
                raise ConfigError('generation.batch_size must be positive')
            if _require(config, 'generation.num_samples') <= 0:
                raise ConfigError('generation.num_samples must be positive')
            if _require(config, 'generation.temperature') <= 0:
                raise ConfigError('generation.temperature must be positive')
            if task == 'imagenet':
                _require(config, 'paths.tokenizer_path')
            else:
                _require(config, 'tokenizer.model_id')
        else:
            _require(config, 'paths.checkpoint')
            if _require(config, 'reconstruction.batch_size') <= 0:
                raise ConfigError('reconstruction.batch_size must be positive')
            num_samples = config['reconstruction'].get('num_samples')
            if num_samples is not None and num_samples <= 0:
                raise ConfigError('reconstruction.num_samples must be null or positive')
            if task == 'imagenet':
                _require(config, 'paths.data_path')
                _require(config, 'paths.tokenizer_path')
            else:
                _require(config, 'paths.cached_path')
                _require(config, 'tokenizer.model_id')
    elif mode == 'cache':
        _require(config, 'paths.data_path')
        _require(config, 'paths.cached_path')
        if _require(config, 'cache.batch_size') <= 0:
            raise ConfigError('cache.batch_size must be positive')
        if task == 'imagenet':
            _require(config, 'paths.tokenizer_path')
        else:
            _require(config, 'tokenizer.model_id')
    else:
        _require(config, 'paths.output_dir')
        _require(config, 'paths.samples_path')
        kind = _require(config, 'metrics.kind')
        if kind not in {'generation', 'reconstruction'}:
            raise ConfigError("metrics.kind must be 'generation' or 'reconstruction'")
        if task == 'imagenet' and kind == 'reconstruction':
            _require(config, 'paths.reference_path')
        else:
            _require(config, 'paths.fid_stats_path')


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if '$' in expanded:
            raise ConfigError(f'unresolved environment variable in config value: {value}')
        return expanded
    return value


def load_config(path: str | Path, expected_mode: str | None = None) -> dict[str, Any]:
    config = _expand_environment(_load_recursive(Path(path), ()))
    validate_config(config, expected_mode=expected_mode)
    return config


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / 'config_resolved.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path

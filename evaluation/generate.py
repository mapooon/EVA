from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tasks import TaskSpec, load_class_mapping
from util.runtime import DistributedContext, autocast_context, barrier

from .artifacts import count_samples, model_provenance, write_manifest
from .io import save_audio, save_images
from .tokenizers import load_tokenizer


def _evaluation_labels(task: TaskSpec, num_samples: int) -> list[int]:
    if task.name == 'vggsound' and num_samples == 15446:
        counts = [50] * 309
        for index in (120, 181, 269, 304):
            counts[index] = 49
        return [label for label, count in enumerate(counts) for _ in range(count)]
    return [index % task.class_num for index in range(num_samples)]


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    *,
    task: TaskSpec,
    config: dict[str, Any],
    context: DistributedContext,
) -> Path:
    generation = config['generation']
    output_dir = Path(config['paths']['output_dir']) / 'samples'
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            'format_version': 1,
            'status': 'in_progress',
            'artifact_type': 'generation',
            'task': task.name,
            'world_size': context.world_size,
            'expected_samples': generation['num_samples'],
            'generation': {
                'guidance_scale': generation['guidance_scale'],
                'temperature': generation['temperature'],
                'seed': config['runtime']['seed'],
            },
            **model_provenance(config),
        }
        write_manifest(Path(config['paths']['output_dir']), manifest)
    barrier()

    tokenizer = load_tokenizer(task, config, context.device)
    class_mapping = load_class_mapping(task)
    model.eval()
    local_ids = list(range(context.rank, generation['num_samples'], context.world_size))
    labels_world = _evaluation_labels(task, generation['num_samples'])
    batch_size = generation['batch_size']

    try:
        from tqdm import tqdm
        batches = range(0, len(local_ids), batch_size)
        batches = tqdm(batches, disable=not context.is_main, desc=f'Generate {task.name}')
    except ImportError:
        batches = range(0, len(local_ids), batch_size)

    for start in batches:
        ids = local_ids[start:start + batch_size]
        labels = torch.tensor(
            [labels_world[sample_id] for sample_id in ids],
            dtype=torch.long,
            device=context.device,
        )
        with autocast_context(context.device, generation['amp_dtype']):
            latents = model.sample_tokens(
                bsz=len(ids),
                labels=labels,
                guidance_scale=generation['guidance_scale'],
                temperature=generation['temperature'],
            )
        decoded = tokenizer.decode(latents.float() / task.latent_scale)
        if task.name == 'imagenet':
            save_images(decoded, ids, output_dir)
        else:
            audio = decoded.sample if hasattr(decoded, 'sample') else decoded
            save_audio(
                audio,
                ids,
                labels,
                output_dir,
                tokenizer.sampling_rate,
                class_mapping,
            )

    barrier()
    if context.is_main:
        sample_count = count_samples(output_dir, task.name)
        if sample_count != generation['num_samples']:
            raise RuntimeError(
                f'generated {sample_count} files, expected {generation["num_samples"]}; '
                'use an empty output directory or remove stale/incomplete artifacts'
            )
        manifest['status'] = 'complete'
        manifest['sample_count'] = sample_count
        write_manifest(Path(config['paths']['output_dir']), manifest)
        print(f'Generated {sample_count} samples in {output_dir}')
    barrier()
    return output_dir

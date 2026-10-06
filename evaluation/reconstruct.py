from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets import CachedLatentDataset
from datasets.cached_latents import DiagonalGaussianDistribution
from tasks import TaskSpec, load_class_mapping
from util.runtime import DistributedContext, autocast_context, barrier

from .artifacts import count_samples, model_provenance, write_manifest
from .io import save_audio, save_images
from .tokenizers import load_tokenizer


class ImageReconstructionDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root: str | Path):
        import torchvision.transforms as transforms
        from util.crop import center_crop_arr

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'ImageNet input directory not found: {self.root}')
        self.paths = sorted(
            path for path in self.root.rglob('*')
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f'no input images found under {self.root}')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        return index, tensor


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


def _reconstruct_latents(model: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    class_embedding = model._label_condition(None, bsz=latents.shape[0])
    tokens = model.patchify(latents)
    z, _mu_q, _logvar_q = model.encode(tokens, class_embedding)
    predictions, _hidden, _mu_p, _logvar_p, _next_mu, _next_logvar = model.decode(
        z,
        class_embedding,
        return_prior_params=False,
    )
    return model.unpatchify(predictions)


def _selected_count(dataset: Dataset, requested: int | None) -> int:
    if requested is None:
        return len(dataset)
    if requested > len(dataset):
        raise ValueError(f'requested {requested} reconstructions, but dataset has {len(dataset)} samples')
    return requested


def _progress(indices: list[int], batch_size: int, *, enabled: bool, task_name: str):
    batches = range(0, len(indices), batch_size)
    try:
        from tqdm import tqdm
        return tqdm(batches, disable=not enabled, desc=f'Reconstruct {task_name}')
    except ImportError:
        return batches


@torch.no_grad()
def _reconstruct_imagenet(
    model: torch.nn.Module,
    tokenizer,
    dataset: ImageReconstructionDataset,
    *,
    task: TaskSpec,
    config: dict[str, Any],
    context: DistributedContext,
    count: int,
    reconstruction_dir: Path,
    target_dir: Path,
) -> None:
    reconstruction = config['reconstruction']
    local_ids = list(range(context.rank, count, context.world_size))
    loader = DataLoader(
        Subset(dataset, local_ids),
        batch_size=reconstruction['batch_size'],
        shuffle=False,
        num_workers=config['runtime']['num_workers'],
        pin_memory=config['runtime']['pin_memory'],
    )
    batches = _progress(local_ids, reconstruction['batch_size'], enabled=context.is_main, task_name=task.name)
    for _start, (ids, images) in zip(batches, loader):
        sample_ids = ids.tolist()
        images = images.to(context.device, non_blocking=True)
        with autocast_context(context.device, reconstruction['amp_dtype']):
            latents = tokenizer.encode(images).sample() * task.latent_scale
            reconstructed = _reconstruct_latents(model, latents)
        decoded = tokenizer.decode(reconstructed.float() / task.latent_scale)
        save_images(decoded, sample_ids, reconstruction_dir)
        save_images(images, sample_ids, target_dir)


@torch.no_grad()
def _reconstruct_vggsound(
    model: torch.nn.Module,
    tokenizer,
    cached: CachedLatentDataset,
    *,
    task: TaskSpec,
    config: dict[str, Any],
    context: DistributedContext,
    count: int,
    reconstruction_dir: Path,
) -> None:
    reconstruction = config['reconstruction']
    dataset = IndexedDataset(cached)
    local_ids = list(range(context.rank, count, context.world_size))
    loader = DataLoader(
        Subset(dataset, local_ids),
        batch_size=reconstruction['batch_size'],
        shuffle=False,
        num_workers=config['runtime']['num_workers'],
        pin_memory=config['runtime']['pin_memory'],
    )
    class_mapping = load_class_mapping(task)
    batches = _progress(local_ids, reconstruction['batch_size'], enabled=context.is_main, task_name=task.name)
    for _start, (ids, moments, labels) in zip(batches, loader):
        sample_ids = ids.tolist()
        moments = moments.to(context.device, non_blocking=True)
        with autocast_context(context.device, reconstruction['amp_dtype']):
            latents = DiagonalGaussianDistribution(moments).sample() * task.latent_scale
            reconstructed = _reconstruct_latents(model, latents)
        decoded = tokenizer.decode(reconstructed.float() / task.latent_scale)
        audio = decoded.sample if hasattr(decoded, 'sample') else decoded
        save_audio(
            audio,
            sample_ids,
            labels,
            reconstruction_dir,
            tokenizer.sampling_rate,
            class_mapping,
        )


@torch.no_grad()
def reconstruct_samples(
    model: torch.nn.Module,
    *,
    task: TaskSpec,
    config: dict[str, Any],
    context: DistributedContext,
) -> Path:
    output_dir = Path(config['paths']['output_dir'])
    reconstruction_dir = output_dir / 'reconstructions'
    target_dir = output_dir / 'targets'
    if task.name == 'imagenet':
        dataset = ImageReconstructionDataset(config['paths']['data_path'])
    else:
        dataset = CachedLatentDataset(config['paths']['cached_path'], task)
        if len(dataset.classes) != task.class_num:
            raise ValueError(
                f'{task.name} expects {task.class_num} class directories, '
                f'found {len(dataset.classes)}'
            )
    count = _selected_count(dataset, config['reconstruction'].get('num_samples'))

    if context.is_main:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        if task.name == 'imagenet':
            target_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            'format_version': 1,
            'status': 'in_progress',
            'artifact_type': 'reconstruction',
            'task': task.name,
            'world_size': context.world_size,
            'expected_samples': count,
            'reconstruction': {
                'seed': config['runtime']['seed'],
                'source_path': config['paths'].get(
                    'data_path' if task.name == 'imagenet' else 'cached_path'
                ),
            },
            **model_provenance(config),
        }
        write_manifest(output_dir, manifest)
    barrier()

    tokenizer = load_tokenizer(task, config, context.device)
    model.eval()
    if task.name == 'imagenet':
        _reconstruct_imagenet(
            model,
            tokenizer,
            dataset,
            task=task,
            config=config,
            context=context,
            count=count,
            reconstruction_dir=reconstruction_dir,
            target_dir=target_dir,
        )
    else:
        _reconstruct_vggsound(
            model,
            tokenizer,
            dataset,
            task=task,
            config=config,
            context=context,
            count=count,
            reconstruction_dir=reconstruction_dir,
        )

    barrier()
    if context.is_main:
        sample_count = count_samples(reconstruction_dir, task.name)
        if sample_count != count:
            raise RuntimeError(f'reconstructed {sample_count} files, expected {count}')
        if task.name == 'imagenet':
            target_count = count_samples(target_dir, task.name)
            if target_count != count:
                raise RuntimeError(f'saved {target_count} targets, expected {count}')
        manifest['status'] = 'complete'
        manifest['sample_count'] = sample_count
        write_manifest(output_dir, manifest)
        print(f'Reconstructed {sample_count} samples in {reconstruction_dir}')
    barrier()
    return reconstruction_dir

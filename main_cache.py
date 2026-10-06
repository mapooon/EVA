from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tasks import get_task_spec
from util.config import load_config
from util.runtime import DistributedContext, autocast_context, barrier, init_distributed, seed_everything


def parse_args():
    parser = argparse.ArgumentParser('Cache tokenizer posterior moments for EVA')
    parser.add_argument('--config', required=True, help='YAML cache config')
    return parser.parse_args()


def _split_root(data_path: str, split: str) -> Path:
    root = Path(data_path)
    candidate = root / split
    return candidate if candidate.is_dir() else root


class ImageFileDataset(Dataset):
    def __init__(self, root: Path):
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from util.crop import center_crop_arr

        transform = transforms.Compose([
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.root = root.resolve()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        source = Path(self.dataset.samples[index][0]).resolve()
        return image, label, source.relative_to(self.root).as_posix()


class AudioFileDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.paths = sorted(self.root.rglob('*.wav'))
        if not self.paths:
            raise RuntimeError(f'no WAV files found under {self.root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return str(path), path.relative_to(self.root).as_posix()


def _audio_collate(batch):
    paths, relative = zip(*batch)
    return list(paths), list(relative)


def _load_audio(path: str, sample_rate: int, channels: int) -> torch.Tensor:
    import soundfile as sf
    from scipy import signal

    waveform, source_rate = sf.read(path, dtype='float32', always_2d=True)
    waveform = waveform.T
    if source_rate != sample_rate:
        waveform = signal.resample_poly(waveform, sample_rate, source_rate, axis=-1).astype(np.float32)
    if waveform.shape[0] == 1 and channels == 2:
        waveform = np.repeat(waveform, 2, axis=0)
    elif waveform.shape[0] >= 2 and channels == 1:
        waveform = waveform[:2].mean(axis=0, keepdims=True)
    elif waveform.shape[0] > channels:
        waveform = waveform[:channels]
    if waveform.shape[0] != channels:
        raise ValueError(f'cannot convert {path} to {channels} channels')
    return torch.from_numpy(np.ascontiguousarray(waveform))


def _sampler(dataset, context: DistributedContext):
    if not context.distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=False,
    )


@torch.no_grad()
def cache_imagenet(config, context: DistributedContext):
    from models.image_vae import AutoencoderKL

    cache = config['cache']
    split = cache['split']
    root = _split_root(config['paths']['data_path'], split)
    dataset = ImageFileDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=cache['batch_size'],
        sampler=_sampler(dataset, context),
        shuffle=False,
        num_workers=cache['num_workers'],
        pin_memory=config['runtime']['pin_memory'],
    )
    tokenizer = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=config['paths']['tokenizer_path'],
    ).to(context.device).eval()
    destination = Path(config['paths']['cached_path']) / split

    for step, (images, _labels, relative_paths) in enumerate(loader):
        pending = [
            not (destination / Path(relative).with_suffix('.npz')).exists()
            for relative in relative_paths
        ]
        if cache['skip_existing'] and not any(pending):
            continue
        images = images.to(context.device, non_blocking=True)
        with autocast_context(context.device, cache['amp_dtype']):
            moments = tokenizer.encode(images).parameters
            moments_flip = tokenizer.encode(images.flip(dims=[3])).parameters
        for index, relative in enumerate(relative_paths):
            save_path = destination / Path(relative).with_suffix('.npz')
            if cache['skip_existing'] and save_path.exists():
                continue
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path,
                moments=moments[index].float().cpu().numpy(),
                moments_flip=moments_flip[index].float().cpu().numpy(),
            )
        if context.is_main and step % 20 == 0:
            print(f'ImageNet cache [{step}/{len(loader)}]')


@torch.no_grad()
def cache_vggsound(config, context: DistributedContext):
    try:
        from diffusers import AutoencoderOobleck
    except ImportError as exc:
        raise RuntimeError('diffusers is required to cache VGGSound latents') from exc

    cache = config['cache']
    split = cache['split']
    root = _split_root(config['paths']['data_path'], split)
    dataset = AudioFileDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=cache['batch_size'],
        sampler=_sampler(dataset, context),
        shuffle=False,
        num_workers=cache['num_workers'],
        pin_memory=config['runtime']['pin_memory'],
        collate_fn=_audio_collate,
    )
    tokenizer = AutoencoderOobleck.from_pretrained(
        config['tokenizer']['model_id'],
        subfolder=config['tokenizer'].get('subfolder', 'vae'),
        revision=config['tokenizer'].get('revision'),
        torch_dtype=torch.float32,
    ).to(context.device).eval()
    destination = Path(config['paths']['cached_path']) / split

    for step, (paths, relative_paths) in enumerate(loader):
        for path, relative in zip(paths, relative_paths):
            save_path = destination / Path(relative).with_suffix('.npy')
            if cache['skip_existing'] and save_path.exists():
                continue
            waveform = _load_audio(path, cache['sample_rate'], cache['audio_channels'])
            waveform = waveform.unsqueeze(0).to(context.device, non_blocking=True)
            posterior = tokenizer.encode(waveform).latent_dist
            moments = torch.cat([posterior.mean[0], posterior.logvar[0]], dim=0)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, moments.float().cpu().numpy().astype(np.float16))
        if context.is_main and step % 20 == 0:
            print(f'VGGSound cache [{step}/{len(loader)}]')


def main():
    args = parse_args()
    config = load_config(args.config, expected_mode='cache')
    context = init_distributed(config['runtime'])
    seed_everything(config['runtime']['seed'], context.rank)
    task = get_task_spec(config['task'])
    if task.name == 'imagenet':
        cache_imagenet(config, context)
    else:
        cache_vggsound(config, context)
    barrier()


if __name__ == '__main__':
    main()

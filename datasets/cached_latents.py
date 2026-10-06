from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tasks import TaskSpec


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> torch.Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean


class CachedLatentDataset(Dataset):
    """Loads cached tokenizer posterior moments from class subdirectories."""

    def __init__(self, root: str | Path, task: TaskSpec):
        self.root = Path(root)
        self.task = task
        if not self.root.is_dir():
            raise FileNotFoundError(f'cached latent directory not found: {self.root}')

        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.classes:
            raise RuntimeError(f'no class directories found under {self.root}')
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.rglob('*')):
                if path.is_file() and path.suffix.lower() in task.cache_extensions:
                    self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            extensions = ', '.join(task.cache_extensions)
            raise RuntimeError(f'no cached latent files ({extensions}) found under {self.root}')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_imagenet(self, path: Path) -> np.ndarray:
        with np.load(path) as data:
            key = 'moments' if torch.rand(()) < 0.5 else 'moments_flip'
            if key not in data:
                raise ValueError(f'{path} does not contain {key!r}')
            moments = np.asarray(data[key], dtype=np.float32)
        return moments

    def _load_vggsound(self, path: Path) -> np.ndarray:
        moments = np.asarray(np.load(path), dtype=np.float32)
        if moments.ndim != 2:
            raise ValueError(f'expected VGGSound moments [2C, T], got {moments.shape} for {path}')
        expected_channels = 2 * self.task.latent_shape[0]
        if moments.shape[0] != expected_channels:
            raise ValueError(
                f'expected {expected_channels} moment channels, got {moments.shape[0]} for {path}'
            )
        target_len = self.task.latent_shape[1]
        length = moments.shape[1]
        if length > target_len:
            start = int(torch.randint(0, length - target_len + 1, ()).item())
            moments = moments[:, start:start + target_len]
        elif length < target_len:
            moments = np.pad(moments, ((0, 0), (0, target_len - length)))
        return moments

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        if self.task.name == 'imagenet':
            moments = self._load_imagenet(path)
        else:
            moments = self._load_vggsound(path)

        expected_shape = (2 * self.task.latent_shape[0], *self.task.latent_shape[1:])
        if moments.shape != expected_shape:
            raise ValueError(f'expected cached moments {expected_shape}, got {moments.shape} for {path}')
        return torch.from_numpy(np.ascontiguousarray(moments)), target

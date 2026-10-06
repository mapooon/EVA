from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_images(images: torch.Tensor, ids: list[int], output_dir: Path) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    images = ((images.float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    for image, sample_id in zip(images, ids):
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(output_dir / f'{sample_id:06d}.png')


def save_audio(
    audio: torch.Tensor,
    ids: list[int],
    labels: torch.Tensor,
    output_dir: Path,
    sample_rate: int,
    class_mapping: dict[int, dict[str, str]],
) -> None:
    import soundfile as sf

    audio = audio.float().cpu()
    labels = labels.cpu().tolist()
    for waveform, sample_id, label in zip(audio, ids, labels):
        class_dir = output_dir / class_mapping[label]['class_id']
        class_dir.mkdir(parents=True, exist_ok=True)
        sf.write(class_dir / f'{sample_id:06d}.wav', waveform.transpose(0, 1).numpy(), sample_rate)

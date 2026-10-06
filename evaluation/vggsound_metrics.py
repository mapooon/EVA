from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy import linalg, signal
from torch.utils.data import DataLoader

from .vggish_preprocess import VGGishExampleDataset, collate_vggish_examples


VGGISH_WEIGHTS_URL = (
    'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth'
)


class VGGish(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for width in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']:
            if width == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ])
                in_channels = width
        self.features = nn.Sequential(*layers)
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.transpose(1, 3).transpose(1, 2).contiguous()
        return self.embeddings(features.view(features.size(0), -1))


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, inputs):
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(outputs + residual)


class VGGSoundClassifier(nn.Module):
    def __init__(self, num_classes=309):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(_conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return self.fc(torch.flatten(outputs, 1))


class AVENet(nn.Module):
    def __init__(self, num_classes=309):
        super().__init__()
        self.audnet = VGGSoundClassifier(num_classes)

    def forward(self, audio):
        return self.audnet(audio)


def _frechet_distance(mu, covariance, reference_mu, reference_covariance):
    covariance_mean = linalg.sqrtm(covariance @ reference_covariance)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    mean_distance = np.sum((mu - reference_mu) ** 2)
    return float(mean_distance + np.trace(covariance + reference_covariance - 2 * covariance_mean))


def _inception_score(probabilities, splits=10):
    splits = max(1, min(splits, len(probabilities)))
    scores = []
    for part in np.array_split(probabilities, splits):
        marginal = part.mean(axis=0, keepdims=True)
        kl = part * (
            np.log(np.clip(part, 1e-12, 1.0)) - np.log(np.clip(marginal, 1e-12, 1.0))
        )
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def _classifier_spectrogram(wav_path: Path):
    samples, sample_rate = sf.read(str(wav_path), dtype='float32')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        samples = signal.resample_poly(samples, 16000, sample_rate).astype(np.float32)
    samples = np.tile(samples, 10)[:160000].clip(-1.0, 1.0)
    _, _, spectrogram = signal.spectrogram(samples, 16000, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-9)
    return torch.tensor(spectrogram, dtype=torch.float32)


@torch.no_grad()
def _vggish_embeddings(wav_files, device):
    model = VGGish()
    weights = torch.hub.load_state_dict_from_url(VGGISH_WEIGHTS_URL, map_location='cpu', progress=True)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    dataset = VGGishExampleDataset(wav_files)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=max(1, min(8, os.cpu_count() or 1)),
        pin_memory=True,
        collate_fn=collate_vggish_examples,
    )
    per_file = [[] for _ in wav_files]
    for examples, owners in loader:
        embeddings = model(examples.to(device, non_blocking=True)).float().cpu()
        for index, owner in enumerate(owners.tolist()):
            per_file[owner].append(embeddings[index])
    return np.stack([torch.stack(chunks).mean(0).numpy() for chunks in per_file])


@torch.no_grad()
def compute_vggsound_metrics(
    wav_root: str | Path,
    *,
    fid_stats_path: str | Path,
    classifier_path: str | Path | None,
    device: torch.device,
):
    wav_files = sorted(Path(wav_root).rglob('*.wav'))
    if not wav_files:
        raise RuntimeError(f'no generated WAV files found under {wav_root}')
    embeddings = _vggish_embeddings(wav_files, device)
    stats = np.load(fid_stats_path)
    reference_mu = stats['mu'] if 'mu' in stats else stats['m']
    reference_covariance = stats['sigma'] if 'sigma' in stats else stats['s']
    fid = _frechet_distance(
        embeddings.mean(axis=0),
        np.cov(embeddings, rowvar=False),
        reference_mu,
        reference_covariance,
    )

    if not classifier_path:
        return {'fid': fid}
    checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if any(name.startswith('module.') for name in state):
        state = {name[7:] if name.startswith('module.') else name: value for name, value in state.items()}
    classifier = AVENet(309).to(device)
    classifier.load_state_dict(state, strict=True)
    classifier.eval()
    probabilities = []
    for wav_file in wav_files:
        inputs = _classifier_spectrogram(wav_file).to(device)[None, None]
        probabilities.append(torch.softmax(classifier(inputs), dim=1)[0].cpu().numpy())
    is_mean, is_std = _inception_score(np.stack(probabilities), splits=10)
    return {'fid': fid, 'is_mean': is_mean, 'is_std': is_std}

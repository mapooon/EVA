import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset

# VGGish feature extraction constants (from torchvggish/vggish_params.py).
VGGISH_SAMPLE_RATE = 16000
VGGISH_STFT_WINDOW_LENGTH_SECONDS = 0.025
VGGISH_STFT_HOP_LENGTH_SECONDS = 0.010
VGGISH_NUM_BANDS = 64
VGGISH_MEL_MIN_HZ = 125
VGGISH_MEL_MAX_HZ = 7500
VGGISH_LOG_OFFSET = 0.01
VGGISH_EXAMPLE_WINDOW_SECONDS = 0.96
VGGISH_EXAMPLE_HOP_SECONDS = 0.96


def _frame_np(data, window_length, hop_length):
    num_samples = data.shape[0]
    if num_samples < window_length:
        return np.empty((0, window_length), dtype=data.dtype)
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def _periodic_hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def _stft_magnitude(signal_np, fft_length, hop_length, window_length):
    frames = _frame_np(signal_np, window_length, hop_length)
    if frames.shape[0] == 0:
        return np.empty((0, fft_length // 2 + 1), dtype=np.float32)
    window = _periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def _hertz_to_mel(frequencies_hertz):
    return 1127.0 * np.log(1.0 + (frequencies_hertz / 700.0))


def _spectrogram_to_mel_matrix(
    num_mel_bins, num_spectrogram_bins, audio_sample_rate, lower_edge_hertz, upper_edge_hertz
):
    nyquist_hertz = audio_sample_rate / 2.0
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = _hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(
        _hertz_to_mel(lower_edge_hertz), _hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def _log_mel_spectrogram(data, audio_sample_rate):
    window_length_samples = int(round(audio_sample_rate * VGGISH_STFT_WINDOW_LENGTH_SECONDS))
    hop_length_samples = int(round(audio_sample_rate * VGGISH_STFT_HOP_LENGTH_SECONDS))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = _stft_magnitude(
        data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples
    )
    if spectrogram.shape[0] == 0:
        return np.empty((0, VGGISH_NUM_BANDS), dtype=np.float32)
    mel_spectrogram = np.dot(
        spectrogram,
        _spectrogram_to_mel_matrix(
            num_mel_bins=VGGISH_NUM_BANDS,
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            lower_edge_hertz=VGGISH_MEL_MIN_HZ,
            upper_edge_hertz=VGGISH_MEL_MAX_HZ,
        ),
    )
    return np.log(mel_spectrogram + VGGISH_LOG_OFFSET)


def wavfile_to_vggish_examples(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    if wav_data.dtype != np.int16:
        raise ValueError(f'Bad sample type for {wav_file}: {wav_data.dtype}')
    samples = wav_data / 32768.0
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sr != VGGISH_SAMPLE_RATE:
        samples = signal.resample_poly(samples, up=VGGISH_SAMPLE_RATE, down=sr).astype(np.float32, copy=False)
    log_mel = _log_mel_spectrogram(samples, audio_sample_rate=VGGISH_SAMPLE_RATE)
    features_sample_rate = 1.0 / VGGISH_STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(VGGISH_EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(VGGISH_EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = _frame_np(log_mel, window_length=example_window_length, hop_length=example_hop_length)
    if log_mel_examples.shape[0] == 0:
        log_mel_examples = np.zeros((1, example_window_length, VGGISH_NUM_BANDS), dtype=np.float32)
    return torch.tensor(log_mel_examples, dtype=torch.float32)[:, None, :, :]


class VGGishExampleDataset(Dataset):
    def __init__(self, wav_files, preprocess_fn=None):
        self.wav_files = list(wav_files)
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else wavfile_to_vggish_examples

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        ex = self.preprocess_fn(str(self.wav_files[idx]))
        return ex, idx


def collate_vggish_examples(batch):
    examples_list = []
    owner = []
    for ex, idx in batch:
        examples_list.append(ex)
        owner.extend([idx] * ex.shape[0])
    examples = torch.cat(examples_list, dim=0)
    owner = torch.tensor(owner, dtype=torch.long)
    return examples, owner

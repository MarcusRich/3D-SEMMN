"""Multimodal AV-MNIST dataset helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from semmn.config import DataConfig, SEMMNConfig


@dataclass
class Batch:
    vision: torch.Tensor
    audio: torch.Tensor
    label: torch.Tensor


class SyntheticAVMNIST(Dataset):
    """Pair MNIST vision with a synthetic spectrogram whose pitch weakly encodes the label.

    Each sample is a 1 s tone (fundamental plus second harmonic) at
    `110 + 22 * (label + jitter)` Hz, with `jitter ~ N(0, audio_pitch_jitter)`, a random
    phase, and additive white noise at `audio_snr_db`. The jitter makes neighbouring
    digits overlap, which caps how well any model can read the label from audio alone;
    the SNR sets how hard the tone is to find. Both are per-sample and fixed by the
    split and index, so a dataset is deterministic.

    The previous generator was a noiseless, fixed-phase tone per label, so a linear
    model read the label off the audio with 100% accuracy and every benchmark
    saturated.
    """

    SAMPLE_RATE = 2048  # Hz; the highest harmonic (2 * 308 Hz) stays below Nyquist.
    N_FFT = 222  # gives exactly 112 frequency bins, the audio encoder's input height.
    HOP = 18

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        audio_snr_db: float = DataConfig.audio_snr_db,
        audio_pitch_jitter: float = DataConfig.audio_pitch_jitter,
    ) -> None:
        tfm = transforms.ToTensor()
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=tfm)
        self.train = train
        self.audio_snr_db = float(audio_snr_db)
        self.audio_pitch_jitter = float(audio_pitch_jitter)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision, label = self.mnist[idx]
        # Seed on (split, index) so the train and test pools never share noise draws.
        seed = 2 * int(idx) + (0 if self.train else 1)
        audio = self._label_to_spectrogram(int(label), seed)
        return vision, audio, torch.tensor(label, dtype=torch.long)

    def _label_to_spectrogram(self, label: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        n = self.SAMPLE_RATE
        t = torch.arange(n, dtype=torch.float32) / self.SAMPLE_RATE
        jitter = self.audio_pitch_jitter * torch.randn((), generator=g).item()
        freq = 110.0 + 22.0 * (float(label) + jitter)
        phase = 2.0 * torch.pi * torch.rand((), generator=g).item()
        tone = torch.sin(2.0 * torch.pi * freq * t + phase) + 0.3 * torch.sin(
            2.0 * torch.pi * 2.0 * freq * t + 2.0 * phase
        )
        noise_power = tone.pow(2).mean() / (10.0 ** (self.audio_snr_db / 10.0))
        wave = tone + noise_power.sqrt() * torch.randn(n, generator=g)
        spec = torch.stft(
            wave,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            win_length=self.N_FFT,
            window=torch.hann_window(self.N_FFT),
            return_complex=True,
        ).abs()  # (112, frames)
        spec = F.interpolate(spec[None, None], size=(112, 112), mode="bilinear", align_corners=False)
        spec = spec.squeeze(0)  # (1, 112, 112)
        return spec / spec.max().clamp_min(1e-6)


class HuggingFaceAVMNIST(Dataset):
    """Attempt to read BLOSSOM-framework/AV-MNIST with robust key fallback."""

    def __init__(self, split: str = "train") -> None:
        from datasets import load_dataset  # lazy import

        self.ds = load_dataset("BLOSSOM-framework/AV-MNIST", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.ds[idx]
        vision = _to_tensor_image(_first_present(row, ("image", "vision", "mnist")))
        audio = _to_tensor_audio(_first_present(row, ("audio", "spectrogram", "sound")))
        label = int(_first_present(row, ("label", "digit", "class")))
        return vision, audio, torch.tensor(label, dtype=torch.long)


def _first_present(row: dict, keys: tuple[str, ...]):
    """Return the value of the first key present in `row`.

    Checks presence (`is not None`), not truthiness: label `0` and array-valued
    fields are valid values, so an `a or b` chain would drop or reject them.
    """
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    raise KeyError(f"row has none of the keys {keys}; available: {sorted(row)}")


def _to_tensor_image(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        x = value.float()
    else:
        x = torch.tensor(value, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = F.interpolate(x.unsqueeze(0), size=(28, 28), mode="bilinear", align_corners=False).squeeze(0)
    return x.clamp(0.0, 1.0)


def _to_tensor_audio(value) -> torch.Tensor:
    if isinstance(value, dict) and "array" in value:
        arr = torch.tensor(value["array"], dtype=torch.float32)
        spec = torch.stft(arr, n_fft=128, hop_length=32, return_complex=True).abs()
        x = spec.unsqueeze(0)
    elif isinstance(value, torch.Tensor):
        x = value.float()
    else:
        x = torch.tensor(value, dtype=torch.float32)
    while x.ndim < 3:
        x = x.unsqueeze(0)
    if x.shape[0] != 1:
        x = x[:1]
    x = F.interpolate(x.unsqueeze(0), size=(112, 112), mode="bilinear", align_corners=False).squeeze(0)
    x = x / (x.max().clamp_min(1e-6))
    return x


def synthetic_avmnist(config: SEMMNConfig, root: str, train: bool, download: bool = True) -> SyntheticAVMNIST:
    return SyntheticAVMNIST(
        root=root,
        train=train,
        download=download,
        audio_snr_db=config.data.audio_snr_db,
        audio_pitch_jitter=config.data.audio_pitch_jitter,
    )


def build_dataloaders(config: SEMMNConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return `(train, val, test)` loaders.

    Val is a fixed subset of the source's training split (`data.val_size`, drawn with
    `data.split_seed`) and is disjoint from train; it is the only split to select on.
    Test is the source's test split (MNIST-t10k for the synthetic data).
    """
    root = "./data"
    source = config.training.dataset_source
    if source == "hf":
        try:
            pool = HuggingFaceAVMNIST(split="train")
            test_ds = HuggingFaceAVMNIST(split="test")
        except Exception as exc:
            if not config.training.allow_synthetic_fallback:
                raise RuntimeError(
                    "training.dataset_source is 'hf' but AV-MNIST could not be loaded "
                    f"({type(exc).__name__}: {exc}). Refusing to train on synthetic audio "
                    "instead. Set training.allow_synthetic_fallback: true to opt into that "
                    "fallback. Loading the Hugging Face dataset contacts the hub."
                ) from exc
            warnings.warn(
                "AV-MNIST failed to load; using synthetic audio because "
                "training.allow_synthetic_fallback is true "
                f"({type(exc).__name__}: {exc})",
                stacklevel=2,
            )
            pool = synthetic_avmnist(config, root, train=True)
            test_ds = synthetic_avmnist(config, root, train=False)
    elif source == "synthetic":
        pool = synthetic_avmnist(config, root, train=True)
        test_ds = synthetic_avmnist(config, root, train=False)
    else:
        raise ValueError(
            "training.dataset_source must be 'synthetic' or 'hf'; "
            f"got {source!r}"
        )
    train_ds, val_ds = split_train_val(pool, config.data.val_size, config.data.split_seed)

    def loader(ds: Dataset, shuffle: bool, workers: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            # The hub read-out uses BatchNorm, which cannot train on a batch of one.
            drop_last=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

    eval_workers = max(1, config.training.num_workers // 2) if config.training.num_workers else 0
    return (
        loader(train_ds, True, config.training.num_workers),
        loader(val_ds, False, eval_workers),
        loader(test_ds, False, eval_workers),
    )


def split_train_val(pool: Dataset, val_size: int, seed: int) -> tuple[Subset, Subset]:
    """Disjoint train/val subsets of `pool`; the val indices depend only on `seed`."""
    n = len(pool)
    if not 0 < val_size < n:
        raise ValueError(f"val_size must be in (0, {n}); got {val_size}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(pool, perm[val_size:]), Subset(pool, perm[:val_size])

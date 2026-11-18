"""Multimodal AV-MNIST dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from semmn.config import SEMMNConfig


@dataclass
class Batch:
    vision: torch.Tensor
    audio: torch.Tensor
    label: torch.Tensor


class SyntheticAVMNIST(Dataset):
    """Pair MNIST vision with synthetic spectrogram-like audio."""

    def __init__(self, root: str, train: bool = True, download: bool = True) -> None:
        tfm = transforms.ToTensor()
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=tfm)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision, label = self.mnist[idx]
        audio = self._label_to_spectrogram(int(label), idx)
        return vision, audio, torch.tensor(label, dtype=torch.long)

    def _label_to_spectrogram(self, label: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        t = torch.linspace(0, 1, steps=224)
        base_freq = 110.0 + 22.0 * float(label)
        sine = (
            torch.sin(2.0 * torch.pi * base_freq * t)
            + 0.3 * torch.sin(2.0 * torch.pi * 2.0 * base_freq * t)
            + 0.2 * torch.randn(224, generator=g)
        )
        spec = torch.stft(
            sine,
            n_fft=64,
            hop_length=16,
            win_length=64,
            return_complex=True,
        ).abs()
        spec = spec.unsqueeze(0)  # (1, F, T)
        spec = F.interpolate(spec.unsqueeze(0), size=(112, 112), mode="bilinear", align_corners=False)
        spec = spec.squeeze(0)
        spec = spec / (spec.max().clamp_min(1e-6))
        return spec


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


def build_dataloaders(config: SEMMNConfig) -> tuple[DataLoader, DataLoader]:
    root = "./data"
    if config.training.dataset_source == "hf":
        try:
            train_ds = HuggingFaceAVMNIST(split="train")
            val_ds = HuggingFaceAVMNIST(split="test")
        except Exception:
            train_ds = SyntheticAVMNIST(root=root, train=True)
            val_ds = SyntheticAVMNIST(root=root, train=False)
    else:
        train_ds = SyntheticAVMNIST(root=root, train=True)
        val_ds = SyntheticAVMNIST(root=root, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=max(1, config.training.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader

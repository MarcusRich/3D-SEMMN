"""Row decoding for the Hugging Face AV-MNIST adapter."""

from __future__ import annotations

import numpy as np
import pytest

from semmn.dataset import HuggingFaceAVMNIST


def _dataset(rows: list[dict]) -> HuggingFaceAVMNIST:
    # Bypass __init__ so the test never downloads anything.
    ds = HuggingFaceAVMNIST.__new__(HuggingFaceAVMNIST)
    ds.ds = rows
    return ds


def _image() -> list[list[float]]:
    return [[0.5] * 28 for _ in range(28)]


def _audio() -> list[list[float]]:
    return [[0.1] * 112 for _ in range(112)]


def test_label_zero_is_loaded() -> None:
    vision, audio, label = _dataset([{"image": _image(), "audio": _audio(), "label": 0}])[0]
    assert label.item() == 0
    assert vision.shape == (1, 28, 28)
    assert audio.shape == (1, 112, 112)


def test_numpy_fields_are_loaded() -> None:
    row = {
        "image": np.full((28, 28), 0.5, dtype=np.float32),
        "audio": np.full((112, 112), 0.1, dtype=np.float32),
        "label": np.int64(0),
    }
    _, _, label = _dataset([row])[0]
    assert label.item() == 0


def test_fallback_keys_are_used_when_primary_missing() -> None:
    row = {"vision": _image(), "spectrogram": _audio(), "digit": 7}
    _, _, label = _dataset([row])[0]
    assert label.item() == 7


def test_hf_source_does_not_silently_train_on_synthetic(tiny_config, monkeypatch) -> None:
    from semmn.dataset import build_dataloaders

    tiny_config.training.dataset_source = "hf"

    class _Boom:
        def __init__(self, split: str = "train") -> None:
            raise OSError("offline")

    monkeypatch.setattr("semmn.dataset.HuggingFaceAVMNIST", _Boom)
    with pytest.raises(RuntimeError, match="synthetic"):
        build_dataloaders(tiny_config)


def test_unknown_dataset_source_is_rejected(tiny_config) -> None:
    from semmn.dataset import build_dataloaders

    tiny_config.training.dataset_source = "HF"
    with pytest.raises(ValueError, match="dataset_source"):
        build_dataloaders(tiny_config)


def test_synthetic_fallback_is_opt_in(tiny_config, monkeypatch) -> None:
    from semmn.dataset import build_dataloaders

    tiny_config.training.dataset_source = "hf"
    tiny_config.training.allow_synthetic_fallback = True

    class _Boom:
        def __init__(self, split: str = "train") -> None:
            raise OSError("offline")

    monkeypatch.setattr("semmn.dataset.HuggingFaceAVMNIST", _Boom)
    with pytest.warns(UserWarning, match="synthetic"):
        loaders = build_dataloaders(tiny_config)
    assert loaders[2].dataset.mnist.train is False


def test_missing_label_raises_key_error() -> None:
    with pytest.raises(KeyError, match="label"):
        _dataset([{"image": _image(), "audio": _audio()}])[0]

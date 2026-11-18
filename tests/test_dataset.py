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


def test_missing_label_raises_key_error() -> None:
    with pytest.raises(KeyError, match="label"):
        _dataset([{"image": _image(), "audio": _audio()}])[0]

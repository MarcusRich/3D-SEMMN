"""The synthetic audio must be informative but must not leak the label."""

from __future__ import annotations

import torch

import eval_small
from semmn.dataset import SyntheticAVMNIST

SIZES = {"data_seed": 7, "train_size": 400, "val_size": 200, "test_size": 20}


def _audio_only_val_accuracy(data_cfg=None) -> float:
    kwargs = {} if data_cfg is None else {"data_cfg": data_cfg}
    splits = eval_small.build_splits(SIZES, **kwargs)
    return eval_small.logreg(splits, "audio")[1]["val_accuracy"]


def test_audio_alone_does_not_saturate_by_default() -> None:
    # The old generator (a noiseless tone per label) gave 1.0 here.
    acc = _audio_only_val_accuracy()
    assert acc < 0.9, f"audio-only logreg reached {acc:.3f}; the audio leaks the label"


def test_snr_knob_controls_how_informative_the_audio_is() -> None:
    from semmn.config import DataConfig

    clean = _audio_only_val_accuracy(DataConfig(audio_snr_db=20.0, audio_pitch_jitter=0.0))
    default = _audio_only_val_accuracy(DataConfig())
    buried = _audio_only_val_accuracy(DataConfig(audio_snr_db=-30.0))
    assert clean > 0.95 > default
    assert default > buried + 0.3
    assert buried < 0.3  # chance is 0.1


def test_train_and_test_pools_draw_different_noise_for_the_same_index() -> None:
    train = SyntheticAVMNIST("./data", train=True, download=False)
    test = SyntheticAVMNIST("./data", train=False, download=False)
    i = next(i for i in range(100) if train.mnist.targets[i] == test.mnist.targets[i])
    assert not torch.allclose(train[i][1], test[i][1])


def test_audio_is_deterministic_per_sample() -> None:
    ds = SyntheticAVMNIST("./data", train=True, download=False)
    assert torch.equal(ds[3][1], ds[3][1])
    assert ds[3][1].shape == (1, 112, 112)

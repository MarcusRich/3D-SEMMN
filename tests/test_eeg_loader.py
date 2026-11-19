"""Broderick loader on synthetic files shaped like the real dataset."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import io as sio

from semmn.eeg import broderick as br

FS = 128.0
N = int(60 * FS)


def _write_condition(root, n_subjects=2, n_runs=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    for s in range(1, n_subjects + 1):
        d = root / "EEG" / f"Subject{s}"
        d.mkdir(parents=True)
        for r in range(1, n_runs + 1):
            common = 50.0 * np.sin(2 * np.pi * 0.3 * t) + 20.0 * rng.standard_normal(N)
            neural = np.sin(2 * np.pi * 4.0 * t)[:, None] * np.linspace(1, 2, 128)
            eeg = neural + common[:, None] + 0.01 * rng.standard_normal((N, 128))
            mast = common[:, None] + 0.01 * rng.standard_normal((N, 2))
            sio.savemat(d / f"Subject{s}_Run{r}.mat", {"eegData": eeg, "mastoids": mast, "fs": FS})
    (root / "Stimuli" / "Text").mkdir(parents=True)
    (root / "Stimuli" / "Envelopes").mkdir(parents=True)
    for r in range(1, n_runs + 1):
        words = np.array(["the", "old", "man", "was", "thin"], dtype=object)
        on = np.array([0.5, 1.0, 1.75, 2.0, 59.99])
        sio.savemat(
            root / "Stimuli" / "Text" / f"Run{r}.mat",
            {"wordVec": words, "onset_time": on, "offset_time": on + 0.2,
             "sentence_boundaries": np.array([1.9, 59.0, 60.0])},
        )
        env = np.abs(np.sin(2 * np.pi * 3.0 * t)) + 0.1
        sio.savemat(root / "Stimuli" / "Envelopes" / f"audio{r}_128Hz.mat", {"env": env})  # no fs, as released


def test_discovery_and_shapes(tmp_path):
    _write_condition(tmp_path)
    eeg_files = br.discover_eeg(tmp_path)
    assert sorted(eeg_files) == [1, 2] and sorted(eeg_files[1]) == [1, 2]
    text, env = br.discover_stimuli(tmp_path)
    assert sorted(text) == [1, 2] and sorted(env) == [1, 2]
    eeg, mast, fs = br.load_eeg_run(eeg_files[1][2])
    assert eeg.shape == (N, 128) and mast.shape == (N, 2) and fs == FS
    words = br.load_words(text[1])
    assert words.words == ("the", "old", "man", "was", "thin")
    np.testing.assert_allclose(words.offsets - words.onsets, 0.2)
    np.testing.assert_array_equal(br.load_sentence_ids(text[1]), [0, 0, 0, 1, 2])
    e, efs = br.load_envelope(env[1])
    assert e.shape == (N,) and efs == FS


def test_preprocess_references_filters_and_resamples(tmp_path):
    _write_condition(tmp_path)
    eeg, mast, fs = br.load_eeg_run(br.discover_eeg(tmp_path)[1][1])
    x = br.preprocess_eeg(eeg, mast, fs, band=(1.0, 8.0), fs_out=64.0, standardise=False)
    assert x.shape == (N // 2, 128) and x.dtype == np.float32
    # The common mode (0.3 Hz drift and broadband noise on every electrode) is removed by
    # the mastoid reference; what remains is the 4 Hz neural sine with its gain.
    t = np.arange(N // 2) / 64.0
    ref = np.sin(2 * np.pi * 4.0 * t)
    mid = slice(200, -200)  # skip filter edges
    for ch, gain in [(0, 1.0), (127, 2.0)]:
        assert np.corrcoef(x[mid, ch], ref[mid])[0, 1] > 0.99
        assert np.std(x[mid, ch]) == pytest.approx(gain / np.sqrt(2), rel=0.05)
    z = br.preprocess_eeg(eeg, mast, fs)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-3)


def test_bandpass_rejects_out_of_band():
    t = np.arange(int(40 * FS)) / FS
    x = np.sin(2 * np.pi * 4 * t) + np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 0.1 * t)
    y = br.bandpass(x[:, None], FS, (1.0, 8.0))[:, 0]
    mid = slice(1000, -1000)
    assert np.corrcoef(y[mid], np.sin(2 * np.pi * 4 * t)[mid])[0, 1] > 0.99


def test_word_onsets_align_to_samples(tmp_path):
    _write_condition(tmp_path)
    words = br.load_words(br.discover_stimuli(tmp_path)[0][1])
    idx = br.onset_samples(words.onsets, 64.0, n=60 * 64)
    np.testing.assert_array_equal(idx, [32, 64, 112, 128, 3839])
    # An onset that rounds past the end of the run is dropped, not wrapped.
    assert len(br.onset_samples(words.onsets, 64.0, n=3839)) == 4


def test_time_reversed_words():
    w = br.Words(("a", "b"), np.array([1.0, 3.0]), np.array([2.0, 3.5]))
    r = w.time_reversed(10.0)
    assert r.words == ("b", "a")
    np.testing.assert_allclose(r.onsets, [6.5, 8.0])
    np.testing.assert_allclose(r.offsets, [7.0, 9.0])


def test_short_recordings_keep_the_channel_axis(tmp_path):
    # 40 samples is shorter than 128 channels. Inferring the channel axis from
    # which side is longer transposed this into 128 time points of 40 channels.
    n = 40
    eeg = np.arange(n * 128, dtype=float).reshape(n, 128)
    mast = np.arange(n * 2, dtype=float).reshape(n, 2)
    sio.savemat(tmp_path / "time_major.mat", {"eegData": eeg, "mastoids": mast, "fs": 128})
    sio.savemat(tmp_path / "channel_major.mat", {"eegData": eeg.T, "mastoids": mast.T, "fs": 128})
    a, ma, fs = br.load_eeg_run(tmp_path / "time_major.mat")
    b, mb, _ = br.load_eeg_run(tmp_path / "channel_major.mat")
    assert a.shape == (n, 128) and ma.shape == (n, 2) and fs == 128
    np.testing.assert_array_equal(b, a)
    np.testing.assert_array_equal(mb, ma)
    square = tmp_path / "square.mat"
    sio.savemat(square, {"eegData": np.zeros((128, 128)), "mastoids": np.zeros((128, 2)), "fs": 128})
    with pytest.raises(ValueError, match="ambiguous"):
        br.load_eeg_run(square)


def test_v73_files_fail_clearly(tmp_path):
    p = tmp_path / "x.mat"
    p.write_bytes(b"MATLAB 7.3 MAT-file, Platform: x" + b"\0" * 200)
    with pytest.raises(ValueError, match="h5py"):
        br.load_mat(p)

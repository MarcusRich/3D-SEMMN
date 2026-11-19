"""End-to-end smoke test of eeg_p3.py on a synthetic dataset with a planted response."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scipy import io as sio
from scipy import signal

import eeg_p3

FS = 128.0
N = int(60 * FS)


def _make_dataset(root, n_subjects=3, n_runs=4, with_embedding=True):
    rng = np.random.default_rng(0)
    cond = root / eeg_p3.CONDITIONS["forward"]
    (cond / "Stimuli" / "Text").mkdir(parents=True)
    (cond / "Stimuli" / "Envelopes").mkdir(parents=True)
    kernel = np.exp(-np.arange(20) / 6.0) * np.sin(np.arange(20) / 3.0)
    n_words = []
    for r in range(1, n_runs + 1):
        env = np.abs(signal.sosfiltfilt(signal.butter(2, 6, fs=FS, output="sos"), rng.standard_normal(N)))
        sio.savemat(cond / "Stimuli" / "Envelopes" / f"audio{r}_128Hz.mat", {"env": env})
        on = np.sort(rng.uniform(0.5, 59, 120))
        words = np.array([f"w{i}" for i in range(len(on))], dtype=object)
        sio.savemat(cond / "Stimuli" / "Text" / f"Run{r}.mat",
                    {"wordVec": words, "onset_time": on, "offset_time": on + 0.2})
        n_words.append(len(on))
        drive = np.convolve(env - env.mean(), kernel)[:N]
        for s in range(1, n_subjects + 1):
            d = cond / "EEG" / f"Subject{s}"
            d.mkdir(parents=True, exist_ok=True)
            gain = np.linspace(0.2, 1.0, 128)
            eeg = drive[:, None] * gain + 0.5 * rng.standard_normal((N, 128)) * drive.std()
            sio.savemat(d / f"Subject{s}_Run{r}.mat",
                        {"eegData": eeg, "mastoids": np.zeros((N, 2)), "fs": FS})
    total = sum(n_words)
    lm_path = root / "cache" / "lm_fake.npz"
    lm_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"embedding": rng.standard_normal((total, 12))} if with_embedding else {}
    np.savez(lm_path, surprisal=rng.gamma(2, 2, total), dissimilarity=rng.uniform(0, 1, total),
             run=np.concatenate([[r] * n for r, n in zip(range(1, n_runs + 1), n_words)]), **extra)
    return lm_path


@pytest.mark.parametrize("with_embedding", [True, False])
def test_pipeline_recovers_planted_envelope_response(tmp_path, with_embedding):
    pytest.importorskip("mne")
    lm_path = _make_dataset(tmp_path, with_embedding=with_embedding)
    eeg_p3.prep(tmp_path, "forward", [1, 2, 3])
    out = tmp_path / "out"
    eeg_p3.encode(tmp_path, "forward", [1, 2, 3], [1, 2, 3], [4], 3, lm_path, 4, out, [5.0],
                  ["A_env", "B_env_onset", "C_env_onset_lm", "L_onset_lm"])
    with (out / "encoding_per_subject_forward.csv").open() as fh:
        enc = list(csv.DictReader(fh))
    a128 = [float(r["mean_r"]) for r in enc if r["tier"] == "A_env" and r["montage"] == "biosemi128"]
    l128 = [float(r["mean_r"]) for r in enc if r["tier"] == "L_onset_lm" and r["montage"] == "biosemi128"]
    assert len(a128) == 3 and min(a128) > 0.5  # the planted envelope response is recovered
    assert max(abs(x) for x in l128) < 0.15  # random word features carry nothing
    with (out / "decoding_per_subject_forward.csv").open() as fh:
        dec = list(csv.DictReader(fh))
    top1 = [float(r["top1"]) for r in dec if r["tier"] == "A_env" and r["montage"] == "biosemi128"]
    assert min(top1) > 0.9 and int(dec[0]["n_candidates"]) == 12
    folds = json.loads((out / "folds_forward.json").read_text())["folds"]
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert len(f["best8"]) == 8
    assert "Top 10 channels" in (out / "scalp_forward.md").read_text()
    eeg_p3.summarize(out)
    summary = (out / "summary.md").read_text()
    assert "word-feature gain" in summary and "segment identification" in summary
    assert "N400 check" in summary and "each montage against 128 channels" in summary
    with (out / "n400_trf_per_subject_forward.csv").open() as fh:
        n4 = list(csv.DictReader(fh))
    assert {r["feature"] for r in n4} == {"surprisal", "dissimilarity"} and len(n4) == 2 * 4 * 3

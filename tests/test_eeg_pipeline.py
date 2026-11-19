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


def _make_dataset(root, n_subjects=3, n_runs=4, with_embedding=True, word_gain=0.0):
    rng = np.random.default_rng(0)
    surprisal = []
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
        sur = rng.gamma(2, 2, len(on))
        surprisal.append(sur)
        if word_gain:  # a planted response to word surprisal, 100-250 ms after onset
            imp = np.zeros(N)
            np.add.at(imp, np.round(on * FS).astype(int), sur - sur.mean())
            drive = drive + word_gain * np.convolve(imp, np.r_[np.zeros(13), np.hanning(20)])[:N]
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
    np.savez(lm_path, surprisal=np.concatenate(surprisal), dissimilarity=rng.uniform(0, 1, total),
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


def _fake_distractors(root, lm_path, subjects, train_runs, test_runs, k, win_s):
    """Distractor file in the distract stage's format, with random candidate features."""
    from semmn.eeg import decode, lm

    rng = np.random.default_rng(5)
    design = eeg_p3.Design(root, "forward", subjects, lm_path, train_runs, n_pcs=0)
    z = np.load(lm_path)
    basis = lm.Pca.fit(z["embedding"][np.isin(z["run"], train_runs)], 64)
    win = int(win_s * eeg_p3.FS)
    rec = {"run": [], "start": [], "lo": [], "hi": []}
    sur, coords = [], []
    for r in test_runs:
        for st in decode.window_starts(design.length[r], win):
            pos = eeg_p3.window_slots(design, r, int(st), win)
            flat = design.word_flat[r][pos]
            for key, v in (("run", r), ("start", int(st)), ("lo", int(flat[0])), ("hi", int(flat[-1]) + 1)):
                rec[key].append(v)
            sur.append(rng.gamma(2, 2, (k, len(pos))))
            coords.append(rng.standard_normal((k, len(pos), basis.components.shape[0])))
    out = root / "cache" / "fake_distractors.npz"
    eeg_p3.save_distractors(out, rec, sur, coords, basis, win, train_runs)
    return out


def test_distractor_ranking_finds_planted_word_responses(tmp_path):
    pytest.importorskip("mne")
    lm_path = _make_dataset(tmp_path, word_gain=3.0)
    eeg_p3.prep(tmp_path, "forward", [1, 2, 3])
    dis = _fake_distractors(tmp_path, lm_path, [1, 2, 3], [1, 2, 3], [4], k=9, win_s=10.0)
    out = tmp_path / "out"
    eeg_p3.encode(tmp_path, "forward", [1, 2, 3], [1, 2, 3], [4], 3, lm_path, 4, out, [10.0],
                  ["A_env", "B_env_onset", "S_env_onset_sur", "E_env_onset_sur_emb"],
                  main_tier="E_env_onset_sur_emb", distractor_path=dis, n_shuffles=50)
    with (out / "distractor_per_subject_forward.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    # Tiers with word features are ranked on the features they use; A and B cannot tell
    # acoustically identical candidates apart, so they are not ranked at all.
    assert {r["tier"] for r in rows} == {"S_env_onset_sur", "E_env_onset_sur_emb"}
    e128 = [r for r in rows if r["montage"] == "biosemi128" and r["tier"] == "S_env_onset_sur"]
    assert len(e128) == 3 and all(int(r["n_candidates"]) == 10 and int(r["n_windows"]) == 6 for r in e128)
    assert min(float(r["top1"]) for r in e128) >= 0.5  # planted surprisal response is found
    assert max(abs(float(r["shuffled_top1"]) - 0.1) for r in e128) < 0.08  # shuffled EEG: chance
    assert {r["prior_top1"] for r in rows} == {rows[0]["prior_top1"]}  # the prior ignores the EEG
    folds = json.loads((out / "folds_forward.json").read_text())
    assert folds["main_tier"] == "E_env_onset_sur_emb" and folds["distractors"] == dis.name
    # The fake word-feature file also has dissimilarity, so tier E is not the full design
    # and the N400 kernels (which index the full design) are skipped rather than mis-read.
    assert not (out / "n400_trf_per_subject_forward.csv").exists()

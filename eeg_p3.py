#!/usr/bin/env python
"""P3: EEG response-surface models on Broderick et al. (2018).

Stages (run in order; each caches what the next needs under <data-root>/cache):
  prep     preprocess EEG and envelopes of a condition to float32 arrays
  words    language-model features for every word of the story
  encode   leave-subjects-out ridge TRFs per feature tier and montage, plus decoding

Only summary tables are written to --out; arrays stay in the cache outside the repo.

Example:
  python -I eeg_p3.py --data-root "$SEMMN_BRODERICK" encode --subjects 1-8
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` drops the script dir

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from semmn.eeg import broderick as br  # noqa: E402
from semmn.eeg import decode, lm, trf  # noqa: E402
from semmn.eeg import features as ft  # noqa: E402
from semmn.eeg import montages as mt  # noqa: E402

FS = 64.0
BAND = (1.0, 8.0)
ALPHAS = [10.0**k for k in range(-3, 4)]
CONDITIONS = {"forward": "Natural Speech", "reverse": "Natural Speech - Reverse"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_ints(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cache_dir(root: Path, condition: str) -> Path:
    d = root / "cache" / f"{condition}_b{BAND[0]:g}-{BAND[1]:g}_fs{FS:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- prep


def prep(root: Path, condition: str, subjects: list[int]) -> None:
    eeg_files = br.discover_eeg(root / CONDITIONS[condition])
    # The reversed-speech release has no stimulus folder: its audio is the forward
    # audio played backwards, so its envelope is the forward envelope reversed.
    _, env = br.discover_stimuli(root / CONDITIONS["forward"])
    cd = cache_dir(root, condition)
    for r, path in sorted(env.items()):
        out = cd / f"env_r{r:02d}.npy"
        if not out.exists():
            e, fs = br.load_envelope(path)
            if condition == "reverse":
                e = e[::-1].copy()
            np.save(out, br.preprocess_envelope(e, fs, BAND, FS))
    for s in subjects:
        if s not in eeg_files:
            log(f"{condition}: subject {s} not found, skipped")
            continue
        for r, path in sorted(eeg_files[s].items()):
            out = cd / f"s{s:02d}_r{r:02d}.npy"
            if out.exists():
                continue
            eeg, mast, fs = br.load_eeg_run(path)
            np.save(out, br.preprocess_eeg(eeg, mast, fs, BAND, FS))
        log(f"{condition}: subject {s} cached ({len(eeg_files[s])} runs)")


# ---------------------------------------------------------------- words


def story_words(root: Path) -> dict[int, br.Words]:
    text, _ = br.discover_stimuli(root / CONDITIONS["forward"])
    return {r: br.load_words(p) for r, p in sorted(text.items())}


def story_sentences(root: Path) -> dict[int, np.ndarray]:
    """Sentence index of every word, unique across the story (runs follow in order)."""
    text, _ = br.discover_stimuli(root / CONDITIONS["forward"])
    out, offset = {}, 0
    for r, p in sorted(text.items()):
        ids = br.load_sentence_ids(p)
        out[r] = ids + offset
        offset = int(out[r].max()) + 1
    return out


def minilm_snapshot() -> Path:
    """The locally cached all-MiniLM-L6-v2 snapshot (read only; nothing is downloaded).

    When several snapshots are cached, the most recently modified one is used.
    Lexicographic order of the snapshot hashes is not recency.
    """
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snaps = list((hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError("all-MiniLM-L6-v2 is not in the local Hugging Face cache")
    return max(snaps, key=lambda p: p.stat().st_mtime)


def words_stage(root: Path, backend: str, layer: int, download: bool = False) -> Path:
    """Per-word features for the whole story, in run order.

    The release lists content words only (``wordVec``), not the full transcript.

    ``local`` (P3 default): leave-run-out story n-gram surprisal, static word vectors
    (mean WordPiece input embedding of the locally cached all-MiniLM-L6-v2), and
    Broderick et al.'s semantic dissimilarity from those vectors. No network forward pass.
    ``gpt2``: GPT-2 small surprisal and layer-``layer`` states (needs transformers). It is
    fed the content-word stream, which is off-distribution for GPT-2; the full transcript
    is not part of the release. The model is read from the local Hugging Face cache unless
    ``download`` is true.
    """
    words = story_words(root)
    run_of = np.concatenate([[r] * len(words[r]) for r in sorted(words)])
    flat = [w for r in sorted(words) for w in words[r].words]
    if backend == "local":
        out = root / "cache" / "lm_local.npz"
        sur = lm.ngram_surprisal({r: words[r].words for r in words})
        snap = minilm_snapshot()
        table = lm.read_safetensors(snap / "model.safetensors", "embeddings.word_embeddings.weight")
        with (snap / "vocab.txt").open(encoding="utf-8") as fh:
            vocab = {t.rstrip("\r\n"): i for i, t in enumerate(fh)}
        vec = lm.static_word_vectors(flat, table, vocab)
        sent = story_sentences(root)
        dis = lm.semantic_dissimilarity(vec, np.concatenate([sent[r] for r in sorted(words)]))
        np.savez(out, surprisal=np.concatenate([sur[r] for r in sorted(words)]), dissimilarity=dis,
                 embedding=vec.astype(np.float32), run=run_of)
        log(f"local word features: {len(flat)} content words")
        return out
    out = root / "cache" / f"lm_gpt2_layer{layer}.npz"
    if out.exists():
        return out
    import torch

    torch.set_num_threads(4)
    log("GPT-2 is fed the content-word stream only (no function words): off-distribution input")
    if download:
        log("GPT-2 download is enabled; from_pretrained may contact the Hugging Face hub")
    model, tok = lm.load_gpt2("gpt2", local_files_only=not download)
    surprisal, emb = lm.gpt2_word_features(flat, model, tok.encode, layer=layer)
    np.savez(out, surprisal=surprisal, embedding=emb.astype(np.float32), run=run_of)
    log(f"GPT-2 features: {len(flat)} words, layer {layer}")
    return out


# ---------------------------------------------------------------- design


class Design:
    """Per-run stimulus features, lagged design and column indices of each tier."""

    def __init__(self, root: Path, condition: str, subjects: list[int], lm_path: Path,
                 train_runs: list[int], n_pcs: int) -> None:
        self.cd = cache_dir(root, condition)
        words = story_words(root)
        z = np.load(lm_path)
        run_of = z["run"]
        tr = np.isin(run_of, train_runs)
        # Word features, scaled with training-run words only.
        word_feats: dict[str, np.ndarray] = {}
        for name in ("surprisal", "dissimilarity"):
            if name in z.files:
                v = z[name]
                word_feats[name] = ft.zscore_with_train(v, v[tr])[:, None]
        if "embedding" in z.files and n_pcs > 0:
            word_feats["embedding"] = lm.Pca.fit(z["embedding"][tr], n_pcs).transform(z["embedding"])
        self.width = {"env": 1, "onset": 1, **{k: v.shape[1] for k, v in word_feats.items()}}
        self.order = [n for n in ("env", "onset") + ft.WORD_FEATURES if n in self.width]
        self.first, self.count = ft.n_lags(FS, 0.0, 0.6)
        self.runs = sorted(r for r in words if (self.cd / f"env_r{r:02d}.npy").exists())
        self.subjects = subjects
        self._memo: dict[tuple[int, int], np.ndarray] = {}
        self.length: dict[int, int] = {}
        self.x: dict[int, np.ndarray] = {}
        for r in self.runs:
            env = np.load(self.cd / f"env_r{r:02d}.npy")
            n = min([len(env)] + [np.load(self.cd / f"s{s:02d}_r{r:02d}.npy", mmap_mode="r").shape[0]
                                  for s in subjects])
            self.length[r] = n
            w = words[r]
            order = np.arange(len(w))
            if condition == "reverse":
                w = w.time_reversed(len(env) / FS)
                order = order[::-1]
            idx = np.round(w.onsets * FS).astype(int)
            sel = np.nonzero(run_of == r)[0][order]
            stim = {"env": env[:n], "onset": ft.impulse_train(n, idx)}
            for name, v in word_feats.items():
                stim[name] = ft.impulse_train(n, idx, v[sel])
            self.x[r] = ft.lag_matrix(ft.stimulus_matrix(stim, tuple(self.order)), self.first, self.count)

    def feature_index(self, name: str) -> list[int]:
        """Positions of a feature's columns in the un-lagged stimulus matrix."""
        start = sum(self.width[n] for n in self.order[: self.order.index(name)])
        return list(range(start, start + self.width[name]))

    def cols(self, tier: str) -> np.ndarray:
        feats = [i for name in ft.TIERS[tier] if name in self.width for i in self.feature_index(name)]
        return np.array([f * self.count + j for f in feats for j in range(self.count)])

    def eeg(self, s: int, r: int, keep: bool = False) -> np.ndarray:
        """EEG of subject s, run r, cut to the run's shared length; ``keep`` memoises it."""
        if (s, r) in self._memo:
            return self._memo[(s, r)]
        y = np.load(self.cd / f"s{s:02d}_r{r:02d}.npy")[: self.length[r]]
        if keep:
            self._memo[(s, r)] = y
        return y


def run_stats(design: Design) -> dict[int, dict[int, trf.Stats]]:
    """Stats per subject and run; X'X and column sums are shared across subjects."""
    shared = {r: (x.T @ x, x.sum(0)) for r, x in design.x.items()}
    out: dict[int, dict[int, trf.Stats]] = {}
    for s in design.subjects:
        out[s] = {}
        for r, x in design.x.items():
            y = design.eeg(s, r).astype(np.float64)
            xtx, sx = shared[r]
            out[s][r] = trf.Stats(xtx, x.T @ y, sx, y.sum(0), (y * y).sum(0), x.shape[0])
    return out


# ---------------------------------------------------------------- encode + decode


def montage_sets(curve_c: np.ndarray) -> dict[str, np.ndarray]:
    sets = {"biosemi128": np.arange(128)}
    for name in mt.STARTER_MONTAGES:
        sets[name] = mt.montage_indices(name)
    sets["best8_train"] = trf.best_channels(curve_c, ALPHAS, 8)
    sets["best16_train"] = trf.best_channels(curve_c, ALPHAS, 16)
    return sets


def decode_subject(design: Design, s: int, test_runs: list[int], preds: dict[int, np.ndarray],
                   chans: np.ndarray, win_s: float, rng: np.random.Generator) -> dict:
    """Rank all test-run windows of subject ``s`` against every candidate window.

    ``preds`` maps each test run to the model's predicted response (all channels).
    """
    win = int(win_s * FS)
    pred_w, obs_w = [], []
    for r in test_runs:
        starts = decode.window_starts(design.length[r], win)
        pred_w.append(decode.cut_windows(preds[r][:, chans], starts, win))
        obs_w.append(decode.cut_windows(design.eeg(s, r, keep=True)[:, chans], starts, win))
    pred_w, obs_w = np.concatenate(pred_w), np.concatenate(obs_w)
    scores = decode.score_matrix(obs_w, pred_w)
    n = len(scores)
    ranks = decode.true_ranks(scores)
    # Null: a random wrong candidate labelled as the truth (a derangement).
    shift = rng.integers(1, n)
    null_ranks = decode.true_ranks(scores, truth=(np.arange(n) + shift) % n)
    return {"n_candidates": n, "top1": decode.topk(ranks, 1), "top10": decode.topk(ranks, 10),
            "null_top1": decode.topk(null_ranks, 1), "null_top10": decode.topk(null_ranks, 10),
            "median_rank": float(np.median(ranks)), "hits1": int((ranks <= 1).sum()),
            "hits10": int((ranks <= 10).sum())}


def encode(root: Path, condition: str, subjects: list[int], train_runs: list[int], test_runs: list[int],
           n_folds: int, lm_path: Path, n_pcs: int, out: Path, windows: list[float],
           tiers: list[str]) -> None:
    t0 = time.time()
    design = Design(root, condition, subjects, lm_path, train_runs, n_pcs)
    train_runs = [r for r in train_runs if r in design.runs]
    test_runs = [r for r in test_runs if r in design.runs]
    log(f"{condition}: design built, runs train={train_runs} test={test_runs}, "
        f"P={design.x[design.runs[0]].shape[1]} ({time.time() - t0:.0f}s)")
    stats = run_stats(design)
    log(f"{condition}: statistics done ({time.time() - t0:.0f}s)")
    s_train = {s: trf.sum_stats(stats[s][r] for r in train_runs) for s in subjects}
    s_test = {s: trf.sum_stats(stats[s][r] for r in test_runs) for s in subjects}
    rng = np.random.default_rng(0)

    rows, dec_rows, chan_r, chosen = [], [], [], []
    alpha_c: dict[int, float] = {}
    for fold, (tr_subs, te_subs) in enumerate(trf.subject_folds(subjects, n_folds, seed=0)):
        curves = {}
        for tier in tiers:
            cols = design.cols(tier)
            curves[tier] = trf.inner_loso_curve({s: s_train[s].select(cols) for s in tr_subs}, ALPHAS)
        sets = montage_sets(curves["C_env_onset_lm"] if "C_env_onset_lm" in curves else curves[tiers[-1]])
        chosen.append({"fold": fold, "train": tr_subs, "test": te_subs,
                       "best8": sets["best8_train"].tolist(), "best16": sets["best16_train"].tolist()})
        for tier in tiers:
            cols = design.cols(tier)
            pooled = trf.sum_stats(s_train[s].select(cols) for s in tr_subs)
            path = trf.RidgePath(pooled)
            pred_cache: dict[float, dict[int, np.ndarray]] = {}
            for mname, chans in sets.items():
                alpha = trf.choose_alpha(curves[tier], ALPHAS, chans)
                w = path.weights(alpha)
                if alpha not in pred_cache:  # predictions depend on the stimulus, not the subject
                    pred_cache[alpha] = {r: trf.predict(design.x[r][:, cols], w, pooled) for r in test_runs}
                for s in te_subs:
                    r = trf.corr_from_stats(w, s_test[s].select(cols))
                    rows.append({"condition": condition, "fold": fold, "subject": s, "tier": tier,
                                 "montage": mname, "alpha": alpha, "mean_r": float(r[chans].mean())})
                    if mname == "biosemi128":
                        chan_r.append({"tier": tier, "subject": s, "r": r.tolist()})
                        if tier == "C_env_onset_lm":
                            alpha_c[s] = alpha
                    for win_s in windows:
                        d = decode_subject(design, s, test_runs, pred_cache[alpha], chans, win_s, rng)
                        dec_rows.append({"condition": condition, "fold": fold, "subject": s, "tier": tier,
                                         "montage": mname, "window_s": win_s, **d})
        log(f"{condition}: fold {fold} done, test subjects {te_subs} ({time.time() - t0:.0f}s)")

    out.mkdir(parents=True, exist_ok=True)
    tag = f"{condition}"
    if "C_env_onset_lm" in tiers and alpha_c:
        _write_csv(out / f"n400_trf_per_subject_{tag}.csv", n400_kernels(design, stats, alpha_c, condition))
    _write_csv(out / f"encoding_per_subject_{tag}.csv", rows)
    _write_csv(out / f"decoding_per_subject_{tag}.csv", dec_rows)
    (out / f"folds_{tag}.json").write_text(json.dumps(
        {"subjects": subjects, "train_runs": train_runs, "test_runs": test_runs, "alphas": ALPHAS,
         "band_hz": BAND, "fs": FS, "n_pcs": n_pcs, "lm": lm_path.name, "folds": chosen}, indent=1))
    _write_scalp(out / f"scalp_{tag}.md", chan_r)
    log(f"{condition}: wrote tables to {out.name}/ ({time.time() - t0:.0f}s)")


def n400_kernels(design: Design, stats: dict[int, dict[int, trf.Stats]], alpha: dict[int, float],
                 condition: str) -> list[dict]:
    """N400 check on continuous speech (the source paper's analysis).

    Per subject, fit the tier-C TRF on all of that subject's runs (alpha: the one chosen
    without that subject, in its held-out fold) and average each word feature's kernel
    over 300-500 ms and the montage's pre-registered centro-parietal region. A negative
    value is an N400-like response to less predictable (or more dissimilar) words.
    """
    from semmn.eeg import n400

    rois = {"biosemi128": mt.n400_roi_128(mt.biosemi_positions())}
    rois.update({m: mt.n400_roi_indices(m) for m in mt.N400_ROI_1020})
    lags = np.arange(int(round(0.3 * FS)) - design.first, int(round(0.5 * FS)) - design.first + 1)
    # Alpha was chosen for tier C. Applying it to a wider design changes the penalty
    # scale (trace(Cxx) / P), so refuse that instead of fitting a different model.
    expected = len(design.cols("C_env_onset_lm"))
    got = next(iter(next(iter(stats.values())).values())).xtx.shape[0]
    if got != expected:
        raise ValueError(
            f"N400 alpha was chosen for tier C ({expected} columns) but the design has {got}"
        )
    rows = []
    for s, a in sorted(alpha.items()):
        w = trf.fit_ridge(trf.sum_stats(stats[s].values()), a)
        for feat in ("surprisal", "dissimilarity"):
            if feat not in design.width:
                continue
            f = design.feature_index(feat)[0]
            kernel = w[f * design.count + lags]  # (lags, channels)
            for m, roi in rois.items():
                rows.append({"condition": condition, "subject": s, "feature": feat, "montage": m,
                             "roi_size": len(roi), "kernel_300_500ms": float(kernel[:, roi].mean())})
    for feat in dict.fromkeys(r["feature"] for r in rows):
        for m in rois:
            g = n400.group_effect(np.array([r["kernel_300_500ms"] for r in rows
                                            if r["feature"] == feat and r["montage"] == m]))
            log(f"{condition} N400 kernel {feat} {m}: mean {g.mean:+.4f} dz {g.dz:+.2f} p {g.p:.2g} "
                f"({g.n_negative}/{g.n} negative)")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def _write_scalp(path: Path, chan_r: list[dict]) -> None:
    pos = mt.biosemi_positions()
    lines = ["# Scalp summary: held-out prediction r per channel (mean over test subjects)", ""]
    for tier in dict.fromkeys(c["tier"] for c in chan_r):
        r = np.mean([c["r"] for c in chan_r if c["tier"] == tier], axis=0)
        top = np.argsort(r)[::-1][:10]
        lines += [f"## {tier}", "", "Top 10 channels: " + ", ".join(
            f"{mt.BIOSEMI_NAMES[i]} ({mt.scalp_region(pos[mt.BIOSEMI_NAMES[i]])}) {r[i]:.3f}" for i in top), ""]
        regions: dict[str, list[float]] = {}
        for i, name in enumerate(mt.BIOSEMI_NAMES):
            regions.setdefault(mt.scalp_region(pos[name]), []).append(r[i])
        lines += ["| region | channels | mean r |", "| --- | --- | --- |"]
        for reg, vals in sorted(regions.items(), key=lambda kv: -np.mean(kv[1])):
            lines.append(f"| {reg} | {len(vals)} | {np.mean(vals):.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- summarize


def _read_csv(path: Path) -> list[dict]:
    import csv

    with path.open() as fh:
        return list(csv.DictReader(fh))


def _paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    t = sstats.ttest_rel(a, b) if len(d) > 1 else None
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    p = f"{t.pvalue:.2g}" if t is not None else "n/a"
    return f"{d.mean():+.4f} | {int((d > 0).sum())}/{len(d)} | {dz:+.2f} | {p}"


def _ci(d: np.ndarray) -> str:
    """Mean and 95% t confidence interval."""
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return "n/a"
    half = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} [{d.mean() - half:+.4f}, {d.mean() + half:+.4f}]"


def _per_subject(rows: list[dict], key: str, **match: str) -> dict[int, float]:
    return {int(r["subject"]): float(r[key]) for r in rows if all(r[k] == v for k, v in match.items())}


def summarize(out: Path) -> None:
    """Write summary.md from the per-subject tables in ``out``."""
    lines = [
        "# P3 summary tables",
        "",
        "Encoding and segment identification use held-out subjects and held-out runs.",
        "The N400 kernels do not: each is fit on all of that subject's runs, and only the",
        "ridge alpha was chosen without that subject. Tier L is word timing plus word",
        "features (it keeps the onset regressor), not semantics alone. Reported p-values",
        "are not corrected for the number of montages or features.",
        "",
    ]
    gains: dict[str, dict[str, np.ndarray]] = {}
    kernels: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        enc_path = out / f"encoding_per_subject_{cond}.csv"
        if not enc_path.exists():
            continue
        enc = _read_csv(enc_path)
        tiers = list(dict.fromkeys(r["tier"] for r in enc))
        monts = list(dict.fromkeys(r["montage"] for r in enc))

        def vals(tier: str, mont: str) -> np.ndarray:
            rows = sorted((r for r in enc if r["tier"] == tier and r["montage"] == mont),
                          key=lambda r: int(r["subject"]))
            return np.array([float(r["mean_r"]) for r in rows])

        n = len(vals(tiers[0], monts[0]))
        lines += [f"## {cond}: encoding, mean r over the montage's channels (mean ± SEM over {n} subjects)", "",
                  "| montage | " + " | ".join(tiers) + " |", "| --- |" + " --- |" * len(tiers)]
        for m in monts:
            cells = []
            for t in tiers:
                v = vals(t, m)
                cells.append(f"{v.mean():.4f} ± {v.std(ddof=1) / np.sqrt(len(v)):.4f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        if "C_env_onset_lm" in tiers and "B_env_onset" in tiers:
            lines += [f"### {cond}: word-feature gain, tier C minus tier B (paired over subjects)", "",
                      "| montage | mean Δr | subjects with Δr > 0 | dz | paired t p |", "| --- | --- | --- | --- | --- |"]
            for m in monts:
                lines.append(f"| {m} | " + _paired(vals("C_env_onset_lm", m), vals("B_env_onset", m)) + " |")
            lines.append("")
            gains[cond] = {m: vals("C_env_onset_lm", m) - vals("B_env_onset", m) for m in monts}
        n4_path = out / f"n400_trf_per_subject_{cond}.csv"
        if n4_path.exists():
            from semmn.eeg import n400

            n4 = _read_csv(n4_path)
            kernels[cond] = n4
            lines += [f"### {cond}: N400 check, word-feature TRF kernel at 300-500 ms, centro-parietal region",
                      "",
                      "Within-subject fit on every run of that subject (not the held-out runs). "
                      "A negative kernel is the N400 direction.",
                      "",
                      "| feature | montage | ROI channels | mean kernel | subjects negative | dz | t | p |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for feat in dict.fromkeys(r["feature"] for r in n4):
                for m in dict.fromkeys(r["montage"] for r in n4):
                    sel = [r for r in n4 if r["feature"] == feat and r["montage"] == m]
                    g = n400.group_effect(np.array([float(r["kernel_300_500ms"]) for r in sel]))
                    lines.append(f"| {feat} | {m} | {sel[0]['roi_size']} | {g.mean:+.4f} | {g.n_negative}/{g.n} | "
                                 f"{g.dz:+.2f} | {g.t:+.2f} | {g.p:.2g} |")
            lines.append("")
        dec_path = out / f"decoding_per_subject_{cond}.csv"
        if dec_path.exists():
            dec = _read_csv(dec_path)
            lines += [f"## {cond}: segment identification", "",
                      "Top-1 and top-10 are pooled over subjects. The binomial p treats every window as "
                      "independent, but all subjects share one candidate set and one set of predictions, so it "
                      "overstates the evidence; the subject-level columns (per-subject top-10 against chance, "
                      "one-sample t) are the conservative test.", "",
                      "| window s | montage | tier | N candidates | top-1 | chance | top-10 | chance | "
                      "binomial p (top-1) | subjects top-10 > chance | subject-level p (top-10) | null top-1 |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for win in dict.fromkeys(r["window_s"] for r in dec):
                for m in monts:
                    for t in tiers:
                        rows = [r for r in dec if r["window_s"] == win and r["montage"] == m and r["tier"] == t]
                        if not rows:
                            continue
                        nc = np.array([int(r["n_candidates"]) for r in rows])
                        hits1 = sum(int(r["hits1"]) for r in rows)
                        hits10 = sum(int(r["hits10"]) for r in rows)
                        trials = int(nc.sum())
                        ncand = int(nc[0])
                        null1 = np.mean([float(r["null_top1"]) for r in rows])
                        chance10 = min(10 / ncand, 1)
                        acc10 = np.array([float(r["top10"]) for r in rows])
                        p_subj = (sstats.ttest_1samp(acc10, chance10, alternative="greater").pvalue
                                  if len(acc10) > 1 and acc10.std() > 0 else float("nan"))
                        lines.append(
                            f"| {float(win):g} | {m} | {t} | {ncand} | {hits1 / trials:.3f} | {1 / ncand:.3f} | "
                            f"{hits10 / trials:.3f} | {chance10:.3f} | "
                            f"{decode.binom_p(hits1, trials, 1 / ncand):.2g} | {int((acc10 > chance10).sum())}/{len(rows)} | "
                            f"{p_subj:.2g} | {null1:.3f} |")
            lines.append("")
            lines += [f"### {cond}: each montage against 128 channels (paired over subjects, mean [95% CI])", "",
                      "Negative means the montage loses against 128 channels.", "",
                      "| montage | word-feature gain (C−B) difference | tier C 10 s top-10 difference |",
                      "| --- | --- | --- |"]
            ref10 = _per_subject(dec, "top10", window_s="10.0", montage="biosemi128", tier="C_env_onset_lm")
            for m in monts:
                if m == "biosemi128" or cond not in gains:
                    continue
                m10 = _per_subject(dec, "top10", window_s="10.0", montage=m, tier="C_env_onset_lm")
                d10 = np.array([m10[k] - ref10[k] for k in sorted(ref10)])
                lines.append(f"| {m} | {_ci(gains[cond][m] - gains[cond]['biosemi128'])} | {_ci(d10)} |")
            lines.append("")
    if len(gains) == 2:
        lines += ["## Forward against reversed speech (Welch t-test on per-subject values)", "",
                  "| measure | montage | forward mean | reverse mean | t | p |", "| --- | --- | --- | --- | --- | --- |"]
        for m in gains["forward"]:
            f, r = gains["forward"][m], gains["reverse"][m]
            w = sstats.ttest_ind(f, r, equal_var=False)
            lines.append(f"| word-feature gain C−B | {m} | {f.mean():+.4f} | {r.mean():+.4f} | {w.statistic:+.2f} | {w.pvalue:.2g} |")
        if len(kernels) == 2:
            for feat in dict.fromkeys(x["feature"] for x in kernels["forward"]):
                for m in dict.fromkeys(x["montage"] for x in kernels["forward"]):
                    f, r = (np.array([float(x["kernel_300_500ms"]) for x in kernels[c]
                                      if x["feature"] == feat and x["montage"] == m]) for c in ("forward", "reverse"))
                    w = sstats.ttest_ind(f, r, equal_var=False)
                    lines.append(f"| N400 kernel, {feat} | {m} | {f.mean():+.4f} | {r.mean():+.4f} | "
                                 f"{w.statistic:+.2f} | {w.pvalue:.2g} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    log(f"wrote {out.name}/summary.md")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=os.environ.get("SEMMN_BRODERICK"),
                    help="directory holding the extracted condition folders (or $SEMMN_BRODERICK)")
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--condition", choices=list(CONDITIONS), default="forward")
    p.add_argument("--subjects", default="1-8")
    p = sub.add_parser("words")
    p.add_argument("--backend", choices=["local", "gpt2"], default="local")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--download", action="store_true",
                   help="allow the gpt2 backend to download model files from the Hugging Face hub; "
                        "by default only the local cache is read")
    p = sub.add_parser("encode")
    p.add_argument("--condition", choices=list(CONDITIONS), default="forward")
    p.add_argument("--subjects", default="1-8")
    p.add_argument("--train-runs", default="1-15")
    p.add_argument("--test-runs", default="16-20")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--lm", type=Path, required=True, help="word-feature .npz from the words stage")
    p.add_argument("--pcs", type=int, default=20)
    p.add_argument("--windows", default="5,10")
    p.add_argument("--tiers", default="A_env,B_env_onset,C_env_onset_lm,L_onset_lm")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "eeg")
    p = sub.add_parser("summarize")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "eeg")
    a = ap.parse_args()
    if a.stage == "summarize":
        summarize(a.out)
        return
    if a.data_root is None:
        ap.error("--data-root or $SEMMN_BRODERICK is required")
    root = Path(a.data_root)
    if a.stage == "prep":
        prep(root, a.condition, parse_ints(a.subjects))
    elif a.stage == "words":
        print(words_stage(root, a.backend, a.layer, download=a.download))
    else:
        encode(root, a.condition, parse_ints(a.subjects), parse_ints(a.train_runs), parse_ints(a.test_runs),
               a.folds, a.lm, a.pcs, a.out, [float(w) for w in a.windows.split(",")], a.tiers.split(","))


if __name__ == "__main__":
    main()

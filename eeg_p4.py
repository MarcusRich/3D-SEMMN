#!/usr/bin/env python
"""P4 summary: GPT-2 word features and GPT-2 distractors on Broderick et al. (2018).

The stages live in eeg_p3.py (``words --backend gpt2``, ``distract``, ``encode
--distractors``). This script only reads their per-subject tables and writes
summary_p4.md: the stepwise tier comparison, the reversed-speech control, GPT-2 against
P3's stand-in features, segment identification against the envelope-only ranker, the
distractor ranking with its authorship controls, and the starter montages.

Example:
  python -I eeg_p4.py --out outputs/eeg_p4 --p3 outputs/eeg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats as sstats

A, B, S, E = "A_env", "B_env_onset", "S_env_onset_sur", "E_env_onset_sur_emb"
P3_C = "C_env_onset_lm"
MONTAGES = ("biosemi128", "cyton8", "cyton_daisy16", "unicorn8", "best8_train", "best16_train")
STARTER = ("cyton8", "cyton_daisy16")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def per_subject(rows: list[dict], key: str, **match) -> dict[int, float]:
    return {int(r["subject"]): float(r[key]) for r in rows if all(r.get(k) == str(v) for k, v in match.items())}


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[np.ndarray, str]:
    """a - b over shared subjects: mean, subjects > 0, dz, paired t p."""
    subs = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in subs])
    if len(d) < 2:
        return d, "n/a | n/a | n/a | n/a"
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else float("nan")
    p = sstats.ttest_1samp(d, 0.0).pvalue if sd > 0 else float("nan")
    return d, f"{d.mean():+.4f} | {int((d > 0).sum())}/{len(d)} | {dz:+.2f} | {p:.2g}"


def ci(d: np.ndarray, scale: float = 1.0, fmt: str = "+.4f") -> str:
    d = np.asarray(d, float) * scale
    if len(d) < 2:
        return "n/a"
    half = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():{fmt}} [{d.mean() - half:{fmt}}, {d.mean() + half:{fmt}}]"


def vs_chance(acc: np.ndarray, chance: float) -> str:
    """Subjects above chance and the one-sample t p (one-sided, greater)."""
    above = int((acc > chance).sum())
    if len(acc) < 2 or acc.std() == 0:
        return f"{above}/{len(acc)} | n/a"
    return f"{above}/{len(acc)} | {sstats.ttest_1samp(acc, chance, alternative='greater').pvalue:.2g}"


def encoding_section(cond: str, enc: list[dict], p3_enc: list[dict]) -> tuple[list[str], dict]:
    tiers = [t for t in (A, B, S, E) if any(r["tier"] == t for r in enc)]
    monts = [m for m in MONTAGES if any(r["montage"] == m for r in enc)]
    v = {(t, m): per_subject(enc, "mean_r", tier=t, montage=m) for t in tiers for m in monts}
    n = len(v[(tiers[0], monts[0])])
    out = [f"## {cond}: encoding, held-out r (mean ± SEM over {n} subjects)", "",
           "| montage | " + " | ".join(tiers) + " |", "| --- |" + " --- |" * len(tiers)]
    for m in monts:
        cells = []
        for t in tiers:
            x = np.array(list(v[(t, m)].values()))
            cells.append(f"{x.mean():.4f} ± {x.std(ddof=1) / np.sqrt(len(x)):.4f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", f"### {cond}: stepwise gains (paired over subjects)", "",
            "| montage | step | mean Δr | subjects Δr > 0 | dz | p |", "| --- | --- | --- | --- | --- | --- |"]
    gains: dict = {}
    for m in monts:
        for name, hi, lo in (("onsets: B − A", B, A), ("GPT-2 surprisal: S − B", S, B),
                             ("GPT-2 embedding: E − S", E, S), ("all GPT-2: E − B", E, B)):
            if (hi, m) in v and (lo, m) in v:
                d, txt = paired(v[(hi, m)], v[(lo, m)])
                gains[(name, m)] = d
                out.append(f"| {m} | {name} | {txt} |")
    out.append("")
    if p3_enc:
        out += [f"### {cond}: GPT-2 against P3's stand-in word features (same folds, same subjects)", "",
                "Tiers A and B must reproduce P3 exactly on the fixed montages. The best-k sets are picked by "
                "the main tier (E here, C in P3), so they hold different channels and A and B differ there.", "",
                "| montage | max abs diff A vs P3 | max abs diff B vs P3 | E (GPT-2) − C (stand-in): mean Δr | "
                "subjects > 0 | dz | p |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for m in monts:
            diffs = []
            for t in (A, B):
                mine, theirs = v.get((t, m), {}), per_subject(p3_enc, "mean_r", tier=t, montage=m)
                shared = set(mine) & set(theirs)
                diffs.append(f"{max(abs(mine[s] - theirs[s]) for s in shared):.1e}" if shared else "n/a")
            _, txt = paired(v.get((E, m), {}), per_subject(p3_enc, "mean_r", tier=P3_C, montage=m))
            out.append(f"| {m} | {diffs[0]} | {diffs[1]} | {txt} |")
        out.append("")
    return out, gains


def segment_section(cond: str, dec: list[dict]) -> list[str]:
    out = [f"## {cond}: segment identification (other held-out segments as candidates)", "",
           "Pooled top-k over subjects; subject-level test is per-subject top-k against chance (one-sided t). "
           "Language against acoustics is the paired per-subject difference from tier A on the same windows.", "",
           "| window s | montage | tier | N | top-1 | top-10 | chance top-1 / top-10 | subjects top-10 > chance | subject-level p | "
           "top-10 − tier A, points [95% CI] | paired p | top-1 − tier A, points [95% CI] |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for win in dict.fromkeys(r["window_s"] for r in dec):
        for m in MONTAGES:
            for t in (A, B, S, E):
                rows = [r for r in dec if r["window_s"] == win and r["montage"] == m and r["tier"] == t]
                if not rows:
                    continue
                ncand = int(rows[0]["n_candidates"])
                trials = sum(int(r["n_candidates"]) for r in rows)
                top1 = sum(int(r["hits1"]) for r in rows) / trials
                top10 = sum(int(r["hits10"]) for r in rows) / trials
                acc10 = np.array([float(r["top10"]) for r in rows])
                c10 = min(10 / ncand, 1.0)
                if t == A:
                    d10 = d1 = "—"
                    pp = "—"
                else:
                    a10 = per_subject(dec, "top10", window_s=win, montage=m, tier=A)
                    a1 = per_subject(dec, "top1", window_s=win, montage=m, tier=A)
                    t10 = per_subject(dec, "top10", window_s=win, montage=m, tier=t)
                    t1 = per_subject(dec, "top1", window_s=win, montage=m, tier=t)
                    dd10 = np.array([t10[s] - a10[s] for s in sorted(a10)])
                    dd1 = np.array([t1[s] - a1[s] for s in sorted(a1)])
                    d10, d1 = ci(dd10, 100, "+.1f"), ci(dd1, 100, "+.1f")
                    pp = f"{sstats.ttest_1samp(dd10, 0).pvalue:.2g}" if dd10.std() > 0 else "n/a"
                out.append(f"| {float(win):g} | {m} | {t} | {ncand} | {top1:.3f} | {top10:.3f} | "
                           f"{1 / ncand:.3f} / {c10:.3f} | {vs_chance(acc10, c10)} | {d10} | {pp} | {d1} |")
    out.append("")
    return out


def distractor_section(cond: str, dis: list[dict]) -> list[str]:
    if not dis:
        return []
    ncand = int(dis[0]["n_candidates"])
    nwin = int(dis[0]["n_windows"])
    out = [f"## {cond}: true words against {ncand - 1} GPT-2 distractors (same audio, same onsets)", "",
           f"{nwin} windows of 10 s per subject. Chance: top-1 {1 / ncand:.3f}, top-10 {10 / ncand:.3f}. "
           "Envelope-only and envelope+onset models score every candidate identically here, so they sit at "
           "chance by construction. 'Shuffled EEG' scores each window's candidates against another window of "
           "the same subject (mean over derangements) and must sit at chance. 'Prior only' ranks by GPT-2 "
           "log-probability with no EEG. A ranker can beat chance without using the EEG at all (for example "
           "when distractor features are atypical), so the EEG-specific evidence is the paired per-subject "
           "excess of the real EEG over the shuffled EEG, not the distance from chance.", "",
           "| montage | tier | top-1 | top-10 | subjects top-1 > chance | p | subjects top-10 > chance | p | "
           "shuffled EEG top-1 / top-10 | prior only top-1 / top-10 | "
           "real − shuffled EEG top-1, points [95% CI] | p | real − shuffled EEG top-10, points [95% CI] | p |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in MONTAGES:
        for t in (S, E):
            rows = [r for r in dis if r["montage"] == m and r["tier"] == t]
            if not rows:
                continue
            trials = sum(int(r["n_windows"]) for r in rows)
            top1 = sum(int(r["hits1"]) for r in rows) / trials
            top10 = sum(int(r["hits10"]) for r in rows) / trials
            a1 = np.array([float(r["top1"]) for r in rows])
            a10 = np.array([float(r["top10"]) for r in rows])
            sh1 = np.mean([float(r["shuffled_top1"]) for r in rows])
            sh10 = np.mean([float(r["shuffled_top10"]) for r in rows])
            out.append(f"| {m} | {t} | {top1:.3f} | {top10:.3f} | {vs_chance(a1, 1 / ncand)} | "
                       f"{vs_chance(a10, 10 / ncand)} | {sh1:.3f} / {sh10:.3f} | "
                       f"{float(rows[0]['prior_top1']):.3f} / {float(rows[0]['prior_top10']):.3f} | "
                       f"{excess(rows, 'top1')} | {excess(rows, 'top10')} |")
    out.append("")
    return out


def excess(rows: list[dict], key: str) -> str:
    """Per-subject real-EEG minus shuffled-EEG accuracy: mean [95% CI] in points | paired t p."""
    d = np.array([float(r[key]) - float(r[f"shuffled_{key}"]) for r in rows])
    p = sstats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else float("nan")
    return f"{ci(d, 100, '+.1f')} | {p:.2g}"


def starter_section(cond: str, gains: dict, dec: list[dict], dis: list[dict]) -> list[str]:
    out = [f"## {cond}: starter montages against 128 channels (paired over subjects, mean [95% CI])", "",
           "Negative means the montage loses against 128 channels. Decoding differences are in points.", "",
           "| montage | GPT-2 gain E − B | segment ID top-10, tier E | distractor top-1, tier E | "
           "distractor top-10, tier E |", "| --- | --- | --- | --- | --- |"]
    for m in STARTER:
        cells = []
        if ("all GPT-2: E − B", m) in gains:
            cells.append(ci(gains[("all GPT-2: E − B", m)] - gains[("all GPT-2: E − B", "biosemi128")]))
        else:
            cells.append("n/a")
        for rows, key, extra in ((dec, "top10", {"window_s": "10.0"}), (dis, "top1", {}), (dis, "top10", {})):
            mine = per_subject(rows, key, montage=m, tier=E, **extra)
            ref = per_subject(rows, key, montage="biosemi128", tier=E, **extra)
            subs = sorted(set(mine) & set(ref))
            cells.append(ci(np.array([mine[s] - ref[s] for s in subs]), 100, "+.1f") if subs else "n/a")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def summarize(out_dir: Path, p3_dir: Path | None) -> Path:
    lines = ["# P4 summary: GPT-2 word features (held-out subjects, held-out runs)", "",
             "GPT-2 small reads each run's content-word stream (the release has no function words), "
             "lower-cased, with a period at each sentence close, from the run's start only.", ""]
    gains_by_cond = {}
    for cond in ("forward", "reverse"):
        enc = read(out_dir / f"encoding_per_subject_{cond}.csv")
        if not enc:
            continue
        p3_enc = read(p3_dir / f"encoding_per_subject_{cond}.csv") if p3_dir else []
        sec, gains = encoding_section(cond, enc, p3_enc)
        gains_by_cond[cond] = gains
        dec = read(out_dir / f"decoding_per_subject_{cond}.csv")
        dis = read(out_dir / f"distractor_per_subject_{cond}.csv")
        lines += sec + segment_section(cond, dec) + distractor_section(cond, dis) + starter_section(cond, gains, dec, dis)
        n4 = read(out_dir / f"n400_trf_per_subject_{cond}.csv")
        if n4:
            lines += [f"## {cond}: N400 check with GPT-2 surprisal (tier E kernel, 300-500 ms, centro-parietal)", "",
                      "| montage | mean kernel | subjects negative | dz | p |", "| --- | --- | --- | --- | --- |"]
            for m in dict.fromkeys(r["montage"] for r in n4):
                k = np.array([float(r["kernel_300_500ms"]) for r in n4 if r["montage"] == m and r["feature"] == "surprisal"])
                if len(k) > 1:
                    lines.append(f"| {m} | {k.mean():+.4f} | {int((k < 0).sum())}/{len(k)} | "
                                 f"{k.mean() / k.std(ddof=1):+.2f} | {sstats.ttest_1samp(k, 0).pvalue:.2g} |")
            lines.append("")
    if len(gains_by_cond) == 2:
        lines += ["## Forward against reversed speech (Welch t on per-subject gains)", "",
                  "| step | montage | forward mean | reverse mean | t | p |", "| --- | --- | --- | --- | --- | --- |"]
        for (name, m), f in gains_by_cond["forward"].items():
            r = gains_by_cond["reverse"].get((name, m))
            if r is None or name.startswith("onsets") or m not in ("biosemi128",) + STARTER:
                continue
            w = sstats.ttest_ind(f, r, equal_var=False)
            lines.append(f"| {name} | {m} | {f.mean():+.4f} | {r.mean():+.4f} | {w.statistic:+.2f} | {w.pvalue:.2g} |")
        lines.append("")
    path = out_dir / "summary_p4.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", type=Path, default=here / "outputs" / "eeg_p4")
    ap.add_argument("--p3", type=Path, default=here / "outputs" / "eeg", help="P3 tables, for the stand-in comparison")
    a = ap.parse_args()
    print(summarize(a.out, a.p3 if a.p3.exists() else None))


if __name__ == "__main__":
    main()

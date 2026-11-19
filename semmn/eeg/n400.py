"""N400 check: does word predictability predict the N400 amplitude?

Per subject: epoch the EEG around target-word onsets, baseline-correct on the
pre-onset interval, and take the mean amplitude over the centro-parietal region and
300-500 ms. Regress single-trial amplitude on a z-scored predictor (surprisal, or
log-probability); the subject's slope is the effect. Across subjects, a one-sample
t-test on the slopes gives the group statistic, and Cohen's dz (mean / SD of the
slopes) the effect size. A larger N400 is more negative, so surprisal should give a
negative slope.

Slow ERPs need a lower high-pass than the 1-8 Hz TRF band; the runner preprocesses the
N400 data at about 0.1-30 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sstats


def epochs(x: np.ndarray, onsets: np.ndarray, fs: float, tmin: float = -0.2, tmax: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """(trials, samples, channels) epochs and the kept-trial mask.

    Trials whose window runs off either end of the recording are dropped.
    """
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    onsets = np.asarray(onsets, dtype=int)
    keep = (onsets + a >= 0) & (onsets + b <= len(x))
    ep = np.stack([x[o + a : o + b] for o in onsets[keep]]) if keep.any() else np.zeros((0, b - a, x.shape[1]))
    return ep, keep


def window_mean(ep: np.ndarray, fs: float, tmin: float, win: tuple[float, float], chans: np.ndarray,
                baseline: tuple[float, float] = (-0.2, 0.0)) -> np.ndarray:
    """Baseline-corrected mean amplitude per trial over ``chans`` and the window ``win`` (s)."""
    def idx(t: float) -> int:
        return int(round((t - tmin) * fs))

    base = ep[:, idx(baseline[0]) : idx(baseline[1])].mean(axis=1, keepdims=True)
    e = ep - base
    return e[:, idx(win[0]) : idx(win[1])][:, :, chans].mean(axis=(1, 2))


def subject_slope(amplitude: np.ndarray, predictor: np.ndarray) -> float:
    """OLS slope of amplitude on the z-scored predictor (units of amplitude per SD)."""
    p = np.asarray(predictor, dtype=float)
    sd = float(p.std())
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError("predictor has no variance; the N400 slope is undefined")
    z = (p - p.mean()) / sd
    return float(np.polyfit(z, np.asarray(amplitude, dtype=float), 1)[0])


@dataclass
class GroupEffect:
    n: int
    mean: float
    sd: float
    dz: float
    t: float
    p: float
    n_negative: int


def group_effect(slopes: np.ndarray) -> GroupEffect:
    """One-sample t-test of subject slopes against zero, with Cohen's dz."""
    s = np.asarray(slopes, dtype=float)
    res = sstats.ttest_1samp(s, 0.0)
    sd = s.std(ddof=1)
    return GroupEffect(len(s), float(s.mean()), float(sd), float(s.mean() / sd) if sd > 0 else float("nan"),
                       float(res.statistic), float(res.pvalue), int((s < 0).sum()))

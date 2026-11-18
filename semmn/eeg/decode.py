"""Guesswork decoding: which script segment was this EEG window listening to?

For a held-out subject, the EEG is cut into non-overlapping windows. Each window is
compared with the predicted response surface of every candidate segment (the encoding
model applied to that segment's stimulus), and the candidates are ranked by the mean
over channels of the Pearson correlation between predicted and observed signals. The
true segment's rank gives top-1 and top-k accuracy; chance is k / N.

This identifies *perceived* speech among a known set of candidates. It is not decoding
of intended speech.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def window_starts(n: int, win: int) -> np.ndarray:
    """Starts of the non-overlapping windows of ``win`` samples that fit in ``n``."""
    return np.arange(0, n - win + 1, win)


def cut_windows(x: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    """(N, win, C) windows of a (n, C) signal."""
    return np.stack([x[s : s + win] for s in starts])


def _zscore_time(w: np.ndarray) -> np.ndarray:
    w = w - w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return w / np.where(sd > 0, sd, 1.0)


def score_matrix(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """S[i, j]: mean over channels of r(observed window i, predicted candidate j).

    Both inputs are (N, T, C) with matching T and C.
    """
    if observed.shape[1:] != predicted.shape[1:]:
        raise ValueError("observed and predicted windows must share (T, C)")
    zo = _zscore_time(np.asarray(observed, dtype=np.float64))
    zp = _zscore_time(np.asarray(predicted, dtype=np.float64))
    n, t, c = observed.shape
    return (zo.reshape(n, t * c) @ zp.reshape(len(zp), t * c).T) / (t * c)


def true_ranks(scores: np.ndarray, truth: np.ndarray | None = None) -> np.ndarray:
    """1-based rank of the true candidate for each observed window.

    ``truth[i]`` is the true candidate of window i (default: the diagonal). Ties count
    against the true candidate, so equal scores never inflate accuracy.
    """
    n = scores.shape[0]
    truth = np.arange(n) if truth is None else np.asarray(truth)
    true_score = scores[np.arange(n), truth]
    others = scores >= true_score[:, None]
    others[np.arange(n), truth] = False
    return 1 + others.sum(axis=1)


def topk(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(np.asarray(ranks) <= k))


def binom_p(hits: int, trials: int, p_chance: float) -> float:
    """One-sided binomial p-value of at least ``hits`` successes at chance ``p_chance``."""
    return float(sstats.binomtest(int(hits), int(trials), min(p_chance, 1.0), alternative="greater").pvalue)

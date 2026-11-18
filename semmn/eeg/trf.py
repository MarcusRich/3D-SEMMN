"""Ridge temporal response functions (mTRF) from sufficient statistics.

Every subject heard the same runs, so the lagged design X of a run is shared and only
the EEG Y differs. The fit needs only X'X, X'Y, the column sums and n, which add across
runs and subjects. Pooling training subjects is then a sum, and the held-out prediction
correlation is exact from the held-out subject's own statistics:

    cov(y_hat, y) = w' Cxy,   var(y_hat) = w' Cxx w,   var(y) = Syy - n my^2

with C the centred statistics of the held-out data. Correlation ignores the intercept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Stats:
    """Sufficient statistics of a design X (n x P) and a response Y (n x C)."""

    xtx: np.ndarray
    xty: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    syy: np.ndarray
    n: int

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray) -> "Stats":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        return cls(x.T @ x, x.T @ y, x.sum(0), y.sum(0), (y * y).sum(0), x.shape[0])

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(
            self.xtx + other.xtx,
            self.xty + other.xty,
            self.sx + other.sx,
            self.sy + other.sy,
            self.syy + other.syy,
            self.n + other.n,
        )

    def select(self, cols: np.ndarray | None = None, chans: np.ndarray | None = None) -> "Stats":
        """Restrict to design columns ``cols`` and response channels ``chans``."""
        c = slice(None) if cols is None else np.asarray(cols)
        h = slice(None) if chans is None else np.asarray(chans)
        xtx = self.xtx if cols is None else self.xtx[np.ix_(c, c)]
        return Stats(xtx, self.xty[c][:, h], self.sx[c], self.sy[h], self.syy[h], self.n)

    def centred(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Cxx, Cxy, var_y * n): the statistics about the means."""
        mx = self.sx / self.n
        my = self.sy / self.n
        cxx = self.xtx - self.n * np.outer(mx, mx)
        cxy = self.xty - self.n * np.outer(mx, my)
        vy = self.syy - self.n * my**2
        return cxx, cxy, vy


def sum_stats(items: Iterable[Stats]) -> Stats:
    items = list(items)
    if not items:
        raise ValueError("no statistics to sum")
    total = items[0]
    for s in items[1:]:
        total = total + s
    return total


class RidgePath:
    """Ridge solutions for several alphas from one eigendecomposition of Cxx.

    ``alpha`` is relative: the penalty is ``alpha * trace(Cxx) / P``, so one grid works
    whatever the feature scale and sample count.
    """

    def __init__(self, train: Stats) -> None:
        cxx, cxy, _ = train.centred()
        self.evals, self.evecs = np.linalg.eigh(cxx)
        self.evals = np.clip(self.evals, 0, None)
        self.proj = self.evecs.T @ cxy
        self.scale = float(np.trace(cxx)) / cxx.shape[0]

    def weights(self, alpha: float) -> np.ndarray:
        d = 1.0 / (self.evals + alpha * self.scale)
        return self.evecs @ (d[:, None] * self.proj)


def fit_ridge(train: Stats, alpha: float) -> np.ndarray:
    """Ridge weights (P x C) for the centred problem."""
    return RidgePath(train).weights(alpha)


def corr_from_stats(w: np.ndarray, test: Stats) -> np.ndarray:
    """Per-channel Pearson r between X_test w and Y_test, from test statistics."""
    cxx, cxy, vy = test.centred()
    cov = np.einsum("pc,pc->c", w, cxy)
    var_hat = np.einsum("pc,pc->c", w, cxx @ w)
    denom = np.sqrt(np.clip(var_hat, 0, None) * np.clip(vy, 0, None))
    return np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), 0.0)


def predict(x: np.ndarray, w: np.ndarray, train: Stats) -> np.ndarray:
    """Predictions for a design ``x`` with the intercept implied by the training means."""
    mx = train.sx / train.n
    my = train.sy / train.n
    return (x - mx) @ w + my


def subject_folds(subjects: Sequence[int], n_folds: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Split subjects into ``n_folds`` disjoint test groups; train on the rest.

    Every subject is tested exactly once, and no subject is in both halves of a fold.
    """
    subjects = list(subjects)
    if len(set(subjects)) != len(subjects):
        raise ValueError("duplicate subjects")
    if not 2 <= n_folds <= len(subjects):
        raise ValueError("need 2 <= n_folds <= number of subjects")
    order = np.random.default_rng(seed).permutation(len(subjects))
    groups = np.array_split(order, n_folds)
    folds = []
    for g in groups:
        test = sorted(subjects[i] for i in g)
        train = sorted(s for s in subjects if s not in test)
        assert not set(train) & set(test)
        folds.append((train, test))
    return folds


def inner_loso_curve(per_subject: dict[int, Stats], alphas: Sequence[float]) -> np.ndarray:
    """Leave-one-subject-out r inside the training subjects: (n_alphas, C), mean over subjects.

    Only the statistics passed in are used, so with training subjects and training runs
    the held-out subjects never inform alpha or channel choices.
    """
    subs = sorted(per_subject)
    if len(subs) < 2:
        raise ValueError("need at least two training subjects for inner cross-validation")
    out = np.zeros((len(alphas), per_subject[subs[0]].xty.shape[1]))
    for s in subs:
        path = RidgePath(sum_stats(per_subject[t] for t in subs if t != s))
        for j, a in enumerate(alphas):
            out[j] += corr_from_stats(path.weights(a), per_subject[s])
    return out / len(subs)


def choose_alpha(curve: np.ndarray, alphas: Sequence[float], chans: np.ndarray | None = None) -> float:
    """The alpha with the best inner-CV r averaged over ``chans`` (all channels if None)."""
    score = curve if chans is None else curve[:, np.asarray(chans)]
    return float(alphas[int(np.argmax(score.mean(axis=1)))])


def best_channels(curve: np.ndarray, alphas: Sequence[float], k: int) -> np.ndarray:
    """The k channels with the highest inner-CV r at the all-channel alpha."""
    a = choose_alpha(curve, alphas)
    r = curve[list(alphas).index(a)]
    return np.sort(np.argsort(r)[::-1][:k])

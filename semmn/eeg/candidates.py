"""Rank a window's true words against acoustically identical distractor words.

Every candidate of a window shares the window's audio, word onsets and the true words
before the window; only the words with onsets inside the window differ. The response
model is a linear TRF, so a candidate's predicted response is

    pred(candidate) = pred(truth) - R(true word features) + R(candidate word features)

where R places each word's feature vector, through the word-feature kernels, at its
onset. The envelope and onset terms cancel, so an envelope-only (or envelope + onset)
model scores every candidate identically: its accuracy is chance by construction, and
anything above chance here comes from the language features.

Three scores per window, for the authorship check:
  * EEG: the window's own EEG against each candidate's predicted response;
  * shuffled EEG: another window's EEG of the same subject against this window's
    candidates (must sit at chance, whatever the language prior does);
  * prior only: the candidates' GPT-2 log-probability, no EEG.
"""

from __future__ import annotations

import numpy as np

from semmn.eeg import decode


def word_responses(onsets: np.ndarray, values: np.ndarray, kernel: np.ndarray, first_lag: int,
                   start: int, win: int) -> np.ndarray:
    """(K, win, C) responses of K word sets sharing ``onsets`` with features ``values`` (K, m, F).

    ``kernel`` is (F, L, C): the TRF weight of feature f at lag ``first_lag + j``.
    """
    values = np.asarray(values, dtype=np.float64)
    contrib = np.einsum("kmf,flc->kmlc", values, kernel)
    out = np.zeros((values.shape[0], win, kernel.shape[2]))
    lags = first_lag + np.arange(kernel.shape[1])
    for m, o in enumerate(np.asarray(onsets, dtype=int)):
        t = o + lags - start
        keep = (t >= 0) & (t < win)
        out[:, t[keep]] += contrib[:, m, keep]
    return out


def candidate_predictions(true_pred: np.ndarray, onsets: np.ndarray, true_values: np.ndarray,
                          cand_values: np.ndarray, kernel: np.ndarray, first_lag: int,
                          start: int) -> np.ndarray:
    """(1 + K, win, C): the truth first (exactly ``true_pred``), then the K candidates (K, m, F)."""
    win = true_pred.shape[0]
    vals = np.concatenate([np.asarray(true_values, float)[None], np.asarray(cand_values, float)])
    resp = word_responses(onsets, vals, kernel, first_lag, start, win)
    out = true_pred[None] - resp[:1] + resp
    out[0] = true_pred
    return out


def rank_of_truth(scores: np.ndarray) -> int:
    """1-based rank of candidate 0 among ``scores``; ties count against it."""
    scores = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(scores[1:] >= scores[0]))


def eeg_scores(observed: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Mean-over-channels Pearson r of one observed window (T, C) with each prediction."""
    return decode.score_matrix(observed[None], preds)[0]


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly drawn permutation of range(n) with no fixed point (n >= 2)."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def shuffled_eeg_ranks(observed: list[np.ndarray], preds: list[np.ndarray], rng: np.random.Generator,
                       n_perm: int) -> np.ndarray:
    """(n_perm, n_windows) ranks of each window's truth scored against another window's EEG."""
    perms = [derangement(len(observed), rng) for _ in range(n_perm)]
    return np.array([[rank_of_truth(eeg_scores(observed[p[i]], preds[i])) for i in range(len(preds))]
                     for p in perms])

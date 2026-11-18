"""Stimulus regressors for the response-surface model.

Word-level features become impulse trains: at each word's onset sample the regressor
takes the word's value (1 for the onset regressor, the surprisal, or the embedding's
PCA coordinates), and 0 elsewhere. The acoustic envelope is continuous. A temporal
response function then convolves each regressor with a learned kernel over lags
0..600 ms; ``lag_matrix`` builds that design explicitly.
"""

from __future__ import annotations

import numpy as np

# Feature tiers compared in P3. "L" drops the acoustics to show what the words alone carry.
# A word feature missing from the word-feature file is simply left out of a tier.
WORD_FEATURES = ("surprisal", "dissimilarity", "embedding")
TIERS: dict[str, tuple[str, ...]] = {
    "A_env": ("env",),
    "B_env_onset": ("env", "onset"),
    "C_env_onset_lm": ("env", "onset") + WORD_FEATURES,
    "L_onset_lm": ("onset",) + WORD_FEATURES,
}


def impulse_train(n: int, onset_idx: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    """(n, D) array with ``values[k]`` at row ``onset_idx[k]`` and zeros elsewhere.

    ``values`` is (k,) or (k, D); None gives unit impulses. Events outside [0, n) are
    dropped. Two events on the same sample add.
    """
    onset_idx = np.asarray(onset_idx, dtype=int)
    if values is None:
        values = np.ones(len(onset_idx))
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(onset_idx):
        raise ValueError("one value row per onset is required")
    keep = (onset_idx >= 0) & (onset_idx < n)
    out = np.zeros((n, values.shape[1]))
    np.add.at(out, onset_idx[keep], values[keep])
    return out


def n_lags(fs: float, tmin: float = 0.0, tmax: float = 0.6) -> tuple[int, int]:
    """(first lag, number of lags) in samples covering [tmin, tmax] seconds inclusive."""
    first = int(round(tmin * fs))
    last = int(round(tmax * fs))
    return first, last - first + 1


def lag_matrix(x: np.ndarray, first: int, count: int) -> np.ndarray:
    """Time-lagged design: column ``f * count + j`` holds ``x[t - (first + j), f]``.

    Positive lags look back in time (the response follows the stimulus). Samples shifted
    in from outside the run are zero, so runs never leak into each other.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, f = x.shape
    out = np.zeros((n, f * count))
    for j in range(count):
        lag = first + j
        cols = slice(j, f * count, count)
        if lag >= 0:
            if lag < n:
                out[lag:, cols] = x[: n - lag]
        else:
            if -lag < n:
                out[: n + lag, cols] = x[-lag:]
    return out


def stimulus_matrix(features: dict[str, np.ndarray], tier: tuple[str, ...]) -> np.ndarray:
    """Stack the named per-sample features of one run into (n, F) in tier order."""
    blocks = []
    for name in tier:
        v = np.asarray(features[name], dtype=np.float64)
        blocks.append(v[:, None] if v.ndim == 1 else v)
    n = {b.shape[0] for b in blocks}
    if len(n) != 1:
        raise ValueError(f"features differ in length: {n}")
    return np.concatenate(blocks, axis=1)

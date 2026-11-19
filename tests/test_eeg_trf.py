"""Lagged design, sufficient-statistic ridge, and leave-subjects-out splits."""

from __future__ import annotations

import numpy as np
import pytest

from semmn.eeg import features as ft
from semmn.eeg import trf


def test_impulse_train_places_values():
    x = ft.impulse_train(10, np.array([1, 4, 4, 12]), np.array([2.0, 1.0, 3.0, 9.0]))
    assert x.shape == (10, 1)
    np.testing.assert_array_equal(x[:, 0], [0, 2, 0, 0, 4, 0, 0, 0, 0, 0])
    e = ft.impulse_train(5, np.array([0, 2]), np.eye(2))
    np.testing.assert_array_equal(e[[0, 2]], np.eye(2))


def test_lag_matrix_shifts_without_wrapping():
    x = np.arange(1, 6, dtype=float)[:, None] * np.array([1.0, 10.0])
    first, count = ft.n_lags(64.0, 0.0, 0.6)
    assert (first, count) == (0, 39)
    d = ft.lag_matrix(x, 0, 3)
    assert d.shape == (5, 6)
    np.testing.assert_array_equal(d[:, 0], [1, 2, 3, 4, 5])  # feature 0, lag 0
    np.testing.assert_array_equal(d[:, 1], [0, 1, 2, 3, 4])  # feature 0, lag 1
    np.testing.assert_array_equal(d[:, 2], [0, 0, 1, 2, 3])  # feature 0, lag 2
    np.testing.assert_array_equal(d[:, 4], [0, 10, 20, 30, 40])  # feature 1, lag 1
    neg = ft.lag_matrix(x[:, :1], -1, 2)
    np.testing.assert_array_equal(neg[:, 0], [2, 3, 4, 5, 0])


def test_ridge_recovers_a_known_kernel_and_r_matches_direct():
    rng = np.random.default_rng(0)
    kernel = np.exp(-np.arange(8) / 2.0) * np.sin(np.arange(8))
    runs = []
    for _ in range(4):
        s = rng.standard_normal(2000)
        x = ft.lag_matrix(s, 0, 8)
        y = (x @ kernel)[:, None] + 0.5 * rng.standard_normal((2000, 1)) + 3.0
        runs.append((x, y))
    train = trf.sum_stats(trf.Stats.from_arrays(x, y) for x, y in runs[:3])
    w = trf.fit_ridge(train, 1e-4)
    np.testing.assert_allclose(w[:, 0], kernel, atol=0.05)
    xt, yt = runs[3]
    r_stats = trf.corr_from_stats(w, trf.Stats.from_arrays(xt, yt))
    r_direct = np.corrcoef(trf.predict(xt, w, train)[:, 0], yt[:, 0])[0, 1]
    assert r_stats[0] == pytest.approx(r_direct, abs=1e-10)


def test_stats_add_equals_concatenation_and_select():
    rng = np.random.default_rng(1)
    x1, y1 = rng.standard_normal((50, 4)), rng.standard_normal((50, 3))
    x2, y2 = rng.standard_normal((30, 4)), rng.standard_normal((30, 3))
    a = trf.Stats.from_arrays(x1, y1) + trf.Stats.from_arrays(x2, y2)
    b = trf.Stats.from_arrays(np.vstack([x1, x2]), np.vstack([y1, y2]))
    np.testing.assert_allclose(a.xtx, b.xtx)
    np.testing.assert_allclose(a.xty, b.xty)
    sub = b.select(np.array([0, 2]), np.array([1]))
    np.testing.assert_allclose(sub.xtx, np.vstack([x1, x2])[:, [0, 2]].T @ np.vstack([x1, x2])[:, [0, 2]])
    assert sub.xty.shape == (2, 1)


def test_subject_folds_never_leak():
    subjects = [3, 5, 7, 9, 11, 13, 17]
    folds = trf.subject_folds(subjects, 3, seed=2)
    tested = [s for _, test in folds for s in test]
    assert sorted(tested) == subjects  # every subject held out exactly once
    for train, test in folds:
        assert not set(train) & set(test)
        assert sorted(train + test) == subjects
    with pytest.raises(ValueError):
        trf.subject_folds([1, 1, 2], 2)


def test_choose_alpha_uses_only_given_subjects():
    rng = np.random.default_rng(3)
    k = np.array([1.0, -0.5, 0.25])
    per = {}
    for s in range(4):
        x = ft.lag_matrix(rng.standard_normal(1500), 0, 3)
        per[s] = trf.Stats.from_arrays(x, x @ k[:, None] + rng.standard_normal((1500, 1)))
    alphas = [1e-3, 1e3]
    curve = trf.inner_loso_curve(per, alphas)
    assert curve.shape == (2, 1)
    assert trf.choose_alpha(curve, alphas) == 1e-3 and curve[0, 0] > curve[1, 0]
    with pytest.raises(ValueError):
        trf.inner_loso_curve({0: per[0]}, [1.0])


def test_best_channels_ranks_by_inner_r():
    curve = np.array([[0.1, 0.5, 0.2, 0.4], [0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(trf.best_channels(curve, [1.0, 10.0], 2), [1, 3])
    assert trf.choose_alpha(curve, [1.0, 10.0], np.array([0])) == 1.0

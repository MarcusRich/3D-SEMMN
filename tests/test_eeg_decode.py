"""Candidate ranking on a synthetic case where the answer is known."""

from __future__ import annotations

import numpy as np

from semmn.eeg import decode


def test_true_segment_ranks_first_when_signal_is_present():
    rng = np.random.default_rng(0)
    n, t, c = 40, 320, 8
    predicted = rng.standard_normal((n, t, c))
    observed = 0.5 * predicted + rng.standard_normal((n, t, c))
    ranks = decode.true_ranks(decode.score_matrix(observed, predicted))
    assert decode.topk(ranks, 1) == 1.0


def test_shuffled_pairing_is_at_chance():
    rng = np.random.default_rng(1)
    n, t, c = 200, 64, 4
    predicted = rng.standard_normal((n, t, c))
    observed = 0.5 * predicted + rng.standard_normal((n, t, c))
    perm = rng.permutation(n)
    ranks = decode.true_ranks(decode.score_matrix(observed[perm], predicted), truth=np.arange(n))
    assert decode.topk(ranks, 10) < 0.15  # chance is 10 / 200 = 0.05
    # With the true labels restored, the same data rank first.
    ranks = decode.true_ranks(decode.score_matrix(observed[perm], predicted), truth=perm)
    assert decode.topk(ranks, 1) == 1.0


def test_ties_count_against_the_true_candidate():
    scores = np.zeros((3, 3))
    np.testing.assert_array_equal(decode.true_ranks(scores), [3, 3, 3])


def test_windows_do_not_overlap():
    starts = decode.window_starts(1000, 320)
    np.testing.assert_array_equal(starts, [0, 320, 640])
    w = decode.cut_windows(np.arange(1000)[:, None], starts, 320)
    assert w.shape == (3, 320, 1) and w[1, 0, 0] == 320


def test_binomial_p():
    assert decode.binom_p(10, 10, 0.1) < 1e-9
    assert decode.binom_p(1, 10, 0.1) > 0.5

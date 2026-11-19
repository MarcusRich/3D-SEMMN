"""Candidate ranking against acoustically identical distractors, and the authorship controls."""

from __future__ import annotations

import numpy as np

from semmn.eeg import candidates as cd
from semmn.eeg import features as ft


def _setup(rng, n=400, f=3, c=5, first=0, count=8):
    kernel = rng.standard_normal((f, count, c))
    onsets = np.sort(rng.choice(np.arange(20, n - 20), 12, replace=False))
    return kernel, onsets, first, count


def test_substitution_equals_rebuilding_the_design():
    """pred(truth) - R(true) + R(candidate) equals predicting from the candidate's own design."""
    rng = np.random.default_rng(0)
    kernel, onsets, first, count = _setup(rng)
    n, f, c = 400, 3, 5
    env = rng.standard_normal(n)
    w_env = rng.standard_normal((count, c))
    true_vals = rng.standard_normal((len(onsets), f))
    cand_vals = rng.standard_normal((len(onsets), f))
    start, win = 150, 120
    inside = (onsets >= start) & (onsets < start + win)

    def predict(vals):
        stim = np.column_stack([env, ft.impulse_train(n, onsets, vals)])
        x = ft.lag_matrix(stim, first, count)
        w = np.concatenate([w_env, kernel.reshape(f * count, c)])
        return x @ w

    mixed = true_vals.copy()
    mixed[inside] = cand_vals[inside]
    full_true, full_cand = predict(true_vals), predict(mixed)
    preds = cd.candidate_predictions(full_true[start : start + win], onsets[inside], true_vals[inside],
                                     cand_vals[inside][None], kernel, first, start)
    np.testing.assert_allclose(preds[0], full_true[start : start + win], atol=1e-10)
    np.testing.assert_allclose(preds[1], full_cand[start : start + win], atol=1e-10)


def test_rank_of_truth_counts_ties_against_the_truth():
    assert cd.rank_of_truth(np.array([0.5, 0.1, 0.2])) == 1
    assert cd.rank_of_truth(np.array([0.5, 0.5, 0.2])) == 2
    assert cd.rank_of_truth(np.array([0.0, 0.0, 0.0, 0.0])) == 4  # identical candidates never win


def test_word_responses_is_linear_in_the_features():
    rng = np.random.default_rng(2)
    kernel = rng.standard_normal((3, 6, 4))
    onsets = np.array([5, 9, 30])
    a, b = rng.standard_normal((2, 3, 3))
    ra, rb, rab = cd.word_responses(onsets, np.stack([a, b, a + 2 * b]), kernel, 1, 3, 25)
    np.testing.assert_allclose(rab, ra + 2 * rb, atol=1e-12)


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10):
        for _ in range(20):
            assert not np.any(cd.derangement(n, rng) == np.arange(n))


def test_shuffled_eeg_is_at_chance_even_when_the_prior_knows_the_answer():
    """The authorship control: with EEG that carries the truth, the ranker finds it; with
    another window's EEG it must fall to chance, whatever a language prior would say."""
    rng = np.random.default_rng(1)
    n_win, n_cand, t, c = 60, 20, 64, 4
    observed, preds, prior = [], [], []
    for _ in range(n_win):
        cand = rng.standard_normal((n_cand, t, c))
        observed.append(cand[0] + 1.0 * rng.standard_normal((t, c)))
        preds.append(cand)
        prior.append(np.r_[10.0, rng.standard_normal(n_cand - 1)])  # the prior always favours the truth
    real = np.array([cd.rank_of_truth(cd.eeg_scores(o, p)) for o, p in zip(observed, preds)])
    prior_ranks = np.array([cd.rank_of_truth(p) for p in prior])
    shuf = cd.shuffled_eeg_ranks(observed, preds, rng, n_perm=30)
    assert np.mean(real == 1) > 0.9
    assert np.all(prior_ranks == 1)
    assert abs(np.mean(shuf == 1) - 1 / n_cand) < 0.02
    assert abs(shuf.mean() - (n_cand + 1) / 2) < 0.5

"""N400 epoching, window amplitude and the subject-level statistic."""

from __future__ import annotations

import numpy as np

from semmn.eeg import n400

FS = 100.0


def test_planted_n400_gives_negative_slopes_and_a_group_effect():
    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(10):
        n = int(300 * FS)
        x = rng.standard_normal((n, 4))
        onsets = np.arange(200, n - 200, 150)
        surprisal = rng.gamma(2.0, 2.0, len(onsets))
        t = np.arange(int(0.6 * FS)) / FS
        shape = np.exp(-((t - 0.4) ** 2) / (2 * 0.06**2))
        for o, s in zip(onsets, surprisal):
            x[o : o + len(t), :2] -= 0.5 * s * shape[:, None]  # centro-parietal channels 0 and 1
        ep, keep = n400.epochs(x, onsets, FS)
        assert keep.all() and ep.shape == (len(onsets), 100, 4)
        amp = n400.window_mean(ep, FS, -0.2, (0.3, 0.5), np.array([0, 1]))
        slopes.append(n400.subject_slope(amp, surprisal))
    g = n400.group_effect(np.array(slopes))
    assert g.n == 10 and g.n_negative == 10 and g.mean < 0 and g.p < 1e-4 and g.dz < -2


def test_epochs_drop_trials_off_the_ends():
    x = np.zeros((1000, 2))
    ep, keep = n400.epochs(x, np.array([5, 500, 995]), FS)
    np.testing.assert_array_equal(keep, [False, True, False])
    assert ep.shape == (1, 100, 2)


def test_baseline_removes_offsets():
    ep = np.ones((3, 100, 2)) * 7.0
    amp = n400.window_mean(ep, FS, -0.2, (0.3, 0.5), np.array([0, 1]))
    np.testing.assert_allclose(amp, 0.0)

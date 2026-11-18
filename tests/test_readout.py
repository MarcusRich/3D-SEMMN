"""The hub read-out must pass on what the hub columns carry about the input.

P1 found that a linear probe on the hub-column states read the label well, while the
model's own read-out (attention mean-pooled over the hub columns) did not. These tests
pin the two properties the P2 read-out adds.
"""

from __future__ import annotations

import torch

from semmn.hub import SharedManifoldHub

STATE_DIM, LATENT_DIM, FEATURE_DIM, N_COLUMNS = 8, 4, 16, 3
HUB_COLUMNS = torch.arange(N_COLUMNS)


def _hub() -> SharedManifoldHub:
    try:
        return SharedManifoldHub(
            STATE_DIM, LATENT_DIM, FEATURE_DIM, attention_heads=2, num_hub_columns=N_COLUMNS
        )
    except TypeError:  # the P1 signature: lets the first two tests fail on the old behaviour
        return SharedManifoldHub(STATE_DIM, LATENT_DIM, FEATURE_DIM, attention_heads=2)


def _features(b: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(b, FEATURE_DIM), torch.rand(b, FEATURE_DIM)


def test_readout_tells_the_hub_columns_apart() -> None:
    # Old behaviour: self-attention without positions, mean-pooled over the columns,
    # is invariant to permuting them, so which column carries a signal was lost.
    hub = _hub().eval()
    states = torch.randn(4, N_COLUMNS, STATE_DIM)
    swapped = states[:, [1, 0, 2], :]
    vision, audio = _features(4)
    a = hub(states, HUB_COLUMNS, vision, audio)["hub_latent"]
    b = hub(swapped, HUB_COLUMNS, vision, audio)["hub_latent"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_readout_does_not_depend_on_the_scale_of_the_input_driven_part() -> None:
    # The hub states are a fixed per-column offset plus a small input-driven part.
    # Old behaviour: the latent's spread across inputs scaled with that small part,
    # so the classifier saw the label at about 1e-3 of its scale and could not learn.
    hub = _hub().train()
    offset = torch.randn(1, N_COLUMNS, STATE_DIM)
    signal = torch.randn(64, N_COLUMNS, STATE_DIM)
    vision, audio = _features(64)
    small = hub(offset + 0.05 * signal, HUB_COLUMNS, vision, audio)["hub_latent"]
    large = hub(offset + signal, HUB_COLUMNS, vision, audio)["hub_latent"]
    assert torch.allclose(small, large, rtol=1e-2, atol=1e-2)
    assert large.std(dim=0).mean() > 0.1


def test_mean_readout_is_still_available_for_old_runs() -> None:
    hub = SharedManifoldHub(STATE_DIM, LATENT_DIM, FEATURE_DIM, attention_heads=2, readout="mean")
    out = hub.eval()(torch.randn(4, N_COLUMNS, STATE_DIM), HUB_COLUMNS, *_features(4))
    assert out["hub_latent"].shape == (4, LATENT_DIM)
    assert out["attention_map"] is not None


def test_model_uses_the_flat_readout_by_default(tiny_config) -> None:
    from semmn.model import SEMMNModel

    assert tiny_config.model.hub_readout == "flat"
    model = SEMMNModel(tiny_config)
    assert model.hub.readout_mode == "flat"
    assert model.hub.readout.in_features == model.grid.hub_columns.numel() * tiny_config.model.state_dim

"""The hub must read inputs through the column grid, not around it."""

from __future__ import annotations

import torch

from semmn.hub import SharedManifoldHub

STATE_DIM, LATENT_DIM, FEATURE_DIM = 8, 4, 16
HUB_COLUMNS = torch.tensor([1, 2])


def _hub(**kwargs) -> SharedManifoldHub:
    return SharedManifoldHub(STATE_DIM, LATENT_DIM, FEATURE_DIM, attention_heads=2, **kwargs).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(4, 3, STATE_DIM), torch.rand(4, FEATURE_DIM), torch.rand(4, FEATURE_DIM)


def test_fused_latent_is_the_hub_latent_by_default() -> None:
    states, vision, audio = _inputs()
    out = _hub()(states, HUB_COLUMNS, vision, audio)
    assert torch.allclose(out["fused_latent"], out["hub_latent"])


def test_features_cannot_reach_fused_latent_around_the_hub_attention() -> None:
    hub = _hub()
    with torch.no_grad():
        hub.attn.out_proj.weight.zero_()  # cut the attention read-out
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["fused_latent"]
    b = hub(states, HUB_COLUMNS, vision * 3.0 + 1.0, audio * 3.0 + 1.0)["fused_latent"]
    assert torch.allclose(a, b)


def test_fused_latent_depends_on_hub_column_states() -> None:
    hub = _hub()
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["fused_latent"]
    perturbed = states.clone()
    perturbed[:, HUB_COLUMNS, :] += 1.0
    b = hub(perturbed, HUB_COLUMNS, vision, audio)["fused_latent"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_direct_fusion_flag_restores_the_legacy_bypass() -> None:
    hub = _hub(direct_fusion=True)
    with torch.no_grad():
        hub.attn.out_proj.weight.zero_()
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["fused_latent"]
    b = hub(states, HUB_COLUMNS, vision * 3.0 + 1.0, audio * 3.0 + 1.0)["fused_latent"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_model_config_disables_direct_fusion_by_default(tiny_config) -> None:
    from semmn.model import SEMMNModel

    assert tiny_config.model.hub_direct_fusion is False
    assert SEMMNModel(tiny_config).hub.direct_fusion is False

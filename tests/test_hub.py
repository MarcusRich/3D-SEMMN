"""The hub must read inputs through the column grid, not around it."""

from __future__ import annotations

import torch

from semmn.hub import SharedManifoldHub

STATE_DIM, LATENT_DIM, FEATURE_DIM = 8, 4, 16
HUB_COLUMNS = torch.tensor([1, 2])


def _hub(**kwargs) -> SharedManifoldHub:
    kwargs.setdefault("num_hub_columns", HUB_COLUMNS.numel())
    return SharedManifoldHub(STATE_DIM, LATENT_DIM, FEATURE_DIM, attention_heads=2, **kwargs).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(4, 3, STATE_DIM), torch.rand(4, FEATURE_DIM), torch.rand(4, FEATURE_DIM)


def test_fused_latent_is_the_hub_latent_by_default() -> None:
    states, vision, audio = _inputs()
    out = _hub()(states, HUB_COLUMNS, vision, audio)
    assert torch.allclose(out["fused_latent"], out["hub_latent"])


def test_features_cannot_reach_fused_latent_around_the_hub_readout() -> None:
    hub = _hub()
    with torch.no_grad():
        hub.readout.weight.zero_()  # cut the read-out of the hub columns
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
        hub.readout.weight.zero_()
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["fused_latent"]
    b = hub(states, HUB_COLUMNS, vision * 3.0 + 1.0, audio * 3.0 + 1.0)["fused_latent"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_model_config_disables_both_hub_bypasses_by_default(tiny_config) -> None:
    from semmn.model import SEMMNModel

    assert tiny_config.model.hub_direct_fusion is False
    assert tiny_config.model.hub_modality_tokens is False
    hub = SEMMNModel(tiny_config).hub
    assert hub.direct_fusion is False and hub.modality_tokens is False


def test_hub_latent_ignores_encoder_features_by_default() -> None:
    # Old behaviour: the features always joined the attention as two tokens.
    hub = _hub()
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["hub_latent"]
    b = hub(states, HUB_COLUMNS, vision * 3.0 + 1.0, audio * 3.0 + 1.0)["hub_latent"]
    assert torch.allclose(a, b)


def test_modality_tokens_flag_restores_the_token_route() -> None:
    hub = _hub(modality_tokens=True)
    states, vision, audio = _inputs()
    a = hub(states, HUB_COLUMNS, vision, audio)["hub_latent"]
    b = hub(states, HUB_COLUMNS, vision * 3.0 + 1.0, audio * 3.0 + 1.0)["hub_latent"]
    assert not torch.allclose(a, b, atol=1e-6)


def _hub_latent_spread(config) -> float:
    from semmn.model import SEMMNModel

    model = SEMMNModel(config).eval()
    vision, audio = torch.rand(8, 1, 28, 28), torch.rand(8, 1, 112, 112)
    with torch.no_grad():
        return model(vision, audio)["hub_latent"].std(dim=0).max().item()


def _eval_config():
    # The eval grid keeps the hub columns disjoint from both input lobes (the tiny
    # 4x4 test grid does not), so the hub can only hear the inputs through the graph.
    import eval_small

    cfg = eval_small.load_eval_config("config/eval-small.yaml")[0]
    from semmn.grid import build_spatial_grid

    grid = build_spatial_grid(cfg)
    lobes = set(grid.vision_columns.tolist()) | set(grid.audio_columns.tolist())
    assert lobes.isdisjoint(grid.hub_columns.tolist())
    return cfg


def test_with_connectivity_lesioned_the_hub_sees_no_input() -> None:
    # A lesion must leave the hub latent constant across inputs. With the token route
    # on (the old behaviour), the encoder features still reached it.
    cfg = _eval_config()
    cfg.connectivity.enabled = False
    assert _hub_latent_spread(cfg) < 1e-6


def test_with_connectivity_the_hub_latent_varies_with_the_input() -> None:
    assert _hub_latent_spread(_eval_config()) > 1e-4

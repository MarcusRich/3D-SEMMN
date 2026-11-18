"""Inter-column messages must reach the column output at initialisation."""

from __future__ import annotations

import torch

from semmn.columns import ColumnarDynamics
from semmn.config import ConnectivityConfig
from semmn.connectivity import SparseInterColumnConnectivity
from semmn.model import SEMMNModel

STATE_DIM = 8


def _connectivity() -> SparseInterColumnConnectivity:
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    conn = SparseInterColumnConnectivity(centers, STATE_DIM, ConnectivityConfig(inter_density=1.0))
    assert conn.row_idx.numel() > 0
    with torch.no_grad():
        conn.edge_weights.fill_(0.5)  # hand-set, so the test does not depend on the random draw
    return conn


def _states() -> torch.Tensor:
    # Non-zero per-column mean, so the pooled activity the messages carry is non-zero.
    return torch.randn(4, 3, STATE_DIM) + 1.0


def test_zeroing_edges_changes_column_output_at_init() -> None:
    conn, columns, states = _connectivity(), ColumnarDynamics(3, STATE_DIM), _states()
    with_edges = columns(states, conn(states)[0])
    with torch.no_grad():
        conn.edge_weights.zero_()
    without_edges = columns(states, conn(states)[0])
    assert not torch.allclose(with_edges, without_edges, atol=1e-5)


def test_scaling_inter_gain_changes_column_output_at_init() -> None:
    conn, columns, states = _connectivity(), ColumnarDynamics(3, STATE_DIM), _states()
    base = columns(states, conn(states)[0])
    with torch.no_grad():
        conn.inter_gain.mul_(2.0)
    scaled = columns(states, conn(states)[0])
    assert not torch.allclose(base, scaled, atol=1e-5)


def test_edge_weights_receive_gradient_at_init() -> None:
    conn, columns, states = _connectivity(), ColumnarDynamics(3, STATE_DIM), _states()
    columns(states, conn(states)[0]).pow(2).sum().backward()
    assert conn.edge_weights.grad is not None
    # On the pre-LayerNorm path this gradient is zero up to float error (~1e-4).
    assert conn.edge_weights.grad.abs().max().item() > 1e-2


def test_model_logits_depend_on_connectivity_at_init(tiny_config) -> None:
    model = SEMMNModel(tiny_config).eval()
    with torch.no_grad():
        model.connectivity.edge_weights.fill_(0.5)
        vision, audio = torch.rand(4, 1, 28, 28), torch.rand(4, 1, 112, 112)
        with_edges = model(vision, audio)["logits"]
        model.connectivity.edge_weights.zero_()
        without_edges = model(vision, audio)["logits"]
    assert not torch.allclose(with_edges, without_edges, atol=1e-5)


def test_disabled_connectivity_sends_no_messages_and_skips_stdp() -> None:
    conn = _connectivity()
    conn.enabled = False
    states = _states()
    messages, activity = conn(states)
    assert torch.count_nonzero(messages) == 0
    assert activity.shape == (4, 3)
    before = conn.edge_weights.detach().clone()
    conn.stdp_update(activity, activity)
    assert torch.equal(conn.edge_weights, before)


def test_connectivity_flag_matches_zeroed_edges(tiny_config) -> None:
    tiny_config.connectivity.enabled = False
    model = SEMMNModel(tiny_config).eval()
    vision, audio = torch.rand(4, 1, 28, 28), torch.rand(4, 1, 112, 112)
    with torch.no_grad():
        lesioned = model(vision, audio)["logits"]
        model.connectivity.enabled = True
        model.connectivity.edge_weights.zero_()
        zeroed = model(vision, audio)["logits"]
    assert torch.allclose(lesioned, zeroed)

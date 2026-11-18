"""The inter-column graph must be dense enough to carry input from the lobes to the hub."""

from __future__ import annotations

import math

import pytest
import torch

import eval_small
from semmn.config import ConnectivityConfig, SEMMNConfig
from semmn.connectivity import SparseInterColumnConnectivity
from semmn.grid import build_spatial_grid
from semmn.model import SEMMNModel

EVAL_CONFIG = "config/eval-small.yaml"


@pytest.fixture(scope="module")
def eval_config() -> SEMMNConfig:
    return eval_small.load_eval_config(EVAL_CONFIG)[0]


def _graph(cfg: SEMMNConfig, seed: int) -> SparseInterColumnConnectivity:
    torch.manual_seed(seed)
    grid = build_spatial_grid(cfg)
    return SparseInterColumnConnectivity(grid.column_centers, cfg.model.state_dim, cfg.connectivity)


def _reachable(conn: SparseInterColumnConnectivity, sources: torch.Tensor, hops: int) -> torch.Tensor:
    """Columns a message from `sources` reaches within `hops` steps (edges run col -> row)."""
    n = conn.num_columns
    adj = torch.zeros(n, n)
    adj[conn.row_idx, conn.col_idx] = 1.0
    reached = torch.zeros(n, dtype=torch.bool)
    reached[sources] = True
    for _ in range(hops):
        reached |= (adj @ reached.float()) > 0
    return reached


@pytest.mark.parametrize("seed", range(5))
def test_hub_is_reachable_from_both_lobes_within_the_recurrent_steps(eval_config, seed) -> None:
    # The old sampler drew 2-11 edges over 100 columns and reached no hub column.
    conn = _graph(eval_config, seed)
    grid = build_spatial_grid(eval_config)
    hops = eval_config.model.recurrent_steps
    for lobe in (grid.vision_columns, grid.audio_columns):
        frac = _reachable(conn, lobe, hops)[grid.hub_columns].float().mean().item()
        assert frac >= 0.9


def test_edge_count_matches_inter_density(eval_config) -> None:
    c = eval_config.n_columns
    expected = eval_config.connectivity.inter_density * c * (c - 1)
    counts = [_graph(eval_config, seed).row_idx.numel() for seed in range(5)]
    mean = sum(counts) / len(counts)
    # Binomial-ish spread on ~1000 edges is a few percent.
    assert abs(mean - expected) < 0.1 * expected


def test_edge_probability_decays_with_distance_in_column_spacings() -> None:
    # Two grids with the same layout in different units must give the same probabilities.
    centers = torch.tensor([[float(i), 0.0, 0.0] for i in range(6)])
    dist_a = torch.cdist(centers, centers)
    dist_b = torch.cdist(centers * 10.0, centers * 10.0)
    p_a = SparseInterColumnConnectivity.edge_probabilities(dist_a, 1.0, 0.3)
    p_b = SparseInterColumnConnectivity.edge_probabilities(dist_b, 1.0, 0.3)
    assert torch.allclose(p_a, p_b, atol=1e-5)
    assert p_a[0, 1] > p_a[0, 2] > p_a[0, 5]
    assert torch.all(p_a.diagonal() == 0)
    assert math.isclose(p_a.sum().item(), 0.3 * 6 * 5, rel_tol=1e-3)


def test_initial_edge_weights_are_all_non_zero(eval_config) -> None:
    # A signed init clamped at min_weight = 0 left about half the edges at exactly 0.
    conn = _graph(eval_config, 0)
    assert torch.all(conn.edge_weights > 0)


def test_hub_states_vary_with_the_input_at_init(eval_config) -> None:
    # The hub only receives input through the graph. With the old sampler the hub
    # columns were constant across inputs (std ~0); they must now vary on the scale
    # of the input lobes.
    torch.manual_seed(0)
    model = SEMMNModel(eval_config).eval()
    vision, audio = torch.rand(32, 1, 28, 28), torch.rand(32, 1, 112, 112)
    with torch.no_grad():
        states = model(vision, audio)["column_states"]
    grid = model.grid
    hub_std = states[:, grid.hub_columns].std(dim=0).mean().item()
    lobe_std = states[:, torch.cat([grid.vision_columns, grid.audio_columns])].std(dim=0).mean().item()
    assert hub_std > 0.1 * lobe_std


def test_shipped_eval_config_runs_stdp_within_the_budget(eval_config) -> None:
    _, eval_cfg = eval_small.load_eval_config(EVAL_CONFIG)
    steps_per_epoch = math.ceil(eval_cfg["train_size"] / eval_config.training.batch_size)
    steps = steps_per_epoch * eval_config.training.epochs
    assert steps // eval_config.connectivity.stdp_interval >= 3


def test_stdp_update_changes_weights_and_reports_the_change() -> None:
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    conn = SparseInterColumnConnectivity(centers, 4, ConnectivityConfig(inter_density=1.0))
    before = conn.edge_weights.detach().clone()
    activity = torch.rand(8, 3) + 0.5
    applied = conn.stdp_update(activity, activity)
    assert applied > 0.0
    assert not torch.equal(before, conn.edge_weights)
    assert math.isclose(applied, (conn.edge_weights - before).abs().mean().item(), rel_tol=1e-5)

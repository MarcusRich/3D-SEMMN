"""Training must not collapse the hub's spread across inputs.

In P1, the grid lost 8-19x of its spread across inputs in the first epoch on each seed
measured (0, 1 and 3), and on two of five seeds the hub latent ended near-constant. This
trains the eval-grid model briefly on a tiny subset and checks the hub-column states
still vary with the input, on held-out samples, in eval mode. It checks the hub-column
states rather than the hub latent: the latent now comes out of a BatchNorm, so its
spread is near unit scale by construction and a threshold on it would test nothing.
"""

from __future__ import annotations

import pytest
import torch

import eval_small
from semmn.losses import compute_losses
from semmn.model import SEMMNModel

# Spread of the hub-column states across inputs, per feature, over their RMS: a
# scale-free number, so a normalisation layer downstream cannot pass it by itself.
# About 0.08 at init. After the run below: 0.001-0.005 on the P1 code, 0.03-0.04 with
# the P2 read-out alone, 0.07-0.09 with the column bias moved after the norm.
MIN_RELATIVE_SPREAD = 0.05
STEPS_PER_EPOCH, EPOCHS = 4, 8  # 32 optimizer steps; P1 collapsed within its first 32


@pytest.fixture(scope="module")
def setup():
    cfg, _ = eval_small.load_eval_config("config/eval-small.yaml")
    sizes = {"data_seed": 1234, "train_size": 64 * STEPS_PER_EPOCH, "val_size": 256, "test_size": 1}
    return cfg, eval_small.build_splits(sizes, data_cfg=cfg.data)


def relative_hub_spread(model: SEMMNModel, split: eval_small.Split) -> float:
    model.eval()
    with torch.no_grad():
        states = model(split.vision, split.audio)["column_states"]
    hub = states[:, model.grid.hub_columns, :].flatten(1)
    return (hub.std(dim=0).mean() / hub.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("seed", [0, 3])  # the seeds whose hub collapsed in the P1 eval
def test_hub_keeps_its_spread_across_inputs_after_a_short_run(setup, seed: int) -> None:
    cfg, splits = setup
    eval_small.set_seed(seed)
    model = SEMMNModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        for vision, audio, label in eval_small._batches(splits["train"], 64, shuffle):
            optimizer.zero_grad(set_to_none=True)
            compute_losses(model(vision, audio), label, cfg, model.connectivity)["total"].backward()
            optimizer.step()
    assert relative_hub_spread(model, splits["val"]) > MIN_RELATIVE_SPREAD

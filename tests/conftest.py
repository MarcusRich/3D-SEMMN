"""Shared pytest setup: keep tests small, CPU-only and deterministic."""

from __future__ import annotations

import pytest
import torch

from semmn.config import SEMMNConfig

torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def tiny_config() -> SEMMNConfig:
    """A 4x4-column model small enough to run a forward pass in milliseconds."""
    return SEMMNConfig.from_dict(
        {
            "grid": {"dims": [20, 20, 4], "n_columns_xy": [4, 4]},
            "model": {
                "state_dim": 8,
                "latent_dim": 4,
                "vision_feature_dim": 16,
                "audio_feature_dim": 16,
                "recurrent_steps": 2,
                "attention_heads": 2,
            },
        }
    )

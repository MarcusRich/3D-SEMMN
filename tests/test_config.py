"""Config loading must fail on typos and must keep the eval-small section."""

from __future__ import annotations

import pytest

from semmn.config import SEMMNConfig


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="modle"):
        SEMMNConfig.from_dict({"modle": {}})
    with pytest.raises(ValueError, match="not_a_real_key"):
        SEMMNConfig.from_dict({"grid": {"not_a_real_key": 1}})


def test_eval_section_is_allowed_and_not_a_model_field() -> None:
    cfg = SEMMNConfig.from_dict({"seed": 3, "eval": {"train_size": 10}})
    assert cfg.seed == 3
    assert not hasattr(cfg, "eval")


def test_shipped_configs_load() -> None:
    for path in ("config/default.yaml", "config/light.yaml", "config/eval-small.yaml"):
        SEMMNConfig.from_yaml(path)

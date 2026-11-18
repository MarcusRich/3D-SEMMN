"""Fixed split and end-to-end smoke test for eval_small.py (reads the local MNIST files)."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

import eval_small

SIZES = {"data_seed": 7, "train_size": 40, "val_size": 20, "test_size": 20}


def test_splits_are_fixed_and_disjoint() -> None:
    a = eval_small.build_splits(SIZES)
    b = eval_small.build_splits(SIZES)
    for name in ("train", "val", "test"):
        assert torch.equal(a[name].source_indices, b[name].source_indices)
        assert torch.equal(a[name].label, b[name].label)
    train_idx = set(a["train"].source_indices.tolist())
    assert train_idx.isdisjoint(a["val"].source_indices.tolist())
    assert len(a["train"]) == 40 and len(a["val"]) == 20 and len(a["test"]) == 20


def test_split_changes_with_data_seed() -> None:
    a = eval_small.build_splits(SIZES)
    b = eval_small.build_splits({**SIZES, "data_seed": 8})
    assert not torch.equal(a["train"].source_indices, b["train"].source_indices)


def test_shipped_config_has_a_fixed_split() -> None:
    _, eval_cfg = eval_small.load_eval_config("config/eval-small.yaml")
    assert {"data_seed", "train_size", "val_size", "test_size", "seeds"} <= set(eval_cfg)
    assert len(eval_cfg["seeds"]) == 5


def test_end_to_end_smoke(tmp_path: Path) -> None:
    config = {
        "seed": 0,
        "grid": {"dims": [20, 20, 4], "n_columns_xy": [4, 4]},
        "model": {"state_dim": 8, "latent_dim": 4, "recurrent_steps": 2, "attention_heads": 2},
        "training": {"batch_size": 16, "epochs": 1, "num_workers": 0, "amp": False},
        "eval": {**SIZES, "seeds": [0, 1], "output_dir": str(tmp_path / "out")},
    }
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = eval_small.run(str(config_path))

    methods = {r["method"] for r in rows}
    assert methods == {
        "semmn",
        "semmn_lesion_test",
        "semmn_lesion_train",
        "mlp",
        "mlp_vision_only",
        "mlp_audio_only",
        "logreg",
        "logreg_vision_only",
        "logreg_audio_only",
    }
    assert all(0.0 <= r["test_accuracy"] <= 1.0 for r in rows)
    table = (tmp_path / "out" / "results.md").read_text(encoding="utf-8")
    assert "| semmn | 2 |" in table
    assert "| logreg | 1 |" in table
    assert (tmp_path / "out" / "results.csv").exists()

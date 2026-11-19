"""train.py must select checkpoints on a held-out validation split, never on test."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import train
from semmn.dataset import build_dataloaders


def _cfg(tiny_config, val_size: int = 50):
    if hasattr(tiny_config, "data"):  # absent before the split existed; lets the tests fail on the old code
        tiny_config.data.val_size = val_size
    tiny_config.training.num_workers = 0
    tiny_config.training.batch_size = 16
    tiny_config.training.amp = False
    return tiny_config


def test_validation_split_is_held_out_from_the_training_pool(tiny_config) -> None:
    # The old build_dataloaders returned MNIST-t10k as the "val" loader.
    loaders = build_dataloaders(_cfg(tiny_config))
    train_ds, val_ds = loaders[0].dataset, loaders[1].dataset
    assert getattr(val_ds, "mnist", None) is None or val_ds.mnist.train, "val is the test split"
    assert isinstance(train_ds, Subset) and isinstance(val_ds, Subset)
    assert train_ds.dataset is val_ds.dataset
    assert train_ds.dataset.mnist.train is True
    assert len(val_ds) == 50
    assert set(train_ds.indices).isdisjoint(val_ds.indices)
    assert len(train_ds) + len(val_ds) == len(train_ds.dataset)


def test_test_split_is_mnist_t10k(tiny_config) -> None:
    loaders = build_dataloaders(_cfg(tiny_config))
    assert len(loaders) == 3, "expected separate train, val and test loaders"
    assert loaders[2].dataset.mnist.train is False


def test_validation_split_is_fixed_by_split_seed(tiny_config) -> None:
    cfg = _cfg(tiny_config)
    a = build_dataloaders(cfg)[1].dataset.indices
    b = build_dataloaders(cfg)[1].dataset.indices
    cfg.data.split_seed += 1
    c = build_dataloaders(cfg)[1].dataset.indices
    assert list(a) == list(b)
    assert list(a) != list(c)


def _tiny_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    small = lambda loader, n: DataLoader(Subset(loader.dataset, range(n)), batch_size=16)
    return small(train_loader, 32), small(val_loader, 16), small(test_loader, 16)


def test_best_checkpoint_is_chosen_on_val_and_test_is_scored_once(tiny_config, tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tiny_config)
    cfg.training.epochs = 3
    loaders = _tiny_loaders(cfg)
    val_scores = iter([0.2, 0.9, 0.5])  # epoch 1 is best on val
    calls: list[str] = []
    tested_weights: dict[str, torch.Tensor] = {}

    def fake_evaluate(model, loader, cfg, device, max_batches):
        if loader is loaders[1]:
            calls.append("val")
            return {"accuracy": next(val_scores), "cross_modal_retrieval": 0.0}
        assert loader is loaders[2], "evaluate called on an unexpected loader"
        calls.append("test")
        tested_weights.update({k: v.detach().clone() for k, v in model.state_dict().items()})
        return {"accuracy": 0.42, "cross_modal_retrieval": 0.0}

    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    summary = train.run(cfg, torch.device("cpu"), tmp_path, loaders=loaders)

    assert calls == ["val", "val", "val", "test"]
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == 1
    assert best["selected_on"] == "val_accuracy"
    assert summary["best_epoch"] == 1
    assert summary["test"]["accuracy"] == 0.42
    # Test is scored with the selected (epoch 1) weights, not the last epoch's.
    assert all(torch.equal(tested_weights[k], v) for k, v in best["model"].items())

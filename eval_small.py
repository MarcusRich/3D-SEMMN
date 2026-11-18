"""Small, reproducible test-set evaluation of 3D-SEMMN against simple baselines.

Every run uses the same fixed data split (config `eval.data_seed`) and repeats
training over `eval.seeds`. Epochs are selected on the validation split only;
the test split is scored once, after selection.

Rows reported:
  semmn                      full model
  semmn_lesion_train         trained and tested with inter-column connectivity off
  semmn_lesion_test          the full model with connectivity switched off at test time
  *_hub_probe                a fixed logistic-regression probe on the frozen model's
                             hub-column states (fit on train, scored on test): how much
                             label information reaches the hub, apart from the read-out
  semmn_hub_tokens           reference: the full model with model.hub_modality_tokens on
  mlp                        one-hidden-layer MLP on the raw vision + audio inputs
  mlp_vision_only            the same MLP on vision alone (reference)
  mlp_audio_only             the same MLP on audio alone (reference)
  logreg                     logistic regression on the raw vision + audio inputs
  logreg_vision_only         reference: vision input alone
  logreg_audio_only          reference: audio input alone. The synthetic audio is only
                             weakly informative (config `data`), so this should sit well
                             below 1.0; a value near 1.0 means the audio leaks the label.

usage: python eval_small.py --config config/eval-small.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from semmn.config import DataConfig, SEMMNConfig
from semmn.dataset import SyntheticAVMNIST
from semmn.losses import compute_losses
from semmn.model import SEMMNModel

DATA_ROOT = "./data"


@dataclass
class Split:
    vision: torch.Tensor  # (N, 1, 28, 28)
    audio: torch.Tensor  # (N, 1, 112, 112)
    label: torch.Tensor  # (N,)
    source_indices: torch.Tensor  # indices into the MNIST split it came from

    def __len__(self) -> int:
        return int(self.label.numel())

    def flat_features(self, which: str = "both") -> np.ndarray:
        parts = []
        if which in ("both", "vision"):
            parts.append(self.vision.flatten(1))
        if which in ("both", "audio"):
            parts.append(self.audio.flatten(1))
        return torch.cat(parts, dim=1).numpy()


def load_eval_config(path: str | Path) -> tuple[SEMMNConfig, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "eval" not in raw:
        raise KeyError(f"{path} has no `eval` section")
    return SEMMNConfig.from_dict(raw), raw["eval"]


def _materialise(ds: SyntheticAVMNIST, indices: torch.Tensor) -> Split:
    items = [ds[int(i)] for i in indices]
    return Split(
        vision=torch.stack([v for v, _, _ in items]),
        audio=torch.stack([a for _, a, _ in items]),
        label=torch.stack([y for _, _, y in items]),
        source_indices=indices.clone(),
    )


def build_splits(
    eval_cfg: dict[str, Any], root: str = DATA_ROOT, data_cfg: DataConfig | None = None
) -> dict[str, Split]:
    """Fixed train/val (disjoint, from MNIST-train) and test (from MNIST-t10k)."""
    data_cfg = data_cfg or DataConfig()
    g = torch.Generator().manual_seed(int(eval_cfg["data_seed"]))
    audio = {"audio_snr_db": data_cfg.audio_snr_db, "audio_pitch_jitter": data_cfg.audio_pitch_jitter}
    train_pool = SyntheticAVMNIST(root=root, train=True, download=False, **audio)
    test_pool = SyntheticAVMNIST(root=root, train=False, download=False, **audio)
    n_train, n_val, n_test = (int(eval_cfg[k]) for k in ("train_size", "val_size", "test_size"))
    if n_train + n_val > len(train_pool) or n_test > len(test_pool):
        raise ValueError("eval split sizes exceed the available MNIST data")

    perm = torch.randperm(len(train_pool), generator=g)
    test_perm = torch.randperm(len(test_pool), generator=g)
    return {
        "train": _materialise(train_pool, perm[:n_train]),
        "val": _materialise(train_pool, perm[n_train : n_train + n_val]),
        "test": _materialise(test_pool, test_perm[:n_test]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(split: Split, batch_size: int, generator: torch.Generator | None = None):
    n = len(split)
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield split.vision[idx], split.audio[idx], split.label[idx]


@torch.no_grad()
def semmn_accuracy(model: SEMMNModel, split: Split, batch_size: int) -> float:
    model.eval()
    correct = 0
    for vision, audio, label in _batches(split, batch_size):
        correct += (model(vision, audio)["logits"].argmax(dim=1) == label).sum().item()
    return correct / len(split)


@torch.no_grad()
def hub_diagnostics(model: SEMMNModel, split: Split, batch_size: int) -> dict[str, float]:
    """How much the hub varies across samples, in absolute terms and against the input lobes.

    `hub_state_std` near 0 means the hub columns see no input; `hub_to_lobe_std` puts it
    on the scale of the vision and audio columns, which receive the input directly.
    `hub_latent_std` is the spread of the hub's output latent.
    """
    model.eval()
    grid = model.grid
    lobes = torch.cat([grid.vision_columns, grid.audio_columns])
    hub, lobe, latent = [], [], []
    for v, a, _ in _batches(split, batch_size):
        out = model(v, a)
        hub.append(out["column_states"][:, grid.hub_columns, :])
        lobe.append(out["column_states"][:, lobes, :])
        latent.append(out["hub_latent"])
    hub_std = torch.cat(hub).std(dim=0).mean().item()
    lobe_std = torch.cat(lobe).std(dim=0).mean().item()
    return {
        "hub_state_std": hub_std,
        "hub_to_lobe_std": hub_std / max(lobe_std, 1e-12),
        "hub_latent_std": torch.cat(latent).std(dim=0).mean().item(),
    }


@torch.no_grad()
def _hub_state_features(model: SEMMNModel, split: Split, batch_size: int) -> np.ndarray:
    model.eval()
    hub = model.grid.hub_columns
    return torch.cat(
        [model(v, a)["column_states"][:, hub, :].flatten(1) for v, a, _ in _batches(split, batch_size)]
    ).numpy()


def hub_probe(model: SEMMNModel, splits: dict[str, Split], batch_size: int) -> float:
    """Test accuracy of a fixed logistic regression on the trained model's hub-column states.

    The probe is fit on train and scored on test, with the model frozen. It measures how
    much label information reaches the hub through the grid, separately from whether the
    model's own read-out can use it. The hub gets no direct input, so with connectivity
    off its states are constant and the probe sits at the majority-class rate.
    """
    x_train = _hub_state_features(model, splits["train"], batch_size)
    x_test = _hub_state_features(model, splits["test"], batch_size)
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=1.0, max_iter=500).fit(scaler.transform(x_train), splits["train"].label.numpy())
    return float(clf.score(scaler.transform(x_test), splits["test"].label.numpy()))


def _train_with_val_selection(
    model: nn.Module,
    epochs: int,
    train_epoch: Callable[[int], None],
    val_score: Callable[[], float],
) -> tuple[nn.Module, int, float]:
    best_state, best_epoch, best_val = copy.deepcopy(model.state_dict()), -1, -1.0
    for epoch in range(epochs):
        train_epoch(epoch)
        score = val_score()
        if score > best_val:
            best_state, best_epoch, best_val = copy.deepcopy(model.state_dict()), epoch, score
    model.load_state_dict(best_state)
    return model, best_epoch, best_val


def train_semmn(
    cfg: SEMMNConfig,
    splits: dict[str, Split],
    seed: int,
    connectivity: bool,
    hub_modality_tokens: bool | None = None,
) -> tuple[SEMMNModel, dict[str, Any]]:
    cfg = copy.deepcopy(cfg)
    cfg.connectivity.enabled = connectivity
    if hub_modality_tokens is not None:
        cfg.model.hub_modality_tokens = hub_modality_tokens
    set_seed(seed)
    model = SEMMNModel(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    shuffle = torch.Generator().manual_seed(seed)
    bs = cfg.training.batch_size
    step = 0
    stdp_deltas: list[float] = []

    def train_epoch(_: int) -> None:
        nonlocal step
        model.train()
        for vision, audio, label in _batches(splits["train"], bs, shuffle):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(vision, audio)
            compute_losses(outputs, label, cfg, model.connectivity)["total"].backward()
            optimizer.step()
            step += 1
            if cfg.connectivity.stdp_due(step):
                # Same schedule as train.py. Skipped when lesioned (it would be a no-op).
                stdp_deltas.append(model.apply_stdp(outputs["pre_activity"], outputs["post_activity"]))

    model, best_epoch, best_val = _train_with_val_selection(
        model, cfg.training.epochs, train_epoch, lambda: semmn_accuracy(model, splits["val"], bs)
    )
    info = {
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "edges": int(model.connectivity.row_idx.numel()),
        "nonzero_edges": model.active_parameters(),
        "stdp_updates": len(stdp_deltas),
        "stdp_mean_abs_dw": float(np.mean(stdp_deltas)) if stdp_deltas else 0.0,
    }
    return model, info


def train_mlp(
    splits: dict[str, Split], seed: int, epochs: int, batch_size: int, lr: float, which: str = "both"
) -> tuple[float, dict[str, Any]]:
    set_seed(seed)
    scaler = StandardScaler().fit(splits["train"].flat_features(which))
    x = {
        k: torch.tensor(scaler.transform(s.flat_features(which)), dtype=torch.float32)
        for k, s in splits.items()
    }
    y = {k: s.label for k, s in splits.items()}
    model = nn.Sequential(nn.Linear(x["train"].shape[1], 256), nn.ReLU(), nn.Linear(256, 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    shuffle = torch.Generator().manual_seed(seed)

    def accuracy(split: str) -> float:
        model.eval()
        with torch.no_grad():
            return (model(x[split]).argmax(dim=1) == y[split]).float().mean().item()

    def train_epoch(_: int) -> None:
        model.train()
        order = torch.randperm(len(y["train"]), generator=shuffle)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x["train"][idx]), y["train"][idx]).backward()
            optimizer.step()

    model, best_epoch, best_val = _train_with_val_selection(
        model, epochs, train_epoch, lambda: accuracy("val")
    )
    return accuracy("test"), {"best_epoch": best_epoch, "val_accuracy": best_val}


def logreg(splits: dict[str, Split], which: str) -> tuple[float, dict[str, Any]]:
    """Fixed hyperparameters (no tuning); lbfgs on a convex loss, so no seed dependence."""
    scaler = StandardScaler().fit(splits["train"].flat_features(which))
    clf = LogisticRegression(C=1.0, max_iter=200)
    clf.fit(scaler.transform(splits["train"].flat_features(which)), splits["train"].label.numpy())

    def score(split: str) -> float:
        return float(clf.score(scaler.transform(splits[split].flat_features(which)), splits[split].label.numpy()))

    return score("test"), {"val_accuracy": score("val")}


def run(config_path: str, output_dir: str | None = None) -> list[dict[str, Any]]:
    cfg, eval_cfg = load_eval_config(config_path)
    torch.set_num_threads(int(eval_cfg.get("threads", 4)))
    out_dir = Path(output_dir or eval_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(eval_cfg, data_cfg=cfg.data)
    bs, epochs = cfg.training.batch_size, cfg.training.epochs
    rows: list[dict[str, Any]] = []

    def record(method: str, seed: int, test_acc: float, **info: Any) -> None:
        row = {"method": method, "seed": seed, "test_accuracy": test_acc, **info}
        rows.append(row)
        print(json.dumps(row), flush=True)

    for seed in eval_cfg["seeds"]:
        model, info = train_semmn(cfg, splits, seed, connectivity=True)
        info.update(hub_diagnostics(model, splits["val"], bs))
        record("semmn", seed, semmn_accuracy(model, splits["test"], bs), **info)
        record("semmn_hub_probe", seed, hub_probe(model, splits, bs))
        model.connectivity.enabled = False
        record("semmn_lesion_test", seed, semmn_accuracy(model, splits["test"], bs))
        record("semmn_lesion_test_hub_probe", seed, hub_probe(model, splits, bs))

        model, info = train_semmn(cfg, splits, seed, connectivity=False)
        record("semmn_lesion_train", seed, semmn_accuracy(model, splits["test"], bs), **info)
        record("semmn_lesion_train_hub_probe", seed, hub_probe(model, splits, bs))

        # Reference: the same model with the encoder-feature tokens back in the hub.
        model, info = train_semmn(cfg, splits, seed, connectivity=True, hub_modality_tokens=True)
        record("semmn_hub_tokens", seed, semmn_accuracy(model, splits["test"], bs), **info)

        for which, name in (("both", "mlp"), ("vision", "mlp_vision_only"), ("audio", "mlp_audio_only")):
            acc, info = train_mlp(splits, seed, epochs, bs, cfg.training.lr, which)
            record(name, seed, acc, **info)

    # Logistic regression is convex with a deterministic solver, so it runs once.
    for which, name in (("both", "logreg"), ("vision", "logreg_vision_only"), ("audio", "logreg_audio_only")):
        acc, info = logreg(splits, which)
        record(name, -1, acc, **info)

    write_outputs(rows, out_dir, config_path, eval_cfg, cfg, splits)
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[tuple[str, int, float, float]]:
    order = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for method in order:
        accs = np.array([r["test_accuracy"] for r in rows if r["method"] == method])
        std = float(accs.std(ddof=1)) if accs.size > 1 else 0.0
        out.append((method, int(accs.size), float(accs.mean()), std))
    return out


def write_outputs(
    rows: list[dict[str, Any]],
    out_dir: Path,
    config_path: str,
    eval_cfg: dict[str, Any],
    cfg: SEMMNConfig,
    splits: dict[str, Split],
) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# eval-small results",
        "",
        f"Config: `{config_path}`. Split seed {eval_cfg['data_seed']}: "
        f"train {len(splits['train'])} / val {len(splits['val'])} (MNIST-train, disjoint), "
        f"test {len(splits['test'])} (MNIST-t10k). {cfg.training.epochs} epochs, "
        "epoch selected on val accuracy, test scored once.",
        "",
        "| method | runs | test accuracy (mean ± std over seeds) |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {m} | {n} | {mean:.4f} ± {std:.4f} |" if n > 1 else f"| {m} | {n} | {mean:.4f} (single deterministic run) |"
        for m, n, mean, std in summarise(rows)
    ]
    semmn_rows = [r for r in rows if r["method"] == "semmn"]
    if semmn_rows:
        lines += [
            "",
            "Per-seed SEMMN diagnostics on the val split: sampled inter-column edges, non-zero "
            "edges after training, hub-column state std across samples (absolute and relative "
            "to the input lobes), hub latent std, STDP updates applied and their mean |dw|:",
            "",
            "| seed | best epoch | edges | non-zero edges | hub state std | hub/lobe std | hub latent std | STDP updates | STDP mean abs dw |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['seed']} | {r['best_epoch']} | {r['edges']} | {r['nonzero_edges']} | {r['hub_state_std']:.2e} "
            f"| {r['hub_to_lobe_std']:.3f} | {r['hub_latent_std']:.2e} | {r['stdp_updates']} | {r['stdp_mean_abs_dw']:.2e} |"
            for r in semmn_rows
        ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small fixed-split evaluation of 3D-SEMMN")
    parser.add_argument("--config", type=str, default="config/eval-small.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()

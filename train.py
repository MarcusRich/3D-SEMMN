"""Training entrypoint for 3D-SEMMN."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from semmn.config import SEMMNConfig
from semmn.dataset import build_dataloaders
from semmn.losses import compute_losses
from semmn.model import SEMMNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D-SEMMN prototype")
    parser.add_argument("--config", type=str, default="config/light.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: SEMMNModel,
    loader,
    cfg: SEMMNConfig,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    retrieval_hits = 0
    retrieval_total = 0
    for bi, (vision, audio, label) in enumerate(loader):
        if bi >= max_batches:
            break
        vision = vision.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        out = model(vision, audio)
        pred = out["logits"].argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()

        v = torch.nn.functional.normalize(out["vision_embed"], dim=-1)
        a = torch.nn.functional.normalize(out["audio_embed"], dim=-1)
        sim = v @ a.t()
        retrieved = sim.argmax(dim=1)
        retrieval_hits += (label == label[retrieved]).sum().item()
        retrieval_total += label.numel()

    return {
        "accuracy": correct / max(total, 1),
        "cross_modal_retrieval": retrieval_hits / max(retrieval_total, 1),
    }


def run(
    cfg: SEMMNConfig,
    device: torch.device,
    out_dir: Path,
    loaders: tuple | None = None,
) -> dict:
    """Train, keep the epoch with the best validation accuracy, then score test once.

    The validation split is held out from MNIST-train (`data.val_size`,
    `data.split_seed`). Test is never used for selection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = loaders or build_dataloaders(cfg)
    model = SEMMNModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scaler = GradScaler(enabled=(cfg.training.amp and device.type == "cuda"))

    history: list[dict[str, float]] = []
    best_val = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        for step, (vision, audio, label) in enumerate(train_loader):
            vision = vision.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(vision, audio)
                losses = compute_losses(outputs, label, cfg, model.connectivity)

            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if cfg.connectivity.stdp_due(global_step):
                model.apply_stdp(outputs["pre_activity"], outputs["post_activity"])

            if step % cfg.training.log_interval == 0:
                row = {
                    "epoch": float(epoch),
                    "step": float(step),
                    "loss_total": float(losses["total"].detach().cpu()),
                    "loss_cls": float(losses["classification"].detach().cpu()),
                    "loss_contrastive": float(losses["contrastive"].detach().cpu()),
                    "loss_manifold": float(losses["manifold"].detach().cpu()),
                    "active_params": float(model.active_parameters()),
                    "energy_proxy": float(model.energy_proxy().detach().cpu()),
                }
                history.append(row)
                print(json.dumps(row))

        val_metrics = evaluate(model, val_loader, cfg, device, cfg.training.val_batches)
        history.append(
            {
                "epoch": float(epoch),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_cross_modal_retrieval": float(val_metrics["cross_modal_retrieval"]),
            }
        )
        print(json.dumps({"epoch": epoch, "split": "val", **val_metrics}))

        # Select on validation classification accuracy: it is the task the model is
        # trained for, and retrieval only probes the encoder projections.
        if val_metrics["accuracy"] > best_val:
            best_val, best_epoch = val_metrics["accuracy"], epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "selected_on": "val_accuracy",
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

    torch.save({"model": model.state_dict()}, out_dir / "last.pt")

    # Score the selected checkpoint on test exactly once.
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, cfg, device, max_batches=len(test_loader))
    history.append(
        {
            "epoch": float(best_epoch),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_cross_modal_retrieval": float(test_metrics["cross_modal_retrieval"]),
        }
    )
    print(json.dumps({"epoch": best_epoch, "split": "test", **test_metrics}))

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"best_epoch": best_epoch, "val_accuracy": best_val, "test": test_metrics}


def main() -> None:
    args = parse_args()
    cfg = SEMMNConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run(cfg, device, Path(args.output_dir or cfg.paths.output_dir))


if __name__ == "__main__":
    main()

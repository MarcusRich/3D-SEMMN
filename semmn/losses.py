"""Losses for multimodal manifold alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from semmn.config import SEMMNConfig
from semmn.connectivity import SparseInterColumnConnectivity


def nt_xent_loss(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.transpose(0, 1) / max(temperature, 1e-6)
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: SEMMNConfig,
    connectivity: SparseInterColumnConnectivity,
) -> dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(outputs["logits"], labels)
    contrastive = nt_xent_loss(
        outputs["vision_embed"], outputs["audio_embed"], config.loss.temperature
    )
    recon = F.mse_loss(outputs["audio_from_vision"], outputs["audio_features"].detach()) + F.mse_loss(
        outputs["vision_from_audio"], outputs["vision_features"].detach()
    )
    imagination = F.mse_loss(
        outputs["cycle_vision"], outputs["vision_features"].detach()
    ) + F.mse_loss(outputs["cycle_audio"], outputs["audio_features"].detach())

    manifold = outputs["manifold_loss"]
    spatial = connectivity.spatial_wiring_cost()
    if not config.loss.use_manifold_loss:
        manifold = manifold * 0.0
    if not config.loss.use_spatial_penalty:
        spatial = spatial * 0.0

    total = (
        cls_loss
        + config.loss.contrastive_weight * contrastive
        + config.loss.reconstruction_weight * recon
        + config.loss.imagination_weight * imagination
        + config.loss.manifold_weight * manifold
        + config.loss.spatial_penalty_weight * spatial
    )
    return {
        "total": total,
        "classification": cls_loss,
        "contrastive": contrastive,
        "reconstruction": recon,
        "imagination": imagination,
        "manifold": manifold,
        "spatial": spatial,
    }

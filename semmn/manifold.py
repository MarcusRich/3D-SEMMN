"""Manifold regularization and geometric twist layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricTwistLayer(nn.Module):
    """Learnable orthogonal transform via skew-symmetric matrix exponential.

    We use `exp(A - A^T)` instead of Cayley solve for MPS stability while
    preserving an orthogonal rotation on the latent manifold.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dim, dim))

    def rotation_matrix(self) -> torch.Tensor:
        skew = self.a - self.a.transpose(0, 1)
        return torch.matrix_exp(skew)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.rotation_matrix().transpose(0, 1)


class PopulationManifoldAE(nn.Module):
    """Low-dimensional manifold constraint on column-pooled activity.

    In line with Gallego et al. (2017), this encourages dynamics to occupy
    low-dimensional trajectories rather than unconstrained high-D activity.
    """

    def __init__(self, num_columns: int, state_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        in_dim = num_columns
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_dim),
        )
        self.twist = GeometricTwistLayer(latent_dim)

    def forward(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = states.mean(dim=-1)
        latent = self.encoder(pooled)
        twisted = self.twist(latent)
        recon = self.decoder(twisted)
        recon_loss = F.mse_loss(recon, pooled)
        return {
            "pooled_activity": pooled,
            "latent": latent,
            "twisted_latent": twisted,
            "manifold_recon": recon,
            "manifold_loss": recon_loss,
        }

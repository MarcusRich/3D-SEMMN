"""Sparse distance-penalized connectivity and STDP rules."""

from __future__ import annotations

import torch
import torch.nn as nn

from semmn.config import ConnectivityConfig


class SparseInterColumnConnectivity(nn.Module):
    """Sparse inter-column graph with distance-decay sampling.

    Inspired by seRNN wiring constraints: long-range links carry a larger cost,
    so we initialize and regularize them with an exponential distance penalty.
    """

    def __init__(
        self,
        column_centers: torch.Tensor,
        state_dim: int,
        config: ConnectivityConfig,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.config = config
        self.num_columns = int(column_centers.shape[0])
        # Instance flag so a trained model can also be lesioned at test time.
        self.enabled = bool(config.enabled)
        self.register_buffer("column_centers", column_centers.clone())

        row_idx, col_idx, distances, weights = self._build_sparse_edges()
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)
        self.register_buffer("edge_distances", distances)
        self.edge_weights = nn.Parameter(weights)
        self.inter_gain = nn.Parameter(torch.ones(state_dim))

    def _build_sparse_edges(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = self.column_centers
        c = centers.shape[0]
        dist = torch.cdist(centers, centers, p=2)
        decay = torch.exp(-dist / max(self.config.lambda_decay, 1e-6))
        eye_mask = ~torch.eye(c, dtype=torch.bool, device=centers.device)
        prob = (decay * self.config.inter_density).clamp(0.0, 1.0)
        sampled = (torch.rand_like(prob) < prob) & eye_mask
        row_idx, col_idx = torch.where(sampled)

        if row_idx.numel() == 0:
            row_idx = torch.arange(0, c - 1, device=centers.device)
            col_idx = torch.arange(1, c, device=centers.device)

        distances = dist[row_idx, col_idx]
        weights = torch.randn(row_idx.numel(), device=centers.device) * self.config.init_weight_scale
        weights = weights.clamp(self.config.min_weight, self.config.max_weight)
        return row_idx.long(), col_idx.long(), distances.float(), weights.float()

    def sparse_matrix(self) -> torch.Tensor:
        indices = torch.stack([self.row_idx, self.col_idx], dim=0)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=self.edge_weights,
            size=(self.num_columns, self.num_columns),
            device=self.edge_weights.device,
        ).coalesce()

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sparse inter-column message passing.

        Args:
            states: `(B, C, D)` column states.
        Returns:
            inter_messages: `(B, C, D)` sparse messages.
            scalar_activity: `(B, C)` pooled activity used for STDP.
        """
        scalar_activity = states.mean(dim=-1)  # (B, C)
        if not self.enabled:
            return torch.zeros_like(states), scalar_activity
        sparse_w = self.sparse_matrix()
        propagated = torch.sparse.mm(sparse_w, scalar_activity.transpose(0, 1)).transpose(0, 1)
        # Per-channel gain on a per-column scalar. ColumnarDynamics adds this after
        # its LayerNorm, so the gain and edge weights both act from initialisation.
        inter_messages = propagated.unsqueeze(-1) * self.inter_gain.view(1, 1, -1)
        return inter_messages, scalar_activity

    @torch.no_grad()
    def stdp_update(self, pre_activity: torch.Tensor, post_activity: torch.Tensor) -> None:
        """Hebbian/STDP-like update on sparse edge values."""
        if not self.enabled or pre_activity.numel() == 0:
            return
        pre = pre_activity.mean(dim=0)  # (C,)
        post = post_activity.mean(dim=0)  # (C,)
        corr = torch.outer(post, pre)
        edge_corr = corr[self.row_idx, self.col_idx]
        anti = self.config.anti_hebbian * self.edge_weights.abs()
        delta = self.config.stdp_lr * (edge_corr - anti)
        new_w = self.edge_weights + delta
        self.edge_weights.copy_(new_w.clamp(self.config.min_weight, self.config.max_weight))

    def spatial_wiring_cost(self) -> torch.Tensor:
        return (self.edge_weights.abs() * self.edge_distances).mean()

    def active_parameters(self) -> int:
        return int((self.edge_weights.abs() > 1e-8).sum().item())

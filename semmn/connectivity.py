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
        """Sample a directed graph whose edge probability decays with distance.

        `P(j -> i) = min(1, s * exp(-d_ij / (lambda_decay * spacing)))`, where `spacing`
        is the typical distance between neighbouring columns, so `lambda_decay` is in
        column spacings and does not depend on the grid's units. The scale `s` is
        solved so the expected number of edges is `inter_density * C * (C - 1)`:
        `inter_density` is the expected fraction of possible directed edges.
        """
        centers = self.column_centers
        c = centers.shape[0]
        dist = torch.cdist(centers, centers, p=2)
        eye = torch.eye(c, dtype=torch.bool, device=centers.device)
        prob = self.edge_probabilities(dist, self.config.lambda_decay, self.config.inter_density)
        sampled = (torch.rand_like(prob) < prob) & ~eye
        row_idx, col_idx = torch.where(sampled)

        if row_idx.numel() == 0:
            row_idx = torch.arange(0, c - 1, device=centers.device)
            col_idx = torch.arange(1, c, device=centers.device)

        distances = dist[row_idx, col_idx]
        # Half-normal init: every sampled edge starts excitatory and non-zero. A signed
        # init clamped at `min_weight = 0` left about half the edges at exactly zero.
        weights = torch.randn(row_idx.numel(), device=centers.device).abs() * self.config.init_weight_scale
        weights = weights.clamp(self.config.min_weight, self.config.max_weight)
        return row_idx.long(), col_idx.long(), distances.float(), weights.float()

    @staticmethod
    def edge_probabilities(dist: torch.Tensor, lambda_decay: float, density: float) -> torch.Tensor:
        """Per-pair edge probability with expected density `density` (diagonal is 0)."""
        c = dist.shape[0]
        eye = torch.eye(c, dtype=torch.bool, device=dist.device)
        if c < 2 or density <= 0.0:
            return torch.zeros_like(dist)
        spacing = dist.masked_fill(eye, float("inf")).min(dim=1).values.median().clamp_min(1e-6)
        decay = torch.exp(-dist / (spacing * max(lambda_decay, 1e-6))).masked_fill(eye, 0.0)
        target = min(float(density), 1.0) * c * (c - 1)
        # Expected edge count is monotone in the scale, so bisect on it.
        lo, hi = 0.0, 1.0
        while (hi * decay).clamp(max=1.0).sum().item() < target and hi < 1e12:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (mid * decay).clamp(max=1.0).sum().item() < target:
                lo = mid
            else:
                hi = mid
        return (hi * decay).clamp(max=1.0)

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
    def stdp_update(self, pre_activity: torch.Tensor, post_activity: torch.Tensor) -> float:
        """Hebbian/STDP-like update on sparse edge values; returns the mean |delta w| applied.

        The result is clamped to `[min_weight, max_weight]`, so with `min_weight = 0` an
        update also zeroes any edge the optimizer has pushed negative.
        """
        if not self.enabled or pre_activity.numel() == 0:
            return 0.0
        pre = pre_activity.mean(dim=0)  # (C,)
        post = post_activity.mean(dim=0)  # (C,)
        corr = torch.outer(post, pre)
        edge_corr = corr[self.row_idx, self.col_idx]
        anti = self.config.anti_hebbian * self.edge_weights.abs()
        delta = self.config.stdp_lr * (edge_corr - anti)
        new_w = (self.edge_weights + delta).clamp(self.config.min_weight, self.config.max_weight)
        applied = (new_w - self.edge_weights).abs().mean().item()
        self.edge_weights.copy_(new_w)
        return applied

    def spatial_wiring_cost(self) -> torch.Tensor:
        return (self.edge_weights.abs() * self.edge_distances).mean()

    def active_parameters(self) -> int:
        return int((self.edge_weights.abs() > 1e-8).sum().item())

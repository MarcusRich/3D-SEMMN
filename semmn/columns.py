"""Columnar dynamics modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class ColumnarDynamics(nn.Module):
    """Dense intra-column dynamics with low-rank factors."""

    def __init__(self, num_columns: int, state_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.state_dim = state_dim
        self.rank = rank

        self.u = nn.Parameter(torch.randn(num_columns, state_dim, rank) * 0.05)
        self.v = nn.Parameter(torch.randn(num_columns, rank, state_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(num_columns, state_dim))
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, states: torch.Tensor, inter_messages: torch.Tensor) -> torch.Tensor:
        # W_col = U @ V in factored form to keep parameters compact.
        w = torch.matmul(self.u, self.v)  # (C, D, D)
        intra = torch.einsum("bcd,cde->bce", states, w)
        # Inter-column messages are added after LayerNorm as a residual. Each
        # message is one pooled scalar per column broadcast across channels, so
        # adding it before the norm is removed exactly by mean subtraction and
        # the connectivity would have no effect (and no gradient) at init.
        local = self.norm(intra + self.bias.unsqueeze(0))
        return torch.tanh(local + inter_messages)


def pool_columns(states: torch.Tensor) -> torch.Tensor:
    """Convert `(B, C, D)` into `(B, C)` pooled activity."""
    return states.mean(dim=-1)

"""3D neuron grid utilities for SEMMN."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from semmn.config import SEMMNConfig


@dataclass
class GridPartition:
    coordinates: torch.Tensor
    column_ids: torch.Tensor
    column_centers: torch.Tensor
    column_sizes: torch.Tensor
    column_to_neuron_indices: list[torch.Tensor]
    vision_mask: torch.Tensor
    audio_mask: torch.Tensor
    hub_mask: torch.Tensor
    vision_columns: torch.Tensor
    audio_columns: torch.Tensor
    hub_columns: torch.Tensor


def build_spatial_grid(config: SEMMNConfig, device: torch.device | None = None) -> GridPartition:
    """Create fixed 3D Euclidean coordinates and column/lobe assignments."""
    gx, gy, gz = config.grid.dims
    nx, ny = config.grid.n_columns_xy
    device = device or torch.device("cpu")

    x = torch.arange(gx, dtype=torch.float32)
    y = torch.arange(gy, dtype=torch.float32)
    z = torch.arange(gz, dtype=torch.float32)
    mesh = torch.meshgrid(x, y, z, indexing="ij")
    coordinates = torch.stack(mesh, dim=-1).reshape(-1, 3).to(device)

    x_bins = torch.clamp((coordinates[:, 0] / max(gx, 1) * nx).long(), max=nx - 1)
    y_bins = torch.clamp((coordinates[:, 1] / max(gy, 1) * ny).long(), max=ny - 1)
    column_ids = x_bins * ny + y_bins
    n_columns = nx * ny

    column_centers = torch.zeros(n_columns, 3, device=device)
    column_sizes = torch.zeros(n_columns, device=device)
    column_to_neuron_indices: list[torch.Tensor] = []
    for cid in range(n_columns):
        idx = torch.where(column_ids == cid)[0]
        column_to_neuron_indices.append(idx)
        if idx.numel() > 0:
            column_centers[cid] = coordinates[idx].mean(dim=0)
            column_sizes[cid] = float(idx.numel())
        else:
            xi = cid // ny
            yi = cid % ny
            column_centers[cid] = torch.tensor(
                [(xi + 0.5) * gx / nx, (yi + 0.5) * gy / ny, 0.5 * gz], device=device
            )

    y_norm = coordinates[:, 1] / max(float(gy), 1.0)
    z_norm = coordinates[:, 2] / max(float(gz), 1.0)
    vision_mask = y_norm < config.grid.vision_y_ratio
    audio_mask = y_norm >= config.grid.audio_y_ratio
    hub_mask = (
        (y_norm >= config.grid.hub_y_range[0])
        & (y_norm <= config.grid.hub_y_range[1])
        & (z_norm >= config.grid.hub_z_range[0])
        & (z_norm <= config.grid.hub_z_range[1])
    )

    vision_columns = _columns_from_mask(column_ids, vision_mask, n_columns)
    audio_columns = _columns_from_mask(column_ids, audio_mask, n_columns)
    hub_columns = _columns_from_mask(column_ids, hub_mask, n_columns)

    if vision_columns.numel() == 0:
        vision_columns = torch.arange(0, n_columns // 3, device=device)
    if audio_columns.numel() == 0:
        audio_columns = torch.arange((2 * n_columns) // 3, n_columns, device=device)
    if hub_columns.numel() == 0:
        start = n_columns // 3
        hub_columns = torch.arange(start, start + max(1, n_columns // 5), device=device)

    return GridPartition(
        coordinates=coordinates,
        column_ids=column_ids,
        column_centers=column_centers,
        column_sizes=column_sizes,
        column_to_neuron_indices=column_to_neuron_indices,
        vision_mask=vision_mask,
        audio_mask=audio_mask,
        hub_mask=hub_mask,
        vision_columns=vision_columns,
        audio_columns=audio_columns,
        hub_columns=hub_columns,
    )


def _columns_from_mask(column_ids: torch.Tensor, mask: torch.Tensor, n_columns: int) -> torch.Tensor:
    hits = torch.bincount(column_ids[mask], minlength=n_columns)
    return torch.where(hits > 0)[0]

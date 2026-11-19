"""Visualization utilities for 3D-SEMMN outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from semmn.config import SEMMNConfig
from semmn.dataset import build_dataloaders
from semmn.model import SEMMNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 3D-SEMMN outputs")
    parser.add_argument("--config", type=str, default="config/light.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    parser.add_argument("--history", type=str, default="outputs/history.json")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def plot_neuron_clusters(model: SEMMNModel, output_dir: Path) -> None:
    coords = model.grid.coordinates.cpu().numpy()
    columns = model.grid.column_ids.cpu().numpy()

    # Downsample for readability.
    max_points = 30000
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]
        columns = columns[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=columns, s=1, cmap="tab20")
    ax.set_title("3D Neuron Grid (colored by column)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output_dir / "neuron_clusters_3d.png", dpi=180)
    plt.close(fig)


def _sample_grid_dataframe(model: SEMMNModel, max_points: int = 40000) -> pd.DataFrame:
    coords = model.grid.coordinates.cpu().numpy()
    vision = model.grid.vision_mask.cpu().numpy()
    audio = model.grid.audio_mask.cpu().numpy()
    hub = model.grid.hub_mask.cpu().numpy()
    columns = model.grid.column_ids.cpu().numpy()

    if coords.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]
        vision = vision[idx]
        audio = audio[idx]
        hub = hub[idx]
        columns = columns[idx]

    module = np.full(coords.shape[0], "other", dtype=object)
    module[audio] = "audio"
    module[vision] = "vision"
    module[hub] = "hub"  # hub has highest priority if overlapping masks
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "column": columns,
            "module": module,
        }
    )


def plot_module_focus_images(model: SEMMNModel, output_dir: Path) -> None:
    df = _sample_grid_dataframe(model, max_points=30000)
    colors = {"vision": "#1f77b4", "audio": "#ff7f0e", "hub": "#2ca02c", "other": "#9467bd"}
    modules = ["vision", "audio", "hub", "other"]

    # Baseline all-modules image.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for m in modules:
        sub = df[df["module"] == m]
        ax.scatter(sub["x"], sub["y"], sub["z"], s=1, alpha=0.65, c=colors[m], label=m)
    ax.set_title("Neuron Grid by Module")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "neuron_modules_all.png", dpi=180)
    plt.close(fig)

    # Focus images with non-focused modules dimmed.
    for focus in modules:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for m in modules:
            sub = df[df["module"] == m]
            if m == focus:
                ax.scatter(sub["x"], sub["y"], sub["z"], s=2, alpha=0.95, c=colors[m], label=f"{m} (focus)")
            else:
                ax.scatter(sub["x"], sub["y"], sub["z"], s=1, alpha=0.05, c="#A0A0A0", label=f"{m} (dim)")
        ax.set_title(f"Neuron Grid Focus: {focus}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"neuron_module_{focus}.png", dpi=180)
        plt.close(fig)


def plot_interactive_module_grid(model: SEMMNModel, output_dir: Path) -> None:
    """Interactive HTML with module selection and dimming."""
    df = _sample_grid_dataframe(model, max_points=35000)
    colors = {"vision": "#1f77b4", "audio": "#ff7f0e", "hub": "#2ca02c", "other": "#9467bd"}
    modules = ["vision", "audio", "hub", "other"]

    fig = go.Figure()
    for m in modules:
        sub = df[df["module"] == m]
        fig.add_trace(
            go.Scatter3d(
                x=sub["x"],
                y=sub["y"],
                z=sub["z"],
                mode="markers",
                name=m,
                marker={"size": 2, "color": colors[m], "opacity": 0.75},
                customdata=np.stack([sub["column"].to_numpy(), sub["module"].to_numpy()], axis=1),
                hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<br>column=%{customdata[0]}<br>module=%{customdata[1]}<extra></extra>",
            )
        )

    def focus_opacity(focus: str | None) -> list[float]:
        if focus is None:
            return [0.75, 0.75, 0.75, 0.75]
        return [0.95 if m == focus else 0.06 for m in modules]

    buttons = [
        {
            "label": "All modules",
            "method": "restyle",
            "args": [{"marker.opacity": focus_opacity(None)}],
        }
    ]
    for m in modules:
        buttons.append(
            {
                "label": f"Focus: {m}",
                "method": "restyle",
                "args": [{"marker.opacity": focus_opacity(m)}],
            }
        )

    fig.update_layout(
        title="3D-SEMMN Neuron Grid (Interactive Module Focus)",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
        legend={"orientation": "h", "x": 0.01, "y": 0.98},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    fig.write_html(str(output_dir / "neuron_grid_interactive.html"), include_plotlyjs="cdn")


def _column_module_labels(model: SEMMNModel) -> np.ndarray:
    n_columns = model.num_columns
    labels = np.full(n_columns, "other", dtype=object)
    labels[model.grid.audio_columns.cpu().numpy()] = "audio"
    labels[model.grid.vision_columns.cpu().numpy()] = "vision"
    labels[model.grid.hub_columns.cpu().numpy()] = "hub"
    return labels


def _simulate_column_route(
    start_activity: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    weights: torch.Tensor,
    steps: int,
    top_edges_per_step: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate signal routing over learned sparse graph.

    This approximates route-of-understanding by propagating activity over
    inter-column edges and keeping strongest contributors each recurrent step.
    """
    act = start_activity.detach().float().clamp_min(0.0)
    act = act / (act.sum().clamp_min(1e-6))

    edge_rows: list[dict[str, float]] = []
    node_rows: list[dict[str, float]] = []

    for step in range(steps):
        node_strength = act.detach().cpu().numpy()
        for node_id, val in enumerate(node_strength):
            if val > 1e-4:
                node_rows.append({"step": float(step), "column": float(node_id), "activity": float(val)})

        edge_strength = weights * act[col_idx]
        if edge_strength.numel() == 0:
            break
        k = min(top_edges_per_step, edge_strength.numel())
        top_vals, top_idx = torch.topk(edge_strength, k=k, largest=True)
        for i in range(k):
            s = int(col_idx[top_idx[i]].item())
            t = int(row_idx[top_idx[i]].item())
            w = float(top_vals[i].item())
            if w > 0:
                edge_rows.append(
                    {"step": float(step), "source": float(s), "target": float(t), "strength": w}
                )

        next_act = torch.zeros_like(act)
        next_act.index_add_(0, row_idx, edge_strength)
        # Small residual keeps originating evidence traceable in later steps.
        act = 0.8 * next_act + 0.2 * act
        act = act / (act.sum().clamp_min(1e-6))

    edge_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["step", "source", "target", "strength"])
    node_df = pd.DataFrame(node_rows) if node_rows else pd.DataFrame(columns=["step", "column", "activity"])
    return edge_df, node_df


@torch.no_grad()
def plot_understanding_routes_2d(
    model: SEMMNModel, cfg: SEMMNConfig, device: torch.device, output_dir: Path
) -> None:
    """2D route maps of how visual/audio evidence traverses column graph."""
    _, val_loader, _ = build_dataloaders(cfg)
    batch = next(iter(val_loader))
    vision, audio, label = batch
    vision = vision[:1].to(device)
    audio = audio[:1].to(device)
    digit = int(label[0].item())

    vision_feat = model.vision_encoder(vision)
    audio_feat = model.audio_encoder(audio)
    vision_init = model.vision_injector(vision_feat)[0].norm(dim=-1)
    audio_init = model.audio_injector(audio_feat)[0].norm(dim=-1)

    row_idx = model.connectivity.row_idx
    col_idx = model.connectivity.col_idx
    weights = model.connectivity.edge_weights.clamp_min(0.0)
    centers = model.grid.column_centers[:, :2].detach().cpu().numpy()
    modules = _column_module_labels(model)
    module_colors = {"vision": "#1f77b4", "audio": "#ff7f0e", "hub": "#2ca02c", "other": "#A8A8A8"}

    vision_edges, vision_nodes = _simulate_column_route(
        vision_init, row_idx, col_idx, weights, steps=cfg.model.recurrent_steps
    )
    audio_edges, audio_nodes = _simulate_column_route(
        audio_init, row_idx, col_idx, weights, steps=cfg.model.recurrent_steps
    )

    vision_edges["modality"] = "vision"
    audio_edges["modality"] = "audio"
    edge_frames = [df for df in (vision_edges, audio_edges) if not df.empty]
    route_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    vision_nodes["modality"] = "vision"
    audio_nodes["modality"] = "audio"
    node_frames = [df for df in (vision_nodes, audio_nodes) if not df.empty]
    route_nodes = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    route_edges.to_csv(output_dir / "route_edges.csv", index=False)
    route_nodes.to_csv(output_dir / "route_nodes.csv", index=False)

    def draw_static(modality: str, edge_df: pd.DataFrame, node_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(centers[:, 0], centers[:, 1], s=14, c="#E0E0E0", alpha=0.35, label="all columns")
        for module_name, color in module_colors.items():
            idx = np.where(modules == module_name)[0]
            if idx.size > 0:
                ax.scatter(
                    centers[idx, 0], centers[idx, 1], s=18, c=color, alpha=0.30, label=f"{module_name} columns"
                )

        if not edge_df.empty:
            max_strength = max(float(edge_df["strength"].max()), 1e-6)
            for _, row in edge_df.iterrows():
                s = int(row["source"])
                t = int(row["target"])
                strength = float(row["strength"])
                alpha = min(0.95, 0.08 + 0.9 * (strength / max_strength))
                width = 0.7 + 2.4 * (strength / max_strength)
                ax.plot(
                    [centers[s, 0], centers[t, 0]],
                    [centers[s, 1], centers[t, 1]],
                    color=module_colors[modality],
                    alpha=alpha,
                    linewidth=width,
                )

        if not node_df.empty:
            final_step = int(node_df["step"].max())
            final_nodes = node_df[node_df["step"] == final_step]
            top_nodes = final_nodes.nlargest(12, "activity")
            idx = top_nodes["column"].astype(int).to_numpy()
            ax.scatter(
                centers[idx, 0],
                centers[idx, 1],
                s=120,
                c=module_colors[modality],
                edgecolors="black",
                linewidths=0.8,
                label=f"{modality} dominant columns",
            )

        hub_idx = np.where(modules == "hub")[0]
        if hub_idx.size > 0:
            ax.scatter(
                centers[hub_idx, 0],
                centers[hub_idx, 1],
                s=40,
                facecolors="none",
                edgecolors="#2ca02c",
                linewidths=1.0,
                label="hub columns",
            )

        ax.set_title(f"2D Understanding Route ({modality}) - sample digit {digit}")
        ax.set_xlabel("Column X")
        ax.set_ylabel("Column Y")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output_dir / f"route_map_{modality}_2d.png", dpi=200)
        plt.close(fig)

    draw_static("vision", vision_edges, vision_nodes)
    draw_static("audio", audio_edges, audio_nodes)

    # Interactive route map with modality focus selector.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"size": 7, "color": "#D0D0D0", "opacity": 0.35},
            name="all_columns",
            hovertemplate="column=%{text}<extra></extra>",
            text=[str(i) for i in range(centers.shape[0])],
        )
    )

    def edge_trace(df: pd.DataFrame, color: str, name: str) -> go.Scatter:
        x_vals: list[float | None] = []
        y_vals: list[float | None] = []
        for _, row in df.iterrows():
            s = int(row["source"])
            t = int(row["target"])
            x_vals.extend([float(centers[s, 0]), float(centers[t, 0]), None])
            y_vals.extend([float(centers[s, 1]), float(centers[t, 1]), None])
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line={"width": 2, "color": color},
            opacity=0.75,
            name=name,
            hoverinfo="skip",
        )

    fig.add_trace(edge_trace(vision_edges, "#1f77b4", "vision_route"))
    fig.add_trace(edge_trace(audio_edges, "#ff7f0e", "audio_route"))
    fig.update_layout(
        title=f"2D Route Simulation Across Column Graph (digit {digit})",
        xaxis_title="Column X",
        yaxis_title="Column Y",
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 1.12,
                "buttons": [
                    {"label": "Show both", "method": "update", "args": [{"visible": [True, True, True]}]},
                    {"label": "Vision route", "method": "update", "args": [{"visible": [True, True, False]}]},
                    {"label": "Audio route", "method": "update", "args": [{"visible": [True, False, True]}]},
                ],
            }
        ],
        legend={"orientation": "h"},
    )
    fig.write_html(str(output_dir / "route_map_interactive.html"), include_plotlyjs="cdn")


@torch.no_grad()
def plot_manifold_trajectories(
    model: SEMMNModel, cfg: SEMMNConfig, device: torch.device, output_dir: Path
) -> None:
    _, val_loader, _ = build_dataloaders(cfg)
    zs = []
    ys = []
    for bi, (vision, audio, label) in enumerate(val_loader):
        if bi >= 20:
            break
        out = model(vision.to(device), audio.to(device))
        zs.append(out["latent"].cpu())
        ys.append(label)
    latent = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(ys, dim=0).numpy()
    proj = PCA(n_components=2).fit_transform(latent)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8)
    ax.set_title("Manifold Trajectories (PCA of latent-8)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Digit")
    fig.tight_layout()
    fig.savefig(output_dir / "manifold_pca.png", dpi=180)
    plt.close(fig)


def plot_metrics(history_path: Path, output_dir: Path) -> None:
    with history_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    train_rows = [r for r in rows if "loss_total" in r]
    val_rows = [r for r in rows if "val_accuracy" in r]
    if not train_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot([r["loss_total"] for r in train_rows], label="total")
    axes[0].plot([r["loss_cls"] for r in train_rows], label="cls")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot([r["energy_proxy"] for r in train_rows], label="energy")
    axes[1].plot([r["active_params"] for r in train_rows], label="active_params")
    axes[1].set_title("Energy/Active Params")
    axes[1].legend()

    if val_rows:
        axes[2].plot([r["val_accuracy"] for r in val_rows], label="val_acc")
        axes[2].plot(
            [r["val_cross_modal_retrieval"] for r in val_rows],
            label="cross_modal",
        )
    axes[2].set_title("Validation Metrics")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SEMMNConfig.from_yaml(args.config)
    if args.device == "mps":
        use_device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif args.device == "cuda":
        use_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        use_device = args.device
    device = torch.device(use_device)
    model = SEMMNModel(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_state = model.state_dict()
    incoming = ckpt["model"]
    filtered = {
        k: v
        for k, v in incoming.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()

    plot_neuron_clusters(model, output_dir)
    plot_module_focus_images(model, output_dir)
    plot_interactive_module_grid(model, output_dir)
    plot_understanding_routes_2d(model, cfg, device, output_dir)
    plot_manifold_trajectories(model, cfg, device, output_dir)
    plot_metrics(Path(args.history), output_dir)


if __name__ == "__main__":
    main()

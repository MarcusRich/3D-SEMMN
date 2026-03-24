"""Top-level SEMMN model."""

from __future__ import annotations

import torch
import torch.nn as nn

from semmn.columns import ColumnarDynamics
from semmn.config import SEMMNConfig
from semmn.connectivity import SparseInterColumnConnectivity
from semmn.encoders import AudioEncoder1D, LobeInjector, VisionEncoder
from semmn.grid import GridPartition, build_spatial_grid
from semmn.hub import SharedManifoldHub
from semmn.manifold import PopulationManifoldAE


class SEMMNModel(nn.Module):
    """3D Spatially-Embedded Multimodal Manifold Network (prototype)."""

    def __init__(self, config: SEMMNConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.grid: GridPartition = build_spatial_grid(config, device=torch.device("cpu"))
        self.num_columns = config.n_columns
        self.state_dim = config.model.state_dim

        self.vision_encoder = VisionEncoder(config.model.vision_feature_dim)
        self.audio_encoder = AudioEncoder1D(config.model.audio_feature_dim)

        self.vision_injector = LobeInjector(
            in_dim=config.model.vision_feature_dim,
            num_columns=self.num_columns,
            state_dim=self.state_dim,
            target_columns=self.grid.vision_columns,
        )
        self.audio_injector = LobeInjector(
            in_dim=config.model.audio_feature_dim,
            num_columns=self.num_columns,
            state_dim=self.state_dim,
            target_columns=self.grid.audio_columns,
        )

        self.connectivity = SparseInterColumnConnectivity(
            column_centers=self.grid.column_centers,
            state_dim=self.state_dim,
            config=config.connectivity,
        )
        self.columnar = ColumnarDynamics(
            num_columns=self.num_columns, state_dim=self.state_dim, rank=8
        )
        self.manifold = PopulationManifoldAE(
            num_columns=self.num_columns,
            state_dim=self.state_dim,
            latent_dim=config.model.latent_dim,
        )
        self.hub = SharedManifoldHub(
            state_dim=self.state_dim,
            latent_dim=config.model.latent_dim,
            feature_dim=config.model.vision_feature_dim,
            attention_heads=config.model.attention_heads,
        )

        self.vision_proj = nn.Linear(config.model.vision_feature_dim, config.model.latent_dim)
        self.audio_proj = nn.Linear(config.model.audio_feature_dim, config.model.latent_dim)
        self.classifier = nn.Linear(config.model.latent_dim, 10)

    def forward(self, vision: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        b = vision.shape[0]
        device = vision.device

        vision_features = self.vision_encoder(vision)
        audio_features = self.audio_encoder(audio)

        state = torch.zeros(b, self.num_columns, self.state_dim, device=device)
        state = state + self.vision_injector(vision_features) + self.audio_injector(audio_features)

        pre_activity = None
        post_activity = None
        for _ in range(self.config.model.recurrent_steps):
            inter_messages, pre_activity = self.connectivity(state)
            state = self.columnar(state, inter_messages)
            post_activity = state.mean(dim=-1)

        manifold_out = self.manifold(state)
        hub_out = self.hub(
            states=state,
            hub_columns=self.grid.hub_columns.to(device),
            vision_features=vision_features,
            audio_features=audio_features,
        )

        vision_embed = self.vision_proj(vision_features)
        audio_embed = self.audio_proj(audio_features)
        fused = (hub_out["fused_latent"] + manifold_out["twisted_latent"]) / 2.0
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "vision_embed": vision_embed,
            "audio_embed": audio_embed,
            "vision_features": vision_features,
            "audio_features": audio_features,
            "manifold_loss": manifold_out["manifold_loss"],
            "latent": manifold_out["twisted_latent"],
            "hub_latent": hub_out["hub_latent"],
            "audio_from_vision": hub_out["audio_from_vision"],
            "vision_from_audio": hub_out["vision_from_audio"],
            "cycle_vision": hub_out["cycle_vision"],
            "cycle_audio": hub_out["cycle_audio"],
            "pre_activity": pre_activity,
            "post_activity": post_activity,
            "column_states": state,
            "reliability_weights": hub_out["reliability_weights"],
        }

    @torch.no_grad()
    def apply_stdp(self, pre_activity: torch.Tensor, post_activity: torch.Tensor) -> None:
        self.connectivity.stdp_update(pre_activity, post_activity)

    def active_parameters(self) -> int:
        return self.connectivity.active_parameters()

    def energy_proxy(self) -> torch.Tensor:
        return self.connectivity.spatial_wiring_cost()

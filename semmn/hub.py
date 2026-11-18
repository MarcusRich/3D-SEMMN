"""Shared manifold hub with cross-modal attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedManifoldHub(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        feature_dim: int,
        attention_heads: int = 4,
        direct_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        # When True, encoder features are also added straight into the fused latent,
        # skipping the column grid. Off by default; kept only to reproduce old runs.
        self.direct_fusion = direct_fusion
        self.modality_proj = nn.Linear(feature_dim, state_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=attention_heads, batch_first=True
        )
        self.hub_to_latent = nn.Linear(state_dim, latent_dim)

        self.reliability = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.audio_imagination_decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.vision_imagination_decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        self.cycle_vision = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.cycle_audio = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        hub_columns: torch.Tensor,
        vision_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hub_states = states[:, hub_columns, :]
        v_token = self.modality_proj(vision_features).unsqueeze(1)
        a_token = self.modality_proj(audio_features).unsqueeze(1)
        tokens = torch.cat([hub_states, v_token, a_token], dim=1)

        attn_out, attn_map = self.attn(tokens, tokens, tokens, need_weights=True)
        pooled = attn_out.mean(dim=1)
        hub_latent = self.hub_to_latent(pooled)

        rel_logits = self.reliability(torch.cat([vision_features, audio_features], dim=-1))
        rel_weights = torch.softmax(rel_logits, dim=-1)
        if self.direct_fusion:
            # Legacy bypass: reliability-weighted modality latents added directly.
            fused_latent = (
                rel_weights[:, 0:1] * self.hub_to_latent(self.modality_proj(vision_features))
                + rel_weights[:, 1:2] * self.hub_to_latent(self.modality_proj(audio_features))
                + hub_latent
            ) / 2.0
        else:
            fused_latent = hub_latent

        audio_from_vision = self.audio_imagination_decoder(fused_latent)
        vision_from_audio = self.vision_imagination_decoder(fused_latent)
        cycle_vision = self.cycle_vision(vision_from_audio)
        cycle_audio = self.cycle_audio(audio_from_vision)

        return {
            "hub_latent": hub_latent,
            "fused_latent": fused_latent,
            "reliability_weights": rel_weights,
            "attention_map": attn_map,
            "audio_from_vision": audio_from_vision,
            "vision_from_audio": vision_from_audio,
            "cycle_vision": cycle_vision,
            "cycle_audio": cycle_audio,
        }

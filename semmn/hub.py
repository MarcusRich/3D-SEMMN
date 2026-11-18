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
        modality_tokens: bool = False,
        num_hub_columns: int | None = None,
        readout: str = "flat",
    ) -> None:
        super().__init__()
        if readout not in ("flat", "mean"):
            raise ValueError(f"readout must be 'flat' or 'mean'; got {readout!r}")
        if readout == "flat" and not num_hub_columns:
            raise ValueError("readout='flat' needs num_hub_columns")
        self.state_dim = state_dim
        # When True, projected encoder features join the hub columns as two extra
        # tokens, a second route from the encoders to the hub latent that
        # skips the column grid. Off by default so the hub reads only its columns.
        self.modality_tokens = modality_tokens
        # When True, encoder features are also added straight into the fused latent,
        # skipping the column grid. Off by default; kept only to reproduce old runs.
        self.direct_fusion = direct_fusion
        self.modality_proj = nn.Linear(feature_dim, state_dim)
        # How the hub tokens become the latent the classifier reads.
        #   "flat": every hub column's state, flattened (column identity kept) and
        #     standardised per feature across the batch, then one linear map. The
        #     input-dependent part of the hub states is small next to their
        #     per-column offsets, so without the standardisation it reaches the
        #     classifier at about 1e-3 of its scale.
        #   "mean": the P1 read-out, self-attention over the tokens, mean-pooled
        #     over tokens. It cannot tell the columns apart (no positions) and keeps
        #     their small input-dependent part small. Kept to reproduce old runs.
        self.readout_mode = readout
        if readout == "flat":
            n_tokens = int(num_hub_columns) + (2 if modality_tokens else 0)
            self.readout_norm = nn.BatchNorm1d(n_tokens * state_dim)
            self.readout = nn.Linear(n_tokens * state_dim, latent_dim)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=state_dim, num_heads=attention_heads, batch_first=True
            )
        # Used by the "mean" read-out and by the legacy `direct_fusion` path.
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
        tokens = states[:, hub_columns, :]
        if self.modality_tokens:
            v_token = self.modality_proj(vision_features).unsqueeze(1)
            a_token = self.modality_proj(audio_features).unsqueeze(1)
            tokens = torch.cat([tokens, v_token, a_token], dim=1)

        if self.readout_mode == "flat":
            attn_map = None
            hub_latent = self.readout(self.readout_norm(tokens.flatten(1)))
        else:
            attn_out, attn_map = self.attn(tokens, tokens, tokens, need_weights=True)
            hub_latent = self.hub_to_latent(attn_out.mean(dim=1))

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

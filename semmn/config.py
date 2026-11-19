"""Configuration handling for 3D-SEMMN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    dims: tuple[int, int, int] = (100, 100, 50)
    n_columns_xy: tuple[int, int] = (10, 10)
    vision_y_ratio: float = 0.30
    audio_y_ratio: float = 0.70
    hub_y_range: tuple[float, float] = (0.40, 0.60)
    hub_z_range: tuple[float, float] = (0.30, 0.70)


@dataclass
class ModelConfig:
    state_dim: int = 32
    latent_dim: int = 8
    vision_feature_dim: int = 256
    audio_feature_dim: int = 256
    recurrent_steps: int = 6
    attention_heads: int = 4
    # Legacy: add encoder features straight into the hub's fused latent (bypasses the grid).
    hub_direct_fusion: bool = False
    # Encoder features as two extra attention tokens in the hub (bypasses the grid).
    # Off: the hub reads only its columns, so it depends on inter-column connectivity.
    hub_modality_tokens: bool = False
    # How the hub columns become the classifier's latent. "flat": all hub column
    # states, standardised per feature across the batch (BatchNorm), then linear.
    # "mean": the P1 attention mean-pool, which loses the label (see the P2 notes).
    hub_readout: str = "flat"


@dataclass
class ConnectivityConfig:
    # False lesions inter-column message passing (messages are zero, STDP is skipped).
    enabled: bool = True
    # Edge probability decays as exp(-d / (lambda_decay * column spacing)), so
    # `lambda_decay` is in column spacings. At 1.0 most nearest neighbours are wired
    # and the mean edge spans about two columns.
    lambda_decay: float = 1.0
    # Expected fraction of the C * (C - 1) possible directed edges (0.08 is about
    # 8 in-edges per column on a 10x10 sheet).
    inter_density: float = 0.08
    # Accepted so older configs still load, and ignored. Intra-column dynamics are
    # the dense low-rank factors in columns.py, not a sampled graph, so there is
    # no intra-column edge density to boost.
    intra_density_boost: float = 2.0
    min_weight: float = 0.0
    max_weight: float = 1.5
    init_weight_scale: float = 0.15
    # Optimizer steps between STDP updates (train.py and eval_small.py); 0 turns STDP
    # off. At the eval-small scale STDP is negligible (see the P2 notes): over a run
    # it changes each edge by about 1% of its weight, about 6% of what the optimizer
    # moves it, and val accuracy with it off is within the seed-to-seed spread.
    stdp_interval: int = 100
    stdp_lr: float = 0.01
    anti_hebbian: float = 0.05

    def stdp_due(self, step: int) -> bool:
        """Whether STDP runs after optimizer step `step` (counted from 1)."""
        return self.enabled and self.stdp_interval > 0 and step % self.stdp_interval == 0


@dataclass
class LossConfig:
    contrastive_weight: float = 1.0
    reconstruction_weight: float = 0.5
    imagination_weight: float = 0.3
    manifold_weight: float = 0.1
    spatial_penalty_weight: float = 5e-4
    temperature: float = 0.1
    use_spatial_penalty: bool = True
    use_manifold_loss: bool = True


@dataclass
class DataConfig:
    # Synthetic audio: a tone whose pitch encodes the digit, buried in white noise.
    # `audio_snr_db` is tone power over noise power in the waveform; lower is less
    # informative. `audio_pitch_jitter` is the per-sample pitch std in units of the
    # spacing between adjacent digits, so neighbouring digits overlap even at high
    # SNR and no model can read the label off the audio perfectly.
    audio_snr_db: float = -10.0
    audio_pitch_jitter: float = 0.35
    # train.py holds out this many MNIST-train samples (fixed by `split_seed`) for
    # checkpoint selection; the source's test split is scored once, after selection.
    val_size: int = 5000
    split_seed: int = 1234


@dataclass
class TrainingConfig:
    # "synthetic" builds the local MNIST + tone spectrograms. "hf" loads
    # BLOSSOM-framework/AV-MNIST, which contacts the Hugging Face hub.
    dataset_source: str = "synthetic"
    # When False (the default), a failed "hf" load raises. True trains on synthetic
    # audio instead, and build_dataloaders warns that it did so.
    allow_synthetic_fallback: bool = False
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    log_interval: int = 50
    val_batches: int = 100
    amp: bool = True


@dataclass
class PathsConfig:
    output_dir: str = "outputs"


@dataclass
class SEMMNConfig:
    seed: int = 42
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def n_columns(self) -> int:
        return self.grid.n_columns_xy[0] * self.grid.n_columns_xy[1]

    @property
    def n_neurons(self) -> int:
        return self.grid.dims[0] * self.grid.dims[1] * self.grid.dims[2]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SEMMNConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SEMMNConfig":
        # `eval` is read by eval_small.py, not by the model. Any other unknown key
        # is a typo and used to be ignored, so a misspelled section silently did nothing.
        unknown = set(raw) - _TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(
            seed=raw.get("seed", 42),
            grid=_merge_dataclass(GridConfig(), raw.get("grid", {})),
            model=_merge_dataclass(ModelConfig(), raw.get("model", {})),
            connectivity=_merge_dataclass(
                ConnectivityConfig(), raw.get("connectivity", {})
            ),
            loss=_merge_dataclass(LossConfig(), raw.get("loss", {})),
            data=_merge_dataclass(DataConfig(), raw.get("data", {})),
            training=_merge_dataclass(TrainingConfig(), raw.get("training", {})),
            paths=_merge_dataclass(PathsConfig(), raw.get("paths", {})),
        )


# Top-level YAML keys. `eval` belongs to eval_small.py and is not a model field.
_TOP_LEVEL_KEYS = {
    "seed",
    "grid",
    "model",
    "connectivity",
    "loss",
    "data",
    "training",
    "paths",
    "eval",
}


def _merge_dataclass(default_obj: Any, overrides: dict[str, Any] | None) -> Any:
    if not overrides:
        return default_obj
    current = default_obj.__dict__.copy()
    unknown = set(overrides) - set(current)
    if unknown:
        raise ValueError(
            f"unknown {type(default_obj).__name__} keys: {sorted(unknown)}"
        )
    for key, value in overrides.items():
        if isinstance(current.get(key), tuple) and isinstance(value, list):
            current[key] = tuple(value)
        else:
            current[key] = value
    return type(default_obj)(**current)

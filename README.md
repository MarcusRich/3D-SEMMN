# 3D-SEMMN (Prototype)

Minimal but fully functional PyTorch 2.3+ prototype of a **3D Spatially-Embedded Multimodal Manifold Network** with:

- fixed 3D neuron coordinates (`100 x 100 x 50` in `config/default.yaml`)
- 100 columnar modules (dense intra-column, sparse inter-column)
- exponential distance-penalized sparse connectivity (`torch.sparse`): `lambda_decay` is in column spacings and `inter_density` is the expected fraction of possible directed edges
- Hebbian/STDP update every `connectivity.stdp_interval` optimizer steps (default 100; `eval-small.yaml` uses 10 so it fires within its 96-step budget)
- shared hub read out from its columns: every hub column's state, flattened, standardised per feature
  (BatchNorm) and mapped linearly to the latent (`model.hub_readout: flat`)
- manifold regularization (latent `d=8`) + geometric twist layer (Cayley rotation)
- multimodal losses: classification + contrastive + reconstruction + imagination cycle
- ablations for spatial penalty and manifold loss

This code is designed for readability and modular experimentation rather than full biological fidelity.

## Biological Inspirations

- **seRNN (Achterberg et al., 2023)**: spatial embedding + wiring-cost constraints for modular sparse topology.
- **MSeNN-style multimodal ideas (2021)**: cross-modal coupling and biologically motivated integration.
- **Gallego et al. (2017)**: low-dimensional population manifolds.

## Project Layout

- `semmn/config.py`: dataclass config + YAML loader
- `semmn/grid.py`: 3D coordinates, lobe masks, column assignment
- `semmn/connectivity.py`: sparse distance-decay edges + STDP + energy proxy
- `semmn/columns.py`: low-rank dense intra-column dynamics
- `semmn/encoders.py`: vision CNN and audio 1D-conv encoder
- `semmn/manifold.py`: manifold autoencoder and geometric twist
- `semmn/hub.py`: shared hub read-out + imagination decoders
- `semmn/losses.py`: full training objective
- `semmn/model.py`: full SEMMN forward pass
- `semmn/dataset.py`: AV-MNIST loading with synthetic fallback
- `train.py`: training/eval/checkpointing
- `visualize.py`: 3D grid, manifold PCA, metrics plots

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (light config, recommended first):

```bash
python train.py --config config/light.yaml --device cuda
```

Train (full 500K-neuron spatial grid metadata):

```bash
python train.py --config config/default.yaml --device cuda
```

Generate figures. No checkpoint is committed: `outputs/best.pt` is written by
`train.py` (the epoch with the best validation accuracy), so train first:

```bash
python train.py --config config/light.yaml --device cpu   # writes outputs/best.pt
python visualize.py \
  --config config/light.yaml \
  --checkpoint outputs/best.pt \
  --history outputs/history.json \
  --device cuda
```

Small fixed-split evaluation (CPU, a few minutes) and tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
python eval_small.py --config config/eval-small.yaml   # writes outputs/eval-small/
```

## Config Notes

- `config/default.yaml`: full spatial grid (`100x100x50`)
- `config/light.yaml`: faster profile (`100x100x10`) for rapid iteration
- Ablations:
  - `loss.use_spatial_penalty: false`
  - `loss.use_manifold_loss: false`
  - `connectivity.enabled: false` (inter-column lesion)
  - `model.hub_modality_tokens: true` (encoder features as two extra hub tokens; off by default)
  - `model.hub_readout: mean` (the earlier read-out: self-attention over the hub columns, mean-pooled;
    it loses the label, see the P2 notes)
- Synthetic audio (`data`): a tone whose pitch encodes the digit, with per-sample pitch jitter
  (`audio_pitch_jitter`) and white noise (`audio_snr_db`), so audio alone is informative but
  does not determine the label

## Reported Metrics

`train.py` holds out `data.val_size` samples of MNIST-train (fixed by
`data.split_seed`) as the validation split. It keeps `best.pt` from the epoch with
the best validation accuracy, then scores that checkpoint on the test split
(MNIST-t10k) once. It logs:

- classification loss and total loss
- validation classification accuracy, and the final test accuracy of `best.pt`
- cross-modal retrieval accuracy (vision->audio nearest match in batch)
- active sparse parameters (`nnz` proxy)
- energy proxy (`mean(|w| * distance)`)

## Extension Notes

### Full spiking variant (Brian2 / snnTorch)

1. Replace rate-based state updates (`tanh`) with LIF/AdEx neuron dynamics.
2. Convert STDP update from rate Hebbian correlation to spike-timing windows.
3. Keep 3D coordinates and sparse wiring graph unchanged for fair ablations.

### Neuromorphic export direction

1. Map columns to compute cores (Loihi/SpiNNaker style partitioning).
2. Convert sparse inter-column edges to routing tables.
3. Quantize state variables and weights (8-bit or mixed precision).
4. Use event-driven updates for power-efficient deployment.

## Practical Runtime Guidance

- Start with `config/light.yaml` to confirm setup and debug quickly.
- Increase `training.batch_size` and `model.recurrent_steps` gradually.
- STDP runs every `connectivity.stdp_interval` optimizer steps (0 turns it off); check that a run is long enough for it to fire.
  At the `eval-small.yaml` scale it is negligible: over a run it changes each edge by about 1% of its weight, about 6% of
  what the optimizer moves it, and turning it off leaves val accuracy within the seed-to-seed spread. Each update is
  `stdp_lr` (0.01) times a Hebbian term built from batch-mean column activities, which are small, minus a weight decay;
  the mean |dw| per update is about 1e-4. It is kept for experiments, not claimed as a contributor.
- Use mixed precision (`training.amp: true`) on CUDA devices.

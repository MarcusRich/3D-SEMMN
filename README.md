# 3D-SEMMN (Prototype)

Minimal but fully functional PyTorch 2.3+ prototype of a **3D Spatially-Embedded Multimodal Manifold Network** with:

- fixed 3D neuron coordinates (`100 x 100 x 50` in `config/default.yaml`)
- 100 columnar modules (dense intra-column, sparse inter-column)
- exponential distance-penalized sparse connectivity (`torch.sparse`)
- Hebbian/STDP update every 100 steps
- shared perceptual manifold hub with cross-modal attention
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
- `semmn/hub.py`: shared manifold hub + attention + imagination decoders
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

Generate figures:

```bash
python visualize.py \
  --config config/light.yaml \
  --checkpoint outputs/best.pt \
  --history outputs/history.json \
  --device cuda
```

## Config Notes

- `config/default.yaml`: full spatial grid (`100x100x50`)
- `config/light.yaml`: faster profile (`100x100x10`) for rapid iteration
- Ablations:
  - `loss.use_spatial_penalty: false`
  - `loss.use_manifold_loss: false`

## Reported Metrics

`train.py` logs:

- classification loss and total loss
- validation classification accuracy
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
- Keep STDP interval at `100` to limit plasticity overhead.
- Use mixed precision (`training.amp: true`) on CUDA devices.

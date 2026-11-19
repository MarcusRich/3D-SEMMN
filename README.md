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

## EEG response surface (P3)

`semmn/eeg/` fits ridge temporal response functions that predict 128-channel EEG
from a story's envelope, word onsets and word features. It evaluates them on held-out
subjects and runs, and simulates 8- and 16-channel starter headsets by subsetting the
cap. The data is Broderick et al. (2018), Dryad doi:10.5061/dryad.070jc (CC0); it is
never copied into the repo. `eeg_p3.py` caches preprocessed arrays next to the data
and writes only summary tables to `outputs/eeg/` (`summary.md` first).

```bash
export SEMMN_BRODERICK=/path/to/broderick2018/extracted   # holds "Natural Speech", ...
python -I eeg_p3.py prep --condition forward --subjects 1-19
python -I eeg_p3.py prep --condition reverse --subjects 1-10
python -I eeg_p3.py words --backend local    # n-gram surprisal + static word vectors
python -I eeg_p3.py encode --condition forward --subjects 1-19 --folds 5 --lm "$SEMMN_BRODERICK/cache/lm_local.npz"
python -I eeg_p3.py encode --condition reverse --subjects 1-10 --folds 5 --lm "$SEMMN_BRODERICK/cache/lm_local.npz"
python -I eeg_p3.py summarize
```

`words --backend local` reads the static embedding table of a locally cached
all-MiniLM-L6-v2 from the Hugging Face cache. `--backend gpt2` needs `transformers`.
The release lists content words only, so neither backend sees the full transcript.

## GPT-2 word features and distractor ranking (P4)

P4 swaps the stand-in word features for GPT-2 small (`openai-community/gpt2`, loaded
from the Hugging Face cache). Each run's content-word stream is read from the run's
start only, lower-cased, with a period at each sentence close. Per word it gives the
surprisal (bits, summed over sub-word tokens) and the layer-6 state at its last token,
reduced to 20 PCs fitted on training-run words. Tiers: `A_env`, `B_env_onset`,
`S_env_onset_sur` (+ GPT-2 surprisal), `E_env_onset_sur_emb` (+ GPT-2 embedding).

`distract` samples 88 GPT-2 distractor word sequences for every held-out 10 s window
(same audio, same word onsets; see `semmn/eeg/distractors.py`). `encode --distractors`
then ranks the true words against them three ways: with the subject's EEG, with
another window's EEG (must be at chance) and with the GPT-2 prior alone.

```bash
python -I eeg_p3.py words --backend gpt2     # writes cache/lm_gpt2_run_l6.npz
LM="$SEMMN_BRODERICK/cache/lm_gpt2_run_l6.npz"
python -I eeg_p3.py distract --condition forward --subjects 1-19 --lm "$LM"
python -I eeg_p3.py distract --condition reverse --subjects 1-10 --lm "$LM"
for c in forward reverse; do
  s=$([ $c = forward ] && echo 1-19 || echo 1-10)
  python -I eeg_p3.py encode --condition $c --subjects $s --folds 5 --lm "$LM" \
    --tiers A_env,B_env_onset,S_env_onset_sur,E_env_onset_sur_emb --main-tier E_env_onset_sur_emb \
    --distractors "$SEMMN_BRODERICK/cache/distractors_${c}_w10_k88_p0.9_l6.npz" --out outputs/eeg_p4
done
python -I eeg_p4.py --out outputs/eeg_p4 --p3 outputs/eeg   # writes outputs/eeg_p4/summary_p4.md
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

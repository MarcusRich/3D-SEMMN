# eval-small results

Config: `config/eval-small.yaml`. Split seed 1234: train 2000 / val 500 (MNIST-train, disjoint), test 1000 (MNIST-t10k). 3 epochs, epoch selected on val accuracy, test scored once.

| method | runs | test accuracy (mean ± std over seeds) |
| --- | --- | --- |
| semmn | 5 | 0.1534 ± 0.0630 |
| semmn_hub_probe | 5 | 0.7970 ± 0.1275 |
| semmn_lesion_test | 5 | 0.1374 ± 0.0551 |
| semmn_lesion_test_hub_probe | 5 | 0.1290 ± 0.0000 |
| semmn_lesion_train | 5 | 0.1230 ± 0.0500 |
| semmn_lesion_train_hub_probe | 5 | 0.1290 ± 0.0000 |
| semmn_hub_tokens | 5 | 0.1420 ± 0.0589 |
| mlp | 5 | 0.9192 ± 0.0044 |
| mlp_vision_only | 5 | 0.9100 ± 0.0066 |
| mlp_audio_only | 5 | 0.8272 ± 0.0044 |
| logreg | 1 | 0.9240 (single deterministic run) |
| logreg_vision_only | 1 | 0.8900 (single deterministic run) |
| logreg_audio_only | 1 | 0.8330 (single deterministic run) |

Per-seed SEMMN diagnostics on the val split: sampled inter-column edges, non-zero edges after training, hub-column state std across samples (absolute and relative to the input lobes), hub latent std, STDP updates applied and their mean |dw|:

| seed | best epoch | edges | non-zero edges | hub state std | hub/lobe std | hub latent std | STDP updates | STDP mean abs dw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 995 | 995 | 1.29e-03 | 0.253 | 2.34e-04 | 9 | 1.83e-04 |
| 1 | 2 | 981 | 981 | 9.56e-02 | 0.453 | 3.55e-02 | 9 | 1.66e-04 |
| 2 | 2 | 956 | 956 | 1.56e-01 | 0.456 | 6.27e-02 | 9 | 2.51e-04 |
| 3 | 2 | 997 | 997 | 3.04e-03 | 0.054 | 7.55e-04 | 9 | 1.48e-04 |
| 4 | 2 | 979 | 979 | 5.95e-02 | 0.497 | 2.89e-02 | 9 | 1.31e-04 |

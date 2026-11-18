# eval-small results

Config: `config/eval-small.yaml`. Split seed 1234: train 2000 / val 500 (MNIST-train, disjoint), test 1000 (MNIST-t10k). 3 epochs, epoch selected on val accuracy, test scored once.

| method | runs | test accuracy (mean ± std over seeds) |
| --- | --- | --- |
| semmn | 5 | 0.5874 ± 0.1128 |
| semmn_lesion_test | 5 | 0.5862 ± 0.1132 |
| semmn_lesion_train | 5 | 0.5994 ± 0.1194 |
| mlp | 5 | 1.0000 ± 0.0000 |
| logreg | 1 | 1.0000 (single deterministic run) |
| logreg_vision_only | 1 | 0.8900 (single deterministic run) |
| logreg_audio_only | 1 | 1.0000 (single deterministic run) |

Per-seed SEMMN diagnostics (sampled inter-column edges, non-zero edges, hub-column state std across test samples, STDP updates applied):

| seed | edges | non-zero edges | hub state std | STDP updates |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 | 0.00e+00 | 0 |
| 1 | 9 | 9 | 1.98e-06 | 0 |
| 2 | 9 | 9 | 2.61e-06 | 0 |
| 3 | 8 | 8 | 0.00e+00 | 0 |
| 4 | 9 | 9 | 2.76e-04 | 0 |

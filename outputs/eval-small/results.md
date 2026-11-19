# eval-small results

Config: `config/eval-small.yaml`. Split seed 1234: train 2000 / val 500 (MNIST-train, disjoint), test 1000 (MNIST-t10k). 3 epochs, epoch selected on val accuracy, test scored once.

The val column is the val accuracy of the selected epoch: the only number model selection saw. Test was scored once per run, after selection.

| method | runs | test accuracy (mean ± std over seeds) | val accuracy, selected epoch |
| --- | --- | --- | --- |
| semmn_init_hub_probe | 5 | 0.9498 ± 0.0107 | – |
| semmn | 5 | 0.8814 ± 0.1312 | 0.8832 ± 0.1271 |
| semmn_hub_probe | 5 | 0.9926 ± 0.0018 | – |
| semmn_lesion_test | 5 | 0.1122 ± 0.0113 | – |
| semmn_lesion_test_hub_probe | 5 | 0.1290 ± 0.0000 | – |
| semmn_lesion_train | 5 | 0.1064 ± 0.0158 | 0.1068 ± 0.0108 |
| semmn_lesion_train_hub_probe | 5 | 0.1290 ± 0.0000 | – |
| semmn_mean_readout | 5 | 0.1718 ± 0.0638 | 0.1764 ± 0.0672 |
| semmn_hub_tokens | 5 | 0.9258 ± 0.0595 | 0.9256 ± 0.0565 |
| mlp | 5 | 0.9192 ± 0.0044 | 0.9440 ± 0.0042 |
| mlp_vision_only | 5 | 0.9100 ± 0.0066 | 0.9284 ± 0.0017 |
| mlp_audio_only | 5 | 0.8272 ± 0.0044 | 0.8536 ± 0.0067 |
| logreg | 1 | 0.9240 (single deterministic run) | 0.9360 |
| logreg_vision_only | 1 | 0.8900 (single deterministic run) | 0.9160 |
| logreg_audio_only | 1 | 0.8330 (single deterministic run) | 0.8480 |

Per-seed SEMMN diagnostics on the val split: selected epoch and its val accuracy, sampled inter-column edges, non-zero edges after training, hub-column state std across samples (absolute, over the hub states' RMS, and over the input lobes' std), hub latent std (after the read-out's BatchNorm, so near unit scale by construction), STDP updates applied and their mean |dw|:

| seed | best epoch | val acc | edges | non-zero edges | hub state std | hub std / RMS | hub/lobe std | hub latent std | STDP updates | STDP mean abs dw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0.982 | 995 | 995 | 7.39e-02 | 0.119 | 0.466 | 9.85e+00 | 9 | 1.09e-04 |
| 1 | 1 | 0.672 | 981 | 981 | 5.05e-02 | 0.081 | 0.799 | 5.76e+00 | 9 | 1.20e-04 |
| 2 | 1 | 0.910 | 956 | 956 | 5.61e-02 | 0.091 | 0.515 | 5.95e+00 | 9 | 9.98e-05 |
| 3 | 1 | 0.872 | 997 | 997 | 8.40e-02 | 0.134 | 0.505 | 7.49e+00 | 9 | 1.01e-04 |
| 4 | 2 | 0.980 | 979 | 979 | 8.54e-02 | 0.135 | 0.516 | 9.28e+00 | 9 | 1.03e-04 |

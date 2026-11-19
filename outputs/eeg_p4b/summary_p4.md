# P4 summary: GPT-2 word features (held-out subjects, held-out runs)

GPT-2 small reads each run's content-word stream (the release has no function words), lower-cased, with a period at each sentence close, from the run's start only.

## forward: encoding, held-out r (mean ± SEM over 19 subjects)

| montage | A_env | B_env_onset | S_env_onset_sur | E_env_onset_sur_emb |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0210 ± 0.0034 | 0.0263 ± 0.0036 | 0.0300 ± 0.0039 | 0.0314 ± 0.0040 |
| cyton8 | 0.0170 ± 0.0026 | 0.0220 ± 0.0030 | 0.0259 ± 0.0033 | 0.0267 ± 0.0034 |
| cyton_daisy16 | 0.0196 ± 0.0034 | 0.0249 ± 0.0035 | 0.0283 ± 0.0037 | 0.0294 ± 0.0039 |
| unicorn8 | 0.0202 ± 0.0028 | 0.0253 ± 0.0033 | 0.0300 ± 0.0037 | 0.0312 ± 0.0036 |
| best8_train | 0.0276 ± 0.0044 | 0.0337 ± 0.0045 | 0.0369 ± 0.0047 | 0.0388 ± 0.0052 |
| best16_train | 0.0280 ± 0.0045 | 0.0340 ± 0.0045 | 0.0372 ± 0.0048 | 0.0394 ± 0.0052 |

### forward: stepwise gains (paired over subjects)

| montage | step | mean Δr | subjects Δr > 0 | dz | p |
| --- | --- | --- | --- | --- | --- |
| biosemi128 | onsets: B − A | +0.0053 | 15/19 | +0.88 | 0.0012 |
| biosemi128 | GPT-2 surprisal: S − B | +0.0036 | 14/19 | +0.91 | 0.00087 |
| biosemi128 | GPT-2 embedding: E − S | +0.0015 | 11/19 | +0.26 | 0.28 |
| biosemi128 | all GPT-2: E − B | +0.0051 | 15/19 | +0.75 | 0.0041 |
| cyton8 | onsets: B − A | +0.0051 | 13/19 | +0.76 | 0.0038 |
| cyton8 | GPT-2 surprisal: S − B | +0.0039 | 16/19 | +0.95 | 0.00059 |
| cyton8 | GPT-2 embedding: E − S | +0.0007 | 10/19 | +0.11 | 0.63 |
| cyton8 | all GPT-2: E − B | +0.0046 | 13/19 | +0.59 | 0.019 |
| cyton_daisy16 | onsets: B − A | +0.0052 | 15/19 | +0.92 | 0.00085 |
| cyton_daisy16 | GPT-2 surprisal: S − B | +0.0034 | 16/19 | +0.91 | 0.00089 |
| cyton_daisy16 | GPT-2 embedding: E − S | +0.0012 | 11/19 | +0.19 | 0.41 |
| cyton_daisy16 | all GPT-2: E − B | +0.0045 | 14/19 | +0.63 | 0.013 |
| unicorn8 | onsets: B − A | +0.0051 | 13/19 | +0.71 | 0.006 |
| unicorn8 | GPT-2 surprisal: S − B | +0.0047 | 15/19 | +0.98 | 0.00045 |
| unicorn8 | GPT-2 embedding: E − S | +0.0012 | 9/19 | +0.22 | 0.35 |
| unicorn8 | all GPT-2: E − B | +0.0059 | 15/19 | +0.87 | 0.0014 |
| best8_train | onsets: B − A | +0.0062 | 18/19 | +1.22 | 4.8e-05 |
| best8_train | GPT-2 surprisal: S − B | +0.0031 | 15/19 | +0.72 | 0.0054 |
| best8_train | GPT-2 embedding: E − S | +0.0019 | 13/19 | +0.32 | 0.18 |
| best8_train | all GPT-2: E − B | +0.0050 | 15/19 | +0.67 | 0.0092 |
| best16_train | onsets: B − A | +0.0060 | 17/19 | +1.11 | 0.00013 |
| best16_train | GPT-2 surprisal: S − B | +0.0032 | 16/19 | +0.73 | 0.0054 |
| best16_train | GPT-2 embedding: E − S | +0.0021 | 14/19 | +0.38 | 0.11 |
| best16_train | all GPT-2: E − B | +0.0053 | 16/19 | +0.78 | 0.0032 |

### forward: GPT-2 against P3's stand-in word features (same folds, same subjects)

Tiers A and B must reproduce P3 exactly on the fixed montages. The best-k sets are picked by the main tier (E here, C in P3), so they hold different channels and A and B differ there.

| montage | max abs diff A vs P3 | max abs diff B vs P3 | E (GPT-2) − C (stand-in): mean Δr | subjects > 0 | dz | p |
| --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | 0.0e+00 | 0.0e+00 | +0.0008 | 10/19 | +0.18 | 0.45 |
| cyton8 | 0.0e+00 | 0.0e+00 | +0.0009 | 10/19 | +0.17 | 0.48 |
| cyton_daisy16 | 0.0e+00 | 0.0e+00 | +0.0008 | 11/19 | +0.17 | 0.46 |
| unicorn8 | 0.0e+00 | 0.0e+00 | +0.0014 | 11/19 | +0.26 | 0.27 |
| best8_train | 5.1e-03 | 5.4e-03 | -0.0000 | 10/19 | -0.00 | 0.99 |
| best16_train | 1.7e-03 | 1.6e-03 | +0.0004 | 10/19 | +0.08 | 0.73 |

## forward: segment identification (other held-out segments as candidates)

Pooled top-k over subjects; subject-level test is per-subject top-k against chance (one-sided t). Language against acoustics is the paired per-subject difference from tier A on the same windows.

| window s | montage | tier | N | top-1 | top-10 | chance top-1 / top-10 | subjects top-10 > chance | subject-level p | top-10 − tier A, points [95% CI] | paired p | top-1 − tier A, points [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.014 | 0.106 | 0.005 / 0.055 | 18/19 | 5.3e-07 | — | — | — |
| 5 | biosemi128 | B_env_onset | 182 | 0.013 | 0.103 | 0.005 / 0.055 | 19/19 | 1e-05 | -0.3 [-1.4, +0.8] | 0.63 | -0.1 [-0.6, +0.3] |
| 5 | biosemi128 | S_env_onset_sur | 182 | 0.013 | 0.116 | 0.005 / 0.055 | 19/19 | 1.9e-06 | +1.0 [-0.3, +2.4] | 0.12 | -0.1 [-0.6, +0.4] |
| 5 | biosemi128 | E_env_onset_sur_emb | 182 | 0.018 | 0.126 | 0.005 / 0.055 | 19/19 | 1.3e-06 | +2.1 [+0.3, +3.8] | 0.027 | +0.4 [-0.1, +1.0] |
| 5 | cyton8 | A_env | 182 | 0.012 | 0.091 | 0.005 / 0.055 | 17/19 | 2.4e-07 | — | — | — |
| 5 | cyton8 | B_env_onset | 182 | 0.010 | 0.097 | 0.005 / 0.055 | 18/19 | 9.4e-07 | +0.6 [-0.4, +1.6] | 0.25 | -0.3 [-0.8, +0.3] |
| 5 | cyton8 | S_env_onset_sur | 182 | 0.011 | 0.104 | 0.005 / 0.055 | 19/19 | 5.1e-07 | +1.3 [+0.1, +2.6] | 0.036 | -0.2 [-0.7, +0.4] |
| 5 | cyton8 | E_env_onset_sur_emb | 182 | 0.016 | 0.103 | 0.005 / 0.055 | 17/19 | 3.2e-06 | +1.2 [-0.0, +2.5] | 0.054 | +0.3 [-0.2, +0.9] |
| 5 | cyton_daisy16 | A_env | 182 | 0.013 | 0.109 | 0.005 / 0.055 | 18/19 | 1.1e-06 | — | — | — |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.012 | 0.105 | 0.005 / 0.055 | 18/19 | 1.6e-05 | -0.5 [-1.7, +0.8] | 0.44 | -0.1 [-0.6, +0.3] |
| 5 | cyton_daisy16 | S_env_onset_sur | 182 | 0.014 | 0.119 | 0.005 / 0.055 | 19/19 | 1.9e-06 | +1.0 [-0.7, +2.6] | 0.24 | +0.1 [-0.4, +0.5] |
| 5 | cyton_daisy16 | E_env_onset_sur_emb | 182 | 0.019 | 0.126 | 0.005 / 0.055 | 18/19 | 2.6e-06 | +1.6 [+0.0, +3.2] | 0.047 | +0.6 [+0.1, +1.1] |
| 5 | unicorn8 | A_env | 182 | 0.012 | 0.091 | 0.005 / 0.055 | 17/19 | 8e-07 | — | — | — |
| 5 | unicorn8 | B_env_onset | 182 | 0.012 | 0.095 | 0.005 / 0.055 | 19/19 | 2.5e-06 | +0.4 [-0.8, +1.6] | 0.47 | -0.1 [-0.5, +0.4] |
| 5 | unicorn8 | S_env_onset_sur | 182 | 0.014 | 0.106 | 0.005 / 0.055 | 19/19 | 2.2e-07 | +1.5 [+0.5, +2.5] | 0.0071 | +0.1 [-0.4, +0.7] |
| 5 | unicorn8 | E_env_onset_sur_emb | 182 | 0.016 | 0.106 | 0.005 / 0.055 | 18/19 | 4.4e-06 | +1.5 [+0.1, +2.8] | 0.034 | +0.3 [-0.1, +0.7] |
| 5 | best8_train | A_env | 182 | 0.015 | 0.102 | 0.005 / 0.055 | 17/19 | 1.1e-05 | — | — | — |
| 5 | best8_train | B_env_onset | 182 | 0.015 | 0.107 | 0.005 / 0.055 | 18/19 | 1.8e-05 | +0.5 [-0.5, +1.5] | 0.3 | -0.0 [-0.5, +0.5] |
| 5 | best8_train | S_env_onset_sur | 182 | 0.016 | 0.113 | 0.005 / 0.055 | 18/19 | 2.6e-05 | +1.2 [-0.3, +2.6] | 0.1 | +0.1 [-0.3, +0.6] |
| 5 | best8_train | E_env_onset_sur_emb | 182 | 0.018 | 0.117 | 0.005 / 0.055 | 18/19 | 1.1e-05 | +1.5 [+0.2, +2.9] | 0.026 | +0.3 [-0.2, +0.8] |
| 5 | best16_train | A_env | 182 | 0.016 | 0.106 | 0.005 / 0.055 | 18/19 | 2.9e-05 | — | — | — |
| 5 | best16_train | B_env_onset | 182 | 0.015 | 0.110 | 0.005 / 0.055 | 19/19 | 1.4e-05 | +0.3 [-0.8, +1.5] | 0.53 | -0.1 [-0.7, +0.4] |
| 5 | best16_train | S_env_onset_sur | 182 | 0.017 | 0.113 | 0.005 / 0.055 | 18/19 | 3.8e-05 | +0.7 [-0.6, +2.0] | 0.28 | +0.1 [-0.5, +0.6] |
| 5 | best16_train | E_env_onset_sur_emb | 182 | 0.018 | 0.123 | 0.005 / 0.055 | 18/19 | 6.5e-06 | +1.7 [+0.6, +2.9] | 0.0056 | +0.2 [-0.4, +0.8] |
| 10 | biosemi128 | A_env | 89 | 0.034 | 0.221 | 0.011 / 0.112 | 17/19 | 6.3e-05 | — | — | — |
| 10 | biosemi128 | B_env_onset | 89 | 0.032 | 0.224 | 0.011 / 0.112 | 18/19 | 9e-05 | +0.2 [-1.9, +2.3] | 0.82 | -0.2 [-1.2, +0.8] |
| 10 | biosemi128 | S_env_onset_sur | 89 | 0.037 | 0.247 | 0.011 / 0.112 | 19/19 | 3.1e-06 | +2.6 [+0.2, +5.0] | 0.032 | +0.3 [-0.7, +1.2] |
| 10 | biosemi128 | E_env_onset_sur_emb | 89 | 0.044 | 0.268 | 0.011 / 0.112 | 19/19 | 6.7e-07 | +4.7 [+2.2, +7.3] | 0.0011 | +1.1 [+0.0, +2.1] |
| 10 | cyton8 | A_env | 89 | 0.028 | 0.192 | 0.011 / 0.112 | 17/19 | 7.9e-05 | — | — | — |
| 10 | cyton8 | B_env_onset | 89 | 0.030 | 0.193 | 0.011 / 0.112 | 18/19 | 1.8e-05 | +0.2 [-1.7, +2.0] | 0.84 | +0.2 [-0.8, +1.3] |
| 10 | cyton8 | S_env_onset_sur | 89 | 0.028 | 0.221 | 0.011 / 0.112 | 19/19 | 1.9e-06 | +3.0 [+0.6, +5.3] | 0.016 | +0.1 [-1.0, +1.1] |
| 10 | cyton8 | E_env_onset_sur_emb | 89 | 0.037 | 0.236 | 0.011 / 0.112 | 18/19 | 5.5e-06 | +4.4 [+1.6, +7.2] | 0.0039 | +0.9 [+0.1, +1.7] |
| 10 | cyton_daisy16 | A_env | 89 | 0.036 | 0.228 | 0.011 / 0.112 | 17/19 | 0.00019 | — | — | — |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.036 | 0.222 | 0.011 / 0.112 | 18/19 | 0.00016 | -0.6 [-2.8, +1.6] | 0.58 | -0.0 [-1.1, +1.1] |
| 10 | cyton_daisy16 | S_env_onset_sur | 89 | 0.040 | 0.253 | 0.011 / 0.112 | 19/19 | 2.5e-06 | +2.5 [-0.2, +5.2] | 0.071 | +0.4 [-0.7, +1.6] |
| 10 | cyton_daisy16 | E_env_onset_sur_emb | 89 | 0.047 | 0.260 | 0.011 / 0.112 | 18/19 | 3.4e-06 | +3.2 [+0.3, +6.1] | 0.032 | +1.1 [-0.2, +2.4] |
| 10 | unicorn8 | A_env | 89 | 0.027 | 0.183 | 0.011 / 0.112 | 17/19 | 1.2e-05 | — | — | — |
| 10 | unicorn8 | B_env_onset | 89 | 0.027 | 0.203 | 0.011 / 0.112 | 19/19 | 1.8e-05 | +2.0 [-0.1, +4.1] | 0.056 | +0.1 [-0.8, +1.0] |
| 10 | unicorn8 | S_env_onset_sur | 89 | 0.027 | 0.228 | 0.011 / 0.112 | 19/19 | 4e-06 | +4.4 [+1.7, +7.2] | 0.0031 | +0.1 [-0.9, +1.0] |
| 10 | unicorn8 | E_env_onset_sur_emb | 89 | 0.033 | 0.235 | 0.011 / 0.112 | 18/19 | 4.6e-06 | +5.2 [+2.4, +8.0] | 0.0011 | +0.7 [-0.4, +1.7] |
| 10 | best8_train | A_env | 89 | 0.037 | 0.216 | 0.011 / 0.112 | 16/19 | 4e-05 | — | — | — |
| 10 | best8_train | B_env_onset | 89 | 0.035 | 0.241 | 0.011 / 0.112 | 18/19 | 2.2e-05 | +2.5 [+0.1, +5.0] | 0.043 | -0.1 [-1.0, +0.8] |
| 10 | best8_train | S_env_onset_sur | 89 | 0.041 | 0.232 | 0.011 / 0.112 | 19/19 | 4.6e-05 | +1.7 [-0.8, +4.1] | 0.17 | +0.4 [-1.1, +1.9] |
| 10 | best8_train | E_env_onset_sur_emb | 89 | 0.047 | 0.253 | 0.011 / 0.112 | 19/19 | 1.6e-05 | +3.7 [+1.0, +6.4] | 0.0096 | +1.0 [-0.3, +2.3] |
| 10 | best16_train | A_env | 89 | 0.038 | 0.229 | 0.011 / 0.112 | 17/19 | 3.2e-05 | — | — | — |
| 10 | best16_train | B_env_onset | 89 | 0.035 | 0.247 | 0.011 / 0.112 | 18/19 | 1.5e-05 | +1.8 [+0.2, +3.3] | 0.028 | -0.3 [-1.3, +0.7] |
| 10 | best16_train | S_env_onset_sur | 89 | 0.042 | 0.250 | 0.011 / 0.112 | 19/19 | 1e-05 | +2.1 [-0.1, +4.3] | 0.062 | +0.4 [-1.1, +1.9] |
| 10 | best16_train | E_env_onset_sur_emb | 89 | 0.046 | 0.258 | 0.011 / 0.112 | 19/19 | 1.7e-05 | +2.9 [+0.2, +5.6] | 0.037 | +0.8 [-0.4, +2.0] |

## forward: true words against 88 GPT-2 distractors (same audio, same onsets)

89 windows of 10 s per subject. Chance: top-1 0.011, top-10 0.112. Envelope-only and envelope+onset models score every candidate identically here, so they sit at chance by construction. 'Shuffled EEG' scores each window's candidates against another window of the same subject (mean over derangements) and must sit at chance. 'Prior only' ranks by GPT-2 log-probability with no EEG. A ranker can beat chance without using the EEG at all (for example when distractor features are atypical), so the EEG-specific evidence is the paired per-subject excess of the real EEG over the shuffled EEG, not the distance from chance.

| montage | tier | top-1 | top-10 | subjects top-1 > chance | p | subjects top-10 > chance | p | shuffled EEG top-1 / top-10 | prior only top-1 / top-10 | real − shuffled EEG top-1, points [95% CI] | p | real − shuffled EEG top-10, points [95% CI] | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | S_env_onset_sur | 0.024 | 0.157 | 12/19 | 0.0042 | 16/19 | 4e-06 | 0.012 / 0.121 | 0.045 / 0.213 | +1.1 [+0.3, +2.0] | 0.014 | +3.7 [+2.3, +5.1] | 3.5e-05 |
| biosemi128 | E_env_onset_sur_emb | 0.014 | 0.174 | 7/19 | 0.17 | 18/19 | 1.8e-06 | 0.013 / 0.118 | 0.045 / 0.213 | +0.1 [-0.5, +0.7] | 0.69 | +5.6 [+3.6, +7.6] | 1.6e-05 |
| cyton8 | S_env_onset_sur | 0.022 | 0.160 | 11/19 | 0.0069 | 17/19 | 2e-06 | 0.012 / 0.122 | 0.045 / 0.213 | +1.0 [+0.2, +1.9] | 0.024 | +3.8 [+2.1, +5.4] | 0.00014 |
| cyton8 | E_env_onset_sur_emb | 0.016 | 0.164 | 8/19 | 0.021 | 17/19 | 2.6e-05 | 0.013 / 0.118 | 0.045 / 0.213 | +0.3 [-0.2, +0.8] | 0.19 | +4.5 [+2.3, +6.8] | 0.00045 |
| cyton_daisy16 | S_env_onset_sur | 0.020 | 0.159 | 11/19 | 0.018 | 16/19 | 1.4e-05 | 0.012 / 0.119 | 0.045 / 0.213 | +0.8 [-0.0, +1.6] | 0.058 | +4.0 [+2.2, +5.8] | 0.00021 |
| cyton_daisy16 | E_env_onset_sur_emb | 0.017 | 0.170 | 9/19 | 0.035 | 18/19 | 1.6e-05 | 0.014 / 0.120 | 0.045 / 0.213 | +0.2 [-0.4, +0.8] | 0.43 | +5.0 [+2.8, +7.3] | 0.00017 |
| unicorn8 | S_env_onset_sur | 0.020 | 0.162 | 10/19 | 0.018 | 16/19 | 9e-06 | 0.013 / 0.116 | 0.045 / 0.213 | +0.8 [-0.1, +1.6] | 0.066 | +4.6 [+2.9, +6.4] | 2.5e-05 |
| unicorn8 | E_env_onset_sur_emb | 0.013 | 0.167 | 7/19 | 0.25 | 17/19 | 9e-06 | 0.012 / 0.116 | 0.045 / 0.213 | +0.1 [-0.4, +0.7] | 0.64 | +5.1 [+3.1, +7.1] | 4.8e-05 |
| best8_train | S_env_onset_sur | 0.017 | 0.152 | 9/19 | 0.043 | 16/19 | 0.00018 | 0.011 / 0.118 | 0.045 / 0.213 | +0.6 [-0.1, +1.3] | 0.092 | +3.4 [+1.6, +5.3] | 0.0011 |
| best8_train | E_env_onset_sur_emb | 0.025 | 0.176 | 16/19 | 0.00016 | 18/19 | 2.5e-05 | 0.014 / 0.123 | 0.045 / 0.213 | +1.2 [+0.4, +1.9] | 0.0047 | +5.3 [+2.7, +7.9] | 0.00048 |
| best16_train | S_env_onset_sur | 0.017 | 0.155 | 6/19 | 0.053 | 17/19 | 0.00011 | 0.013 / 0.119 | 0.045 / 0.213 | +0.5 [-0.3, +1.2] | 0.21 | +3.6 [+1.6, +5.7] | 0.0016 |
| best16_train | E_env_onset_sur_emb | 0.024 | 0.169 | 12/19 | 0.0012 | 18/19 | 9.3e-06 | 0.013 / 0.121 | 0.045 / 0.213 | +1.1 [+0.3, +1.9] | 0.01 | +4.7 [+2.6, +6.9] | 0.00018 |

## forward: starter montages against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels. Decoding differences are in points.

| montage | GPT-2 gain E − B | segment ID top-10, tier E | distractor top-1, tier E | distractor top-10, tier E |
| --- | --- | --- | --- | --- |
| cyton8 | -0.0005 [-0.0014, +0.0005] | -3.3 [-4.9, -1.6] | +0.2 [-0.4, +0.8] | -1.0 [-2.5, +0.5] |
| cyton_daisy16 | -0.0006 [-0.0011, -0.0000] | -0.9 [-2.1, +0.4] | +0.2 [-0.3, +0.7] | -0.4 [-1.5, +0.6] |

## forward: N400 check with GPT-2 surprisal (tier E kernel, 300-500 ms, centro-parietal)

| montage | mean kernel | subjects negative | dz | p |
| --- | --- | --- | --- | --- |
| biosemi128 | -0.0146 | 17/19 | -1.24 | 3.8e-05 |
| cyton8 | -0.0115 | 14/19 | -0.89 | 0.0011 |
| cyton_daisy16 | -0.0134 | 16/19 | -1.12 | 0.00012 |
| unicorn8 | -0.0129 | 17/19 | -1.08 | 0.00018 |

## reverse: encoding, held-out r (mean ± SEM over 10 subjects)

| montage | A_env | B_env_onset | S_env_onset_sur | E_env_onset_sur_emb |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0205 ± 0.0027 | 0.0212 ± 0.0031 | 0.0208 ± 0.0029 | 0.0217 ± 0.0027 |
| cyton8 | 0.0186 ± 0.0031 | 0.0190 ± 0.0033 | 0.0187 ± 0.0032 | 0.0192 ± 0.0029 |
| cyton_daisy16 | 0.0200 ± 0.0032 | 0.0205 ± 0.0034 | 0.0200 ± 0.0032 | 0.0209 ± 0.0030 |
| unicorn8 | 0.0192 ± 0.0026 | 0.0196 ± 0.0031 | 0.0194 ± 0.0030 | 0.0205 ± 0.0027 |
| best8_train | 0.0248 ± 0.0038 | 0.0258 ± 0.0039 | 0.0254 ± 0.0036 | 0.0259 ± 0.0037 |
| best16_train | 0.0256 ± 0.0033 | 0.0267 ± 0.0033 | 0.0264 ± 0.0030 | 0.0267 ± 0.0031 |

### reverse: stepwise gains (paired over subjects)

| montage | step | mean Δr | subjects Δr > 0 | dz | p |
| --- | --- | --- | --- | --- | --- |
| biosemi128 | onsets: B − A | +0.0007 | 5/10 | +0.20 | 0.54 |
| biosemi128 | GPT-2 surprisal: S − B | -0.0005 | 4/10 | -0.31 | 0.36 |
| biosemi128 | GPT-2 embedding: E − S | +0.0009 | 6/10 | +0.27 | 0.42 |
| biosemi128 | all GPT-2: E − B | +0.0004 | 6/10 | +0.10 | 0.76 |
| cyton8 | onsets: B − A | +0.0005 | 6/10 | +0.17 | 0.6 |
| cyton8 | GPT-2 surprisal: S − B | -0.0003 | 6/10 | -0.21 | 0.53 |
| cyton8 | GPT-2 embedding: E − S | +0.0005 | 6/10 | +0.21 | 0.53 |
| cyton8 | all GPT-2: E − B | +0.0002 | 7/10 | +0.06 | 0.86 |
| cyton_daisy16 | onsets: B − A | +0.0005 | 5/10 | +0.17 | 0.61 |
| cyton_daisy16 | GPT-2 surprisal: S − B | -0.0005 | 5/10 | -0.32 | 0.34 |
| cyton_daisy16 | GPT-2 embedding: E − S | +0.0008 | 6/10 | +0.28 | 0.4 |
| cyton_daisy16 | all GPT-2: E − B | +0.0004 | 7/10 | +0.10 | 0.76 |
| unicorn8 | onsets: B − A | +0.0004 | 5/10 | +0.10 | 0.75 |
| unicorn8 | GPT-2 surprisal: S − B | -0.0002 | 5/10 | -0.12 | 0.72 |
| unicorn8 | GPT-2 embedding: E − S | +0.0011 | 7/10 | +0.30 | 0.37 |
| unicorn8 | all GPT-2: E − B | +0.0009 | 7/10 | +0.20 | 0.54 |
| best8_train | onsets: B − A | +0.0010 | 6/10 | +0.23 | 0.48 |
| best8_train | GPT-2 surprisal: S − B | -0.0004 | 3/10 | -0.28 | 0.4 |
| best8_train | GPT-2 embedding: E − S | +0.0004 | 6/10 | +0.11 | 0.75 |
| best8_train | all GPT-2: E − B | +0.0001 | 6/10 | +0.01 | 0.97 |
| best16_train | onsets: B − A | +0.0011 | 5/10 | +0.24 | 0.46 |
| best16_train | GPT-2 surprisal: S − B | -0.0003 | 5/10 | -0.22 | 0.5 |
| best16_train | GPT-2 embedding: E − S | +0.0004 | 6/10 | +0.08 | 0.8 |
| best16_train | all GPT-2: E − B | +0.0000 | 7/10 | +0.00 | 0.99 |

### reverse: GPT-2 against P3's stand-in word features (same folds, same subjects)

Tiers A and B must reproduce P3 exactly on the fixed montages. The best-k sets are picked by the main tier (E here, C in P3), so they hold different channels and A and B differ there.

| montage | max abs diff A vs P3 | max abs diff B vs P3 | E (GPT-2) − C (stand-in): mean Δr | subjects > 0 | dz | p |
| --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | 0.0e+00 | 0.0e+00 | +0.0002 | 6/10 | +0.10 | 0.75 |
| cyton8 | 0.0e+00 | 0.0e+00 | +0.0001 | 6/10 | +0.06 | 0.85 |
| cyton_daisy16 | 0.0e+00 | 0.0e+00 | +0.0002 | 5/10 | +0.08 | 0.81 |
| unicorn8 | 0.0e+00 | 0.0e+00 | +0.0002 | 6/10 | +0.08 | 0.81 |
| best8_train | 5.2e-03 | 5.0e-03 | +0.0003 | 6/10 | +0.05 | 0.87 |
| best16_train | 1.1e-03 | 8.2e-04 | +0.0003 | 5/10 | +0.05 | 0.87 |

## reverse: segment identification (other held-out segments as candidates)

Pooled top-k over subjects; subject-level test is per-subject top-k against chance (one-sided t). Language against acoustics is the paired per-subject difference from tier A on the same windows.

| window s | montage | tier | N | top-1 | top-10 | chance top-1 / top-10 | subjects top-10 > chance | subject-level p | top-10 − tier A, points [95% CI] | paired p | top-1 − tier A, points [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.009 | 0.090 | 0.005 / 0.055 | 10/10 | 2.6e-06 | — | — | — |
| 5 | biosemi128 | B_env_onset | 182 | 0.005 | 0.076 | 0.005 / 0.055 | 8/10 | 0.0045 | -1.4 [-2.9, +0.2] | 0.08 | -0.4 [-0.9, +0.0] |
| 5 | biosemi128 | S_env_onset_sur | 182 | 0.006 | 0.082 | 0.005 / 0.055 | 8/10 | 0.0033 | -0.8 [-2.6, +1.1] | 0.37 | -0.3 [-0.8, +0.1] |
| 5 | biosemi128 | E_env_onset_sur_emb | 182 | 0.009 | 0.082 | 0.005 / 0.055 | 10/10 | 0.00094 | -0.8 [-2.0, +0.4] | 0.16 | -0.1 [-0.5, +0.4] |
| 5 | cyton8 | A_env | 182 | 0.009 | 0.086 | 0.005 / 0.055 | 10/10 | 0.00049 | — | — | — |
| 5 | cyton8 | B_env_onset | 182 | 0.005 | 0.080 | 0.005 / 0.055 | 9/10 | 0.0024 | -0.5 [-1.8, +0.7] | 0.33 | -0.3 [-0.7, +0.0] |
| 5 | cyton8 | S_env_onset_sur | 182 | 0.007 | 0.084 | 0.005 / 0.055 | 9/10 | 0.003 | -0.2 [-1.6, +1.3] | 0.8 | -0.2 [-0.4, -0.0] |
| 5 | cyton8 | E_env_onset_sur_emb | 182 | 0.007 | 0.088 | 0.005 / 0.055 | 9/10 | 0.00058 | +0.3 [-1.2, +1.7] | 0.67 | -0.2 [-0.6, +0.3] |
| 5 | cyton_daisy16 | A_env | 182 | 0.012 | 0.085 | 0.005 / 0.055 | 10/10 | 0.00039 | — | — | — |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.005 | 0.080 | 0.005 / 0.055 | 8/10 | 0.0062 | -0.4 [-1.7, +0.9] | 0.46 | -0.7 [-1.1, -0.3] |
| 5 | cyton_daisy16 | S_env_onset_sur | 182 | 0.008 | 0.078 | 0.005 / 0.055 | 9/10 | 0.0034 | -0.7 [-2.0, +0.7] | 0.29 | -0.4 [-0.8, -0.0] |
| 5 | cyton_daisy16 | E_env_onset_sur_emb | 182 | 0.010 | 0.082 | 0.005 / 0.055 | 8/10 | 0.0049 | -0.2 [-1.9, +1.4] | 0.77 | -0.2 [-0.7, +0.4] |
| 5 | unicorn8 | A_env | 182 | 0.007 | 0.080 | 0.005 / 0.055 | 9/10 | 0.00061 | — | — | — |
| 5 | unicorn8 | B_env_onset | 182 | 0.004 | 0.070 | 0.005 / 0.055 | 8/10 | 0.028 | -1.0 [-2.4, +0.5] | 0.16 | -0.3 [-0.6, +0.0] |
| 5 | unicorn8 | S_env_onset_sur | 182 | 0.007 | 0.073 | 0.005 / 0.055 | 9/10 | 0.0049 | -0.7 [-2.1, +0.7] | 0.28 | -0.1 [-0.4, +0.3] |
| 5 | unicorn8 | E_env_onset_sur_emb | 182 | 0.007 | 0.080 | 0.005 / 0.055 | 10/10 | 0.00015 | -0.1 [-1.0, +0.9] | 0.9 | -0.1 [-0.5, +0.4] |
| 5 | best8_train | A_env | 182 | 0.012 | 0.073 | 0.005 / 0.055 | 7/10 | 0.0084 | — | — | — |
| 5 | best8_train | B_env_onset | 182 | 0.011 | 0.071 | 0.005 / 0.055 | 8/10 | 0.0021 | -0.2 [-0.9, +0.6] | 0.64 | -0.1 [-0.5, +0.4] |
| 5 | best8_train | S_env_onset_sur | 182 | 0.010 | 0.074 | 0.005 / 0.055 | 8/10 | 0.0031 | +0.1 [-0.9, +1.0] | 0.9 | -0.2 [-0.6, +0.3] |
| 5 | best8_train | E_env_onset_sur_emb | 182 | 0.012 | 0.074 | 0.005 / 0.055 | 7/10 | 0.014 | +0.1 [-0.9, +1.1] | 0.81 | +0.1 [-0.5, +0.6] |
| 5 | best16_train | A_env | 182 | 0.011 | 0.074 | 0.005 / 0.055 | 8/10 | 0.0048 | — | — | — |
| 5 | best16_train | B_env_onset | 182 | 0.011 | 0.072 | 0.005 / 0.055 | 8/10 | 0.0055 | -0.2 [-1.4, +1.0] | 0.69 | -0.0 [-0.5, +0.5] |
| 5 | best16_train | S_env_onset_sur | 182 | 0.011 | 0.075 | 0.005 / 0.055 | 8/10 | 0.0041 | +0.1 [-0.9, +1.0] | 0.9 | -0.0 [-0.5, +0.5] |
| 5 | best16_train | E_env_onset_sur_emb | 182 | 0.012 | 0.079 | 0.005 / 0.055 | 7/10 | 0.0099 | +0.4 [-1.0, +1.8] | 0.5 | +0.1 [-0.3, +0.5] |
| 10 | biosemi128 | A_env | 89 | 0.021 | 0.193 | 0.011 / 0.112 | 9/10 | 0.00021 | — | — | — |
| 10 | biosemi128 | B_env_onset | 89 | 0.015 | 0.197 | 0.011 / 0.112 | 9/10 | 0.00035 | +0.3 [-1.3, +2.0] | 0.66 | -0.7 [-1.7, +0.3] |
| 10 | biosemi128 | S_env_onset_sur | 89 | 0.017 | 0.197 | 0.011 / 0.112 | 9/10 | 0.00024 | +0.3 [-1.2, +1.9] | 0.64 | -0.4 [-1.1, +0.2] |
| 10 | biosemi128 | E_env_onset_sur_emb | 89 | 0.024 | 0.199 | 0.011 / 0.112 | 8/10 | 0.0016 | +0.6 [-1.8, +2.9] | 0.6 | +0.2 [-0.4, +0.9] |
| 10 | cyton8 | A_env | 89 | 0.027 | 0.181 | 0.011 / 0.112 | 9/10 | 0.0013 | — | — | — |
| 10 | cyton8 | B_env_onset | 89 | 0.021 | 0.184 | 0.011 / 0.112 | 9/10 | 0.0019 | +0.3 [-2.3, +3.0] | 0.78 | -0.6 [-1.5, +0.4] |
| 10 | cyton8 | S_env_onset_sur | 89 | 0.020 | 0.190 | 0.011 / 0.112 | 9/10 | 0.0012 | +0.9 [-0.7, +2.5] | 0.25 | -0.7 [-1.5, +0.1] |
| 10 | cyton8 | E_env_onset_sur_emb | 89 | 0.029 | 0.189 | 0.011 / 0.112 | 9/10 | 0.0016 | +0.8 [-0.9, +2.5] | 0.32 | +0.2 [-0.7, +1.1] |
| 10 | cyton_daisy16 | A_env | 89 | 0.017 | 0.189 | 0.011 / 0.112 | 10/10 | 0.0014 | — | — | — |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.015 | 0.184 | 0.011 / 0.112 | 9/10 | 0.0014 | -0.4 [-2.5, +1.6] | 0.64 | -0.2 [-1.0, +0.5] |
| 10 | cyton_daisy16 | S_env_onset_sur | 89 | 0.013 | 0.184 | 0.011 / 0.112 | 9/10 | 0.0014 | -0.4 [-1.9, +1.0] | 0.51 | -0.3 [-1.1, +0.4] |
| 10 | cyton_daisy16 | E_env_onset_sur_emb | 89 | 0.022 | 0.185 | 0.011 / 0.112 | 10/10 | 0.0031 | -0.3 [-1.5, +0.8] | 0.52 | +0.6 [-0.4, +1.5] |
| 10 | unicorn8 | A_env | 89 | 0.024 | 0.187 | 0.011 / 0.112 | 9/10 | 0.00017 | — | — | — |
| 10 | unicorn8 | B_env_onset | 89 | 0.016 | 0.183 | 0.011 / 0.112 | 10/10 | 0.00033 | -0.3 [-2.2, +1.5] | 0.69 | -0.8 [-1.8, +0.2] |
| 10 | unicorn8 | S_env_onset_sur | 89 | 0.020 | 0.187 | 0.011 / 0.112 | 10/10 | 0.00028 | +0.0 [-1.8, +1.8] | 1 | -0.3 [-0.9, +0.2] |
| 10 | unicorn8 | E_env_onset_sur_emb | 89 | 0.021 | 0.191 | 0.011 / 0.112 | 10/10 | 0.00023 | +0.4 [-1.9, +2.8] | 0.67 | -0.2 [-1.3, +0.8] |
| 10 | best8_train | A_env | 89 | 0.016 | 0.173 | 0.011 / 0.112 | 10/10 | 0.00062 | — | — | — |
| 10 | best8_train | B_env_onset | 89 | 0.016 | 0.166 | 0.011 / 0.112 | 9/10 | 0.00082 | -0.7 [-3.3, +2.0] | 0.58 | +0.0 [-1.0, +1.0] |
| 10 | best8_train | S_env_onset_sur | 89 | 0.018 | 0.172 | 0.011 / 0.112 | 10/10 | 0.00049 | -0.1 [-2.4, +2.1] | 0.91 | +0.2 [-0.6, +1.1] |
| 10 | best8_train | E_env_onset_sur_emb | 89 | 0.016 | 0.188 | 0.011 / 0.112 | 10/10 | 4.4e-05 | +1.5 [-0.1, +3.0] | 0.057 | -0.0 [-1.1, +1.1] |
| 10 | best16_train | A_env | 89 | 0.016 | 0.181 | 0.011 / 0.112 | 9/10 | 0.00036 | — | — | — |
| 10 | best16_train | B_env_onset | 89 | 0.020 | 0.163 | 0.011 / 0.112 | 9/10 | 0.0019 | -1.8 [-4.0, +0.4] | 0.095 | +0.4 [-0.7, +1.6] |
| 10 | best16_train | S_env_onset_sur | 89 | 0.020 | 0.166 | 0.011 / 0.112 | 10/10 | 0.00016 | -1.5 [-3.3, +0.4] | 0.1 | +0.4 [-0.4, +1.3] |
| 10 | best16_train | E_env_onset_sur_emb | 89 | 0.022 | 0.188 | 0.011 / 0.112 | 10/10 | 1.7e-05 | +0.7 [-1.0, +2.4] | 0.39 | +0.7 [-0.5, +1.9] |

## reverse: true words against 88 GPT-2 distractors (same audio, same onsets)

89 windows of 10 s per subject. Chance: top-1 0.011, top-10 0.112. Envelope-only and envelope+onset models score every candidate identically here, so they sit at chance by construction. 'Shuffled EEG' scores each window's candidates against another window of the same subject (mean over derangements) and must sit at chance. 'Prior only' ranks by GPT-2 log-probability with no EEG. A ranker can beat chance without using the EEG at all (for example when distractor features are atypical), so the EEG-specific evidence is the paired per-subject excess of the real EEG over the shuffled EEG, not the distance from chance.

| montage | tier | top-1 | top-10 | subjects top-1 > chance | p | subjects top-10 > chance | p | shuffled EEG top-1 / top-10 | prior only top-1 / top-10 | real − shuffled EEG top-1, points [95% CI] | p | real − shuffled EEG top-10, points [95% CI] | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | S_env_onset_sur | 0.012 | 0.127 | 3/10 | 0.4 | 7/10 | 0.042 | 0.010 / 0.120 | 0.011 / 0.213 | +0.3 [-0.7, +1.2] | 0.53 | +0.7 [-1.0, +2.5] | 0.37 |
| biosemi128 | E_env_onset_sur_emb | 0.022 | 0.139 | 5/10 | 0.026 | 7/10 | 0.028 | 0.017 / 0.134 | 0.011 / 0.213 | +0.5 [-0.7, +1.7] | 0.35 | +0.5 [-2.1, +3.1] | 0.66 |
| cyton8 | S_env_onset_sur | 0.008 | 0.118 | 2/10 | 0.86 | 5/10 | 0.22 | 0.012 / 0.118 | 0.011 / 0.213 | -0.4 [-1.2, +0.4] | 0.3 | -0.0 [-1.6, +1.6] | 0.97 |
| cyton8 | E_env_onset_sur_emb | 0.019 | 0.126 | 5/10 | 0.012 | 6/10 | 0.1 | 0.017 / 0.130 | 0.011 / 0.213 | +0.3 [-0.6, +1.1] | 0.5 | -0.4 [-2.8, +2.1] | 0.73 |
| cyton_daisy16 | S_env_onset_sur | 0.013 | 0.127 | 5/10 | 0.3 | 6/10 | 0.014 | 0.012 / 0.120 | 0.011 / 0.213 | +0.2 [-0.8, +1.1] | 0.7 | +0.7 [-0.4, +1.8] | 0.17 |
| cyton_daisy16 | E_env_onset_sur_emb | 0.021 | 0.131 | 6/10 | 0.021 | 7/10 | 0.062 | 0.017 / 0.128 | 0.011 / 0.213 | +0.4 [-0.7, +1.5] | 0.4 | +0.4 [-2.5, +3.2] | 0.78 |
| unicorn8 | S_env_onset_sur | 0.013 | 0.126 | 2/10 | 0.22 | 5/10 | 0.11 | 0.010 / 0.119 | 0.011 / 0.213 | +0.4 [-0.3, +1.0] | 0.23 | +0.7 [-1.5, +2.9] | 0.49 |
| unicorn8 | E_env_onset_sur_emb | 0.020 | 0.148 | 5/10 | 0.035 | 8/10 | 0.0057 | 0.018 / 0.138 | 0.011 / 0.213 | +0.2 [-0.9, +1.3] | 0.71 | +1.1 [-1.8, +3.9] | 0.42 |
| best8_train | S_env_onset_sur | 0.008 | 0.122 | 1/10 | 0.9 | 6/10 | 0.12 | 0.010 / 0.118 | 0.011 / 0.213 | -0.2 [-0.8, +0.4] | 0.5 | +0.4 [-1.5, +2.4] | 0.61 |
| best8_train | E_env_onset_sur_emb | 0.026 | 0.142 | 6/10 | 0.019 | 7/10 | 0.061 | 0.018 / 0.134 | 0.011 / 0.213 | +0.8 [-0.6, +2.2] | 0.24 | +0.7 [-3.4, +4.9] | 0.7 |
| best16_train | S_env_onset_sur | 0.011 | 0.126 | 2/10 | 0.5 | 6/10 | 0.059 | 0.010 / 0.117 | 0.011 / 0.213 | +0.2 [-0.4, +0.7] | 0.52 | +0.9 [-1.3, +3.2] | 0.38 |
| best16_train | E_env_onset_sur_emb | 0.021 | 0.147 | 7/10 | 0.021 | 7/10 | 0.027 | 0.019 / 0.136 | 0.011 / 0.213 | +0.2 [-0.8, +1.2] | 0.65 | +1.1 [-2.5, +4.7] | 0.5 |

## reverse: starter montages against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels. Decoding differences are in points.

| montage | GPT-2 gain E − B | segment ID top-10, tier E | distractor top-1, tier E | distractor top-10, tier E |
| --- | --- | --- | --- | --- |
| cyton8 | -0.0002 [-0.0013, +0.0008] | -1.0 [-3.0, +1.0] | -0.3 [-1.5, +0.9] | -1.3 [-2.7, +0.0] |
| cyton_daisy16 | -0.0001 [-0.0006, +0.0004] | -1.3 [-3.1, +0.4] | -0.1 [-0.9, +0.7] | -0.8 [-1.9, +0.4] |

## reverse: N400 check with GPT-2 surprisal (tier E kernel, 300-500 ms, centro-parietal)

| montage | mean kernel | subjects negative | dz | p |
| --- | --- | --- | --- | --- |
| biosemi128 | +0.0001 | 4/10 | +0.09 | 0.78 |
| cyton8 | +0.0000 | 4/10 | +0.00 | 1 |
| cyton_daisy16 | +0.0001 | 4/10 | +0.09 | 0.79 |
| unicorn8 | +0.0002 | 4/10 | +0.11 | 0.73 |

## Forward against reversed speech (Welch t on per-subject gains)

| step | montage | forward mean | reverse mean | t | p |
| --- | --- | --- | --- | --- | --- |
| GPT-2 surprisal: S − B | biosemi128 | +0.0036 | -0.0005 | +3.97 | 0.00052 |
| GPT-2 embedding: E − S | biosemi128 | +0.0015 | +0.0009 | +0.34 | 0.74 |
| all GPT-2: E − B | biosemi128 | +0.0051 | +0.0004 | +2.30 | 0.029 |
| GPT-2 surprisal: S − B | cyton8 | +0.0039 | -0.0003 | +4.01 | 0.00048 |
| GPT-2 embedding: E − S | cyton8 | +0.0007 | +0.0005 | +0.14 | 0.89 |
| all GPT-2: E − B | cyton8 | +0.0046 | +0.0002 | +2.16 | 0.04 |
| GPT-2 surprisal: S − B | cyton_daisy16 | +0.0034 | -0.0005 | +3.96 | 0.00052 |
| GPT-2 embedding: E − S | cyton_daisy16 | +0.0012 | +0.0008 | +0.20 | 0.85 |
| all GPT-2: E − B | cyton_daisy16 | +0.0045 | +0.0004 | +2.06 | 0.049 |

## P4 against P4b (same subjects, folds and windows; paired over subjects)

P4: GPT-2 on the content-word stream. P4b: GPT-2 on the full transcript, features kept at the same content words; distractors keep the true function words as context.

| condition | montage | measure | P4 | P4b | P4b − P4 [95% CI] | p |
| --- | --- | --- | --- | --- | --- | --- |
| forward | biosemi128 | surprisal gain S − B (Δr) | +0.0026 | +0.0036 | +0.0010 [-0.0005, +0.0026] | 0.18 |
| forward | biosemi128 | all GPT-2 gain E − B (Δr) | +0.0047 | +0.0051 | +0.0004 [-0.0009, +0.0017] | 0.51 |
| forward | biosemi128 | N400 surprisal kernel 300–500 ms | -0.0121 | -0.0146 | -0.0025 [-0.0058, +0.0008] | 0.13 |
| forward | biosemi128 | segment ID top10, E − A (points) | +3.3 | +4.7 | +1.4 [+0.0, +2.8] | 0.043 |
| forward | biosemi128 | segment ID top1, E − A (points) | +0.2 | +1.1 | +0.8 [-0.4, +2.0] | 0.16 |
| forward | biosemi128 | distractor top10, S: real − shuffled EEG (points) | +3.0 | +3.7 | +0.6 [-1.5, +2.7] | 0.54 |
| forward | biosemi128 | distractor top1, S: real − shuffled EEG (points) | +0.4 | +1.1 | +0.7 [-0.2, +1.5] | 0.11 |
| forward | biosemi128 | distractor top1, S: shuffled EEG (%) | +1.0 | +1.2 | +0.3 [+0.1, +0.4] | 0.0046 |
| forward | biosemi128 | distractor top10, E: real − shuffled EEG (points) | +5.2 | +5.6 | +0.4 [-1.8, +2.6] | 0.7 |
| forward | biosemi128 | distractor top1, E: real − shuffled EEG (points) | +1.1 | +0.1 | -1.0 [-2.0, +0.0] | 0.058 |
| forward | biosemi128 | distractor top1, E: shuffled EEG (%) | +1.2 | +1.3 | +0.1 [-0.1, +0.3] | 0.17 |
| forward | biosemi128 | distractor top1: prior only (%) | +1.1 | +4.5 | n/a | n/a |
| forward | cyton8 | surprisal gain S − B (Δr) | +0.0028 | +0.0039 | +0.0011 [-0.0005, +0.0026] | 0.16 |
| forward | cyton8 | all GPT-2 gain E − B (Δr) | +0.0039 | +0.0046 | +0.0007 [-0.0007, +0.0021] | 0.3 |
| forward | cyton8 | N400 surprisal kernel 300–500 ms | -0.0098 | -0.0115 | -0.0017 [-0.0052, +0.0019] | 0.34 |
| forward | cyton8 | segment ID top10, E − A (points) | +4.1 | +4.4 | +0.4 [-0.8, +1.6] | 0.54 |
| forward | cyton8 | segment ID top1, E − A (points) | +0.8 | +0.9 | +0.1 [-0.9, +1.2] | 0.82 |
| forward | cyton8 | distractor top10, S: real − shuffled EEG (points) | +2.9 | +3.8 | +0.9 [-1.4, +3.1] | 0.42 |
| forward | cyton8 | distractor top1, S: real − shuffled EEG (points) | +0.3 | +1.0 | +0.7 [-0.3, +1.7] | 0.18 |
| forward | cyton8 | distractor top1, S: shuffled EEG (%) | +1.1 | +1.2 | +0.1 [-0.0, +0.3] | 0.089 |
| forward | cyton8 | distractor top10, E: real − shuffled EEG (points) | +4.8 | +4.5 | -0.2 [-2.2, +1.7] | 0.8 |
| forward | cyton8 | distractor top1, E: real − shuffled EEG (points) | +1.0 | +0.3 | -0.7 [-1.5, +0.1] | 0.1 |
| forward | cyton8 | distractor top1, E: shuffled EEG (%) | +1.2 | +1.3 | +0.1 [-0.1, +0.2] | 0.38 |
| forward | cyton8 | distractor top1: prior only (%) | +1.1 | +4.5 | n/a | n/a |
| forward | cyton_daisy16 | surprisal gain S − B (Δr) | +0.0024 | +0.0034 | +0.0010 [-0.0004, +0.0025] | 0.16 |
| forward | cyton_daisy16 | all GPT-2 gain E − B (Δr) | +0.0042 | +0.0045 | +0.0004 [-0.0009, +0.0016] | 0.53 |
| forward | cyton_daisy16 | N400 surprisal kernel 300–500 ms | -0.0118 | -0.0134 | -0.0016 [-0.0050, +0.0019] | 0.35 |
| forward | cyton_daisy16 | segment ID top10, E − A (points) | +3.0 | +3.2 | +0.2 [-1.1, +1.5] | 0.78 |
| forward | cyton_daisy16 | segment ID top1, E − A (points) | +0.6 | +1.1 | +0.5 [-0.8, +1.8] | 0.4 |
| forward | cyton_daisy16 | distractor top10, S: real − shuffled EEG (points) | +2.5 | +4.0 | +1.4 [-0.6, +3.4] | 0.15 |
| forward | cyton_daisy16 | distractor top1, S: real − shuffled EEG (points) | +0.5 | +0.8 | +0.3 [-0.6, +1.3] | 0.48 |
| forward | cyton_daisy16 | distractor top1, S: shuffled EEG (%) | +1.1 | +1.2 | +0.1 [+0.0, +0.3] | 0.049 |
| forward | cyton_daisy16 | distractor top10, E: real − shuffled EEG (points) | +4.5 | +5.0 | +0.5 [-1.2, +2.3] | 0.54 |
| forward | cyton_daisy16 | distractor top1, E: real − shuffled EEG (points) | +1.1 | +0.2 | -0.9 [-1.6, -0.2] | 0.021 |
| forward | cyton_daisy16 | distractor top1, E: shuffled EEG (%) | +1.2 | +1.4 | +0.2 [-0.0, +0.5] | 0.051 |
| forward | cyton_daisy16 | distractor top1: prior only (%) | +1.1 | +4.5 | n/a | n/a |
| reverse | biosemi128 | surprisal gain S − B (Δr) | -0.0006 | -0.0005 | +0.0001 [-0.0002, +0.0005] | 0.53 |
| reverse | biosemi128 | all GPT-2 gain E − B (Δr) | -0.0005 | +0.0004 | +0.0009 [+0.0005, +0.0014] | 0.00086 |
| reverse | biosemi128 | N400 surprisal kernel 300–500 ms | +0.0001 | +0.0001 | +0.0001 [-0.0007, +0.0009] | 0.85 |
| reverse | biosemi128 | segment ID top10, E − A (points) | -0.1 | +0.6 | +0.7 [-0.9, +2.2] | 0.34 |
| reverse | biosemi128 | segment ID top1, E − A (points) | -0.1 | +0.2 | +0.3 [-0.1, +0.7] | 0.081 |
| reverse | biosemi128 | distractor top10, S: real − shuffled EEG (points) | +1.1 | +0.7 | -0.3 [-3.0, +2.4] | 0.8 |
| reverse | biosemi128 | distractor top1, S: real − shuffled EEG (points) | +0.3 | +0.3 | +0.0 [-1.3, +1.3] | 0.98 |
| reverse | biosemi128 | distractor top1, S: shuffled EEG (%) | +1.0 | +1.0 | -0.0 [-0.3, +0.2] | 0.92 |
| reverse | biosemi128 | distractor top10, E: real − shuffled EEG (points) | +0.3 | +0.5 | +0.2 [-1.6, +2.1] | 0.79 |
| reverse | biosemi128 | distractor top1, E: real − shuffled EEG (points) | +0.3 | +0.5 | +0.2 [-1.4, +1.8] | 0.79 |
| reverse | biosemi128 | distractor top1, E: shuffled EEG (%) | +1.8 | +1.7 | -0.1 [-0.4, +0.3] | 0.63 |
| reverse | biosemi128 | distractor top1: prior only (%) | +0.0 | +1.1 | n/a | n/a |
| reverse | cyton8 | surprisal gain S − B (Δr) | -0.0004 | -0.0003 | +0.0001 [-0.0003, +0.0005] | 0.48 |
| reverse | cyton8 | all GPT-2 gain E − B (Δr) | -0.0008 | +0.0002 | +0.0010 [+0.0006, +0.0014] | 0.00034 |
| reverse | cyton8 | N400 surprisal kernel 300–500 ms | +0.0001 | +0.0000 | -0.0001 [-0.0008, +0.0006] | 0.8 |
| reverse | cyton8 | segment ID top10, E − A (points) | +0.0 | +0.8 | +0.8 [-1.2, +2.8] | 0.4 |
| reverse | cyton8 | segment ID top1, E − A (points) | -0.1 | +0.2 | +0.3 [-0.3, +1.0] | 0.28 |
| reverse | cyton8 | distractor top10, S: real − shuffled EEG (points) | +0.1 | -0.0 | -0.1 [-2.6, +2.4] | 0.93 |
| reverse | cyton8 | distractor top1, S: real − shuffled EEG (points) | +0.3 | -0.4 | -0.7 [-2.0, +0.6] | 0.27 |
| reverse | cyton8 | distractor top1, S: shuffled EEG (%) | +0.8 | +1.2 | +0.3 [+0.1, +0.6] | 0.025 |
| reverse | cyton8 | distractor top10, E: real − shuffled EEG (points) | -0.3 | -0.4 | -0.1 [-2.0, +1.8] | 0.9 |
| reverse | cyton8 | distractor top1, E: real − shuffled EEG (points) | +0.3 | +0.3 | -0.1 [-0.9, +0.7] | 0.86 |
| reverse | cyton8 | distractor top1, E: shuffled EEG (%) | +1.7 | +1.7 | -0.0 [-0.2, +0.1] | 0.6 |
| reverse | cyton8 | distractor top1: prior only (%) | +0.0 | +1.1 | n/a | n/a |
| reverse | cyton_daisy16 | surprisal gain S − B (Δr) | -0.0005 | -0.0005 | +0.0001 [-0.0003, +0.0005] | 0.62 |
| reverse | cyton_daisy16 | all GPT-2 gain E − B (Δr) | -0.0007 | +0.0004 | +0.0011 [+0.0006, +0.0015] | 0.00049 |
| reverse | cyton_daisy16 | N400 surprisal kernel 300–500 ms | +0.0000 | +0.0001 | +0.0001 [-0.0006, +0.0008] | 0.71 |
| reverse | cyton_daisy16 | segment ID top10, E − A (points) | -0.6 | -0.3 | +0.2 [-1.2, +1.6] | 0.73 |
| reverse | cyton_daisy16 | segment ID top1, E − A (points) | +0.6 | +0.6 | +0.0 [-0.5, +0.5] | 1 |
| reverse | cyton_daisy16 | distractor top10, S: real − shuffled EEG (points) | +0.6 | +0.7 | +0.1 [-1.7, +2.0] | 0.87 |
| reverse | cyton_daisy16 | distractor top1, S: real − shuffled EEG (points) | +0.3 | +0.2 | -0.2 [-1.0, +0.7] | 0.64 |
| reverse | cyton_daisy16 | distractor top1, S: shuffled EEG (%) | +1.0 | +1.2 | +0.2 [-0.1, +0.5] | 0.19 |
| reverse | cyton_daisy16 | distractor top10, E: real − shuffled EEG (points) | +0.8 | +0.4 | -0.4 [-2.5, +1.7] | 0.67 |
| reverse | cyton_daisy16 | distractor top1, E: real − shuffled EEG (points) | -0.4 | +0.4 | +0.8 [-0.6, +2.3] | 0.22 |
| reverse | cyton_daisy16 | distractor top1, E: shuffled EEG (%) | +1.9 | +1.7 | -0.2 [-0.6, +0.3] | 0.42 |
| reverse | cyton_daisy16 | distractor top1: prior only (%) | +0.0 | +1.1 | n/a | n/a |

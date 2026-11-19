# P4 summary: GPT-2 word features (held-out subjects, held-out runs)

GPT-2 small reads each run's content-word stream (the release has no function words), lower-cased, with a period at each sentence close, from the run's start only.

## forward: encoding, held-out r (mean ± SEM over 19 subjects)

| montage | A_env | B_env_onset | S_env_onset_sur | E_env_onset_sur_emb |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0210 ± 0.0034 | 0.0263 ± 0.0036 | 0.0289 ± 0.0037 | 0.0310 ± 0.0038 |
| cyton8 | 0.0170 ± 0.0026 | 0.0220 ± 0.0030 | 0.0249 ± 0.0031 | 0.0260 ± 0.0032 |
| cyton_daisy16 | 0.0196 ± 0.0034 | 0.0249 ± 0.0035 | 0.0272 ± 0.0035 | 0.0290 ± 0.0038 |
| unicorn8 | 0.0202 ± 0.0028 | 0.0253 ± 0.0033 | 0.0287 ± 0.0033 | 0.0305 ± 0.0034 |
| best8_train | 0.0283 ± 0.0045 | 0.0345 ± 0.0045 | 0.0365 ± 0.0045 | 0.0395 ± 0.0051 |
| best16_train | 0.0280 ± 0.0046 | 0.0341 ± 0.0046 | 0.0362 ± 0.0046 | 0.0393 ± 0.0051 |

### forward: stepwise gains (paired over subjects)

| montage | step | mean Δr | subjects Δr > 0 | dz | p |
| --- | --- | --- | --- | --- | --- |
| biosemi128 | onsets: B − A | +0.0053 | 15/19 | +0.88 | 0.0012 |
| biosemi128 | GPT-2 surprisal: S − B | +0.0026 | 18/19 | +1.09 | 0.00016 |
| biosemi128 | GPT-2 embedding: E − S | +0.0021 | 12/19 | +0.41 | 0.092 |
| biosemi128 | all GPT-2: E − B | +0.0047 | 14/19 | +0.80 | 0.0027 |
| cyton8 | onsets: B − A | +0.0051 | 13/19 | +0.76 | 0.0038 |
| cyton8 | GPT-2 surprisal: S − B | +0.0028 | 15/19 | +0.99 | 0.00041 |
| cyton8 | GPT-2 embedding: E − S | +0.0011 | 11/19 | +0.20 | 0.39 |
| cyton8 | all GPT-2: E − B | +0.0039 | 13/19 | +0.60 | 0.018 |
| cyton_daisy16 | onsets: B − A | +0.0052 | 15/19 | +0.92 | 0.00085 |
| cyton_daisy16 | GPT-2 surprisal: S − B | +0.0024 | 15/19 | +0.97 | 0.00051 |
| cyton_daisy16 | GPT-2 embedding: E − S | +0.0018 | 13/19 | +0.33 | 0.16 |
| cyton_daisy16 | all GPT-2: E − B | +0.0042 | 13/19 | +0.66 | 0.01 |
| unicorn8 | onsets: B − A | +0.0051 | 13/19 | +0.71 | 0.006 |
| unicorn8 | GPT-2 surprisal: S − B | +0.0034 | 17/19 | +1.21 | 5.4e-05 |
| unicorn8 | GPT-2 embedding: E − S | +0.0018 | 12/19 | +0.35 | 0.14 |
| unicorn8 | all GPT-2: E − B | +0.0052 | 14/19 | +0.89 | 0.0011 |
| best8_train | onsets: B − A | +0.0062 | 18/19 | +1.22 | 4.6e-05 |
| best8_train | GPT-2 surprisal: S − B | +0.0020 | 17/19 | +0.87 | 0.0013 |
| best8_train | GPT-2 embedding: E − S | +0.0030 | 14/19 | +0.50 | 0.044 |
| best8_train | all GPT-2: E − B | +0.0050 | 16/19 | +0.71 | 0.0062 |
| best16_train | onsets: B − A | +0.0061 | 17/19 | +1.11 | 0.00013 |
| best16_train | GPT-2 surprisal: S − B | +0.0021 | 17/19 | +0.94 | 0.00066 |
| best16_train | GPT-2 embedding: E − S | +0.0031 | 14/19 | +0.55 | 0.028 |
| best16_train | all GPT-2: E − B | +0.0052 | 16/19 | +0.82 | 0.0023 |

### forward: GPT-2 against P3's stand-in word features (same folds, same subjects)

Tiers A and B must reproduce P3 exactly on the fixed montages. The best-k sets are picked by the main tier (E here, C in P3), so they hold different channels and A and B differ there.

| montage | max abs diff A vs P3 | max abs diff B vs P3 | E (GPT-2) − C (stand-in): mean Δr | subjects > 0 | dz | p |
| --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | 0.0e+00 | 0.0e+00 | +0.0004 | 8/19 | +0.12 | 0.62 |
| cyton8 | 0.0e+00 | 0.0e+00 | +0.0002 | 9/19 | +0.04 | 0.86 |
| cyton_daisy16 | 0.0e+00 | 0.0e+00 | +0.0004 | 7/19 | +0.11 | 0.62 |
| unicorn8 | 0.0e+00 | 0.0e+00 | +0.0006 | 10/19 | +0.17 | 0.47 |
| best8_train | 3.1e-03 | 2.7e-03 | +0.0008 | 12/19 | +0.16 | 0.49 |
| best16_train | 1.7e-03 | 1.6e-03 | +0.0003 | 10/19 | +0.09 | 0.71 |

## forward: segment identification (other held-out segments as candidates)

Pooled top-k over subjects; subject-level test is per-subject top-k against chance (one-sided t). Language against acoustics is the paired per-subject difference from tier A on the same windows.

| window s | montage | tier | N | top-1 | top-10 | chance top-1 / top-10 | subjects top-10 > chance | subject-level p | top-10 − tier A, points [95% CI] | paired p | top-1 − tier A, points [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.014 | 0.106 | 0.005 / 0.055 | 18/19 | 5.3e-07 | — | — | — |
| 5 | biosemi128 | B_env_onset | 182 | 0.013 | 0.103 | 0.005 / 0.055 | 19/19 | 1e-05 | -0.3 [-1.4, +0.8] | 0.63 | -0.1 [-0.6, +0.3] |
| 5 | biosemi128 | S_env_onset_sur | 182 | 0.013 | 0.106 | 0.005 / 0.055 | 18/19 | 3.1e-06 | +0.0 [-1.1, +1.1] | 0.96 | -0.1 [-0.5, +0.4] |
| 5 | biosemi128 | E_env_onset_sur_emb | 182 | 0.017 | 0.111 | 0.005 / 0.055 | 19/19 | 7.8e-07 | +0.5 [-0.6, +1.6] | 0.35 | +0.3 [-0.2, +0.7] |
| 5 | cyton8 | A_env | 182 | 0.012 | 0.091 | 0.005 / 0.055 | 17/19 | 2.4e-07 | — | — | — |
| 5 | cyton8 | B_env_onset | 182 | 0.010 | 0.097 | 0.005 / 0.055 | 18/19 | 9.4e-07 | +0.6 [-0.4, +1.6] | 0.25 | -0.3 [-0.8, +0.3] |
| 5 | cyton8 | S_env_onset_sur | 182 | 0.011 | 0.096 | 0.005 / 0.055 | 18/19 | 2.7e-07 | +0.5 [-0.5, +1.5] | 0.3 | -0.1 [-0.6, +0.4] |
| 5 | cyton8 | E_env_onset_sur_emb | 182 | 0.015 | 0.101 | 0.005 / 0.055 | 17/19 | 1.7e-06 | +1.0 [-0.2, +2.2] | 0.094 | +0.2 [-0.3, +0.8] |
| 5 | cyton_daisy16 | A_env | 182 | 0.013 | 0.109 | 0.005 / 0.055 | 18/19 | 1.1e-06 | — | — | — |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.012 | 0.105 | 0.005 / 0.055 | 18/19 | 1.6e-05 | -0.5 [-1.7, +0.8] | 0.44 | -0.1 [-0.6, +0.3] |
| 5 | cyton_daisy16 | S_env_onset_sur | 182 | 0.014 | 0.107 | 0.005 / 0.055 | 18/19 | 2.6e-06 | -0.2 [-1.5, +1.1] | 0.74 | +0.1 [-0.3, +0.6] |
| 5 | cyton_daisy16 | E_env_onset_sur_emb | 182 | 0.017 | 0.115 | 0.005 / 0.055 | 19/19 | 2.7e-06 | +0.5 [-0.7, +1.8] | 0.4 | +0.4 [-0.2, +0.9] |
| 5 | unicorn8 | A_env | 182 | 0.012 | 0.091 | 0.005 / 0.055 | 17/19 | 8e-07 | — | — | — |
| 5 | unicorn8 | B_env_onset | 182 | 0.012 | 0.095 | 0.005 / 0.055 | 19/19 | 2.5e-06 | +0.4 [-0.8, +1.6] | 0.47 | -0.1 [-0.5, +0.4] |
| 5 | unicorn8 | S_env_onset_sur | 182 | 0.013 | 0.102 | 0.005 / 0.055 | 19/19 | 1e-07 | +1.1 [+0.2, +1.9] | 0.014 | +0.1 [-0.4, +0.5] |
| 5 | unicorn8 | E_env_onset_sur_emb | 182 | 0.013 | 0.102 | 0.005 / 0.055 | 19/19 | 2.5e-07 | +1.1 [+0.1, +2.1] | 0.038 | +0.1 [-0.2, +0.4] |
| 5 | best8_train | A_env | 182 | 0.016 | 0.105 | 0.005 / 0.055 | 17/19 | 1.1e-05 | — | — | — |
| 5 | best8_train | B_env_onset | 182 | 0.017 | 0.110 | 0.005 / 0.055 | 17/19 | 1.6e-05 | +0.5 [-0.6, +1.6] | 0.34 | +0.1 [-0.4, +0.6] |
| 5 | best8_train | S_env_onset_sur | 182 | 0.017 | 0.112 | 0.005 / 0.055 | 17/19 | 6.7e-05 | +0.8 [-0.7, +2.2] | 0.27 | +0.1 [-0.4, +0.6] |
| 5 | best8_train | E_env_onset_sur_emb | 182 | 0.017 | 0.114 | 0.005 / 0.055 | 19/19 | 1.7e-06 | +0.9 [-0.0, +1.9] | 0.053 | +0.1 [-0.5, +0.7] |
| 5 | best16_train | A_env | 182 | 0.016 | 0.106 | 0.005 / 0.055 | 18/19 | 3.8e-05 | — | — | — |
| 5 | best16_train | B_env_onset | 182 | 0.015 | 0.111 | 0.005 / 0.055 | 19/19 | 1.2e-05 | +0.5 [-0.6, +1.7] | 0.36 | -0.1 [-0.7, +0.4] |
| 5 | best16_train | S_env_onset_sur | 182 | 0.018 | 0.110 | 0.005 / 0.055 | 19/19 | 5.6e-05 | +0.5 [-0.8, +1.8] | 0.45 | +0.1 [-0.3, +0.6] |
| 5 | best16_train | E_env_onset_sur_emb | 182 | 0.019 | 0.118 | 0.005 / 0.055 | 18/19 | 1e-05 | +1.2 [+0.2, +2.3] | 0.026 | +0.3 [-0.3, +0.9] |
| 10 | biosemi128 | A_env | 89 | 0.034 | 0.221 | 0.011 / 0.112 | 17/19 | 6.3e-05 | — | — | — |
| 10 | biosemi128 | B_env_onset | 89 | 0.032 | 0.224 | 0.011 / 0.112 | 18/19 | 9e-05 | +0.2 [-1.9, +2.3] | 0.82 | -0.2 [-1.2, +0.8] |
| 10 | biosemi128 | S_env_onset_sur | 89 | 0.031 | 0.245 | 0.011 / 0.112 | 19/19 | 1.1e-05 | +2.4 [+0.2, +4.5] | 0.035 | -0.3 [-1.1, +0.5] |
| 10 | biosemi128 | E_env_onset_sur_emb | 89 | 0.036 | 0.254 | 0.011 / 0.112 | 19/19 | 1.6e-06 | +3.3 [+1.1, +5.5] | 0.0054 | +0.2 [-0.8, +1.3] |
| 10 | cyton8 | A_env | 89 | 0.028 | 0.192 | 0.011 / 0.112 | 17/19 | 7.9e-05 | — | — | — |
| 10 | cyton8 | B_env_onset | 89 | 0.030 | 0.193 | 0.011 / 0.112 | 18/19 | 1.8e-05 | +0.2 [-1.7, +2.0] | 0.84 | +0.2 [-0.8, +1.3] |
| 10 | cyton8 | S_env_onset_sur | 89 | 0.028 | 0.213 | 0.011 / 0.112 | 19/19 | 1.7e-05 | +2.2 [+0.4, +4.0] | 0.022 | -0.0 [-1.0, +1.0] |
| 10 | cyton8 | E_env_onset_sur_emb | 89 | 0.035 | 0.232 | 0.011 / 0.112 | 19/19 | 1.2e-06 | +4.1 [+1.9, +6.2] | 0.0009 | +0.8 [-0.3, +1.8] |
| 10 | cyton_daisy16 | A_env | 89 | 0.036 | 0.228 | 0.011 / 0.112 | 17/19 | 0.00019 | — | — | — |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.036 | 0.222 | 0.011 / 0.112 | 18/19 | 0.00016 | -0.6 [-2.8, +1.6] | 0.58 | -0.0 [-1.1, +1.1] |
| 10 | cyton_daisy16 | S_env_onset_sur | 89 | 0.035 | 0.242 | 0.011 / 0.112 | 18/19 | 2.3e-05 | +1.5 [-1.2, +4.2] | 0.26 | -0.1 [-1.1, +0.9] |
| 10 | cyton_daisy16 | E_env_onset_sur_emb | 89 | 0.042 | 0.258 | 0.011 / 0.112 | 18/19 | 2.3e-06 | +3.0 [+0.1, +6.0] | 0.046 | +0.6 [-0.5, +1.7] |
| 10 | unicorn8 | A_env | 89 | 0.027 | 0.183 | 0.011 / 0.112 | 17/19 | 1.2e-05 | — | — | — |
| 10 | unicorn8 | B_env_onset | 89 | 0.027 | 0.203 | 0.011 / 0.112 | 19/19 | 1.8e-05 | +2.0 [-0.1, +4.1] | 0.056 | +0.1 [-0.8, +1.0] |
| 10 | unicorn8 | S_env_onset_sur | 89 | 0.028 | 0.220 | 0.011 / 0.112 | 19/19 | 1.7e-06 | +3.7 [+1.7, +5.7] | 0.0011 | +0.1 [-0.8, +1.0] |
| 10 | unicorn8 | E_env_onset_sur_emb | 89 | 0.033 | 0.231 | 0.011 / 0.112 | 19/19 | 1.7e-06 | +4.7 [+2.4, +7.1] | 0.00056 | +0.6 [-0.1, +1.3] |
| 10 | best8_train | A_env | 89 | 0.042 | 0.216 | 0.011 / 0.112 | 15/19 | 4.3e-05 | — | — | — |
| 10 | best8_train | B_env_onset | 89 | 0.037 | 0.241 | 0.011 / 0.112 | 18/19 | 1.7e-05 | +2.4 [+0.0, +4.8] | 0.046 | -0.5 [-1.5, +0.4] |
| 10 | best8_train | S_env_onset_sur | 89 | 0.039 | 0.235 | 0.011 / 0.112 | 18/19 | 2.3e-05 | +1.9 [-0.5, +4.3] | 0.11 | -0.3 [-1.5, +1.0] |
| 10 | best8_train | E_env_onset_sur_emb | 89 | 0.044 | 0.260 | 0.011 / 0.112 | 19/19 | 4e-06 | +4.3 [+2.0, +6.6] | 0.0009 | +0.2 [-0.9, +1.3] |
| 10 | best16_train | A_env | 89 | 0.041 | 0.229 | 0.011 / 0.112 | 17/19 | 3.3e-05 | — | — | — |
| 10 | best16_train | B_env_onset | 89 | 0.034 | 0.245 | 0.011 / 0.112 | 18/19 | 1.9e-05 | +1.5 [-0.2, +3.3] | 0.086 | -0.7 [-1.7, +0.2] |
| 10 | best16_train | S_env_onset_sur | 89 | 0.044 | 0.251 | 0.011 / 0.112 | 18/19 | 1.3e-05 | +2.2 [-0.1, +4.4] | 0.056 | +0.4 [-0.9, +1.6] |
| 10 | best16_train | E_env_onset_sur_emb | 89 | 0.047 | 0.265 | 0.011 / 0.112 | 19/19 | 4.9e-06 | +3.5 [+1.3, +5.8] | 0.004 | +0.6 [-0.7, +1.9] |

## forward: true words against 88 GPT-2 distractors (same audio, same onsets)

89 windows of 10 s per subject. Chance: top-1 0.011, top-10 0.112. Envelope-only and envelope+onset models score every candidate identically here, so they sit at chance by construction. 'Shuffled EEG' scores each window's candidates against another window of the same subject (mean over derangements) and must sit at chance. 'Prior only' ranks by GPT-2 log-probability with no EEG. A ranker can beat chance without using the EEG at all (for example when distractor features are atypical), so the EEG-specific evidence is the paired per-subject excess of the real EEG over the shuffled EEG, not the distance from chance.

| montage | tier | top-1 | top-10 | subjects top-1 > chance | p | subjects top-10 > chance | p | shuffled EEG top-1 / top-10 | prior only top-1 / top-10 | real − shuffled EEG top-1, points [95% CI] | p | real − shuffled EEG top-10, points [95% CI] | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | S_env_onset_sur | 0.014 | 0.141 | 9/19 | 0.19 | 15/19 | 0.00027 | 0.010 / 0.111 | 0.011 / 0.090 | +0.4 [-0.3, +1.2] | 0.23 | +3.0 [+1.5, +4.6] | 0.00069 |
| biosemi128 | E_env_onset_sur_emb | 0.023 | 0.168 | 12/19 | 0.0016 | 18/19 | 8.1e-06 | 0.012 / 0.116 | 0.011 / 0.090 | +1.1 [+0.4, +1.8] | 0.0038 | +5.2 [+3.3, +7.1] | 2.2e-05 |
| cyton8 | S_env_onset_sur | 0.014 | 0.141 | 6/19 | 0.17 | 15/19 | 0.00061 | 0.011 / 0.112 | 0.011 / 0.090 | +0.3 [-0.3, +1.0] | 0.27 | +2.9 [+1.4, +4.5] | 0.00094 |
| cyton8 | E_env_onset_sur_emb | 0.022 | 0.164 | 13/19 | 0.0013 | 16/19 | 1.8e-05 | 0.012 / 0.117 | 0.011 / 0.090 | +1.0 [+0.3, +1.6] | 0.0056 | +4.8 [+2.8, +6.7] | 6.5e-05 |
| cyton_daisy16 | S_env_onset_sur | 0.015 | 0.136 | 9/19 | 0.12 | 14/19 | 0.0032 | 0.011 / 0.111 | 0.011 / 0.090 | +0.5 [-0.3, +1.2] | 0.2 | +2.5 [+0.7, +4.4] | 0.011 |
| cyton_daisy16 | E_env_onset_sur_emb | 0.023 | 0.164 | 14/19 | 0.00036 | 17/19 | 3e-05 | 0.012 / 0.120 | 0.011 / 0.090 | +1.1 [+0.4, +1.8] | 0.0024 | +4.5 [+2.3, +6.6] | 0.00036 |
| unicorn8 | S_env_onset_sur | 0.013 | 0.145 | 6/19 | 0.25 | 16/19 | 0.00029 | 0.009 / 0.107 | 0.011 / 0.090 | +0.4 [-0.2, +1.0] | 0.17 | +3.9 [+2.0, +5.7] | 0.00035 |
| unicorn8 | E_env_onset_sur_emb | 0.020 | 0.174 | 12/19 | 0.0004 | 19/19 | 1.6e-06 | 0.012 / 0.116 | 0.011 / 0.090 | +0.8 [+0.3, +1.4] | 0.0028 | +5.8 [+4.0, +7.6] | 2.8e-06 |
| best8_train | S_env_onset_sur | 0.011 | 0.141 | 3/19 | 0.61 | 15/19 | 0.00098 | 0.010 / 0.113 | 0.011 / 0.090 | +0.0 [-0.5, +0.5] | 0.95 | +2.8 [+1.0, +4.5] | 0.0036 |
| best8_train | E_env_onset_sur_emb | 0.021 | 0.161 | 10/19 | 0.012 | 17/19 | 0.00017 | 0.014 / 0.123 | 0.011 / 0.090 | +0.7 [-0.1, +1.5] | 0.085 | +3.8 [+1.5, +6.1] | 0.0024 |
| best16_train | S_env_onset_sur | 0.012 | 0.144 | 5/19 | 0.32 | 12/19 | 0.0034 | 0.009 / 0.110 | 0.011 / 0.090 | +0.3 [-0.2, +0.9] | 0.2 | +3.4 [+1.0, +5.8] | 0.0074 |
| best16_train | E_env_onset_sur_emb | 0.022 | 0.167 | 11/19 | 0.0053 | 17/19 | 7.2e-05 | 0.014 / 0.120 | 0.011 / 0.090 | +0.9 [+0.1, +1.7] | 0.034 | +4.7 [+2.2, +7.2] | 0.00079 |

## forward: starter montages against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels. Decoding differences are in points.

| montage | GPT-2 gain E − B | segment ID top-10, tier E | distractor top-1, tier E | distractor top-10, tier E |
| --- | --- | --- | --- | --- |
| cyton8 | -0.0008 [-0.0016, +0.0001] | -2.2 [-4.2, -0.2] | -0.1 [-0.8, +0.6] | -0.4 [-1.7, +0.9] |
| cyton_daisy16 | -0.0005 [-0.0010, -0.0000] | +0.4 [-0.8, +1.5] | +0.0 [-0.5, +0.5] | -0.4 [-1.5, +0.8] |

## forward: N400 check with GPT-2 surprisal (tier E kernel, 300-500 ms, centro-parietal)

| montage | mean kernel | subjects negative | dz | p |
| --- | --- | --- | --- | --- |
| biosemi128 | -0.0121 | 17/19 | -1.30 | 2.2e-05 |
| cyton8 | -0.0098 | 16/19 | -1.04 | 0.00026 |
| cyton_daisy16 | -0.0118 | 18/19 | -1.29 | 2.5e-05 |
| unicorn8 | -0.0108 | 17/19 | -1.15 | 9.1e-05 |

## reverse: encoding, held-out r (mean ± SEM over 10 subjects)

| montage | A_env | B_env_onset | S_env_onset_sur | E_env_onset_sur_emb |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0205 ± 0.0027 | 0.0212 ± 0.0031 | 0.0207 ± 0.0029 | 0.0207 ± 0.0026 |
| cyton8 | 0.0186 ± 0.0031 | 0.0190 ± 0.0033 | 0.0186 ± 0.0032 | 0.0182 ± 0.0028 |
| cyton_daisy16 | 0.0200 ± 0.0032 | 0.0205 ± 0.0034 | 0.0199 ± 0.0032 | 0.0198 ± 0.0029 |
| unicorn8 | 0.0192 ± 0.0026 | 0.0196 ± 0.0031 | 0.0193 ± 0.0030 | 0.0199 ± 0.0026 |
| best8_train | 0.0247 ± 0.0038 | 0.0258 ± 0.0039 | 0.0254 ± 0.0037 | 0.0240 ± 0.0034 |
| best16_train | 0.0254 ± 0.0033 | 0.0266 ± 0.0033 | 0.0263 ± 0.0031 | 0.0248 ± 0.0029 |

### reverse: stepwise gains (paired over subjects)

| montage | step | mean Δr | subjects Δr > 0 | dz | p |
| --- | --- | --- | --- | --- | --- |
| biosemi128 | onsets: B − A | +0.0007 | 5/10 | +0.20 | 0.54 |
| biosemi128 | GPT-2 surprisal: S − B | -0.0006 | 3/10 | -0.42 | 0.21 |
| biosemi128 | GPT-2 embedding: E − S | +0.0001 | 6/10 | +0.02 | 0.96 |
| biosemi128 | all GPT-2: E − B | -0.0005 | 5/10 | -0.12 | 0.72 |
| cyton8 | onsets: B − A | +0.0005 | 6/10 | +0.17 | 0.6 |
| cyton8 | GPT-2 surprisal: S − B | -0.0004 | 4/10 | -0.39 | 0.25 |
| cyton8 | GPT-2 embedding: E − S | -0.0004 | 6/10 | -0.14 | 0.66 |
| cyton8 | all GPT-2: E − B | -0.0008 | 4/10 | -0.25 | 0.45 |
| cyton_daisy16 | onsets: B − A | +0.0005 | 5/10 | +0.17 | 0.61 |
| cyton_daisy16 | GPT-2 surprisal: S − B | -0.0005 | 3/10 | -0.46 | 0.18 |
| cyton_daisy16 | GPT-2 embedding: E − S | -0.0001 | 5/10 | -0.04 | 0.89 |
| cyton_daisy16 | all GPT-2: E − B | -0.0007 | 4/10 | -0.18 | 0.59 |
| unicorn8 | onsets: B − A | +0.0004 | 5/10 | +0.10 | 0.75 |
| unicorn8 | GPT-2 surprisal: S − B | -0.0003 | 4/10 | -0.19 | 0.56 |
| unicorn8 | GPT-2 embedding: E − S | +0.0006 | 6/10 | +0.16 | 0.63 |
| unicorn8 | all GPT-2: E − B | +0.0004 | 6/10 | +0.08 | 0.81 |
| best8_train | onsets: B − A | +0.0010 | 6/10 | +0.23 | 0.48 |
| best8_train | GPT-2 surprisal: S − B | -0.0004 | 4/10 | -0.22 | 0.5 |
| best8_train | GPT-2 embedding: E − S | -0.0014 | 4/10 | -0.35 | 0.3 |
| best8_train | all GPT-2: E − B | -0.0017 | 5/10 | -0.33 | 0.32 |
| best16_train | onsets: B − A | +0.0012 | 5/10 | +0.25 | 0.46 |
| best16_train | GPT-2 surprisal: S − B | -0.0003 | 5/10 | -0.19 | 0.55 |
| best16_train | GPT-2 embedding: E − S | -0.0015 | 4/10 | -0.37 | 0.27 |
| best16_train | all GPT-2: E − B | -0.0018 | 6/10 | -0.34 | 0.31 |

### reverse: GPT-2 against P3's stand-in word features (same folds, same subjects)

Tiers A and B must reproduce P3 exactly on the fixed montages. The best-k sets are picked by the main tier (E here, C in P3), so they hold different channels and A and B differ there.

| montage | max abs diff A vs P3 | max abs diff B vs P3 | E (GPT-2) − C (stand-in): mean Δr | subjects > 0 | dz | p |
| --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | 0.0e+00 | 0.0e+00 | -0.0007 | 2/10 | -0.35 | 0.3 |
| cyton8 | 0.0e+00 | 0.0e+00 | -0.0009 | 2/10 | -0.68 | 0.061 |
| cyton_daisy16 | 0.0e+00 | 0.0e+00 | -0.0009 | 2/10 | -0.56 | 0.11 |
| unicorn8 | 0.0e+00 | 0.0e+00 | -0.0004 | 4/10 | -0.21 | 0.52 |
| best8_train | 5.2e-03 | 5.0e-03 | -0.0016 | 5/10 | -0.30 | 0.37 |
| best16_train | 1.1e-03 | 8.2e-04 | -0.0017 | 4/10 | -0.33 | 0.32 |

## reverse: segment identification (other held-out segments as candidates)

Pooled top-k over subjects; subject-level test is per-subject top-k against chance (one-sided t). Language against acoustics is the paired per-subject difference from tier A on the same windows.

| window s | montage | tier | N | top-1 | top-10 | chance top-1 / top-10 | subjects top-10 > chance | subject-level p | top-10 − tier A, points [95% CI] | paired p | top-1 − tier A, points [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.009 | 0.090 | 0.005 / 0.055 | 10/10 | 2.6e-06 | — | — | — |
| 5 | biosemi128 | B_env_onset | 182 | 0.005 | 0.076 | 0.005 / 0.055 | 8/10 | 0.0045 | -1.4 [-2.9, +0.2] | 0.08 | -0.4 [-0.9, +0.0] |
| 5 | biosemi128 | S_env_onset_sur | 182 | 0.007 | 0.080 | 0.005 / 0.055 | 9/10 | 0.0055 | -1.0 [-2.9, +0.8] | 0.23 | -0.3 [-0.6, +0.1] |
| 5 | biosemi128 | E_env_onset_sur_emb | 182 | 0.010 | 0.077 | 0.005 / 0.055 | 9/10 | 0.0042 | -1.3 [-2.6, -0.0] | 0.049 | +0.1 [-0.4, +0.6] |
| 5 | cyton8 | A_env | 182 | 0.009 | 0.086 | 0.005 / 0.055 | 10/10 | 0.00049 | — | — | — |
| 5 | cyton8 | B_env_onset | 182 | 0.005 | 0.080 | 0.005 / 0.055 | 9/10 | 0.0024 | -0.5 [-1.8, +0.7] | 0.33 | -0.3 [-0.7, +0.0] |
| 5 | cyton8 | S_env_onset_sur | 182 | 0.006 | 0.084 | 0.005 / 0.055 | 9/10 | 0.0028 | -0.2 [-1.5, +1.2] | 0.79 | -0.3 [-0.5, -0.1] |
| 5 | cyton8 | E_env_onset_sur_emb | 182 | 0.008 | 0.086 | 0.005 / 0.055 | 9/10 | 0.00048 | +0.1 [-1.3, +1.4] | 0.93 | -0.1 [-0.5, +0.4] |
| 5 | cyton_daisy16 | A_env | 182 | 0.012 | 0.085 | 0.005 / 0.055 | 10/10 | 0.00039 | — | — | — |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.005 | 0.080 | 0.005 / 0.055 | 8/10 | 0.0062 | -0.4 [-1.7, +0.9] | 0.46 | -0.7 [-1.1, -0.3] |
| 5 | cyton_daisy16 | S_env_onset_sur | 182 | 0.007 | 0.076 | 0.005 / 0.055 | 7/10 | 0.0093 | -0.8 [-2.2, +0.5] | 0.21 | -0.5 [-0.9, -0.2] |
| 5 | cyton_daisy16 | E_env_onset_sur_emb | 182 | 0.012 | 0.082 | 0.005 / 0.055 | 7/10 | 0.01 | -0.2 [-1.7, +1.3] | 0.75 | -0.1 [-0.4, +0.3] |
| 5 | unicorn8 | A_env | 182 | 0.007 | 0.080 | 0.005 / 0.055 | 9/10 | 0.00061 | — | — | — |
| 5 | unicorn8 | B_env_onset | 182 | 0.004 | 0.070 | 0.005 / 0.055 | 8/10 | 0.028 | -1.0 [-2.4, +0.5] | 0.16 | -0.3 [-0.6, +0.0] |
| 5 | unicorn8 | S_env_onset_sur | 182 | 0.007 | 0.074 | 0.005 / 0.055 | 9/10 | 0.0047 | -0.7 [-1.8, +0.4] | 0.21 | -0.1 [-0.4, +0.3] |
| 5 | unicorn8 | E_env_onset_sur_emb | 182 | 0.005 | 0.076 | 0.005 / 0.055 | 8/10 | 0.00079 | -0.4 [-1.6, +0.9] | 0.51 | -0.2 [-0.5, +0.2] |
| 5 | best8_train | A_env | 182 | 0.012 | 0.074 | 0.005 / 0.055 | 7/10 | 0.0079 | — | — | — |
| 5 | best8_train | B_env_onset | 182 | 0.010 | 0.071 | 0.005 / 0.055 | 8/10 | 0.0022 | -0.3 [-1.2, +0.7] | 0.54 | -0.1 [-0.5, +0.3] |
| 5 | best8_train | S_env_onset_sur | 182 | 0.009 | 0.072 | 0.005 / 0.055 | 8/10 | 0.0062 | -0.2 [-1.4, +1.1] | 0.77 | -0.2 [-0.6, +0.1] |
| 5 | best8_train | E_env_onset_sur_emb | 182 | 0.010 | 0.074 | 0.005 / 0.055 | 7/10 | 0.012 | +0.1 [-1.0, +1.1] | 0.91 | -0.2 [-0.6, +0.3] |
| 5 | best16_train | A_env | 182 | 0.010 | 0.074 | 0.005 / 0.055 | 7/10 | 0.0064 | — | — | — |
| 5 | best16_train | B_env_onset | 182 | 0.011 | 0.073 | 0.005 / 0.055 | 8/10 | 0.0042 | -0.1 [-1.2, +1.0] | 0.83 | +0.1 [-0.4, +0.5] |
| 5 | best16_train | S_env_onset_sur | 182 | 0.011 | 0.072 | 0.005 / 0.055 | 8/10 | 0.0013 | -0.2 [-1.1, +0.8] | 0.71 | +0.1 [-0.2, +0.3] |
| 5 | best16_train | E_env_onset_sur_emb | 182 | 0.010 | 0.079 | 0.005 / 0.055 | 8/10 | 0.003 | +0.5 [-0.4, +1.5] | 0.24 | -0.1 [-0.3, +0.2] |
| 10 | biosemi128 | A_env | 89 | 0.021 | 0.193 | 0.011 / 0.112 | 9/10 | 0.00021 | — | — | — |
| 10 | biosemi128 | B_env_onset | 89 | 0.015 | 0.197 | 0.011 / 0.112 | 9/10 | 0.00035 | +0.3 [-1.3, +2.0] | 0.66 | -0.7 [-1.7, +0.3] |
| 10 | biosemi128 | S_env_onset_sur | 89 | 0.015 | 0.184 | 0.011 / 0.112 | 8/10 | 0.00095 | -0.9 [-2.7, +0.9] | 0.29 | -0.7 [-1.4, +0.0] |
| 10 | biosemi128 | E_env_onset_sur_emb | 89 | 0.020 | 0.192 | 0.011 / 0.112 | 9/10 | 0.00069 | -0.1 [-1.9, +1.7] | 0.89 | -0.1 [-0.7, +0.5] |
| 10 | cyton8 | A_env | 89 | 0.027 | 0.181 | 0.011 / 0.112 | 9/10 | 0.0013 | — | — | — |
| 10 | cyton8 | B_env_onset | 89 | 0.021 | 0.184 | 0.011 / 0.112 | 9/10 | 0.0019 | +0.3 [-2.3, +3.0] | 0.78 | -0.6 [-1.5, +0.4] |
| 10 | cyton8 | S_env_onset_sur | 89 | 0.020 | 0.182 | 0.011 / 0.112 | 9/10 | 0.0032 | +0.1 [-2.0, +2.2] | 0.91 | -0.7 [-1.4, +0.0] |
| 10 | cyton8 | E_env_onset_sur_emb | 89 | 0.026 | 0.181 | 0.011 / 0.112 | 9/10 | 0.0015 | +0.0 [-2.1, +2.1] | 1 | -0.1 [-1.0, +0.8] |
| 10 | cyton_daisy16 | A_env | 89 | 0.017 | 0.189 | 0.011 / 0.112 | 10/10 | 0.0014 | — | — | — |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.015 | 0.184 | 0.011 / 0.112 | 9/10 | 0.0014 | -0.4 [-2.5, +1.6] | 0.64 | -0.2 [-1.0, +0.5] |
| 10 | cyton_daisy16 | S_env_onset_sur | 89 | 0.013 | 0.182 | 0.011 / 0.112 | 9/10 | 0.0033 | -0.7 [-2.1, +0.8] | 0.31 | -0.3 [-1.1, +0.4] |
| 10 | cyton_daisy16 | E_env_onset_sur_emb | 89 | 0.022 | 0.183 | 0.011 / 0.112 | 9/10 | 0.002 | -0.6 [-2.1, +1.0] | 0.44 | +0.6 [-0.7, +1.8] |
| 10 | unicorn8 | A_env | 89 | 0.024 | 0.187 | 0.011 / 0.112 | 9/10 | 0.00017 | — | — | — |
| 10 | unicorn8 | B_env_onset | 89 | 0.016 | 0.183 | 0.011 / 0.112 | 10/10 | 0.00033 | -0.3 [-2.2, +1.5] | 0.69 | -0.8 [-1.8, +0.2] |
| 10 | unicorn8 | S_env_onset_sur | 89 | 0.020 | 0.185 | 0.011 / 0.112 | 9/10 | 0.00075 | -0.1 [-1.6, +1.4] | 0.87 | -0.3 [-0.9, +0.2] |
| 10 | unicorn8 | E_env_onset_sur_emb | 89 | 0.027 | 0.184 | 0.011 / 0.112 | 10/10 | 0.00024 | -0.2 [-2.9, +2.4] | 0.85 | +0.3 [-0.4, +1.1] |
| 10 | best8_train | A_env | 89 | 0.016 | 0.173 | 0.011 / 0.112 | 10/10 | 0.00062 | — | — | — |
| 10 | best8_train | B_env_onset | 89 | 0.016 | 0.167 | 0.011 / 0.112 | 9/10 | 0.0007 | -0.6 [-3.2, +2.1] | 0.65 | +0.0 [-1.0, +1.0] |
| 10 | best8_train | S_env_onset_sur | 89 | 0.016 | 0.172 | 0.011 / 0.112 | 10/10 | 0.00038 | -0.1 [-2.1, +1.9] | 0.9 | +0.0 [-0.4, +0.4] |
| 10 | best8_train | E_env_onset_sur_emb | 89 | 0.016 | 0.169 | 0.011 / 0.112 | 10/10 | 3.8e-05 | -0.4 [-2.5, +1.6] | 0.64 | +0.0 [-1.0, +1.0] |
| 10 | best16_train | A_env | 89 | 0.016 | 0.180 | 0.011 / 0.112 | 9/10 | 0.00046 | — | — | — |
| 10 | best16_train | B_env_onset | 89 | 0.022 | 0.162 | 0.011 / 0.112 | 9/10 | 0.003 | -1.8 [-4.3, +0.7] | 0.13 | +0.7 [-0.5, +1.9] |
| 10 | best16_train | S_env_onset_sur | 89 | 0.019 | 0.167 | 0.011 / 0.112 | 9/10 | 0.00029 | -1.2 [-2.9, +0.4] | 0.12 | +0.3 [-0.6, +1.3] |
| 10 | best16_train | E_env_onset_sur_emb | 89 | 0.019 | 0.174 | 0.011 / 0.112 | 10/10 | 0.00011 | -0.6 [-2.6, +1.5] | 0.54 | +0.3 [-0.9, +1.5] |

## reverse: true words against 88 GPT-2 distractors (same audio, same onsets)

89 windows of 10 s per subject. Chance: top-1 0.011, top-10 0.112. Envelope-only and envelope+onset models score every candidate identically here, so they sit at chance by construction. 'Shuffled EEG' scores each window's candidates against another window of the same subject (mean over derangements) and must sit at chance. 'Prior only' ranks by GPT-2 log-probability with no EEG. A ranker can beat chance without using the EEG at all (for example when distractor features are atypical), so the EEG-specific evidence is the paired per-subject excess of the real EEG over the shuffled EEG, not the distance from chance.

| montage | tier | top-1 | top-10 | subjects top-1 > chance | p | subjects top-10 > chance | p | shuffled EEG top-1 / top-10 | prior only top-1 / top-10 | real − shuffled EEG top-1, points [95% CI] | p | real − shuffled EEG top-10, points [95% CI] | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biosemi128 | S_env_onset_sur | 0.012 | 0.117 | 3/10 | 0.34 | 4/10 | 0.32 | 0.010 / 0.106 | 0.000 / 0.079 | +0.3 [-0.4, +0.9] | 0.37 | +1.1 [-1.3, +3.4] | 0.33 |
| biosemi128 | E_env_onset_sur_emb | 0.021 | 0.138 | 6/10 | 0.021 | 7/10 | 0.055 | 0.018 / 0.135 | 0.000 / 0.079 | +0.3 [-0.5, +1.2] | 0.41 | +0.3 [-2.8, +3.4] | 0.83 |
| cyton8 | S_env_onset_sur | 0.011 | 0.113 | 3/10 | 0.5 | 4/10 | 0.45 | 0.008 / 0.113 | 0.000 / 0.079 | +0.3 [-0.6, +1.2] | 0.47 | +0.1 [-2.3, +2.4] | 0.95 |
| cyton8 | E_env_onset_sur_emb | 0.020 | 0.128 | 7/10 | 0.0016 | 6/10 | 0.14 | 0.017 / 0.131 | 0.000 / 0.079 | +0.3 [-0.2, +0.8] | 0.18 | -0.3 [-3.5, +3.0] | 0.85 |
| cyton_daisy16 | S_env_onset_sur | 0.013 | 0.118 | 4/10 | 0.22 | 5/10 | 0.31 | 0.010 / 0.112 | 0.000 / 0.079 | +0.3 [-0.4, +1.1] | 0.34 | +0.6 [-1.8, +3.0] | 0.6 |
| cyton_daisy16 | E_env_onset_sur_emb | 0.015 | 0.144 | 3/10 | 0.041 | 7/10 | 0.037 | 0.019 / 0.136 | 0.000 / 0.079 | -0.4 [-1.0, +0.2] | 0.15 | +0.8 [-3.2, +4.7] | 0.67 |
| unicorn8 | S_env_onset_sur | 0.013 | 0.117 | 3/10 | 0.17 | 3/10 | 0.31 | 0.008 / 0.111 | 0.000 / 0.079 | +0.6 [+0.1, +1.0] | 0.024 | +0.6 [-1.4, +2.6] | 0.51 |
| unicorn8 | E_env_onset_sur_emb | 0.025 | 0.142 | 5/10 | 0.015 | 8/10 | 0.015 | 0.020 / 0.137 | 0.000 / 0.079 | +0.5 [-0.7, +1.7] | 0.34 | +0.5 [-2.6, +3.5] | 0.73 |
| best8_train | S_env_onset_sur | 0.016 | 0.120 | 5/10 | 0.15 | 6/10 | 0.27 | 0.010 / 0.110 | 0.000 / 0.079 | +0.6 [-0.4, +1.6] | 0.24 | +1.0 [-2.1, +4.2] | 0.48 |
| best8_train | E_env_onset_sur_emb | 0.021 | 0.147 | 4/10 | 0.041 | 8/10 | 0.014 | 0.020 / 0.131 | 0.000 / 0.079 | +0.2 [-1.2, +1.5] | 0.8 | +1.6 [-1.5, +4.6] | 0.27 |
| best16_train | S_env_onset_sur | 0.013 | 0.122 | 4/10 | 0.28 | 6/10 | 0.17 | 0.008 / 0.110 | 0.000 / 0.079 | +0.6 [-0.3, +1.4] | 0.16 | +1.2 [-1.5, +4.0] | 0.34 |
| best16_train | E_env_onset_sur_emb | 0.025 | 0.142 | 6/10 | 0.018 | 8/10 | 0.042 | 0.017 / 0.137 | 0.000 / 0.079 | +0.7 [-0.4, +1.9] | 0.18 | +0.4 [-2.9, +3.8] | 0.78 |

## reverse: starter montages against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels. Decoding differences are in points.

| montage | GPT-2 gain E − B | segment ID top-10, tier E | distractor top-1, tier E | distractor top-10, tier E |
| --- | --- | --- | --- | --- |
| cyton8 | -0.0003 [-0.0015, +0.0008] | -1.1 [-3.9, +1.7] | -0.1 [-1.0, +0.8] | -1.0 [-2.5, +0.5] |
| cyton_daisy16 | -0.0002 [-0.0007, +0.0003] | -0.9 [-3.1, +1.3] | -0.7 [-1.7, +0.3] | +0.6 [-0.9, +2.0] |

## reverse: N400 check with GPT-2 surprisal (tier E kernel, 300-500 ms, centro-parietal)

| montage | mean kernel | subjects negative | dz | p |
| --- | --- | --- | --- | --- |
| biosemi128 | +0.0001 | 5/10 | +0.04 | 0.89 |
| cyton8 | +0.0001 | 3/10 | +0.05 | 0.89 |
| cyton_daisy16 | +0.0000 | 5/10 | +0.01 | 0.99 |
| unicorn8 | +0.0001 | 4/10 | +0.06 | 0.85 |

## Forward against reversed speech (Welch t on per-subject gains)

| step | montage | forward mean | reverse mean | t | p |
| --- | --- | --- | --- | --- | --- |
| GPT-2 surprisal: S − B | biosemi128 | +0.0026 | -0.0006 | +4.58 | 9.5e-05 |
| GPT-2 embedding: E − S | biosemi128 | +0.0021 | +0.0001 | +1.23 | 0.23 |
| all GPT-2: E − B | biosemi128 | +0.0047 | -0.0005 | +2.69 | 0.013 |
| GPT-2 surprisal: S − B | cyton8 | +0.0028 | -0.0004 | +4.39 | 0.00017 |
| GPT-2 embedding: E − S | cyton8 | +0.0011 | -0.0004 | +0.98 | 0.34 |
| all GPT-2: E − B | cyton8 | +0.0039 | -0.0008 | +2.57 | 0.016 |
| GPT-2 surprisal: S − B | cyton_daisy16 | +0.0024 | -0.0005 | +4.30 | 0.0002 |
| GPT-2 embedding: E − S | cyton_daisy16 | +0.0018 | -0.0001 | +1.21 | 0.24 |
| all GPT-2: E − B | cyton_daisy16 | +0.0042 | -0.0007 | +2.55 | 0.017 |

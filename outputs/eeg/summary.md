# P3 summary tables (held-out subjects, held-out runs)

Tier L is word timing plus word features (it keeps the onset regressor), not semantics alone.

## forward: encoding, mean r over the montage's channels (mean ± SEM over 19 subjects)

| montage | A_env | B_env_onset | C_env_onset_lm | L_onset_lm |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0210 ± 0.0034 | 0.0263 ± 0.0036 | 0.0306 ± 0.0035 | 0.0234 ± 0.0028 |
| cyton8 | 0.0170 ± 0.0026 | 0.0220 ± 0.0030 | 0.0258 ± 0.0028 | 0.0209 ± 0.0027 |
| cyton_daisy16 | 0.0196 ± 0.0034 | 0.0249 ± 0.0035 | 0.0286 ± 0.0034 | 0.0213 ± 0.0027 |
| unicorn8 | 0.0202 ± 0.0028 | 0.0253 ± 0.0033 | 0.0299 ± 0.0031 | 0.0242 ± 0.0029 |
| best8_train | 0.0279 ± 0.0045 | 0.0342 ± 0.0045 | 0.0388 ± 0.0048 | 0.0265 ± 0.0031 |
| best16_train | 0.0280 ± 0.0046 | 0.0340 ± 0.0046 | 0.0390 ± 0.0048 | 0.0272 ± 0.0031 |

### forward: word-feature gain, tier C minus tier B (paired over subjects)

| montage | mean Δr | subjects with Δr > 0 | dz | paired t p |
| --- | --- | --- | --- | --- |
| biosemi128 | +0.0043 | 14/19 | +0.87 | 0.0013 |
| cyton8 | +0.0037 | 13/19 | +0.62 | 0.014 |
| cyton_daisy16 | +0.0038 | 13/19 | +0.71 | 0.0064 |
| unicorn8 | +0.0045 | 15/19 | +0.86 | 0.0014 |
| best8_train | +0.0046 | 17/19 | +1.06 | 0.00022 |
| best16_train | +0.0049 | 17/19 | +1.18 | 7.1e-05 |

### forward: N400 check, word-feature TRF kernel at 300-500 ms, centro-parietal region

| feature | montage | ROI channels | mean kernel | subjects negative | dz | t | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surprisal | biosemi128 | 13 | -0.0096 | 16/19 | -1.11 | -4.83 | 0.00013 |
| surprisal | cyton8 | 2 | -0.0085 | 16/19 | -1.14 | -4.97 | 0.0001 |
| surprisal | cyton_daisy16 | 4 | -0.0094 | 16/19 | -1.19 | -5.20 | 6e-05 |
| surprisal | unicorn8 | 4 | -0.0088 | 16/19 | -1.08 | -4.72 | 0.00017 |
| dissimilarity | biosemi128 | 13 | -0.0057 | 17/19 | -1.16 | -5.06 | 8.2e-05 |
| dissimilarity | cyton8 | 2 | -0.0043 | 18/19 | -0.87 | -3.80 | 0.0013 |
| dissimilarity | cyton_daisy16 | 4 | -0.0054 | 17/19 | -1.00 | -4.34 | 0.0004 |
| dissimilarity | unicorn8 | 4 | -0.0050 | 17/19 | -1.01 | -4.39 | 0.00036 |

## forward: segment identification

Top-1 and top-10 are pooled over subjects. The binomial p treats every window as independent, but all subjects share one candidate set and one set of predictions, so it overstates the evidence; the subject-level columns (per-subject top-10 against chance, one-sample t) are the conservative test.

| window s | montage | tier | N candidates | top-1 | chance | top-10 | chance | binomial p (top-1) | subjects top-10 > chance | subject-level p (top-10) | null top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.014 | 0.005 | 0.106 | 0.055 | 1.6e-08 | 18/19 | 5.3e-07 | 0.004 |
| 5 | biosemi128 | B_env_onset | 182 | 0.013 | 0.005 | 0.103 | 0.055 | 6.1e-07 | 19/19 | 1e-05 | 0.003 |
| 5 | biosemi128 | C_env_onset_lm | 182 | 0.021 | 0.005 | 0.119 | 0.055 | 9.7e-21 | 19/19 | 6.8e-08 | 0.006 |
| 5 | biosemi128 | L_onset_lm | 182 | 0.010 | 0.005 | 0.090 | 0.055 | 0.00031 | 18/19 | 2.3e-06 | 0.006 |
| 5 | cyton8 | A_env | 182 | 0.012 | 0.005 | 0.091 | 0.055 | 1.4e-06 | 17/19 | 2.4e-07 | 0.005 |
| 5 | cyton8 | B_env_onset | 182 | 0.010 | 0.005 | 0.097 | 0.055 | 0.0012 | 18/19 | 9.4e-07 | 0.005 |
| 5 | cyton8 | C_env_onset_lm | 182 | 0.014 | 0.005 | 0.100 | 0.055 | 5.9e-09 | 18/19 | 1.4e-06 | 0.004 |
| 5 | cyton8 | L_onset_lm | 182 | 0.010 | 0.005 | 0.085 | 0.055 | 0.0022 | 17/19 | 2.5e-05 | 0.006 |
| 5 | cyton_daisy16 | A_env | 182 | 0.013 | 0.005 | 0.109 | 0.055 | 1e-07 | 18/19 | 1.1e-06 | 0.007 |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.012 | 0.005 | 0.105 | 0.055 | 7.5e-06 | 18/19 | 1.6e-05 | 0.005 |
| 5 | cyton_daisy16 | C_env_onset_lm | 182 | 0.019 | 0.005 | 0.119 | 0.055 | 9.2e-17 | 19/19 | 1e-06 | 0.002 |
| 5 | cyton_daisy16 | L_onset_lm | 182 | 0.012 | 0.005 | 0.088 | 0.055 | 1.7e-05 | 18/19 | 1.3e-06 | 0.005 |
| 5 | unicorn8 | A_env | 182 | 0.012 | 0.005 | 0.091 | 0.055 | 1.4e-06 | 17/19 | 8e-07 | 0.005 |
| 5 | unicorn8 | B_env_onset | 182 | 0.012 | 0.005 | 0.095 | 0.055 | 7.5e-06 | 19/19 | 2.5e-06 | 0.004 |
| 5 | unicorn8 | C_env_onset_lm | 182 | 0.016 | 0.005 | 0.104 | 0.055 | 1.2e-11 | 18/19 | 1.1e-07 | 0.005 |
| 5 | unicorn8 | L_onset_lm | 182 | 0.012 | 0.005 | 0.086 | 0.055 | 1.7e-05 | 15/19 | 2.6e-05 | 0.004 |
| 5 | best8_train | A_env | 182 | 0.016 | 0.005 | 0.102 | 0.055 | 1.3e-12 | 18/19 | 1.3e-05 | 0.006 |
| 5 | best8_train | B_env_onset | 182 | 0.017 | 0.005 | 0.110 | 0.055 | 4.1e-13 | 17/19 | 3.3e-05 | 0.005 |
| 5 | best8_train | C_env_onset_lm | 182 | 0.016 | 0.005 | 0.119 | 0.055 | 3.9e-12 | 19/19 | 1.3e-06 | 0.005 |
| 5 | best8_train | L_onset_lm | 182 | 0.011 | 0.005 | 0.091 | 0.055 | 0.00016 | 17/19 | 4.4e-06 | 0.005 |
| 5 | best16_train | A_env | 182 | 0.017 | 0.005 | 0.106 | 0.055 | 1.3e-13 | 17/19 | 5.1e-05 | 0.006 |
| 5 | best16_train | B_env_onset | 182 | 0.015 | 0.005 | 0.109 | 0.055 | 2.9e-10 | 18/19 | 1.1e-05 | 0.005 |
| 5 | best16_train | C_env_onset_lm | 182 | 0.019 | 0.005 | 0.119 | 0.055 | 7.2e-18 | 19/19 | 5.2e-06 | 0.003 |
| 5 | best16_train | L_onset_lm | 182 | 0.009 | 0.005 | 0.091 | 0.055 | 0.0039 | 18/19 | 2.4e-06 | 0.005 |
| 10 | biosemi128 | A_env | 89 | 0.034 | 0.011 | 0.221 | 0.112 | 1e-12 | 17/19 | 6.3e-05 | 0.011 |
| 10 | biosemi128 | B_env_onset | 89 | 0.032 | 0.011 | 0.224 | 0.112 | 2.9e-11 | 18/19 | 9e-05 | 0.013 |
| 10 | biosemi128 | C_env_onset_lm | 89 | 0.038 | 0.011 | 0.253 | 0.112 | 2.4e-16 | 19/19 | 3.3e-06 | 0.008 |
| 10 | biosemi128 | L_onset_lm | 89 | 0.024 | 0.011 | 0.190 | 0.112 | 1.6e-05 | 16/19 | 3.2e-05 | 0.012 |
| 10 | cyton8 | A_env | 89 | 0.028 | 0.011 | 0.192 | 0.112 | 3.6e-08 | 17/19 | 7.9e-05 | 0.015 |
| 10 | cyton8 | B_env_onset | 89 | 0.030 | 0.011 | 0.193 | 0.112 | 6.9e-10 | 18/19 | 1.8e-05 | 0.011 |
| 10 | cyton8 | C_env_onset_lm | 89 | 0.030 | 0.011 | 0.229 | 0.112 | 6.9e-10 | 19/19 | 2.8e-06 | 0.013 |
| 10 | cyton8 | L_onset_lm | 89 | 0.022 | 0.011 | 0.177 | 0.112 | 0.00015 | 16/19 | 0.00031 | 0.013 |
| 10 | cyton_daisy16 | A_env | 89 | 0.036 | 0.011 | 0.228 | 0.112 | 9.5e-15 | 17/19 | 0.00019 | 0.009 |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.036 | 0.011 | 0.222 | 0.112 | 9.5e-15 | 18/19 | 0.00016 | 0.011 |
| 10 | cyton_daisy16 | C_env_onset_lm | 89 | 0.036 | 0.011 | 0.257 | 0.112 | 9.5e-15 | 18/19 | 4.2e-06 | 0.014 |
| 10 | cyton_daisy16 | L_onset_lm | 89 | 0.023 | 0.011 | 0.182 | 0.112 | 3.4e-05 | 17/19 | 2.3e-05 | 0.011 |
| 10 | unicorn8 | A_env | 89 | 0.027 | 0.011 | 0.183 | 0.112 | 2.3e-07 | 17/19 | 1.2e-05 | 0.009 |
| 10 | unicorn8 | B_env_onset | 89 | 0.027 | 0.011 | 0.203 | 0.112 | 9.1e-08 | 19/19 | 1.8e-05 | 0.009 |
| 10 | unicorn8 | C_env_onset_lm | 89 | 0.032 | 0.011 | 0.241 | 0.112 | 2.9e-11 | 19/19 | 1.4e-06 | 0.009 |
| 10 | unicorn8 | L_onset_lm | 89 | 0.021 | 0.011 | 0.185 | 0.112 | 0.0003 | 18/19 | 2.6e-06 | 0.009 |
| 10 | best8_train | A_env | 89 | 0.043 | 0.011 | 0.219 | 0.112 | 6.3e-21 | 16/19 | 7.3e-05 | 0.012 |
| 10 | best8_train | B_env_onset | 89 | 0.037 | 0.011 | 0.235 | 0.112 | 8.2e-16 | 18/19 | 2.5e-05 | 0.010 |
| 10 | best8_train | C_env_onset_lm | 89 | 0.046 | 0.011 | 0.263 | 0.112 | 1.3e-24 | 19/19 | 3.3e-06 | 0.011 |
| 10 | best8_train | L_onset_lm | 89 | 0.025 | 0.011 | 0.185 | 0.112 | 1.3e-06 | 18/19 | 1.2e-05 | 0.009 |
| 10 | best16_train | A_env | 89 | 0.041 | 0.011 | 0.231 | 0.112 | 3.6e-19 | 16/19 | 3.1e-05 | 0.007 |
| 10 | best16_train | B_env_onset | 89 | 0.034 | 0.011 | 0.243 | 0.112 | 3.3e-13 | 18/19 | 1.6e-05 | 0.010 |
| 10 | best16_train | C_env_onset_lm | 89 | 0.046 | 0.011 | 0.267 | 0.112 | 5.4e-24 | 19/19 | 6.7e-06 | 0.011 |
| 10 | best16_train | L_onset_lm | 89 | 0.027 | 0.011 | 0.185 | 0.112 | 9.1e-08 | 17/19 | 1.8e-05 | 0.011 |

### forward: each montage against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels.

| montage | word-feature gain (C−B) difference | tier C 10 s top-10 difference |
| --- | --- | --- |
| cyton8 | -0.0006 [-0.0014, +0.0003] | -0.0237 [-0.0422, -0.0051] |
| cyton_daisy16 | -0.0005 [-0.0010, -0.0001] | +0.0035 [-0.0070, +0.0141] |
| unicorn8 | +0.0002 [-0.0004, +0.0009] | -0.0124 [-0.0357, +0.0109] |
| best8_train | +0.0003 [-0.0013, +0.0018] | +0.0101 [-0.0101, +0.0302] |
| best16_train | +0.0006 [-0.0007, +0.0020] | +0.0136 [-0.0056, +0.0328] |

## reverse: encoding, mean r over the montage's channels (mean ± SEM over 10 subjects)

| montage | A_env | B_env_onset | C_env_onset_lm | L_onset_lm |
| --- | --- | --- | --- | --- |
| biosemi128 | 0.0205 ± 0.0027 | 0.0212 ± 0.0031 | 0.0214 ± 0.0023 | 0.0074 ± 0.0025 |
| cyton8 | 0.0186 ± 0.0031 | 0.0190 ± 0.0033 | 0.0191 ± 0.0026 | 0.0066 ± 0.0023 |
| cyton_daisy16 | 0.0200 ± 0.0032 | 0.0205 ± 0.0034 | 0.0207 ± 0.0027 | 0.0067 ± 0.0024 |
| unicorn8 | 0.0192 ± 0.0026 | 0.0196 ± 0.0031 | 0.0203 ± 0.0023 | 0.0079 ± 0.0027 |
| best8_train | 0.0257 ± 0.0035 | 0.0268 ± 0.0035 | 0.0256 ± 0.0028 | 0.0111 ± 0.0036 |
| best16_train | 0.0254 ± 0.0034 | 0.0266 ± 0.0034 | 0.0265 ± 0.0026 | 0.0111 ± 0.0034 |

### reverse: word-feature gain, tier C minus tier B (paired over subjects)

| montage | mean Δr | subjects with Δr > 0 | dz | paired t p |
| --- | --- | --- | --- | --- |
| biosemi128 | +0.0002 | 7/10 | +0.05 | 0.88 |
| cyton8 | +0.0001 | 6/10 | +0.02 | 0.95 |
| cyton_daisy16 | +0.0002 | 7/10 | +0.06 | 0.86 |
| unicorn8 | +0.0007 | 7/10 | +0.16 | 0.62 |
| best8_train | -0.0012 | 3/10 | -0.18 | 0.58 |
| best16_train | -0.0001 | 3/10 | -0.03 | 0.94 |

### reverse: N400 check, word-feature TRF kernel at 300-500 ms, centro-parietal region

| feature | montage | ROI channels | mean kernel | subjects negative | dz | t | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surprisal | biosemi128 | 13 | +0.0005 | 3/10 | +0.52 | +1.66 | 0.13 |
| surprisal | cyton8 | 2 | +0.0004 | 2/10 | +0.45 | +1.41 | 0.19 |
| surprisal | cyton_daisy16 | 4 | +0.0005 | 4/10 | +0.54 | +1.72 | 0.12 |
| surprisal | unicorn8 | 4 | +0.0004 | 2/10 | +0.41 | +1.29 | 0.23 |
| dissimilarity | biosemi128 | 13 | +0.0002 | 6/10 | +0.11 | +0.36 | 0.73 |
| dissimilarity | cyton8 | 2 | -0.0002 | 6/10 | -0.08 | -0.26 | 0.8 |
| dissimilarity | cyton_daisy16 | 4 | -0.0002 | 6/10 | -0.11 | -0.35 | 0.74 |
| dissimilarity | unicorn8 | 4 | +0.0000 | 6/10 | +0.03 | +0.09 | 0.93 |

## reverse: segment identification

Top-1 and top-10 are pooled over subjects. The binomial p treats every window as independent, but all subjects share one candidate set and one set of predictions, so it overstates the evidence; the subject-level columns (per-subject top-10 against chance, one-sample t) are the conservative test.

| window s | montage | tier | N candidates | top-1 | chance | top-10 | chance | binomial p (top-1) | subjects top-10 > chance | subject-level p (top-10) | null top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | biosemi128 | A_env | 182 | 0.009 | 0.005 | 0.090 | 0.055 | 0.027 | 10/10 | 2.6e-06 | 0.005 |
| 5 | biosemi128 | B_env_onset | 182 | 0.005 | 0.005 | 0.076 | 0.055 | 0.67 | 8/10 | 0.0045 | 0.006 |
| 5 | biosemi128 | C_env_onset_lm | 182 | 0.010 | 0.005 | 0.087 | 0.055 | 0.014 | 10/10 | 0.0002 | 0.006 |
| 5 | biosemi128 | L_onset_lm | 182 | 0.005 | 0.005 | 0.062 | 0.055 | 0.54 | 5/10 | 0.23 | 0.008 |
| 5 | cyton8 | A_env | 182 | 0.009 | 0.005 | 0.086 | 0.055 | 0.048 | 10/10 | 0.00049 | 0.005 |
| 5 | cyton8 | B_env_onset | 182 | 0.005 | 0.005 | 0.080 | 0.055 | 0.54 | 9/10 | 0.0024 | 0.004 |
| 5 | cyton8 | C_env_onset_lm | 182 | 0.010 | 0.005 | 0.085 | 0.055 | 0.007 | 8/10 | 0.0045 | 0.004 |
| 5 | cyton8 | L_onset_lm | 182 | 0.007 | 0.005 | 0.052 | 0.055 | 0.21 | 4/10 | 0.73 | 0.004 |
| 5 | cyton_daisy16 | A_env | 182 | 0.012 | 0.005 | 0.085 | 0.055 | 0.00067 | 10/10 | 0.00039 | 0.005 |
| 5 | cyton_daisy16 | B_env_onset | 182 | 0.005 | 0.005 | 0.080 | 0.055 | 0.67 | 8/10 | 0.0062 | 0.003 |
| 5 | cyton_daisy16 | C_env_onset_lm | 182 | 0.010 | 0.005 | 0.086 | 0.055 | 0.014 | 9/10 | 0.0012 | 0.004 |
| 5 | cyton_daisy16 | L_onset_lm | 182 | 0.007 | 0.005 | 0.063 | 0.055 | 0.21 | 5/10 | 0.18 | 0.006 |
| 5 | unicorn8 | A_env | 182 | 0.007 | 0.005 | 0.080 | 0.055 | 0.21 | 9/10 | 0.00061 | 0.004 |
| 5 | unicorn8 | B_env_onset | 182 | 0.004 | 0.005 | 0.070 | 0.055 | 0.78 | 8/10 | 0.028 | 0.006 |
| 5 | unicorn8 | C_env_onset_lm | 182 | 0.009 | 0.005 | 0.089 | 0.055 | 0.048 | 10/10 | 0.00033 | 0.003 |
| 5 | unicorn8 | L_onset_lm | 182 | 0.006 | 0.005 | 0.061 | 0.055 | 0.42 | 4/10 | 0.25 | 0.003 |
| 5 | best8_train | A_env | 182 | 0.012 | 0.005 | 0.077 | 0.055 | 0.0015 | 7/10 | 0.0073 | 0.005 |
| 5 | best8_train | B_env_onset | 182 | 0.012 | 0.005 | 0.069 | 0.055 | 0.0015 | 8/10 | 0.0077 | 0.006 |
| 5 | best8_train | C_env_onset_lm | 182 | 0.007 | 0.005 | 0.077 | 0.055 | 0.21 | 8/10 | 0.0066 | 0.007 |
| 5 | best8_train | L_onset_lm | 182 | 0.005 | 0.005 | 0.062 | 0.055 | 0.54 | 6/10 | 0.13 | 0.005 |
| 5 | best16_train | A_env | 182 | 0.012 | 0.005 | 0.074 | 0.055 | 0.0015 | 7/10 | 0.0061 | 0.008 |
| 5 | best16_train | B_env_onset | 182 | 0.012 | 0.005 | 0.071 | 0.055 | 0.0015 | 8/10 | 0.0077 | 0.007 |
| 5 | best16_train | C_env_onset_lm | 182 | 0.006 | 0.005 | 0.082 | 0.055 | 0.42 | 8/10 | 0.00089 | 0.005 |
| 5 | best16_train | L_onset_lm | 182 | 0.004 | 0.005 | 0.059 | 0.055 | 0.78 | 6/10 | 0.29 | 0.007 |
| 10 | biosemi128 | A_env | 89 | 0.021 | 0.011 | 0.193 | 0.112 | 0.0069 | 9/10 | 0.00021 | 0.012 |
| 10 | biosemi128 | B_env_onset | 89 | 0.015 | 0.011 | 0.197 | 0.112 | 0.21 | 9/10 | 0.00035 | 0.018 |
| 10 | biosemi128 | C_env_onset_lm | 89 | 0.022 | 0.011 | 0.194 | 0.112 | 0.0033 | 9/10 | 0.00059 | 0.009 |
| 10 | biosemi128 | L_onset_lm | 89 | 0.010 | 0.011 | 0.144 | 0.112 | 0.67 | 7/10 | 0.059 | 0.012 |
| 10 | cyton8 | A_env | 89 | 0.027 | 0.011 | 0.181 | 0.112 | 0.00011 | 9/10 | 0.0013 | 0.012 |
| 10 | cyton8 | B_env_onset | 89 | 0.021 | 0.011 | 0.184 | 0.112 | 0.0069 | 9/10 | 0.0019 | 0.013 |
| 10 | cyton8 | C_env_onset_lm | 89 | 0.022 | 0.011 | 0.185 | 0.112 | 0.0033 | 9/10 | 0.00054 | 0.012 |
| 10 | cyton8 | L_onset_lm | 89 | 0.011 | 0.011 | 0.133 | 0.112 | 0.54 | 5/10 | 0.13 | 0.006 |
| 10 | cyton_daisy16 | A_env | 89 | 0.017 | 0.011 | 0.189 | 0.112 | 0.082 | 10/10 | 0.0014 | 0.011 |
| 10 | cyton_daisy16 | B_env_onset | 89 | 0.015 | 0.011 | 0.184 | 0.112 | 0.21 | 9/10 | 0.0014 | 0.015 |
| 10 | cyton_daisy16 | C_env_onset_lm | 89 | 0.021 | 0.011 | 0.181 | 0.112 | 0.0069 | 9/10 | 0.0014 | 0.010 |
| 10 | cyton_daisy16 | L_onset_lm | 89 | 0.012 | 0.011 | 0.134 | 0.112 | 0.42 | 7/10 | 0.08 | 0.016 |
| 10 | unicorn8 | A_env | 89 | 0.024 | 0.011 | 0.187 | 0.112 | 0.0015 | 9/10 | 0.00017 | 0.009 |
| 10 | unicorn8 | B_env_onset | 89 | 0.016 | 0.011 | 0.183 | 0.112 | 0.13 | 10/10 | 0.00033 | 0.015 |
| 10 | unicorn8 | C_env_onset_lm | 89 | 0.026 | 0.011 | 0.182 | 0.112 | 0.00027 | 10/10 | 0.00025 | 0.006 |
| 10 | unicorn8 | L_onset_lm | 89 | 0.011 | 0.011 | 0.126 | 0.112 | 0.54 | 4/10 | 0.2 | 0.011 |
| 10 | best8_train | A_env | 89 | 0.016 | 0.011 | 0.180 | 0.112 | 0.13 | 10/10 | 0.00046 | 0.009 |
| 10 | best8_train | B_env_onset | 89 | 0.017 | 0.011 | 0.161 | 0.112 | 0.082 | 7/10 | 0.0037 | 0.011 |
| 10 | best8_train | C_env_onset_lm | 89 | 0.016 | 0.011 | 0.173 | 0.112 | 0.13 | 10/10 | 0.00039 | 0.009 |
| 10 | best8_train | L_onset_lm | 89 | 0.008 | 0.011 | 0.125 | 0.112 | 0.87 | 6/10 | 0.24 | 0.019 |
| 10 | best16_train | A_env | 89 | 0.017 | 0.011 | 0.178 | 0.112 | 0.082 | 9/10 | 0.00065 | 0.012 |
| 10 | best16_train | B_env_onset | 89 | 0.020 | 0.011 | 0.164 | 0.112 | 0.014 | 8/10 | 0.0028 | 0.006 |
| 10 | best16_train | C_env_onset_lm | 89 | 0.017 | 0.011 | 0.167 | 0.112 | 0.082 | 9/10 | 0.00051 | 0.017 |
| 10 | best16_train | L_onset_lm | 89 | 0.007 | 0.011 | 0.131 | 0.112 | 0.93 | 6/10 | 0.12 | 0.007 |

### reverse: each montage against 128 channels (paired over subjects, mean [95% CI])

Negative means the montage loses against 128 channels.

| montage | word-feature gain (C−B) difference | tier C 10 s top-10 difference |
| --- | --- | --- |
| cyton8 | -0.0001 [-0.0009, +0.0007] | -0.0090 [-0.0214, +0.0035] |
| cyton_daisy16 | +0.0000 [-0.0004, +0.0004] | -0.0135 [-0.0259, -0.0010] |
| unicorn8 | +0.0005 [-0.0005, +0.0016] | -0.0124 [-0.0299, +0.0052] |
| best8_train | -0.0014 [-0.0045, +0.0017] | -0.0213 [-0.0582, +0.0155] |
| best16_train | -0.0003 [-0.0026, +0.0020] | -0.0270 [-0.0655, +0.0115] |

## Forward against reversed speech (Welch t-test on per-subject values)

| measure | montage | forward mean | reverse mean | t | p |
| --- | --- | --- | --- | --- | --- |
| word-feature gain C−B | biosemi128 | +0.0043 | +0.0002 | +2.39 | 0.026 |
| word-feature gain C−B | cyton8 | +0.0037 | +0.0001 | +2.08 | 0.047 |
| word-feature gain C−B | cyton_daisy16 | +0.0038 | +0.0002 | +2.08 | 0.048 |
| word-feature gain C−B | unicorn8 | +0.0045 | +0.0007 | +2.02 | 0.056 |
| word-feature gain C−B | best8_train | +0.0046 | -0.0012 | +2.51 | 0.026 |
| word-feature gain C−B | best16_train | +0.0049 | -0.0001 | +2.59 | 0.021 |
| N400 kernel, surprisal | biosemi128 | -0.0096 | +0.0005 | -5.01 | 8.1e-05 |
| N400 kernel, surprisal | cyton8 | -0.0085 | +0.0004 | -5.14 | 5.7e-05 |
| N400 kernel, surprisal | cyton_daisy16 | -0.0094 | +0.0005 | -5.40 | 3.4e-05 |
| N400 kernel, surprisal | unicorn8 | -0.0088 | +0.0004 | -4.87 | 0.00011 |
| N400 kernel, dissimilarity | biosemi128 | -0.0057 | +0.0002 | -4.85 | 7e-05 |
| N400 kernel, dissimilarity | cyton8 | -0.0043 | -0.0002 | -3.25 | 0.0033 |
| N400 kernel, dissimilarity | cyton_daisy16 | -0.0054 | -0.0002 | -3.75 | 0.00096 |
| N400 kernel, dissimilarity | unicorn8 | -0.0050 | +0.0000 | -4.06 | 0.00046 |

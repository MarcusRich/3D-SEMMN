"""EEG response-surface modelling on public natural-speech data (P3).

Modules:
- ``broderick``: loader and preprocessing for the Broderick et al. (2018) Dryad dataset.
- ``montages``: BioSemi-128 to 10-20 mapping and the starter-hardware montages.
- ``features``: word-level stimulus features as impulse trains, and the time-lagged design.
- ``lm``: language-model word features (GPT-2 when available).
- ``trf``: ridge temporal response functions fitted from per-run sufficient statistics,
  with leave-subjects-out cross-validation.
- ``decode``: rank candidate script segments by predicted-vs-observed response correlation.
"""

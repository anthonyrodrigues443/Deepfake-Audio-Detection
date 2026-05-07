"""Source-of-truth for the Phase 4 main notebook (Optuna + stacking + error analysis).

Phase 3 result: XGBoost + per-sample combo augmentation hit 36.0% Hemg EER (down from 48% in
Phase 2). Anti-predictive -> meaningfully predictive on a held-out distribution. Below the 25%
target, but the right direction.

Phase 4 question: can we push Hemg EER below 25%? Three angles:

  4.1  Reproduce the Phase 3 winner on a tighter protocol (held-out Hemg val + Hemg test).
  4.2  Optuna tuning of XGBoost + combo aug on Hemg val EER (50 trials, TPE, ASHA pruner).
       Reports tuned XGBoost on Hemg test (untouched during search).
  4.3  Stack tuned XGBoost (handcrafted features) with Wav2Vec2 + LogReg (deep features)
       via simple-average and LogReg-meta-learned probabilities.
  4.4  Per-clip Hemg error analysis: which 36% does the Phase 3 winner miss, and does the
       stacked model save them? Look at amplitude / duration / F0 of misses vs hits.
  4.5  Confusion matrix + DET curve at the operating points that pass each model's Hemg val
       criterion.
  4.6  Final leaderboard. LLM head-to-head deferred to Phase 5 per roadmap.

Notebook 1 of 2. Notebook 2 (`phase4_w2v2_extract.ipynb`) extracts W2V2 features in a fresh
kernel and saves them to `results/phase4_w2v2_*.npy` for this notebook to read — that fix is
the thing that unblocks experiments 3.4 / 3.5 from Phase 3.
"""

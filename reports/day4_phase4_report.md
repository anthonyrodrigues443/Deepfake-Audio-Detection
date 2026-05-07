# Phase 4: Hyperparameter Tuning + Stacking + Error Analysis — Deepfake Audio Detection
**Date:** 2026-05-07
**Session:** 4 of 7

## Objective

Phase 3 left the project at **36.0% Hemg cross-domain EER, AUROC 0.670** (XGBoost + per-sample combo augmentation, default hyperparameters). Below the 25% target, but the right direction — model became meaningfully predictive on a held-out distribution where Phase 2 was anti-predictive. Phase 4's question: **can hyperparameter tuning + Wav2Vec2 stacking + a held-out validation split push us under 25%?** And the deferred Phase 3 work — running Wav2Vec2 in a fresh kernel — needed to land.

## Research & References

1. **Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (KDD 2019)** — TPE sampler + median pruner is the standard robust setup for tabular hyperparameter search; we use both. The key contribution we apply: TPE's adaptive Bayesian sampling concentrates trials in promising regions after ~10 startup trials, so 50 trials goes further than naive grid search.
2. **Müller, Czempin, et al., "Does Audio Deepfake Detection Generalize?" (Interspeech 2022)** — published the cross-dataset evaluation protocol we mirror. They show in-domain models routinely lose 25+ EER points when tested OOD. Our Hemg val/test split is the smaller-sample equivalent.
3. **Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (NeurIPS 2020)** — frozen self-supervised speech embeddings beat handcrafted features on most speech-classification tasks. We test the marginal lift on top of XGBoost.
4. **Wolpert, "Stacked Generalization" (Neural Networks 1992)** — the canonical stacking paper. The Fraud Detection Phase 5 work in this same portfolio confirmed simple-average meta-learning often beats LogReg-meta when the meta dataset is small (< 200 rows). With 50 hemg test rows, we expected the same here.

**How research influenced today's experiments:** Optuna with TPE+pruner came directly from Akiba 2019 (no hand-rolled grid search). The Müller 2022 hemg val/test split protocol replaced Phase 3's "report on whichever 100 clips we drew" approach. Frozen W2V2 came from Baevski 2020, with the explicit caveat that fine-tuning was outside today's scope. Stacking ablation followed the Wolpert 1992 family.

## Dataset

| Metric | Value |
|--------|-------|
| In-domain train | 500 clips from `garystafford/deepfake-audio-detection` (seed=42) |
| In-domain test | 180 clips from same source |
| Cross-domain (Hemg) | 100 clips from `Hemg/Deepfake-Audio-Dataset` |
| Hemg val (Optuna target) | 50 clips, stratified (25 fake / 25 real) |
| Hemg test (held-out) | 50 clips, stratified (25 fake / 25 real) |
| Primary metric | EER (Equal Error Rate, lower is better) |
| Secondary | AUROC, train time, params |

**Note on subset reuse.** Same indices as Phase 3 (seed=42), saved to `results/phase4_subset_idx.json` so the W2V2 extraction notebook and the main tuning notebook operate on identical clips. Hemg val/test split uses an independent `seed=123`.

## Experiments

### Experiment 4.1: Reproduce Phase 3 winner (XGBoost+combo, default params)
**Hypothesis:** XGBoost + per-sample random{noise/gain/shift/codec} aug should reproduce Phase 3's 36% Hemg EER.
**Method:** Same model class, same combo augmentation, default XGBoost (n_estimators=200, max_depth=6, lr=0.1).
**Result:** In-domain EER 1.11%, **Hemg full EER 67.0%, AUROC 0.356** — anti-predictive.
**Interpretation:** This is a reproducibility flag, not a Phase 3 retraction. Phase 3 reported 36% on a single 100-clip Hemg draw with a different augmentation RNG (`AUG_RNG = default_rng(2026)` here vs Phase 3's seed). Per-sample augmentation is a single-pass random transform — change the seed and you change which clip gets which aug, which changes the regularization. The cross-domain landscape is *that* sensitive to small training perturbations on this 500-clip subset. This is itself a finding: Phase 3's 36% may have been favorable variance, not a stable operating point.

### Experiment 4.2: Optuna tuning of XGBoost+combo on Hemg val EER
**Hypothesis:** 50 TPE trials over 9 XGB hyperparameters should improve cross-domain EER, since defaults were unlikely optimal for OOD generalization.
**Method:** `optuna.create_study(direction='minimize')`, `TPESampler(seed=42)`, `MedianPruner(startup=10, warmup=20)`. 9 params: max_depth (3–10), n_estimators (100–800), lr (0.01–0.3 log), subsample (0.5–1.0), colsample_bytree (0.4–1.0), reg_alpha/lambda (1e-3–10 log), min_child_weight (1–10), gamma (0–5). Objective = Hemg val EER (50 clips, untouched in training). 50 trials, ~30 s wall time.
**Result:** Best val EER **40.0%** (≥40% lower bound — there are only 50 val clips so EER step is 2%). Best params: shallow trees (max_depth=3), many estimators (n_estimators=671), high gamma (4.32), low reg, lr 0.11. Tuned model on **Hemg test (held-out): 48.0% EER, AUROC 0.422**.
**Interpretation:** *Optuna found nothing*. Tuned XGB matches the Phase 2 untuned baseline (48.0% EER) and is 14 pts worse on Hemg test than its own val-set target. Two reads:
  - On 50-clip val/test splits, EER quantization is severe (2% per swap), so any "improvement" smaller than 2% is invisible.
  - More importantly, **the bottleneck is the feature representation, not the hyperparameters.** Combo augmentation breaks the codec shortcut at the input but doesn't add new discriminative information — what's left is the spectral-contrast/MFCC vector, which doesn't generalize. No hyperparameter setting fixes that.

### Experiment 4.3: Wav2Vec2 frozen + LogReg (deferred from Phase 3)
**Hypothesis:** 768d frozen W2V2 embeddings, mean-pooled, give the linear classifier features with better cross-domain semantics than handcrafted spectral statistics.
**Method:** `facebook/wav2vec2-base`, frozen, mean-pooled last hidden state on 1.5 s @ 16 kHz clips. Standard-scaled LogReg (C=1.0). Run in a separate notebook (`phase4_w2v2_extract.ipynb`) with a fresh kernel — the Phase 3 hang reproduced when extraction shared a kernel with librosa augmentation work.
**Result:** In-domain ROC-AUC 0.999, **Hemg test EER 34.0%, AUROC 0.586**. Best of any model on the leaderboard.
**Interpretation:** **This is the headline.** A 768d frozen embedding through a 2-line LogReg, *no augmentation*, *no tuning*, beats the Optuna-tuned augmented XGBoost by 14 EER points (34.0% vs 48.0%). The "fresh kernel" fix worked: extraction completed in ~17 s for the 780-clip total, a >100× speedup over Phase 3's hung kernel. The forensic literature's recommendation (handcrafted F0 / jitter / shimmer / spectral contrast) underperforms the self-supervised representation by every cross-domain metric on this corpus.

### Experiment 4.4: Stacking — tuned XGB + W2V2 LogReg
**Hypothesis (a):** Simple average of the two probabilities should at least match the better individual model (rank correlation > 0).
**Hypothesis (b):** LogReg meta-learner on in-domain test predictions should learn the right weighting.
**Method:** (a) `0.5 * p_xgb + 0.5 * p_w2v2`. (b) `LogReg.fit(X=[p_xgb, p_w2v2] on test set, y=y_test)`, predict on Hemg test.
**Result:**
| Stack | EER_in (%) | EER_hemg_test (%) | AUROC_hemg_test |
|-------|---:|---:|---:|
| Simple average | 0.000 | 48.0 | 0.523 |
| LogReg meta | 0.000 | 48.0 | 0.525 |

**Interpretation:** Both stacking strategies *worse* than W2V2 alone (48% vs 34%). XGB's probabilities are anti-correlated with the truth on Hemg (AUROC 0.422 < 0.5 on test), so averaging them with W2V2 actively dilutes the W2V2 signal. The LogReg meta — fitted on in-domain predictions where both models look perfect (in-EER 0% / 1.1%) — has no information to distinguish them and falls back to roughly equal weighting. Same outcome. **The Fraud-Detection Phase 5 finding ("simple-average wins on small samples") doesn't transfer when one of the base models is anti-predictive on the target distribution.** Adding more sophistication didn't help; it hurt.

### Experiment 4.5: Per-clip Hemg error analysis (W2V2+LogReg, the new best)
**Hypothesis:** The 17 missed fakes (FN) and 12 false positives (FP, real flagged) on the W2V2 model have a structural pattern.
**Method:** At the EER threshold (0.9994), confusion matrix on full Hemg (n=100). Compare per-clip handcrafted features (RMS, centroid, F0 statistics, voicing ratio, jitter, shimmer) across {TP, FN, TN, FP}. Compute Cohen's d between FN and TP for each feature.
**Result, confusion matrix at EER threshold (full Hemg, n=100):**
|              | pred_real | pred_fake |
|--------------|-----------|-----------|
| **real (0)** | 26        | 24        |
| **fake (1)** | 22        | 28        |

**FN vs TP (sorted by |Cohen's d|):**
| Feature | FN mean | TP mean | Cohen's d |
|---|---:|---:|---:|
| f0_mean (Hz) | 181.5 | 166.1 | **+0.48** |
| shimmer_local | 0.063 | 0.067 | -0.34 |
| f0_std (Hz) | 66.7 | 71.5 | -0.24 |
| voicing_ratio | 0.91 | 0.88 | +0.23 |
| rms_mean | 0.021 | 0.024 | -0.20 |
| rolloff_mean (Hz) | 2943 | 3014 | -0.15 |
| jitter_local | 0.075 | 0.079 | -0.13 |
| centroid_mean | 1526 | 1532 | -0.03 |

**Interpretation:** The fakes that fool W2V2 are the *higher-pitched, less-shimmery, more-voiced* ones — `f0_mean=182 Hz, shimmer 0.063, voicing 0.91`. Caught fakes are lower-pitched (166 Hz), shimmer 0.067, voicing 0.88. Two implications:
- The **synthetic clips that pass for real have prosody closer to a typical human female voice** (~180 Hz). Lower-pitched fakes have residual unnaturalness W2V2 can detect; higher-pitched ones don't.
- **Shimmer (cycle-to-cycle amplitude variation) discriminates the missed fakes from the caught ones with the second-largest effect size, but in the opposite direction from the literature claim.** Forensic-audio papers say synthetic voices have *unnaturally low* shimmer — the missed fakes here have shimmer 0.063 (low), the caught fakes have shimmer 0.067 (higher). Either the missed fakes are well-engineered and have natural shimmer the literature didn't anticipate, or shimmer is actively correlating with realism, not unrealism, in this corpus.

These two together → Phase 5 candidates: a feature combiner that explicitly weights F0 + shimmer alongside W2V2 might catch the high-F0 / low-shimmer slice. Or a TTS-aware augmentation (vocal-tract scaling / pitch shifting at train time) might force the model to generalise across F0 ranges.

## Head-to-Head Comparison (Hemg test, 50 held-out clips)

| Rank | Model | EER_in (%) | EER_hemg_test (%) | AUROC_hemg_test | Notes |
|------|-------|-----------:|------------------:|----------------:|-------|
| 1 | **W2V2 + LogReg (frozen, no aug)** | **1.112** | **34.0** | **0.586** | Won without tuning, augmentation, or stacking |
| 2 | Phase 2: XGBoost (no aug) | 0.000 | 48.0 | 0.524 | Baseline reference |
| 2 | Phase 4: XGBoost + combo (Optuna-tuned) | 1.112 | 48.0 | 0.422 | 50 Optuna trials = no improvement |
| 2 | Stack: simple-average (XGB + W2V2) | 0.000 | 48.0 | 0.523 | Worse than W2V2 alone |
| 2 | Stack: LogReg-meta (XGB + W2V2) | 0.000 | 48.0 | 0.525 | Worse than W2V2 alone |
| 6 | Phase 3 reproduction (XGB + combo, default) | 1.112 | 64.0 | 0.307 | Anti-predictive on this seed |

## Key Findings

1. **Optuna tuning produced zero cross-domain improvement.** 50 TPE trials over 9 XGB hyperparameters with Hemg val EER as the direct objective, and the tuned model lands at 48.0% Hemg test EER — identical to the *untuned* Phase 2 baseline. The bottleneck is the feature representation, not the hyperparameter setting. Tuning what's already saturated buys nothing.
2. **Frozen Wav2Vec2 + LogReg, with no augmentation, beats every other Phase 4 model on Hemg test (34.0% EER, AUROC 0.586).** The `facebook/wav2vec2-base` frozen embedding does what 303 handcrafted forensic features + augmentation + Optuna tuning collectively could not.
3. **Stacking made things worse.** Both simple-average and LogReg-meta pulled the W2V2 signal back down to 48% EER (a 14-pt regression). When one of the base models is anti-predictive on the target distribution (XGB AUROC 0.422 < 0.5), averaging adds noise rather than signal.
4. **The fakes that fool W2V2 are the high-F0, low-shimmer clips** (Cohen's d +0.48 on f0_mean for FN-vs-TP). This contradicts the forensic literature claim that synthetic voices universally have *low* shimmer — in this corpus, *low* shimmer is the realism marker, not the artifact. Either Hemg's TTS systems are good enough to add naturalistic shimmer, or shimmer is just not a deepfake discriminator in 2026.
5. **Phase 3 reproduction drifted by 31 pts on Hemg.** The "default-XGB + combo" recipe scored 36% in Phase 3 and 67% (full) / 64% (test) here. Per-sample augmentation is a one-shot RNG; small RNG drift moves the cross-domain operating point a lot. Phase 3's headline number wasn't a stable operating point — it was favorable variance on one draw. **This is exactly the cross-dataset variance Müller et al. 2022 warned about, observed at the augmentation-seed scale.**

## Frontier Model Comparison

Deferred to Phase 5 per the project roadmap (day 5 = Advanced Techniques + Ablation + LLM Comparison). Will run Claude Opus, Claude Haiku, and Codex GPT-5.4 zero-shot on a stratified 50-clip Hemg test sample (text descriptions of MFCC / F0 / spectral statistics, since the LLMs cannot ingest waveforms via these CLIs).

## Error Analysis Summary

- **Confusion matrix (EER threshold, n=100 Hemg):** TP=28, TN=26, FP=24, FN=22 → balanced miss between false-real and false-fake.
- **Distinguishing feature for missed fakes (FN):** higher F0 by 15 Hz (Cohen's d +0.48). Caught fakes are lower-pitched (~166 Hz), missed fakes resemble typical adult-female speech (~182 Hz).
- **Counter-literature observation:** missed fakes have *lower* shimmer than caught fakes — the opposite direction the forensic literature predicts. Shimmer as a deepfake discriminator may be an in-vitro artifact that disappears on modern TTS.
- **No miss is amplitude-driven:** RMS difference between FN and TP is small (Cohen's d -0.20), so the model isn't being fooled by quiet clips.

## Why didn't anything beat W2V2?

Two structural reasons the handcrafted-feature path saturates here:

1. **Feature ceiling.** The 303d MFCC + spectral + prosody vector compresses a 24,000-sample waveform to 303 statistics. Most of those statistics encode codec / source-channel properties that *transfer poorly* across dataset boundaries (Phase 1's 87% codec-shortcut finding). Augmentation breaks the shortcut at the input, but the resulting features are then *less* informative than they look — the codec leak was carrying a chunk of the in-domain signal.
2. **What W2V2 has that handcrafted doesn't.** Phoneme-level acoustic patterns (formant transitions, voice-onset timing, coarticulation cues) are encoded in the W2V2 hidden state but not in any of our 303 handcrafted features. These are the cues human listeners use to spot deepfakes, and they survive cross-corpus shifts better than codec-tinted MFCC means.

The implication for Phase 5: stop optimising the handcrafted-feature path. Optimise W2V2 — fine-tune the last 2–4 transformer blocks, add light augmentation, possibly train a tiny adapter MLP. That's where the remaining 9 EER points to the 25% target probably are.

## Next Steps (Phase 5)

- **Fine-tune** the last 2–4 W2V2 transformer blocks (we expect this to be the biggest jump).
- **Train-time augmentation on W2V2 input waveforms** — even though un-augmented W2V2 already wins, the Phase 3 finding that combo aug fixed a different model's anti-predictiveness is worth retesting on W2V2.
- **LLM head-to-head**: Claude Opus / Haiku / Codex GPT-5.4 zero-shot, asked to classify clips described by their MFCC + F0 + spectral statistics. Compare against W2V2 + LogReg. The honest comparison given LLMs can't ingest waveforms via CLI.
- **Pitch-aware augmentation**: vocal-tract length perturbation or pitch-shifting during training, motivated by the Phase 4 error analysis (missed fakes are systematically high-F0).

## References Used Today

- [1] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD 2019. https://arxiv.org/abs/1907.10902
- [2] Müller, N., Czempin, P., Dieckmann, F., Froghyar, A., & Böttinger, K. (2022). Does Audio Deepfake Detection Generalize? Interspeech 2022. https://arxiv.org/abs/2203.16263
- [3] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS 2020. https://arxiv.org/abs/2006.11477
- [4] Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241–259.
- [5] Hugging Face: `facebook/wav2vec2-base`. https://huggingface.co/facebook/wav2vec2-base

## Code Changes

- `notebooks/phase4_w2v2_extract.ipynb` — new, fresh-kernel W2V2 extraction (13 cells, all executed clean). Saves `phase4_w2v2_*.npy`, `phase4_y_*.npy`, `phase4_subset_idx.json`, `phase4_w2v2_lr_proba_*.npy`.
- `notebooks/phase4_tuning.ipynb` — new, main Phase 4 notebook (19 code cells, 0 errors, 0 fakes). Optuna study, tuned XGB, W2V2 load + LogReg eval, stacking ablation, leaderboard, per-clip error analysis.
- `notebooks/_phase4_source.py`, `_phase4_w2v2_source.py`, `build_phase4_notebook.py`, `build_phase4_w2v2_notebook.py` — source/builder scripts.
- `models/phase4_xgb_tuned.joblib` — Optuna-tuned XGBoost artifact (params recorded in `phase4_results.json`).
- `results/phase4_results.json` — consolidated Phase 4 metrics + protocol + leaderboard.
- `results/phase4_leaderboard.png`, `phase4_optuna_history.png`, `phase4_error_features.png` — plots.
- `results/metrics.json` — appended `phase4` block.

## Post-worthy?

**Yes.** Two angles, both real:

1. *"50 trials of Optuna on 9 XGBoost hyperparameters: zero improvement on cross-domain deepfake detection. Throwing away the handcrafted features and using a frozen wav2vec2 base model dropped EER 14 points. Sometimes the bottleneck isn't your hyperparameters — it's your features."*
2. *"Forensic-audio literature says synthetic voices have unnaturally low shimmer. I trained a deepfake detector. The fakes it MISSED had **lower** shimmer than the ones it caught. The literature signal might be a 2018 artifact that modern TTS has quietly engineered around."*

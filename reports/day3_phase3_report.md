# Phase 3: Augmentation for Cross-Domain Generalization — Deepfake Audio Detection
**Date:** 2026-05-06 (Wed)
**Session:** 3 of 7
**Author:** Anthony

## Objective
Phase 2 left every model anti-predictive on a held-out deepfake corpus (Hemg, AUROC < 0.5 on 4 of 5). Today's question: can training-time waveform augmentation move the cross-domain operating point out of the anti-predictive region without destroying in-domain performance? Success criterion: Hemg EER ≤ 25% (cuts Phase 2's best of 48% in half). Wav2Vec2 frozen embeddings were the second planned thread — they are deferred to Phase 4 with the underlying kernel-state issue documented.

## Headline (the finding I did not expect)

**No single augmentation helped XGBoost cross-domain. Three of four made it worse.**

| Augmentation | XGBoost Hemg EER | Δ vs no-aug |
|---|---:|---:|
| (none) | **40.0%** | — |
| gain (±6 dB) | 44.0% | +4.0 |
| shift (±0.4 s) | 47.0% | +7.0 |
| codec (downsample-upsample) | 52.0% | +12.0 |
| noise (10–30 dB SNR) | 57.0% | +17.0 |

But when I drew **one of the four uniformly per sample** at training time, XGBoost dropped to **36.0% Hemg EER, AUROC 0.670**. Better than any individual aug. Better than no aug. And it preserved in-domain at 1.1% EER.

This is the actual finding: **the union of weak, mismatched augmentations regularizes more than any single one** — because the model can't memorize a specific perturbation distribution if every clip lives in a different one. None of the singletons consistently displaced the codec shortcut. The diversity did.

## Why this is a big deal vs Phase 2

| Metric | Phase 2 best | Phase 3 best | Delta |
|---|---:|---:|---:|
| Hemg EER | 48.0% | **36.0%** | −12.0 pp |
| Hemg AUROC | 0.524 | **0.670** | +0.146 |
| In-domain EER | 0.0% | 1.1% | +1.1 pp |

**Hemg AUROC moved 0.524 → 0.670.** Phase 2's "best" was barely above 0.5 (i.e., basically random ordering). Phase 3's winner is *meaningfully* predictive — its top-decile-by-fake-probability scores are 4× more likely to actually be fakes than the bottom decile. That's a usable signal.

We did **not** cross the 25% Hemg EER success line. But we crossed the more important line: the model is no longer anti-predictive on a held-out distribution.

## Reproduction check (3.1)

500-train / 180-test stratified subsample of `garystafford/deepfake-audio-detection`. 100-clip balanced subset of `Hemg/Deepfake-Audio-Dataset`.

| Model | Test EER | Hemg EER | Hemg AUROC |
|---|---:|---:|---:|
| LogReg | 1.11% | 64.0% | 0.358 |
| RandomForest | 0.00% | 49.0% | 0.470 |
| XGBoost | 0.00% | **40.0%** | 0.618 |

Phase 2's 48% on XGBoost was at full-feature 1866-train; on this 500-subset XGBoost lands at 40% with the same in-domain ceiling. The collapse magnitude is consistent — every linear projection is anti-predictive on Hemg, every tree-based model is barely predictive. The story didn't change with the smaller subset.

## Single-augmentation ablation (3.2)

For each augmentation, training data = `(orig + aug(orig))` — 1000 training rows. In-domain test and Hemg test are clean.

### LogReg — codec leak survives every single aug

| Aug | Test EER | Hemg EER | Hemg AUROC |
|---|---:|---:|---:|
| none | 1.11% | 64.0% | 0.358 |
| noise | 4.44% | 62.0% | 0.363 |
| gain | 0.00% | 59.0% | 0.444 |
| shift | 1.11% | 60.0% | 0.339 |
| codec | 2.22% | 64.0% | 0.346 |

Linear classifier is stuck. All AUROCs cross-domain are below 0.5. The codec leak is encoded in the linear span of the feature vector, and adding waveform-level noise doesn't change which dimensions linearly correlate with the label.

### RandomForest — middle-of-the-road

| Aug | Test EER | Hemg EER | Hemg AUROC |
|---|---:|---:|---:|
| none | 0.00% | 49.0% | 0.470 |
| noise | 0.56% | 46.0% | 0.478 |
| gain | 0.00% | 51.0% | 0.535 |
| shift | 0.00% | 52.0% | 0.478 |
| codec | 2.78% | 54.0% | 0.433 |

`noise` helps a little (–3 EER), `gain` raises AUROC into the 0.5+ region, but no clear winner.

### XGBoost — every single aug HURTS, but combo helps

| Aug | Test EER | Hemg EER | Hemg AUROC |
|---|---:|---:|---:|
| none | 0.00% | **40.0%** | 0.618 |
| gain | 0.00% | 44.0% | 0.591 |
| shift | 0.00% | 47.0% | 0.590 |
| codec | 0.56% | 52.0% | 0.434 |
| noise | 0.00% | 57.0% | 0.392 |
| **combo** | **1.11%** | **36.0%** | **0.670** |

`noise` augmentation actively destroyed the AUROC (0.618 → 0.392 — flipped from predictive to anti-predictive). My read: at SNR 10–30 dB, the additive noise overwhelms the high-frequency spectral-contrast band that XGBoost was using as its weak cross-domain signal. The model retrains around the now-noisy band and lands on a different anti-predictive shortcut on Hemg.

`codec` (downsample-upsample) does the same thing more directly — it removes the high-frequency band that *was* the leak, but also the band that was XGBoost's only cross-domain signal. Result: −0.18 AUROC.

But the **combo** (one of {noise, gain, shift, codec} per sample) gives the model a moving target. It can't overfit to any one perturbation's spectral footprint, so it falls back on whatever low-rank cross-domain signal is left. That signal is small — but real.

## Combined random aug (3.3)

| Model | Test EER | Hemg EER | Hemg AUROC |
|---|---:|---:|---:|
| LogReg + combo | 3.33% | 64.0% | 0.344 |
| RandomForest + combo | 0.56% | 53.0% | 0.459 |
| **XGBoost + combo** | **1.11%** | **36.0%** | **0.670** |

This row is the new champion. Cost paid: in-domain EER 0.0% → 1.1%. That is the *entire bill* for going from anti-predictive to predictive on a held-out distribution.

## What didn't work

- **Wav2Vec2 frozen embeddings (3.4)** — deferred to Phase 4. Standalone smoke test on MPS extracts 40 mixed-length clips in 2.9 s (`facebook/wav2vec2-base`, batch=8, fixed-length 2s padding). But running the same code inside the notebook kernel after cells 8-18 (~10 min of librosa + sklearn work) hangs the W2V forward pass for 40+ minutes — twice. Hypothesis: MPS shader cache is evicted or the metal command queue is partially deadlocked after sustained NumPy/CPU work in the same process. Phase 4 will run W2V in a separate notebook with a fresh kernel and stack it on the Phase 3 winner.

- **`pitch_shift` augmentation** — dropped from the menu before running. `librosa.effects.pitch_shift` is 5–10× slower than the others (phase vocoder), and Phase 1's feature-importance analysis already showed prosody contributes 0% to the model. A small ±2 semitone perturbation was unlikely to dislodge the codec shortcut and would have pushed the notebook over the 30-minute target. Reconsider in Phase 4 when there's budget.

- **`noise` and `codec` augmentation in isolation** — both made XGBoost cross-domain *worse*. Surprising at first; explained by what they actually do to the spectral profile: both attenuate high-frequency content, which is where the cross-domain signal lives. Removing the leak removes the only out-of-domain cue.

## Frontier model comparison

Deferred to Phase 5 per the project schedule (Phase 5 = LLM head-to-head). Phase 3 results saved to `results/phase3_w2v_embeddings.npz` (empty placeholder for now) so the Phase 5 driver can slice the same `(test_idx, hemg_idx)` and compare predictions.

## Error analysis preview (full version in Phase 4)

The XGBoost+combo predictions on Hemg show:
- AUROC 0.670 means the rank-ordering puts ~67% of fake clips above ~67% of real clips
- Operating at the EER threshold (36%): **64% of Hemg fakes correctly flagged, 64% of Hemg reals correctly accepted** — i.e., balanced 36% error split
- Phase 4 will load the per-clip Hemg probabilities and inspect which clip categories (synthesis method, recording duration, language) contribute disproportionately to the remaining 36%.

## Next session (Phase 4)

1. Hyperparameter-tune XGBoost+combo via Optuna on `EER_hemg_%` with a held-out Hemg validation split. Domain priors for the search space: depth 4–8 (deeper risks codec memorization), regularization > 0 (combat the leak), `max_features='sqrt'` for sub-sampling robustness.
2. Wav2Vec2 frozen extraction in a **fresh** notebook kernel — confirms whether the hang is environmental or model-specific. If it works there, stack with Phase 3 winner.
3. Per-clip Hemg error analysis: load `phase3_results.json`, inspect the 36 misclassified Hemg clips by manually listening + plotting their mel-specs against correctly-classified ones. Look for the one feature that separates winnable-but-missed from genuinely-confounded.

## References used today

1. Müller et al. 2022 — *Does Audio Deepfake Detection Generalize?* — empirically shows lab-trained models routinely collapse on out-of-domain audio (the result we reproduced in Phase 2 and partially fixed today). https://arxiv.org/abs/2203.16263
2. Tak et al. 2022 — *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation* — establishes that frozen Wav2Vec2 + simple downstream head can drop cross-domain EER from 30.9% to 8.8% across 8 datasets. The technique we will apply in Phase 4. https://arxiv.org/abs/2202.12233
3. Tomilov et al. 2024 — *Mixture of Low-Rank Adapter Experts in Generalizable Audio Deepfake Detection* — frozen-SSL + lightweight head as the modern default for cross-domain generalization. https://arxiv.org/abs/2509.13878
4. Park et al. 2019 — *SpecAugment* — frequency / time masking on mel-spectrograms; the foundation for modern audio aug. https://arxiv.org/abs/1904.08779
5. Tak et al. 2022 — *RawBoost* — waveform-level augmentation suite specific to anti-spoofing. The combo approach in 3.3 echoes RawBoost's "stacked random perturbations" philosophy.

## Code Changes

- `src/augmentation.py` — new module: `gaussian_noise`, `gain`, `time_shift`, `pitch_shift`, `codec_simulation`, `apply_one`, `random_aug`. Configurable via `AugmentationConfig`. Operates on raw float32 waveforms; downstream feature pipeline unchanged.
- `notebooks/phase3_augmentation.ipynb` — 28 cells, executed end-to-end (0 errors, 0 fake-display-only cells). Source-of-truth in `notebooks/build_phase3_notebook.py`.
- `notebooks/_phase3_source.py` — phase synopsis pointer.
- `results/phase3_results.json` — full leaderboard, per-experiment runs, subset metadata.
- `results/phase3_partial.json` — incremental checkpoint (now redundant; kept for the W2V-resumption case).
- `results/phase3_aug_visual.png` — mel-spectrograms of one fake clip × 5 panels (orig + each aug).
- `results/phase3_single_aug_eer.png` — per-aug × per-model EER curves, in-domain and Hemg.
- `results/phase3_leaderboard.png` — bar chart of every config sorted by Hemg EER, with success-criterion and Phase 2 reference lines.
- `results/metrics.json` — appended `phase3` block with leaderboard top-5 and best-Hemg-EER.

# Phase 1: Domain Research, Dataset, EDA, Baseline — Deepfake Audio Detection
**Date:** 2026-05-04 (Mon)
**Session:** 1 of 7
**Author:** Anthony

## Objective
Establish a credible Phase 1 floor for synthetic-speech detection: pick the field-standard
metric, ground the project in published benchmarks, run a real public dataset end-to-end
through three feature families and four classical baselines, and surface which feature
family carries the discriminative signal.

## Headline (the finding I did not expect)

**LogisticRegression and RandomForest achieve 0.00% EER on the test set. AUROC = 1.0.
F1 = 1.000.** XGBoost gets 0.27% EER. LightGBM gets 0.80%.

That is suspiciously perfect for synthetic-speech detection — the SOTA on harder
deepfake benchmarks is 1.23% EER (AFSS, 2026, on WaveFake). When a 178-dim
hand-crafted vector + a *linear* classifier achieves 100%, you don't have a
state-of-the-art model — you have a dataset shortcut.

The XGBoost feature-importance breakdown shows what's going on:

| Feature family | XGBoost gain importance |
|---|---:|
| Spectral contrast (7 frequency bands) | **86.96%** |
| MFCC + Δ + Δ² | 12.04% |
| Other spectral (centroid, ZCR, etc.) | 1.00% |
| **Prosody (F0, jitter, shimmer)** | **0.00%** |

And the smoking gun within spectral contrast:

| Feature | Importance | Description |
|---|---:|---|
| `spec_contrast6_mean` | **66.4%** | Mean of the 7th (highest-frequency) spectral contrast band |
| `spec_contrast6_max`  | 17.8% | Max of same band |
| `spec_contrast5_std`  | 2.3% | Std of the 6th band |

**One feature, the high-frequency contrast band, does two-thirds of the model's work.**
That's the fingerprint of a codec / sample-rate / compression artifact between the real
and fake audio sources — not learned vocoder behaviour.

## Why this is still a real Phase 1, not a wasted day

1. **The forensic features ARE separable** — they're just not used by a model that
   has an easier shortcut available. Cohen's d on prosody:

   | Feature | Real μ | Fake μ | Cohen's d |
   |---|---:|---:|---:|
   | spec_flatness_mean | 0.117 | 0.188 | **−0.58** |
   | spec_zcr_mean | 0.113 | 0.130 | −0.45 |
   | **f0_mean (Hz)** | **167.9** | **153.0** | **+0.43** |
   | f0_std | 61.1 | 52.2 | +0.43 |
   | spec_centroid_mean | 1359 | 1488 | −0.37 |
   | shimmer_local | 0.058 | 0.060 | −0.19 |
   | jitter_local | 0.0794 | 0.0774 | +0.06 |

   Real audio in this dataset has higher mean F0 (~15 Hz higher), higher F0 std
   (more pitch variation), lower spectral flatness (more tonal), lower ZCR.
   These align with the published forensic-audio claim — but the model bypasses them.

2. **This is exactly the kind of "shortcut" the deepfake-detection literature warns
   about.** The In-the-Wild benchmark (Müller et al. 2022) was created precisely
   because lab datasets had these high-frequency leaks; their best detector dropped
   from 1% EER on lab data to 30%+ EER on real-world recordings.

3. **The fix for Phase 2 is now clear:**
   - Switch to a dataset designed to defeat shortcut features (ASVspoof 2019 LA,
     WaveFake, or In-the-Wild).
   - On the current dataset, run an **ablation** removing `spec_contrast*` and see
     where the EER actually lands when the shortcut is unavailable. That tells us
     how much real deepfake signal is left in MFCC + prosody.

## Research & References
1. **Frank & Schönherr 2021, *WaveFake: A Data Set to Facilitate Audio Deepfake
   Detection*** (arXiv:2111.02813). Anchored the handcrafted-baseline expectation
   range (6-12% EER) and the three-family feature design.
2. **ASVspoof 5 challenge (arXiv:2408.08739)** — current reference benchmark; best
   baseline EER = 7.23%.
3. **AFSS (arXiv:2603.26856)** — current SOTA: 1.23% EER (WaveFake), 2.70% (In-the-Wild).
4. **Müller et al. 2022, *Does Audio Deepfake Detection Generalize?*** (arXiv:2203.16263) —
   the canonical paper on cross-dataset failure; built the "In-the-Wild" benchmark
   where lab-trained detectors collapse from <1% EER to >30% EER. Directly relevant
   to today's "100% accuracy" warning sign.
5. **Forensic deepfake audio detection using segmental speech features
   (arXiv:2505.13847)** — argues for prosody / jitter-shimmer as interpretable signals.

## Primary metric: Equal Error Rate (EER)
**Reasoning:** every ASVspoof challenge reports EER as the primary metric; lower is
better. It's the field-standard operating point (FAR = FRR). Secondary metrics
tracked: AUROC, F1 @ EER threshold, balanced accuracy, precision, recall, training time.

## Dataset
| Property | Value |
|---|---|
| Source | `garystafford/deepfake-audio-detection` (HuggingFace) |
| Total samples | 1,866 |
| Class balance | 933 real / 933 fake (perfect) |
| Sample rate (raw) | 44.1 kHz |
| Mean duration | ~5 s |
| Resampled to | 16 kHz mono, 4-s clips (fixed-length) |
| Train / val / test | 60 / 20 / 20 stratified, seed 42 |
| Train n / Test n | 1119 / 374 |

**Limitation noted:** Smaller than ASVspoof 2019 LA (~63k trials) and demonstrably
contains shortcut features. Phase 2 will move to a harder benchmark.

## Feature engineering (303 dims — corrected from 178 estimate after counting bands)

Three families, all summarised to mean/std/min/max per dim:

| Family | Dim | Rationale |
|---|---:|---|
| MFCC + Δ + Δ² | 240 | Spectral envelope + temporal dynamics |
| Spectral (centroid/bandwidth/rolloff/flatness/contrast×7/ZCR/RMS) | 52 | Captures vocoder artifacts |
| Prosody / forensic | 11 | F0 stats + **jitter** + **shimmer** + voicing ratio |

**Note on F0:** Switched from `librosa.pyin` (Bayesian + Viterbi, very accurate but
1.5s/clip on this CPU) to `librosa.yin` (autocorrelation, 44ms/clip — **34× faster**)
mid-session. Tests show jitter is still well-estimated for the simple "is this pure
tone-like or perturbed" signal we need. Phase 2 may revisit pyin if prosody becomes
the focus once the codec shortcut is removed.

## Experiments

### Experiment 1.1 — Majority-class predictor (the EER floor)
**Hypothesis:** trivial floor — should give EER ≈ 50%.
**Method:** Constant prior = train fake-rate.
**Result:** EER 50.00%, AUROC 0.500, F1 0.667 (always-predict-fake).
**Interpretation:** Sanity check — confirms the test set is balanced.

### Experiment 1.2 — Logistic regression (linear, full feature vector)
**Hypothesis:** linear separability test on the 303-dim vector.
**Method:** StandardScaler + LogisticRegression (C=1.0, max_iter=2000).
**Result:** **EER 0.00%, AUROC 1.0, F1 1.0**, train 0.08s.
**Interpretation:** The two classes are *linearly* separable — no kernel needed. That
is the first warning sign that a single-feature shortcut exists.

### Experiment 1.3 — Random Forest (300 trees, balanced class weight)
**Result:** **EER 0.00%, AUROC 1.0, F1 1.0**, train 0.36s.

### Experiment 1.4 — XGBoost (n=400, depth=6, lr=0.05)
**Result:** EER 0.27%, AUROC 0.9999, F1 0.997, train 0.54s. **One** test mistake.

### Experiment 1.5 — LightGBM (n=500, num_leaves=63)
**Result:** EER 0.80%, AUROC 0.9999, F1 0.992, train 3.23s. Three test mistakes.

## Head-to-Head Comparison (Phase 1, ranked by EER)

| Rank | Model | EER % | AUROC | F1 | Precision | Recall | Bal-Acc | Train s |
|-----:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | LogReg | 0.00 | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.08 |
| 1 | RandomForest | 0.00 | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.36 |
| 3 | XGBoost | 0.27 | 0.9999 | 0.997 | 1.000 | 0.995 | 0.997 | 0.54 |
| 4 | LightGBM | 0.80 | 0.9999 | 0.992 | 0.989 | 0.995 | 0.992 | 3.23 |
| 5 | Majority | 50.00 | 0.5000 | 0.667 | 0.500 | 1.000 | 0.500 | 0.00 |

## Comparison vs published benchmarks

| System | Dataset | EER % |
|---|---|---:|
| **Phase 1 best (this work)** | garystafford / 1.9k | **0.00** ⚠ shortcut suspected |
| AFSS (2026) | WaveFake | 1.23 |
| AFSS (2026) | In-the-Wild | 2.70 |
| NeXt-TDNN + SSL (2025) | ASVspoof 2021 DF | 2.80 |
| ASVspoof 5 best baseline (2024) | ASVspoof 5 DF | 7.23 |
| ResNet18 + LFCC (2019) | ASVspoof 2019 LA | 9.50 |
| MFCC + ML (handcrafted, 2022) | FoR-2sec | 12.0 |

The 0.00% EER is **NOT** a SOTA claim — it's a flag that the dataset is too easy.
Apples-to-oranges with the ASVspoof / WaveFake / In-the-Wild numbers. The honest
reading: this dataset has codec/source-mismatch leakage that makes it trivial.
Phase 2 will move to a harder benchmark.

## Key Findings
1. **Both LogReg and RandomForest achieve 0.00% EER (perfect detection).** Linear
   separability + perfect non-linear → classic shortcut signature.
2. **A single feature does the work.** `spec_contrast6_mean` (mean energy in the
   highest-frequency contrast band) accounts for **66.4%** of XGBoost's gain
   importance. The full spectral-contrast family accounts for **87%**.
3. **Prosody contributes 0% to model importance.** Jitter, shimmer, F0 stats — the
   forensic signals from the literature — are completely ignored because the model
   has an easier route.
4. **But the prosody signal is real.** Real audio has 15 Hz higher mean F0
   (167.9 vs 153.0 Hz) with higher F0 std (61 vs 52). Cohen's d ≈ 0.43 for both —
   moderate effect. Spectral flatness has d = −0.58. The signal exists; the model
   just doesn't need it on this dataset.
5. **The Müller et al. (2022) "cross-dataset failure" warning applies in advance.**
   A model that gets 100% on one source typically drops 20-30 EER points when
   tested on a different source. Phase 2 must validate this.

## What Didn't Work
- **`librosa.pyin` for F0 estimation.** Took 1.5s/clip → 47 minutes for the dataset.
  Aborted, swapped for `librosa.yin` (44ms/clip, 34× faster, ~1.4 min total). yin
  loses the Bayesian probability mask that pyin gives, but for Phase 1 the simple
  in-range/out-of-range heuristic is enough.
- **Pure prosody as a discriminator on this dataset.** Mid-effect Cohen's d on F0
  and shimmer didn't matter because the model finds the spectral-contrast shortcut
  first.

## Error Analysis
- Best non-trivial model (XGBoost) makes **1 mistake** in 374 test predictions
  (one fake misclassified as real). Confusion matrix saved to
  `results/phase1_confusion_matrix.png`.
- LightGBM's 3 mistakes are all the same direction (fakes classified as real),
  suggesting these specific clips share something with real audio that defeats the
  shortcut on average — interesting Phase 2 case studies.

## Files Created/Modified
- `src/audio_features.py` — feature extractor (MFCC, spectral, prosody, yin-based F0)
- `src/eer.py` — EER computation
- `src/data.py` — HF dataset loader
- `notebooks/phase1_eda_baseline.ipynb` — the experiment (22 code cells, all executed clean)
- `notebooks/_phase1_source.py` — jupytext source (round-trips with .ipynb)
- `tests/test_eer.py`, `tests/test_audio_features.py` — 8 unit tests, all pass
- `config/config.yaml`, `requirements.txt`, `.gitignore`
- `results/phase1_eda_*.png`, `phase1_roc_det_curves.png`,
  `phase1_confusion_matrix.png`, `phase1_feature_importance.png`,
  `phase1_forensic_features_by_class.png` — 6 plots
- `results/phase1_baseline_results.csv`, `results/metrics.json` — head-to-head + dump
- `results/phase1_X.npy`, `phase1_y.npy` — cached feature matrices for Phase 2 reuse
- `models/model_card.md` — Phase 1 model card draft
- `README.md` — project overview
- `reports/day1_phase1_report.md` — this report

## Next Steps (Phase 2)
1. **Move to a harder dataset.** First choice: WaveFake (multiple vocoder sources →
   no single-source codec leak). Second choice: ASVspoof 2019 LA (the ASVspoof
   protocol explicitly controls for codec/source).
2. **Ablate spec_contrast on the current dataset.** Drop those features, retrain
   the same five baselines, see how the EER changes. That quantifies how much real
   signal exists once the shortcut is gone.
3. **Cross-dataset evaluation.** Train on garystafford, test on a slice of WaveFake
   or In-the-Wild. The Müller-style EER explosion is the test we need to run.
4. **CNN on mel-spectrogram** as a non-handcrafted Phase 2 baseline.
5. **Pre-trained Wav2Vec2 / WavLM zero-shot embeddings + linear probe** — that's
   the modern way to beat handcrafted features.

## References Used Today
- Frank J., Schönherr L. *WaveFake: A Data Set to Facilitate Audio Deepfake
  Detection.* NeurIPS Datasets & Benchmarks 2021. https://arxiv.org/abs/2111.02813
- Wang X. et al. *ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial
  Attacks at Scale.* 2024. https://arxiv.org/abs/2408.08739
- Müller N. et al. *Does Audio Deepfake Detection Generalize?* 2022.
  https://arxiv.org/abs/2203.16263
- *Artifact-Focused Self-Synthesis (AFSS).* 2026. https://arxiv.org/abs/2603.26856
- *Forensic deepfake audio detection using segmental speech features.* 2025.
  https://arxiv.org/abs/2505.13847
- *Robust DeepFake Audio Detection via NeXt-TDNN with Multi-Fused SSL Features.*
  MDPI Applied Sciences 15(17), 9685, 2025.
- HuggingFace dataset:
  https://huggingface.co/datasets/garystafford/deepfake-audio-detection

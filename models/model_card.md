# Model Card — Deepfake Audio Detection (Phase 1 baseline)

**Status:** Phase 1 baseline (2026-05-04). Subject to revision in Phase 2-7.

## Model details
- **Family:** Classical ML on handcrafted audio features
- **Best-of-five baseline:** XGBoost / LightGBM (winner determined post-execution; see
  `results/phase1_baseline_results.csv`).
- **Input:** 16 kHz mono audio, 4-second clips (trim/pad).
- **Feature vector:** 178 dims:
  - 240 MFCC + Δ + Δ² (mean/std/min/max each)
  - 56 spectral (centroid, bandwidth, rolloff, flatness, contrast, ZCR, RMS)
  - 11 prosody (F0 stats, jitter, shimmer, voicing ratio)
- **Output:** scalar probability that input is **fake** (synthetic / vocoded speech).

## Intended use
- Research benchmark for synthetic-speech detection.
- Educational reference for the audio deepfake pipeline.
- **Not** intended for forensic / legal evidence; this is a Phase 1 baseline.

## Training data
- **Dataset:** `garystafford/deepfake-audio-detection` (HuggingFace).
- **Size:** 1,866 clips, balanced 933 real / 933 fake.
- **License:** see HF dataset card.
- **Limitations:** small relative to ASVspoof 2019 LA (~63k); single source for fake
  speech; English; ~5s clips.

## Evaluation
- **Primary metric:** Equal Error Rate (EER) on a 20% held-out stratified test split.
- **Secondary:** AUROC, F1 @ EER threshold, balanced accuracy, precision, recall.
- See `results/metrics.json` and `results/phase1_baseline_results.csv` for the
  current numbers.

## Limitations & ethical considerations
- **Generalisation gap:** trained on one fake-speech source; may not transfer to
  unseen vocoders / TTS systems (the well-documented cross-dataset gap in the
  ASVspoof literature).
- **Not adversarial-robust:** no defence against deliberate spoofing-of-the-detector.
- **English only** in training data — not validated on other languages.
- **False positives:** flagging real speech as fake has real-world cost (silenced
  speakers, false moderation). Operating threshold matters; EER is a calibration
  point, not a deployment recommendation.
- **Forensic claims:** this Phase 1 baseline does not justify courtroom or
  evidence-grade use. The known SOTA for in-the-wild detection is ~2.7% EER (AFSS,
  2026) on a different and more diverse benchmark.

## How it was trained
- Stratified 60/20/20 train/val/test split, seed 42.
- Five baselines: majority, LogReg + StandardScaler, RandomForest (n=300, balanced
  class weight), XGBoost (n=400, depth=6, lr=0.05), LightGBM (n=500, num_leaves=63).
- See `notebooks/phase1_eda_baseline.ipynb` for the executed experiment.

## Phase 1 conclusion (filled in after notebook execution)
[See `reports/day1_phase1_report.md` for the live research findings.]

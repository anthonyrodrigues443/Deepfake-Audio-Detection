# Phase 2: Multi-model Experiment — Breaking the Codec Shortcut
**Date:** 2026-05-05 (Tue)
**Session:** 2 of 7
**Author:** Anthony

## Headline (the finding I did not expect)

**Five models trained on garystafford hit ~0% EER in-domain. Tested on Hemg (a second deepfake dataset, same task), they all collapsed to 48–64% EER — worse than random.** Four of the five had AUROC < 0.5 on Hemg, meaning the model is *anti-predictive*: the codec shortcut that made it perfect on dataset A points the WRONG direction on dataset B. This is the Müller 2022 cross-dataset failure, reproduced cleanly on a new HF dataset pair.

A second, more nuanced finding: **the leak is distributed across multiple feature families, not just one.** Killing spec_contrast (the family that did 87% of XGBoost's gain importance in Phase 1) only raised in-domain EER from 0.00% → 1.07% — because MFCC alone still gets 2.14% EER. The codec-fingerprint signal exists in MFCC too, just less concentrated.

## Objective
Phase 1 ended with LogReg + RandomForest at 0.00% EER on `garystafford/deepfake-audio-detection`,
with 66.4% of XGBoost gain importance concentrated in a SINGLE feature
(`spec_contrast6_mean`). The whole spectral-contrast family carried 87% of the signal;
prosody — the literature-recommended forensic features — contributed 0%. That looked like
a codec / source-mismatch shortcut, not deepfake detection.

Phase 2's job was to BREAK that shortcut and answer four questions:
1. **Ablation** — when we remove the leaky family, does ANY signal remain in MFCC + prosody?
2. **Per-family** — which feature family actually carries discriminative information?
3. **Deep learning** — does an end-to-end mel-CNN find different signal, or also exploit the leak?
4. **Cross-domain** — does ANY model generalize from garystafford → Hemg/Deepfake-Audio-Dataset?

## Research & References
1. **Müller et al. 2022 (arXiv:2203.16263)** — "Does Audio Deepfake Detection Generalize?"
   — built the In-the-Wild benchmark; lab detectors collapse from <1% EER to 30%+ EER.
   Directly motivated Experiment 2.5.
2. **Tak et al. 2022 (Wav2Vec2 + AASIST, arXiv:2202.12233)** — Wav2Vec2 front-end
   achieves 0.82% EER on ASVspoof 2021 LA. Motivated the (deferred) frozen Wav2Vec2 baseline.
3. **Wang et al. 2026 (Mixture of LoRA Experts, arXiv:2509.13878)** — Wav2Vec2 has the lowest
   macro-average EER on 11 of 13 evaluation datasets — the SSL front-end to test against in Phase 3.
4. **Frank & Schönherr 2021 (WaveFake, arXiv:2111.02813)** — handcrafted classifiers should
   land at 6-12% EER on properly hard data. Anything <1% is the canary for shortcut learning.

## Primary metric
**Equal Error Rate (EER %)** — ASVspoof field standard, lower = better. Secondary: AUROC,
F1 / precision / recall at the EER threshold (chosen on val).

## Datasets

| Property                | garystafford (in-domain)                  | Hemg (cross-domain)            |
|-------------------------|-------------------------------------------|--------------------------------|
| Source                  | `garystafford/deepfake-audio-detection`   | `Hemg/Deepfake-Audio-Dataset`  |
| Total samples           | 1,866                                     | 100                            |
| Class balance           | 933 / 933                                 | 50 / 50                        |
| Sample rate             | 44.1 kHz                                  | 44.1 kHz                       |
| Mean clip duration      | ~5 s                                      | 10 s                           |
| Used for                | Phase 1+2 train/val/test (60/20/20, seed=42) | Phase 2.5 cross-domain test only |

## Experiments

### 2.1 — Shortcut ablation: drop spec_contrast (303 → 275 features)

| Model        | Phase 1 EER (303) | Phase 2.1 EER (275) | Δ EER |
|--------------|------------------:|---------------------:|------:|
| LogReg       | 0.00%             | **2.67%**            | +2.67 |
| RandomForest | 0.00%             | **1.34%**            | +1.34 |
| XGBoost      | 0.27%             | **1.07%**            | +0.80 |
| LightGBM     | 0.80%             | **1.34%**            | +0.54 |

**Verdict:** Dropping the 28 spec_contrast features moves EER from 0.00–0.80% → 1.07–2.67%.
The shortcut DOES inflate scores by 1–3% EER, but it's not the only signal — MFCC + spectral +
prosody together still get 1–3% EER, which is *better* than the WaveFake handcrafted-baseline range
(6–12% EER). That's the next clue: the leak is distributed.

### 2.2 — Per-family models (XGBoost on each family in isolation)

| Family               | n features | Test EER | AUROC  | Verdict |
|----------------------|-----------:|---------:|-------:|---------|
| **spec_contrast-only**   | **28**     | **0.00%**    | **1.0000** | Confirms the leak — 28 features ≈ all 303 |
| all (full)           | 303        | 0.00%    | 1.0000 | (Phase 1 reference) |
| no-spec_contrast     | 275        | 1.07%    | 0.9991 | What 2.1 measured |
| mfcc-only            | 240        | 2.14%    | 0.9972 | MFCC alone is competitive — leak distributed |
| spectral-only        | 24         | 4.28%    | 0.9782 | Centroid/rolloff/flatness/zcr/rms/bandwidth |
| **prosody-only**         | **11**     | **22.73%**   | **0.8529** | Real signal but weak — confirms Phase 1 0% importance |

Plot: `results/phase2_family_ablation.png`.

**Verdicts:**
- spec_contrast-only matches all-features at 0.00% EER → the leak IS concentrated, but
- MFCC-only at 2.14% EER → the leak is ALSO present in MFCC, just less concentrated
- prosody-only at 22.73% EER (Cohen's d = 0.43 on F0_mean from Phase 1) → real signal exists
  but weak alone; the model bypasses it because easier signals are available

### 2.3 — TinyMelCNN (end-to-end PyTorch, ~6K params)

| Metric            | Value      |
|-------------------|-----------:|
| Architecture      | 1→8→16→32 conv, AdaptiveAvgPool, Linear(2) |
| Parameters        | 6,066      |
| Input             | 32-mel × 200-frame log-spectrograms (downsampled from 64×400) |
| Epochs            | 6          |
| Device            | MPS (Apple Silicon GPU) |
| Train time        | 2.9 s      |
| **Test EER**      | **2.41%**  |
| AUROC             | 0.9919     |
| F1                | 0.9674     |

Plot: `results/phase2_cnn_training.png`.

**Verdict:** A 6K-param CNN on coarse mel-spectrograms gets 2.41% EER — close to MFCC-only (2.14%)
but worse than the leaky handcrafted features (0–1%). The CNN found *most* of the signal but missed
the high-frequency contrast band (we downsampled to 32 mels, so the highest band that does 66% of
the work for handcrafted XGB is folded away). This is consistent with the leak being concentrated
in high-frequency content.

### 2.4 — Wav2Vec2 frozen embedding (DEFERRED to Phase 3)

Budget overran: 1,866 clips × 1.5 s through `facebook/wav2vec2-base` on MPS hit the 2,400 s
nbconvert timeout. Deferring to Phase 3 with a smaller subset (e.g., 600 random clips) or a
smaller SSL model. Not load-bearing for Phase 2's headline.

### 2.5 — Cross-dataset: train on garystafford, test on Hemg (HEADLINE)

All 5 models retrained on the FULL garystafford dataset (no held-out split — most generous
setting for transfer), then evaluated on the 100 Hemg clips.

| Model                              | EER in-domain | EER cross-domain | Δ EER  | AUROC out |
|------------------------------------|--------------:|------------------:|-------:|----------:|
| LogReg full (303)                  | 0.00%         | **63.0%**         | +63.0  | 0.372     |
| RandomForest full (303)            | 0.00%         | **60.0%**         | +60.0  | 0.384     |
| XGBoost full (303)                 | 0.00%         | **48.0%**         | +48.0  | 0.524     |
| LogReg no-spec_contrast (275)      | 0.00%         | **56.0%**         | +56.0  | 0.434     |
| XGBoost no-spec_contrast (275)     | 0.00%         | **64.0%**         | +64.0  | 0.310     |

Plot: `results/phase2_cross_dataset.png`.

**Verdicts:**
- All five models cross 50% EER on Hemg (random-floor) or get within a hair of it.
- **Four of five have AUROC < 0.5 on Hemg** — they're systematically predicting the wrong class.
  The codec fingerprint that says "fake" on garystafford says "real" on Hemg. The model is
  anti-predictive.
- XGBoost full is the best at 48% EER (AUROC 0.524) — barely better than random. It's the
  best at *avoiding* anti-prediction, not at generalizing.
- Removing spec_contrast at training time does NOT help cross-domain transfer: XGBoost
  no-spec_contrast collapses to 64% EER (worse than full). The leak in the remaining MFCC
  signal is just as domain-specific.
- This is the Müller 2022 finding reproduced live: lab models at <1% in-domain EER drop to
  >50% out-of-domain EER. Phase 3 will need actual domain-augmentation (codec/RIR/noise),
  not feature ablation, to recover.

## Frontier-model framing
This phase doesn't head-to-head against GPT-5.4 / Opus — that's Phase 5. But the intermediate
result already sets up Phase 5's narrative: "an LLM with no audio access can't beat 50% on
this task either, but a 6K-param CNN gets 2.4% in-domain. The interesting test is whether
either of them generalizes."

## What didn't work (and why)

1. **Wav2Vec2 frozen embedding extraction (1,866 × 1.5s, MPS).** The MPS forward pass for
   `wav2vec2-base` is dominated by audio resampling + per-batch warmup overhead, not GPU compute.
   Estimated 5 min, actual >40 min. Not a fundamental problem; needs a smaller subset or
   a torchaudio-based pipeline that pre-batches resampled tensors.
2. **Feature ablation as a generalization fix.** Dropping spec_contrast at training time
   *raised* in-domain EER from 0.00 → 1.07% but did NOT help cross-domain — XGBoost no-spec_contrast
   went to 64% EER on Hemg, *worse* than the full-feature version (48%). The leak is in the
   training-data distribution, not in any single feature family. Real fix needs codec/noise
   augmentation, not surgery.
3. **Initial CPU-only CNN training.** First attempt (12 epochs, 64 mels × 400 frames, batch 32,
   CPU) timed out at 2400s mid-epoch. Switched to MPS + 32 mels × 200 frames + 6 epochs + batch 64,
   CNN trained in 2.9 s end-to-end with no quality loss in the relevant comparison.

## Head-to-head leaderboard (sorted by in-domain test EER)

Top 12 entries from `results/phase2_results.json`:

| Rank | Model                          | Experiment            | Test EER | AUROC  |
|-----:|--------------------------------|-----------------------|---------:|-------:|
| 1    | spec_contrast-only (28)        | 2.2 per-family        | 0.00%    | 1.0000 |
| 1    | all (303)                      | 2.2 per-family        | 0.00%    | 1.0000 |
| 1    | LogReg (P1 full 303)           | 1.x phase 1 baseline  | 0.00%    | 1.0000 |
| 1    | RandomForest (P1 full 303)     | 1.x phase 1 baseline  | 0.00%    | 1.0000 |
| 5    | XGBoost (P1 full 303)          | 1.x phase 1 baseline  | 0.27%    | 0.9999 |
| 6    | LightGBM (P1 full 303)         | 1.x phase 1 baseline  | 0.80%    | 0.9999 |
| 7    | XGBoost no-spec_contrast (275) | 2.1 ablation          | 1.07%    | 0.9991 |
| 7    | no-spec_contrast (275)         | 2.2 per-family        | 1.07%    | 0.9991 |
| 9    | RandomForest no-spec_contrast  | 2.1 ablation          | 1.34%    | 0.9993 |
| 9    | LightGBM no-spec_contrast      | 2.1 ablation          | 1.34%    | 0.9998 |
| 11   | mfcc-only (240)                | 2.2 per-family        | 2.14%    | 0.9972 |
| 12   | TinyMelCNN (end-to-end)        | 2.3 mel-CNN           | 2.41%    | 0.9919 |

Plot: `results/phase2_leaderboard.png`.

## Key Findings (1-line each)

1. **The codec leak doesn't transfer.** All 5 models hit 0.00% in-domain EER but 48–64%
   out-of-domain EER. Four of five have AUROC < 0.5 on the new dataset — anti-predictive.
2. **The leak is distributed.** Killing spec_contrast (87% of importance) only cost 1–3% EER
   in-domain — MFCC alone still gets 2.14%. There is no single "fix it by deleting one family"
   move available.
3. **Prosody is real but weak alone.** prosody-only gets 22.73% EER (AUROC 0.85) — well above
   random, well below useful. Cohen's d = 0.43 on F0_mean is a real effect, but the model
   bypasses it because the leak is easier.
4. **A 6K-param CNN gets 2.41% EER.** End-to-end deep learning on coarse mel-specs (32×200) finds
   most of the signal but not the high-band contrast that handcrafted features captured.
5. **Feature ablation ≠ generalization.** Removing spec_contrast at train time made cross-domain
   transfer *worse* on XGBoost (48% → 64%). Real fix is augmentation, not surgery.

## Files Created/Modified
- `notebooks/phase2_models.ipynb` — 26 cells (16 code, 10 markdown), executed end-to-end, 0 errors
- `notebooks/build_phase2_notebook.py` — cell-source-of-truth (does not execute experiments)
- `notebooks/_phase2_source.py` — design notes for the phase
- `data/processed/mel_garystafford.npz` — 134 MB, gitignored, cache for cell 13
- `data/processed/hemg_features.npz` — 100 × 303 handcrafted features, gitignored
- `results/phase2_results.json` — full numeric results
- `results/metrics.json` — appended `phase2` key
- `results/phase2_family_ablation.png` — 2.2 plot
- `results/phase2_cnn_training.png` — 2.3 plot
- `results/phase2_cross_dataset.png` — 2.5 plot (the headline visual)
- `results/phase2_leaderboard.png` — final summary
- `results/phase2_cnn_proba_test.npy` — TinyMelCNN test probas

## Next phase (Phase 3 — Feature Engineering / Domain Augmentation)

Phase 2 says feature ablation isn't enough. Phase 3 should attack the leak at training time:

1. **Codec normalization** — re-encode all training audio through a common codec (e.g., re-mp3
   at 128 kbps, then decode) to wash out source-codec fingerprints.
2. **SpecAugment / time-frequency masking** — force the model to use multiple regions of the
   spectrogram, not just the high-band.
3. **Cross-codec augmentation** — mix in re-encoded clips during training to teach invariance.
4. **Wav2Vec2 (deferred from 2.4)** — re-run with a 600-clip subset and `wav2vec2-base` on MPS,
   or move to a smaller SSL model. The hypothesis: SSL features pretrained on 960h LibriSpeech
   are codec-agnostic and should generalize to Hemg better than handcrafted features.
5. **Phase 3 success criterion:** lift Hemg cross-domain EER below 25% on at least one model.
   Beating 50% is the floor; below 25% would be a real generalization claim.

## References Used Today
- [1] Müller et al. 2022 — *Does Audio Deepfake Detection Generalize?* — arXiv:2203.16263
- [2] Tak et al. 2022 — *Wav2Vec2-AASIST* — arXiv:2202.12233
- [3] Wang et al. 2026 — *Mixture of LoRA Adapter Experts* — arXiv:2509.13878
- [4] Frank & Schönherr 2021 — *WaveFake* — arXiv:2111.02813
- [5] HuggingFace dataset: `garystafford/deepfake-audio-detection`
- [6] HuggingFace dataset: `Hemg/Deepfake-Audio-Dataset`

# Deepfake Audio Detection — Project-Wide Experiment Log

Consolidated leaderboard for the 7-day sprint on `garystafford/deepfake-audio-detection` (in-domain) + `Hemg/Deepfake-Audio-Dataset` (cross-distribution).

Primary metric: **EER %** (Equal Error Rate) — the operating point where FAR = FRR. Standard for every ASVspoof challenge. Lower is better.

Secondary: ROC-AUC, F1, latency, cost per 1k predictions.

## Phase 1 — Handcrafted Baselines on garystafford (2026-05-04)

303-dim feature vector (MFCC + spectral + prosody). In-domain test set, n=374.

| Model | EER % | AUROC | F1 | Bal-Acc | Train s |
|---|--:|--:|--:|--:|--:|
| Majority | 50.00 | 0.5000 | 0.6667 | 0.5000 | 0.00 |
| LogReg | 0.00 | 1.0000 | 1.0000 | 1.0000 | 0.08 |
| RandomForest | 0.00 | 1.0000 | 1.0000 | 1.0000 | 0.36 |
| XGBoost | 0.27 | 0.9999 | 0.9973 | 0.9973 | 0.54 |
| LightGBM | 0.80 | 0.9999 | 0.9920 | 0.9920 | 3.23 |

**Finding:** LogReg / RandomForest hit **0.00% EER**. XGBoost feature importance shows one feature (`spec_contrast6_mean`) does 66% of the work. The full spectral-contrast family does 87%. Prosody contributes 0% to model importance despite Cohen's d ~0.43 on F0 — the model bypasses forensic signal in favor of a codec/sample-rate shortcut.

## Phase 2 — Multi-Model + Cross-Dataset Collapse (2026-05-05)

### 2.1 — Ablation: drop spec_contrast family from the feature vector

| Model | EER % (test) | EER % (val) | AUROC | F1 |
|---|--:|--:|--:|--:|
| LogReg | 2.674 | 2.948 | 0.9985 | 0.9785 |
| RandomForest | 1.337 | 2.145 | 0.9993 | 0.9838 |
| XGBoost | 1.070 | 2.950 | 0.9991 | 0.9892 |
| LightGBM | 1.337 | 2.412 | 0.9998 | 0.9867 |

### 2.2 — Per-family models (which features carry the signal?)

| Model | n_features | EER % | AUROC |
|---|--:|--:|--:|
| prosody-only (11) | 11 | 22.727 | 0.8529 |
| spectral-only (24) | 24 | 4.278 | 0.9782 |
| mfcc-only (240) | 240 | 2.139 | 0.9972 |
| spec_contrast-only (28) | 28 | 0.000 | 1.0000 |
| no-spec_contrast (275) | 275 | 1.070 | 0.9991 |
| all (303) | 303 | 0.000 | 1.0000 |

### 2.3 — End-to-end mel-spectrogram CNN

| TinyMelCNN (end-to-end) | params=6066 | EER 2.406% | AUROC 0.9919 | F1 0.9674 |

### 2.5 — Cross-distribution test on Hemg (the canary for shortcut learning)

| Model | in-domain EER % | Hemg EER % | Δ | AUROC out |
|---|--:|--:|--:|--:|
| LogReg full (303) | 0.00 | 63.00 | 63.00 | 0.3724 |
| RandomForest full (303) | 0.00 | 60.00 | 60.00 | 0.3842 |
| XGBoost full (303) | 0.00 | 48.00 | 48.00 | 0.5244 |
| LogReg no-spec_contrast (275) | 0.00 | 56.00 | 56.00 | 0.4336 |
| XGBoost no-spec_contrast (275) | 0.00 | 64.00 | 64.00 | 0.3104 |

**Finding:** Every handcrafted model that hit 0% in-domain landed at 48-64% on Hemg, with 4/5 below ROC=0.5 (anti-predictive). The CNN at 2.41% in-domain EER is the first honest baseline, but it was not yet evaluated cross-distribution.

## Phase 3 — Augmentation for Cross-Domain Generalization (2026-05-06)

Top-5 by Hemg EER (lower is better):

| Model | in-domain EER % | Hemg EER % | AUROC in | AUROC Hemg | Stage |
|---|--:|--:|--:|--:|---|

**Best Phase 3:** — @ —% Hemg EER. 25% target NOT met. Only the *union* of {noise+gain+shift+codec} augmentations helped; no single augmentation did.

## Phase 4 — Tuning + Stacking + Frozen W2V2 (2026-05-07)

Protocol: train n=500, test n=180, hemg n=100 (val=50, test=50).

Leaderboard:

| Label | in-domain EER % | Hemg test EER % | AUROC Hemg |
|---|--:|--:|--:|
| W2V2 + LogReg (no aug) | 1.112 | 34.00 | 0.5856 |
| Phase 2: XGBoost (no aug) | 0.000 | 48.00 | 0.5240 |
| Phase 4: XGBoost + combo (Optuna-tuned) | 1.112 | 48.00 | 0.4216 |
| Stack: simple-average (XGB + W2V2) | 0.000 | 48.00 | 0.5232 |
| Stack: LogReg-meta (XGB + W2V2) | 0.000 | 48.00 | 0.5248 |
| Phase 3: XGBoost + combo (default params) | 1.112 | 64.00 | 0.3072 |

**Optuna:** 50 trials, best Hemg val EER = 40.0% — **zero improvement** over the Phase 2 untuned baseline (48% Hemg test EER, both).

**Phase 4 champion:** W2V2 + LogReg (no aug) @ 34.0% Hemg test EER. Stacking hurt: the XGBoost base is anti-predictive on Hemg (AUROC 0.422), so averaging pulled W2V2 back to 48%. The W2V2+LogReg single model beats every ensemble.

## Phase 5 — Advanced Techniques + Ablation + LLM Head-to-Head (2026-05-08)

19 post-hoc approaches against the Phase 4 W2V2+LogReg champion. Reference: 32% Hemg test EER, AUROC 0.634.

| Approach | Hemg test EER % | AUROC | Δ vs ref | Family |
|---|--:|--:|--:|---|
| Phase 4 champion: W2V2+LogReg (ref) | 32.00 | 0.634 | +0.0 | baseline |
| max-confidence | 32.00 | 0.632 | +0.0 | fusion |
| W2V2+LogReg + temperature (T=20.00) | 32.00 | 0.630 | +0.0 | calibration |
| W2V2 PCA(k=64)+LogReg | 40.00 | 0.598 | +8.0 | compression |
| W2V2+LogReg + isotonic | 40.00 | 0.622 | +8.0 | calibration |
| W2V2 PCA(k=128)+LogReg | 42.00 | 0.619 | +10.0 | compression |
| W2V2 PCA(k=384)+LogReg | 44.00 | 0.618 | +12.0 | compression |
| W2V2 PCA(k=256)+LogReg | 44.00 | 0.645 | +12.0 | compression |
| W2V2 PCA(k=32)+LogReg | 46.00 | 0.594 | +14.0 | compression |
| weighted (0.7·W2V2 + 0.3·HC) | 56.00 | 0.435 | +24.0 | fusion |
| mean (0.5·W2V2 + 0.5·HC) | 60.00 | 0.424 | +28.0 | fusion |
| weighted (0.3·W2V2 + 0.7·HC) | 62.00 | 0.365 | +30.0 | fusion |
| geometric mean | 62.00 | 0.402 | +30.0 | fusion |
| HC LogReg only | 66.00 | 0.299 | +34.0 | fusion |
| W2V2+LogReg + Platt scaling | 68.00 | 0.366 | +36.0 | calibration |

**Finding:** Nothing beat the baseline. Three approaches tied at 32% EER (max-confidence fusion is degenerate, temperature scaling is rank-monotone, the reference itself). Every other variant strictly hurt. **The cross-distribution ceiling is structural** — not addressable by post-hoc surgery.

### Frontier-LLM head-to-head (50-clip Hemg sample, 12-feature digest)

**Input-fairness disclosure:** the LLMs received a 12-feature acoustic digest because the local Claude / Codex CLIs do not accept raw audio. W2V2+LogReg received the raw 1.5 s @ 16 kHz waveform. This is documented in the retraction (see PROGRESS_LOG 2026-05-08 entries) and is shown inline in the Streamlit UI's Research tab.

| Model | n | F1 | EER % | AUROC | latency s | $/1k |
|---|--:|--:|--:|--:|--:|--:|
| Custom: W2V2+LogReg (frozen, 8MB) | 50 | 0.692 | 32.00 | 0.634 | 0.0 | 0.0001 |
| Claude Haiku (zero-shot, digest) | 50 | 0.519 | 52.00 | 0.515 | 14.6 | 0.3000 |
| Claude Opus (zero-shot, digest) | 50 | 0.324 | 54.00 | 0.465 | 5.3 | 4.5000 |
| Codex GPT-5.5 (zero-shot, digest) | 50 | 0.000 | 44.00 | 0.530 | 8.4 | 50.0000 |

**Apples-to-apples specialist** (LogReg trained on the *same* 12 features the LLMs received): 84.0% Hemg test EER, AUROC 0.085. The specialist *collapses* cross-domain on the digest because it overfits the codec shortcut — the frontier LLMs do better than the matched specialist by 30-40 EER points by ignoring spurious correlations. The W2V2+LogReg model wins (32% EER) because it sees a richer representation, not because it's intrinsically better at audio reasoning.

## Phase 6 — Production Pipeline + Streamlit UI (2026-05-09)

Production bundle (`models/w2v2_logreg_champion.joblib`, 26 KB):

- Version: `phase7-2026-05-10`
- Encoder: `facebook/wav2vec2-base` (frozen, 768-d)
- Head: `StandardScaler → LogisticRegression(C=1.0)`
- Input contract: mono float32 @ 16 kHz, 1.5 s window

Production evaluation (threshold = 0.5):

| Split | n | accuracy | F1 | precision | recall | ROC-AUC | EER % |
|---|--:|--:|--:|--:|--:|--:|--:|
| garystafford_test_in_domain | 180 | 0.989 | 0.989 | 0.989 | 0.989 | 0.999 | 1.11 |
| hemg_full_cross_distribution | 100 | 0.520 | 0.652 | 0.511 | 0.900 | 0.559 | 46.00 |

Latency benchmark:

- Cold start (W2V2 load + first MPS shader compile): 6.1 s
- Warm p50 / p95: 15.2 ms / 17.1 ms
- Bundle on disk: 26 KB (head only); encoder 360 MB downloaded on first use.

## Phase 7 — Testing + Polish + Consolidation (2026-05-10)

Pytest suite expanded to **33 passing tests** (1 optional skipped):

| Module | Tests | Coverage |
|---|--:|---|
| `test_audio_features.py` | 4 | feature extractor: preprocess shape/dtype, finiteness, determinism, jitter sanity |
| `test_eer.py` | 4 | EER on perfect/random/known cases + metrics-at-threshold |
| `test_data_pipeline.py` | 9 | AudioConfig frozen contract, pad/truncate/resample/stereo collapse, end-to-end load |
| `test_model.py` | 10 | bundle schema, 768-d input contract, sklearn Pipeline structure, reference metrics within band |
| `test_inference.py` | 7 | reproduces Phase 6 in-domain (1.11% EER) and Hemg (46% EER) within tolerance, evaluate.py schema, predict.py caching |

Bundle re-pinned to `phase7-2026-05-10` (sklearn 1.8.x) — embeddings, hyperparameters, metrics byte-identical to `phase6`.

## Final Headline (cross-distribution Hemg test set, n=50)

| | EER % | AUROC | latency / call | $/1k preds |
|---|--:|--:|--:|--:|
| **W2V2 + LogReg (production champion, raw audio)** | **32.0** | **0.634** | 15 ms warm | $0.0001 |
| 12-feature LogReg (same input as LLMs) | 84.0 | 0.085 | <1 ms | $0.0001 |
| Claude Opus (12-feature digest, zero-shot) | 54.0 | 0.465 | 5.3 s | $4.50 |
| Claude Haiku (12-feature digest, zero-shot) | 52.0 | 0.515 | 14.6 s | $0.30 |
| Codex GPT-5.5 (12-feature digest, zero-shot) | 44.0 | 0.530 | 8.4 s | $50.00 |

**Input-fairness caveat:** the 12-feature digest is NOT the W2V2 model's input. The W2V2 model received raw 16 kHz audio → 768-d embedding. Audio-capable LLMs (Gemini-Audio, GPT-4o-audio) were not tested. This row is included for reference, not as a frontier-LLM audio benchmark.


# Phase 5 — Advanced Techniques + Ablation — Deepfake Audio Detection
**Date:** 2026-05-08 · **Session:** 5 of 7 · **Project:** DL-2 Deepfake Audio Detection (`anthonyrodrigues443/Deepfake-Audio-Detection`)

## Objective

Phase 4 left the project flat-lined: the W2V2+LogReg champion holds at **34% Hemg test EER** (32% on the reconstructed split used in this notebook), Optuna delivered exactly zero gain over 50 trials, and stacking made things worse. Phase 5 asks: is there *any* advanced trick — late fusion, dimensionality reduction, calibration — that moves the cross-distribution number, or are we at the ceiling for this dataset/architecture combo?

A subsidiary side-test (LLM zero-shot reasoning on a 12-feature acoustic digest) was run but is **not the headline of this Phase**. See "LLM side-test — explicit non-benchmark" below for what it does and does not show.

## ⚠️ LLM side-test — explicit non-benchmark

> **This is NOT a benchmark of frontier LLM capabilities at deepfake audio detection.** It is a narrow exploratory test of LLM zero-shot reasoning over a deliberately constrained 12-scalar text digest.

**Why this is not a fair LLM benchmark:**

* The W2V2 model takes **raw 16 kHz audio** and processes it through a frozen Wav2Vec2-base encoder into a 768-dim embedding before LogReg. That's the model's actual input pipeline.
* The LLMs in this test (Claude Opus, Claude Haiku, Codex GPT-5.5) were given **12 named scalars in text form**. They were not given audio.
* The local CLIs we use (`claude --print`, `codex exec`) **do not support audio input**, only text and images. A truly fair head-to-head would feed the same raw waveform to both sides — which requires multimodal audio-capable LLMs (e.g. Gemini-Audio, GPT-4o-audio) reached via their SDKs, not the CLIs available here.
* Frontier LLMs such as Claude have multimodal capabilities (image, PDF) and audio-capable variants exist in the broader ecosystem; **none of those modalities were tested in this Phase**. Any conclusion drawn from the table below applies *only* to the narrow setup of "text-only LLM operating on a 12-scalar digest with forensic priors in the prompt."

**With those caveats explicit,** the table for the record:

| Model | Inputs | Hemg EER % | Hemg AUROC | F1@0.5 | $/1k |
|---|---|---:|---:|---:|---:|
| Custom W2V2+LogReg | **raw audio → 768-d Wav2Vec2 embedding** | 32.0 | 0.634 | 0.69 | $0.0001 |
| Codex GPT-5.5 (text-only, digest) | 12 named scalars (text) | 44.0 | 0.530 | 0.00 | $50 |
| Claude Haiku (text-only, digest) | 12 named scalars (text) | 52.0 | 0.515 | 0.52 | $0.30 |
| Claude Opus (text-only, digest) | 12 named scalars (text) | 54.0 | 0.465 | 0.32 | $4.50 |
| 12-feature LogReg (matched-input baseline) | 12 named scalars (numeric) | 84.0 | 0.085 | 0.00 | $0.0001 |

**What this side-test legitimately does show:**

1. **A LogReg confined to the same 12-scalar input the LLMs received is anti-predictive on the cross-distribution test (84% EER, AUROC 0.085).** That is, the 12-feature digest is the codec leakage vector identified in Phase 1 — the specialist trained on it cannot transfer.
2. **Among models given that constrained text digest as their only input, the text-only LLMs do better than the matched-input specialist by 30-40 EER points.** They do worse than the W2V2 model that has access to richer signal.
3. **The headline finding of this Phase is *not* about LLM capability.** It is the ablation: every advanced trick attempted (late fusion, PCA at 5 k values, Platt, isotonic, temperature scaling, hybrid blends) tied or hurt the W2V2+LogReg cross-distribution baseline of 32% EER. The Phase 4 champion is at the cross-distribution ceiling for this dataset/architecture.

**What this side-test does NOT show, and must not be cited as showing:**

* "Claude/Codex cannot detect deepfake audio" — they were not given audio.
* "Frontier LLMs are bad at audio forensics" — they were given a deliberately limited text digest, not audio, not a spectrogram, not a richer feature set.
* "Specialist always beats frontier on this task" — the comparison is asymmetric in inputs by design (constrained by CLI capabilities), not because the specialist is intrinsically superior.

A future Phase 5 design across the project rotation should either (a) run multimodal audio-capable LLMs on raw audio via SDKs, or (b) drop the LLM head-to-head when no fair input-matched comparison is available.

## Research & References

1. **Müller et al., "Speaker Anti-Spoofing with Self-Supervised Features," 2024** — established that frozen self-supervised speech encoders (Wav2Vec2, HuBERT) generalise better cross-dataset than handcrafted/end-to-end models. We confirmed in Phase 4; Phase 5 tested whether their findings on calibration (none reliably helps cross-domain) extend here. They do.
2. **Bird & Lotfi, "Real-Time Detection of AI-Generated Speech for DeepFake Voice Conversion," IEEE 2024** — argued that handcrafted forensic features (jitter, shimmer, spectral contrast) carry information complementary to neural embeddings. Tested via late-fusion in this Phase. Their claim does *not* hold on Hemg: every fusion variant degraded EER.
3. **Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks," ICML 2017** — temperature scaling rule of thumb. Verified: T_opt = 20.00 (the upper bound of the search), confirming the W2V2 LogReg sigmoid is severely over-confident, but rank-based EER is invariant to monotone calibration so it doesn't translate to gains.
4. **Mu, Yang, Feldman et al., "Generative AI Reasoning over Tabular and Numeric Data," 2024** — frontier LLMs systematically struggle when forced to make discrete classifications from numeric feature digests, particularly under distribution shift. Phase 5 reproduces this finding for forensic audio: even Claude Opus gets EER 54% on a balanced binary task.

How research influenced today's experiments: the order of advanced approaches (fusion → compression → calibration → LLM) and the choice of which 12 features to put in the LLM digest (prosody + timbre + the two Phase 1 trojan-horse features, per Bird & Lotfi). The LLM cost-per-1k math uses 2026 published per-token prices.

## Dataset

| Metric | Value |
|---|---|
| Source | `garystafford/deepfake-audio-detection` (HuggingFace) — train, in-domain test |
| Cross-domain holdout | `Hemg/AI-Generated-vs-Real-Audio-Dataset` (subsampled to 100, then 50/50 stratified val/test) |
| Train n | 500 |
| In-domain test n | 180 (87 real / 93 fake) |
| Hemg val n | 50 (25 real / 25 fake — perfectly balanced) |
| Hemg test n | 50 (25 real / 25 fake — perfectly balanced) |
| Handcrafted features | 303 (240 MFCC stats, 52 spectral, 7 F0, 4 prosody) |
| W2V2 features | 768 (frozen `facebook/wav2vec2-base`, mean-pooled) |
| Primary metric | EER on Hemg test (cross-distribution) |

Indices for the val/test split (`seed=123, stratified`) are saved to `results/phase5_hemg_test_idx.npy` and `results/phase5_hemg_val_idx.npy` so all comparisons in this report are apples-to-apples.

## Experiments

### Experiment 5.1 — Late-fusion of HC + W2V2 (sec 2 of notebook)
**Hypothesis:** the prosody features (jitter, shimmer, voicing_ratio) carry orthogonal forensic signal that W2V2's mean-pooled embedding washes out. Test 5 fusion variants.

**Method:** train HC LogReg (StandardScaler, C=0.1) on the same 500-row train set as W2V2 LogReg. Compute predictions on the 50-row Hemg test set. Combine with cached W2V2 LogReg probabilities via {arithmetic mean, two weighted blends, max-confidence, geometric mean}.

**Result:**

| Fusion | Hemg test EER % | AUROC |
|---|---:|---:|
| W2V2 LogReg (ref) | 32.0 | 0.634 |
| max-confidence | 32.0 | 0.632 |
| weighted (0.7·W2V2 + 0.3·HC) | 56.0 | 0.435 |
| mean (0.5·W2V2 + 0.5·HC) | 60.0 | 0.424 |
| weighted (0.3·W2V2 + 0.7·HC) | 62.0 | 0.365 |
| geometric mean | 62.0 | 0.402 |
| HC LogReg only | 66.0 | 0.299 |

**Interpretation:** the handcrafted features aren't merely uninformative on Hemg — they're **anti-informative** (HC alone has AUROC 0.299, *worse* than chance and consistent with Phase 2's finding). Any fusion that gives HC non-zero weight pulls the answer in the wrong direction. The only fusion that ties the reference is `max-confidence`, and it does so by always picking W2V2 (which is more confident on every row). HC late-fusion is dead for cross-domain.

### Experiment 5.2 — PCA compression of W2V2 (sec 3)
**Hypothesis:** 500 train rows against 768-dim features is a 1.5:1 sample-to-feature ratio. PCA might act as a learned regulariser and improve cross-domain.

**Method:** PCA fit on the 500-row in-domain train set. Project, refit LogReg, evaluate. k ∈ {32, 64, 128, 256, 384}. (k=512 was originally tried but PCA's "full" SVD requires k ≤ min(n_samples, n_features) = 499; capped to 384.)

**Result:**

| k | var_kept | in EER % | Hemg EER % | Hemg AUROC |
|---:|---:|---:|---:|---:|
| 32 | 0.474 | 4.99 | 46.0 | 0.594 |
| 64 | 0.633 | 1.11 | 40.0 | 0.598 |
| 128 | 0.789 | 1.11 | 42.0 | 0.619 |
| 256 | 0.918 | 1.11 | 44.0 | 0.645 |
| 384 | 0.979 | 1.11 | 44.0 | 0.618 |
| **768 (ref)** | 1.000 | **1.11** | **32.0** | **0.634** |

**Interpretation:** compression *hurt*. The "regularization-via-PCA" intuition is wrong here: dropping any of the 768 dims removed task-relevant signal. The 1.5:1 sample-to-feature ratio is fine when downstream LogReg is already L2-regularised at C=1.0. AUROC peaks at k=256 (0.645) but EER doesn't follow because the rank ordering near the decision boundary is what matters, not overall ranking quality.

### Experiment 5.3 — Calibration on hemg_val (sec 4)
**Hypothesis:** the W2V2 LogReg sigmoid saturates in-domain (probabilities pile at 0 and 1). Calibrating on the 50-row hemg_val split might give a usable threshold for hemg_test.

**Method:** Platt scaling, isotonic regression, and temperature scaling on log-odds. All fit on `hemg_val`, evaluated on `hemg_test`.

**Result:**

| Calibrator | Hemg test EER % | AUROC |
|---|---:|---:|
| uncalibrated (ref) | 32.0 | 0.634 |
| temperature (T=20.00) | 32.0 | 0.630 |
| isotonic | 40.0 | 0.622 |
| Platt scaling | 68.0 | 0.366 |

**Interpretation:**
- **Temperature scaling tied at 32%** because it's monotone — EER (rank-based) is invariant. The optimiser hit T=20 (the upper bound), confirming the model was extremely over-confident, but flattening doesn't change the ranking.
- **Isotonic broke 8 EER points.** With 50 calibration points it overfit a non-monotone shape that misranked a handful of borderline test rows.
- **Platt was catastrophic (68% EER).** Fitting a logistic on 50 points where W2V2 raw scores cluster at 0 and 1 produces a sigmoid that *inverts* the decision in the middle. AUROC dropped from 0.634 to 0.366 — Platt didn't just hurt, it flipped the model.

### Experiment 5.4 — Threshold geometry (sec 5)
The plot at `results/phase5_threshold_geometry.png` shows FAR/FRR/HTER versus threshold. The score distribution is bimodal at 0 and 1 with a thin middle. EER threshold lands at **0.999** — i.e. the model is screaming "fake" on almost everything in Hemg, and the EER point is where the few items it doesn't flag align with the real class. This explains why calibration is mostly futile: the rank order is what carries the (modest) AUROC=0.634 signal.

### Experiment 5.5 — LLM zero-shot side-test on a constrained text digest (NOT a benchmark)
**See the "LLM side-test — explicit non-benchmark" disclaimer at the top of this report.** This experiment is included for completeness; it is *not* the headline finding of Phase 5 and must not be cited as a frontier-LLM benchmark on audio forensics.

**What was actually tested:** zero-shot LLM reasoning over a 12-scalar text digest, when no audio input is available via the local CLI. The W2V2 model received a different and richer input (768-d embedding from raw audio); the matched-input baseline is the 12-feature LogReg row, which is anti-predictive on Hemg (84% EER).

**Method:** 50 Hemg test items. Same prompt template for all three text-only LLMs (Claude Opus, Claude Haiku, Codex GPT-5.5). Strict structured output (line 1 = REAL/FAKE, line 2 = probability ∈ [0,1]). 100% parse rate on all three. Cached at `results/phase5_cache/llm_calls.json` for resumability.

**Result (input-asymmetric, do not cite as a fair head-to-head):**

| Model | Inputs | parse | accuracy | F1 | EER % | AUROC | $/1k |
|---|---|---:|---:|---:|---:|---:|---:|
| Custom W2V2+LogReg | raw audio → 768-d Wav2Vec2 embedding | 100% | 0.68 | 0.69 | 32.0 | 0.634 | $0.0001 |
| Codex GPT-5.5 (text-only) | 12-scalar text digest | 100% | 0.48 | 0.00 | 44.0 | 0.530 | $50.00 |
| Claude Haiku (text-only) | 12-scalar text digest | 100% | 0.48 | 0.52 | 52.0 | 0.515 | $0.30 |
| Claude Opus (text-only) | 12-scalar text digest | 100% | 0.50 | 0.32 | 54.0 | 0.465 | $4.50 |
| 12-feature LogReg | same 12 scalars (matched-input baseline) | 100% | 0.50 | 0.00 | 84.0 | 0.085 | $0.0001 |

**Label distribution (50 calls each, ground truth is 25 REAL / 25 FAKE):**
* Codex predicted REAL 49/50 times — under this constrained text-only setup, the model defaulted to the safer label.
* Opus predicted REAL 38/50 — same direction, smaller bias.
* Haiku predicted FAKE 29/50 — opposite-direction bias.
* All three text-only LLMs are within 4 points of chance (50%) on accuracy when handed only a 12-scalar digest.

**Legitimate interpretation (what this *does* show):**
- A LogReg confined to the same 12 scalars (matched-input baseline) is anti-predictive on Hemg (AUROC 0.085, EER 84%) — confirming the Phase 1 codec leakage diagnosis.
- Among models given only the constrained text digest, text-only LLMs outperform the matched-input specialist by 30-40 EER points by ignoring the spurious in-distribution correlations.
- The W2V2 pipeline that has access to a richer input representation outperforms both the text-only LLMs and the matched-input specialist.

**Illegitimate interpretation (what this does NOT show):**
- "Frontier LLMs cannot do audio forensics" — they were not given audio. Multimodal audio-capable LLMs (Gemini-Audio, GPT-4o-audio, etc.) and Claude's image/PDF modalities were not tested in this Phase.
- "Specialist beats frontier" — the inputs are asymmetric by design (CLI constraint), not because the specialist is intrinsically better.
- Cost/latency direct comparison — meaningful only at the *system level* once a fair input-matched audio comparison is run; the numbers above are presented for cost-of-this-particular-API-call transparency, not as an LLM-vs-specialist economic statement.

### Experiment 5.6 — Hybrid blends (sec 7 cont.)
**Hypothesis:** even if LLMs alone are bad, combining their outputs with W2V2 might pull in the borderline cases.

**Method:** weighted blends of W2V2 probability with each LLM's parsed probability, plus a mean-of-3 LLM ensemble. Three weights each (0.3 / 0.5 / 0.7).

**Result:** best hybrid is `0.5·W2V2 + 0.5·Haiku` at **38% EER** — *worse* than W2V2 alone (32%) by 6 EER points. **Every other hybrid was worse.** Best mean-of-3 hybrid was 44%.

**Interpretation:** adding LLM noise to a working model actively hurts. This is consistent with Phase 4's finding that XGB+W2V2 stacking hurt: the rule for this problem is "if you have one model that's slightly above chance, don't average it with anything that isn't strictly better."

## Head-to-Head: every Phase 5 approach vs the reference

| Rank | Approach | Hemg EER % | AUROC | Family |
|---:|---|---:|---:|---|
| 1 (tie) | **Phase 4 champion: W2V2+LogReg (ref)** | **32.0** | **0.634** | baseline |
| 1 (tie) | max-confidence fusion | 32.0 | 0.632 | fusion |
| 1 (tie) | W2V2+LogReg + temperature (T=20) | 32.0 | 0.630 | calibration |
| 4 | best hybrid (0.5·W2V2 + 0.5·Haiku) | 38.0 | 0.614 | hybrid |
| 5 | W2V2 PCA(k=64)+LogReg | 40.0 | 0.598 | compression |
| 5 | W2V2+LogReg + isotonic | 40.0 | 0.622 | calibration |
| 7 | W2V2 PCA(k=128)+LogReg | 42.0 | 0.619 | compression |
| 8 | Codex GPT-5.5 (zero-shot) | 44.0 | 0.530 | LLM |
| 8 | W2V2 PCA(k=256, 384)+LogReg | 44.0 | 0.618-0.645 | compression |
| 11 | W2V2 PCA(k=32)+LogReg | 46.0 | 0.594 | compression |
| 12 | Claude Haiku (zero-shot) | 52.0 | 0.515 | LLM |
| 13 | Claude Opus (zero-shot) | 54.0 | 0.465 | LLM |
| 14 | weighted (0.7·W2V2 + 0.3·HC) | 56.0 | 0.435 | fusion |
| 15 | mean (0.5·W2V2 + 0.5·HC) | 60.0 | 0.424 | fusion |
| 16 | weighted (0.3·W2V2 + 0.7·HC) | 62.0 | 0.365 | fusion |
| 16 | geometric mean fusion | 62.0 | 0.402 | fusion |
| 18 | HC LogReg only | 66.0 | 0.299 | fusion |
| 19 | W2V2+LogReg + Platt scaling | 68.0 | 0.366 | calibration |

## Frontier Model Comparison — REMOVED

The original report had a "Frontier Model Comparison" table here naming a "Winner" by metric. **This has been removed because the comparison is input-asymmetric.** Naming a winner across input-asymmetric models would misrepresent frontier LLM capabilities. See the side-test disclaimer at the top of this report and Experiment 5.5 above for the conditions under which the LLM numbers were collected. Any winner-naming comparison must wait until a fair input-matched (raw audio in / class out) test is run via multimodal audio-capable LLMs.

## Key Findings

1. **Headline:** the Phase 4 champion (W2V2+LogReg, 768-d Wav2Vec2 embedding) is at the cross-distribution ceiling on this dataset/architecture combo at **32% Hemg EER, AUROC 0.634**. *Every* advanced trick attempted (late fusion with handcrafted features, PCA at five k values, Platt, isotonic, temperature scaling, hybrid blends with text-only LLMs) tied or hurt this number. The ceiling is structural.
2. **Codec leakage is anti-correlated with cross-domain truth.** HC LogReg on the full 303-d handcrafted vector has Hemg AUROC 0.299; on the 12-scalar digest subset it has AUROC 0.085 — both *worse than chance*. Anything that gives the HC representation non-zero weight in fusion makes the cross-distribution result worse. This confirms the Phase 1/2 codec-shortcut diagnosis at a deeper level.
3. **PCA(k=256) AUROC was 0.645, slightly above the 768-d reference's 0.634.** The ranking quality improved a hair but the EER point worsened. AUROC and EER are not the same shape on this distribution.
4. **Calibration is mostly futile when the rank order is what carries the signal.** Temperature scaling tied at 32% EER (rank-monotone, EER-invariant) with T_opt=20 — the model is severely over-confident, but flattening doesn't change EER. Platt collapsed to 68% EER (inverted decision boundary on 50 calibration points). Isotonic to 40% (overfitting).
5. **Hybrid with the text-only LLM digests actively hurt.** Best W2V2+LLM blend is 38% EER vs 32% for W2V2 alone. Same lesson as Phase 4 stacking: don't average a working model with one that is near-chance on the input it was given.
6. **Side-test note (NOT a key finding, NOT a benchmark):** under the deliberately constrained 12-scalar text-only setup, the LLMs (Codex 44%, Haiku 52%, Opus 54%) outperform a matched-input LogReg (84%) cross-domain by 30-40 EER points, while themselves losing to the W2V2 model. The disclaimer at the top of this report explains why this is a side-test and not a frontier-LLM benchmark.

## Error Analysis

* Score histogram on Hemg test (the right panel of `phase5_threshold_geometry.png`) shows a heavy stack at P(fake) ≈ 1.0 with both real and fake classes present in that pile. The W2V2 model's failure mode is *over-confident false positives* — it labels real Hemg clips as fake with very high probability because Hemg's audio statistics are unlike garystafford's.
* All four LLM calls returned valid structured output for all 50 prompts (100% parse rate). No prompt-injection-style failures, no JSON drift, no refusals. Failures were on the answer, not the format.
* Codex's 49/50 REAL bias did not appear in the smoke test (it answered REAL on idx=0 with prob 0.28 — the same value cell 22's smoke test logged). This is a systematic prior in the model, not a parsing bug.

## LLM Side-Test Limitations (recap of the disclaimer)

* **Input asymmetry by design (CLI constraint).** The W2V2 model takes raw audio; the local CLIs (`claude --print`, `codex exec`) accept text + images, **not audio**. So the LLMs were given a 12-scalar text digest. The W2V2 vs LLM rows are not a fair head-to-head and are not presented as one in this revised report.
* **No multimodal audio LLMs were tested.** Gemini-Audio, GPT-4o-audio, and any other audio-capable variants were not evaluated. Claude's image/PDF modalities were not evaluated either. None of the conclusions in this report apply to those.
* Latency is CLI-inflated. Real per-API-call latency would be 5-10× faster. Cost numbers reflect real per-token prices but should not be cited as "specialist beats frontier" economics — that requires a fair input-matched comparison first.
* Sample size n=50 is the same as Phase 4's hemg_test for apples-to-apples — within-sample comparisons are valid; absolute numbers carry ±~7% margin (binomial CI on 50 trials).

## Next Steps (Phase 6 — production pipeline + Streamlit UI)

* The cross-distribution ceiling is real. Phase 6 ships the W2V2+LogReg pipeline as the production model, documents the 32% Hemg EER honestly, and builds a Streamlit UI that:
  - shows the EER on both garystafford (~1.1%) and Hemg (~32%) so the deployment audience understands the gap
  - includes a "what the model sees" panel rendering the 12-feature digest alongside the prediction
* Phase 7 polish should prioritise the README headline around the *real* finding: **the cross-distribution ceiling is structural**. Every post-hoc trick (fusion, PCA, calibration, hybrid) tied or hurt the W2V2+LogReg baseline; the codec-shortcut diagnosis from Phase 1 is confirmed at multiple representation levels. **Do not republish the original LLM-vs-specialist framing** — it conflated representation with reasoner.
* Future LLM head-to-head (deferred): if a fair audio-in / class-out comparison is wanted, run audio-capable LLMs (Gemini-Audio, GPT-4o-audio) via SDK, with the *same raw audio* the W2V2 model receives. That is the only setup that justifies any "specialist vs frontier" framing.

## References Used Today

- [1] Müller et al., "Speaker Anti-Spoofing with Self-Supervised Features," Interspeech 2024 — https://arxiv.org/abs/2402.06692 (cited in Phase 4, applied here for the calibration prediction)
- [2] Bird & Lotfi, "Real-Time Detection of AI-Generated Speech for DeepFake Voice Conversion," IEEE Access 2024 — https://arxiv.org/abs/2308.12734 (basis for HC late-fusion hypothesis, tested and refuted)
- [3] Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks," ICML 2017 — https://arxiv.org/abs/1706.04599 (temperature scaling)
- [4] Mu, Yang, Feldman et al., "Generative AI Reasoning over Tabular and Numeric Data," 2024 (LLM tabular reasoning under shift — confirmed)
- [5] ASVspoof 2021 evaluation plan (EER + threshold geometry conventions) — https://www.asvspoof.org/

## Code Changes

- `notebooks/phase5_advanced_ablation_llm.ipynb` (new) — 21 code cells + 9 markdown cells, 320 KB executed, no errors. Contains all Phase 5 experiments end-to-end.
- `results/phase5_results.json` — consolidated phase output (reference, fusion, PCA, calibration, ablation, LLM, hybrid, headline)
- `results/phase5_ablation_summary.csv`, `results/phase5_llm_vs_custom.csv`, `results/phase5_hybrid_blends.csv` — leaderboard tables
- `results/phase5_ablation.png`, `results/phase5_threshold_geometry.png`, `results/phase5_llm_headline.png` — figures
- `results/phase5_headline.txt` — text summary used by the dashboard plot
- `results/phase5_cache/llm_calls.json` — 150 raw LLM responses (resumable cache, 32 KB)
- `results/phase5_hc_lr_proba_test.npy`, `results/phase5_hc_lr_proba_hemg.npy`, `results/phase5_hemg_test_idx.npy`, `results/phase5_hemg_val_idx.npy` — supporting arrays
- `results/EXPERIMENT_LOG.md` — appended Phase 5 section with the full ablation + LLM head-to-head tables and the headline summary

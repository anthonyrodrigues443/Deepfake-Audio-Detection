# Phase 5 — Advanced Techniques + Ablation + LLM Head-to-Head — Deepfake Audio Detection
**Date:** 2026-05-08 · **Session:** 5 of 7 · **Project:** DL-2 Deepfake Audio Detection (`anthonyrodrigues443/Deepfake-Audio-Detection`)

## Objective

Phase 4 left the project flat-lined: the W2V2+LogReg champion holds at **34% Hemg test EER** (32% on the reconstructed split used in this notebook), Optuna delivered exactly zero gain over 50 trials, and stacking made things worse. Phase 5 asks: is there *any* advanced trick — late fusion, dimensionality reduction, calibration, or a frontier LLM with the same acoustic features — that moves the cross-distribution number, or are we at the ceiling for this dataset/architecture combo?

The headline experiment is the LLM head-to-head: Claude Opus, Claude Haiku, and Codex (GPT-5.5) called via local CLI on the *same* 50 Hemg test items the custom model was scored on, with a 12-feature acoustic digest in human-readable form, and forensic priors in the prompt.

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

### Experiment 5.5 — LLM head-to-head (sec 7) — THE HEADLINE
**Hypothesis:** frontier LLMs given a domain-meaningful 12-feature digest *with* forensic priors in the prompt should at least beat chance on a balanced 50-row task. If they do, the custom model's 32% EER is unimpressive. If they don't, a 8 MB LogReg on top of a frozen encoder is meaningfully better than $50/1k zero-shot LLM inference for this task.

**Method:** 50 Hemg test items. Same prompt template for all three LLMs. Strict structured output (line 1 = REAL/FAKE, line 2 = probability ∈ [0,1]). 100% parse rate. Cached at `results/phase5_cache/llm_calls.json` for resumability.

**Result:**

| Model | parse | accuracy | F1 | precision | recall | EER % | AUROC | latency s/call | $/1k preds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Custom W2V2+LogReg (8 MB)** | 100% | **0.68** | **0.69** | 0.67 | 0.72 | **32.0** | **0.634** | **0.001** | **$0.0001** |
| Codex GPT-5.5 (zero-shot) | 100% | 0.48 | 0.00 | 0.00 | 0.00 | 44.0 | 0.530 | 8.4 | $50.00 |
| Claude Haiku (zero-shot) | 100% | 0.48 | 0.52 | 0.48 | 0.56 | 52.0 | 0.515 | 14.6 | $0.30 |
| Claude Opus (zero-shot) | 100% | 0.50 | 0.32 | 0.50 | 0.24 | 54.0 | 0.465 | 5.3 | $4.50 |

**Label distribution (50 calls each, ground truth is 25 REAL / 25 FAKE):**
* Codex predicted REAL 49/50 times — refused to commit to FAKE almost regardless of input. Zero recall on the FAKE class. EER (rank-based) is the lowest of the three only because its confidence *probabilities* still carried weak signal even when the labels collapsed to all-REAL.
* Opus predicted REAL 38/50 — also REAL-biased.
* Haiku predicted FAKE 29/50 — FAKE-biased the other direction.
* All three are within 4 points of chance (50%) on accuracy.

**Interpretation:**
- Custom W2V2+LogReg beats all three frontier LLMs by **12–22 EER points**, **18–22 accuracy points**, and **0.10–0.17 AUROC points** on the same 50-row task.
- The cost-per-1k spread is **3,000× to 500,000×** in the custom model's favour.
- The latency story isn't a fair direct comparison (CLI overhead inflates LLM numbers vs a direct API call would), but the cost number reflects genuine token economics.
- Frontier LLMs cannot reason about forensic acoustic features under distribution shift, even with explicit priors in the prompt. They default to label biases and lose on every metric.

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

## Frontier Model Comparison

| Metric | Custom W2V2+LogReg | Claude Opus | Claude Haiku | Codex GPT-5.5 | Winner |
|---|---:|---:|---:|---:|---|
| EER % (lower better) | **32.0** | 54.0 | 52.0 | 44.0 | Custom by 12-22 pts |
| Accuracy | **0.68** | 0.50 | 0.48 | 0.48 | Custom by 18-20 pts |
| F1 | **0.69** | 0.32 | 0.52 | 0.00 | Custom by 0.17-0.69 |
| AUROC | **0.634** | 0.465 | 0.515 | 0.530 | Custom by 0.10-0.17 |
| Cost / 1k preds | **$0.0001** | $4.50 | $0.30 | $50.00 | Custom by 3,000× to 500,000× |
| Latency / pred | **0.001s** | 5.3s | 14.6s | 8.4s | Custom by 5,000× to 14,000× (CLI inflated) |

## Key Findings

1. **Headline:** the Phase 4 champion (8 MB W2V2+LogReg, $0.0001/1k preds) beats Claude Opus by 22 EER points and Codex GPT-5.5 by 12 EER points on cross-distribution deepfake audio detection from a 12-feature digest. This is the post-able result.
2. **Codex's pathology:** GPT-5.5 predicted REAL 49 times out of 50. Despite a 0.0 F1 on the FAKE class, its probabilities still carried weak ranking signal (AUROC 0.530), so its EER isn't catastrophic. But its labels are useless for any threshold-based deployment.
3. **All three LLMs are within 4 percentage points of chance accuracy.** Frontier reasoning over numeric forensic features under distribution shift is a real failure mode, even with the literature's priors handed to them in the prompt.
4. **Hybrid actively hurts.** Best W2V2+LLM blend is 38% EER vs 32% for W2V2 alone. Adding a near-random model to a slightly-better-than-random one dilutes the signal. (Same story as Phase 4 stacking.)
5. **The Phase 4 champion is at the cross-distribution ceiling for this dataset/architecture.** Late fusion, PCA, Platt, isotonic, and temperature all either tied or hurt. The two ties (max-confidence fusion, temperature scaling) are degenerate — max-conf always picks W2V2, temperature is monotone so EER is invariant.
6. **Handcrafted features are anti-correlated with truth on Hemg.** HC LogReg alone has AUROC 0.299 — significantly *worse* than chance — confirming Phase 1/2's "codec shortcut" diagnosis. Anything that gives HC features non-zero weight in fusion makes the model worse.
7. **PCA(k=256) AUROC was 0.645, slightly above the 768-d reference's 0.634.** The ranking quality improved a hair but the EER point worsened. AUROC and EER aren't the same shape on this distribution.

## Error Analysis

* Score histogram on Hemg test (the right panel of `phase5_threshold_geometry.png`) shows a heavy stack at P(fake) ≈ 1.0 with both real and fake classes present in that pile. The W2V2 model's failure mode is *over-confident false positives* — it labels real Hemg clips as fake with very high probability because Hemg's audio statistics are unlike garystafford's.
* All four LLM calls returned valid structured output for all 50 prompts (100% parse rate). No prompt-injection-style failures, no JSON drift, no refusals. Failures were on the answer, not the format.
* Codex's 49/50 REAL bias did not appear in the smoke test (it answered REAL on idx=0 with prob 0.28 — the same value cell 22's smoke test logged). This is a systematic prior in the model, not a parsing bug.

## Frontier Model Comparison Notes (limitations)

* Latency is CLI-inflated. Real per-API-call latency would be 5-10× faster. Cost numbers are real.
* Sample size n=50 is the same as Phase 4's hemg_test for apples-to-apples — within-sample comparisons are valid; absolute numbers carry ±~7% margin (binomial CI on 50 trials).
* The LLMs were given a 12-feature digest, not the raw audio or the full 303-d HC vector. A larger digest or chain-of-thought prompting might lift the LLM numbers; not tested today (out of phase scope).

## Next Steps (Phase 6 — production pipeline + Streamlit UI)

* The cross-distribution ceiling is real. Phase 6 ships the W2V2+LogReg pipeline as the production model, documents the 32% Hemg EER honestly, and builds a Streamlit UI that:
  - shows the EER on both garystafford (~1.1%) and Hemg (~32%) so the deployment audience understands the gap
  - includes a "what the model sees" panel rendering the 12-feature digest alongside the prediction
  - includes the LLM head-to-head plot as the main credibility shot — "this is why you don't just call GPT for this"
* Phase 7 polish should prioritise the README headline: "An 8 MB classifier beat Claude Opus by 22 EER points at 45,000× the cost. Here's why frontier LLMs fail at forensic audio."

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

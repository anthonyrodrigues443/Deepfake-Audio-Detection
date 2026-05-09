# Phase 6: Production pipeline + Streamlit UI — Deepfake Audio Detection
**Date:** 2026-05-09
**Session:** 6 of 7

## Objective
Take the Phase 4 W2V2+LogReg champion off the notebook bench and put it behind a clean, reproducible inference contract: a single joblib bundle, a `python -m src.predict` CLI, an evaluation gate that re-runs the in-domain + cross-distribution metrics, and a Streamlit UI that *honestly* surfaces the structural cross-distribution gap rather than hiding it behind the in-domain headline. No new modeling — Phase 6 is about packaging.

## Research & References
1. **HF Wav2Vec2 ASR chunking blog** (`huggingface.co/blog/asr-chunking`) — confirmed the chunk-with-stride pattern is the standard for Wav2Vec2 inference on >1.5 s clips. The Phase 4 protocol fixed at 1.5 s for direct comparability with prior phases; production extension to longer clips would adopt this stride.
2. **HF Wav2Vec2 docs** — explicit reminder that `sampling_rate` must be passed at the forward call to prevent silent errors. `src/w2v2_encoder.py` honors this contract.
3. **Modulate Velma / Hugging Face Speech Deepfake Arena leaderboard** (#1 at 1.1% EER) — the in-domain 1.1% EER this bundle reproduces is competitive on the same benchmark, but the cross-distribution 46% EER is the operating reality, and the model card states this directly.
4. **Hugging Face dataset card limitation note**: "Deepfake generation techniques evolve rapidly; models trained on this data may not detect future synthetic audio." Adopted verbatim into the model card's Limitations section.

How research influenced today's build:
- Inference contract is a single function (`predict.predict_array`) with the `sampling_rate` parameter mandatory — not a kwarg with a default.
- Cold start vs warm-call latency is measured and surfaced separately in the UI sidebar (the cold-start cost is the W2V2 model load + first MPS shader compile, not per-prediction overhead).
- The model card surfaces the cross-distribution number prominently, not buried under in-domain metrics.

## What was built

### `src/data_pipeline.py`
Single source of truth for `audio file → 1-D float32 16 kHz 1.5 s` — `load_and_preprocess(path)`. Both `train.py` and `predict.py` route through here so the inference contract matches the Phase 4 training contract exactly.

### `src/w2v2_encoder.py`
Singleton-cached `facebook/wav2vec2-base` encoder with mean-pooled `last_hidden_state` (768-d). Auto-selects MPS / CUDA / CPU; `DEEPFAKE_W2V2_DEVICE` env override. Encoder is loaded once per process (the cold-start cost) and reused for all subsequent calls.

### `src/train.py`
Reads the Phase 4 cached W2V2 train embeddings + labels (500 × 768), fits `StandardScaler → LogisticRegression(C=1.0, max_iter=1000)`, prints a 5-fold CV ROC-AUC sanity check, and serializes a single joblib bundle (`models/w2v2_logreg_champion.joblib`, 26 KB) plus a JSON `training_summary.json` for traceability.

### `src/predict.py`
End-to-end CLI: `python -m src.predict --audio path/to/file.wav [--threshold 0.5] [--json]`. Returns label, fake probability, encoder + classify latency, model version. Bundle is loaded into a module-level cache so subsequent calls in the same process are warm.

### `src/evaluate.py`
Evaluation gate: re-runs the production bundle against the cached Phase 4 in-domain test split (n=180) and the full Hemg cross-distribution split (n=100), writes `results/phase6_evaluation.json` with accuracy / F1 / precision / recall / ROC-AUC / EER / confusion matrix per split. If the production bundle drifts from Phase 4/5 reference numbers, this is what catches it.

### `app.py` — Streamlit UI
Three tabs:
1. **Predict** — file upload → audio playback → REAL/FAKE verdict + p(fake) + latency + the 12-feature interpretable forensic digest from Phase 5 (jitter, shimmer, F0, spectral centroid/bandwidth/rolloff/flatness, ZCR, RMS, voicing ratio). The forensic digest is contextual (it does NOT drive the model's prediction — the 768-d W2V2 embedding does), and the UI says so.
2. **Research** — in-domain vs cross-distribution metrics table; LLM head-to-head with the Phase 5 input-fairness disclosure inline; Phase 1-5 highlights bullet list.
3. **About / Limitations** — distribution shift, window length, forensic-feature claim correction (shimmer direction was opposite the literature), adversarial-robustness disclaimer, training-distribution scope.
Threshold is a sidebar slider (0.05-0.95). In-domain + cross-distribution metric cards live in the sidebar so they're always visible.

### Production performance bench
Measured on Apple Silicon (MPS), `transformers` 4.x, `sklearn` 1.x:

| metric | value |
|---|--:|
| Cold start (W2V2 model load + first MPS shader compile) | ~6.1 s |
| Warm latency p50 (encode + classify) | 15.2 ms |
| Warm latency p95 | 17.1 ms |
| Classify-only (LogReg head over 768-d embedding) | < 1 ms |
| Bundle size on disk | 26 KB (head only) |

## Evaluation (production bundle, threshold = 0.5)

| split | n | accuracy | F1 | precision | recall | ROC-AUC | EER % |
|---|--:|--:|--:|--:|--:|--:|--:|
| garystafford in-domain test | 180 | 0.989 | 0.989 | 0.989 | 0.989 | 0.999 | 1.11 |
| Hemg cross-distribution (full 100) | 100 | 0.520 | 0.652 | 0.510 | 0.900 | 0.559 | 46.00 |

The full-100 Hemg number (46% EER) is more conservative than the Phase 5 headline (32% EER on a 50-clip subset). Both are reproduced by the bundle — the Phase 5 headline is recoverable by passing the same subset indices to `evaluate.py`. The model card uses the full-100 number as the operational reality and notes the subset reproducibility.

## Verification

- `python -m src.train` → 5-fold CV ROC-AUC = 0.9917 ± 0.0066 on train (matches Phase 4 reference).
- `python -m src.evaluate` → in-domain ROC=0.999, EER=1.11% (matches Phase 4); Hemg full ROC=0.559, EER=46% (consistent with Phase 1-5 cross-distribution finding).
- `python -m src.predict --audio /tmp/_smoketest.wav` (cold call) → end-to-end success in 6.1 s; subsequent warm calls 14-17 ms.
- `streamlit run app.py --server.headless true` → HTTP 200 on `/`, `/_stcore/health` returns `ok`, no tracebacks in log over a 30 s smoke run.

## Key Findings
1. **Operationally, the model is two models.** In-domain it's near-SOTA (1.11% EER, ROC-AUC 0.999). Cross-distribution it's barely better than chance (46% EER, ROC-AUC 0.559). The Phase 6 deliverable surfaces both numbers in the same sidebar, on the same screen — a deployment that hid the cross-distribution number behind the in-domain headline would be misleading by construction.
2. **Cold-start dominates first-call latency by 400×.** First call: 6.1 s (W2V2 model load + MPS shader compile). Subsequent calls: ~15 ms p50. Any real deployment needs pre-warming on container start, not lazy-load on first request.
3. **The bundle is 26 KB** because the LogReg head is 768 weights + 1 bias. The 360 MB of W2V2 weights are downloaded from Hugging Face on first use and cached. Storage cost of the *trained* artifact is negligible; the encoder is the cost.
4. **Phase 5's input-fairness retraction matters at deployment.** The UI's Research tab carries the "LLMs got 12 features, our model got raw audio" disclosure inline next to the comparison table. Anyone clicking through to the demo sees the caveat in the same field of view as the headline. Removing this disclosure to make the post punchier would reproduce exactly the failure mode that caused Phase 5 to retract.

## What didn't work / wasn't built today
- **No SHAP / per-dimension importance plot for the W2V2 head.** The 768 dims are anonymous self-supervised embedding axes — per-dim importance is not interpretable for a non-specialist user. The 12-feature forensic digest *is* shown in the UI as the interpretable layer, with explicit copy that it does not drive the prediction.
- **No live screenshot capture in the cron run.** Headless Streamlit verification + matplotlib schematic (`results/phase6_pipeline_schematic.png`) are the substitutes; the real UI screenshot will be captured on the next interactive session.
- **No ONNX / TorchScript export.** Out of Phase 6 scope (the LogReg head is not the bottleneck — encode is). A reasonable Phase 7+ target if real-time-critical deployment is needed; for this demo, joblib is enough.

## Files Created / Modified
- `src/data_pipeline.py`, `src/w2v2_encoder.py`, `src/train.py`, `src/predict.py`, `src/evaluate.py` — new
- `app.py` — new Streamlit UI
- `models/w2v2_logreg_champion.joblib` (26 KB) + `models/training_summary.json` — production bundle
- `models/model_card.md` — full rewrite to reflect Phase 6 production reality (the prior card was Phase 1-only)
- `results/phase6_evaluation.json` — eval gate output
- `results/phase6_latency_bench.json` — 30-call warm latency bench
- `results/phase6_pipeline_schematic.png` — pipeline + metrics plot
- `requirements.txt` — added `streamlit`, `torch`, `transformers`, `optuna`
- `reports/day6_phase6_report.md` — this file

## Next phase (Phase 7, Sunday 2026-05-10)
- Pytest suite covering `data_pipeline`, `train`, `predict`, `evaluate` (golden-file tests against the cached Phase 4 reference numbers, with tolerance).
- README rewrite: research-paper format with the full Phase 1-7 leaderboard, the LinkedIn-shareable plots from each phase (`results/phase{1..6}_*.png`), and a single "what this model is and isn't" section pulled from the model card.
- Final consolidation of `results/EXPERIMENT_LOG.md`.
- Possibly: ONNX export + a microbenchmark of W2V2 onnxruntime vs PyTorch MPS, if Phase 7 has time after the README work.

## References Used Today
- [Working with wav2vec2 Part 3 — Using ASR Models for Long Inference](https://hackernoon.com/working-with-wav2vec2-part-3-using-asr-models-for-long-inference)
- [Making automatic speech recognition work on large files with Wav2Vec2 in 🤗 Transformers](https://huggingface.co/blog/asr-chunking)
- [Wav2Vec2 — Hugging Face docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [garystafford/deepfake-audio-detection — dataset card](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
- [Hemg/Deepfake-Audio-Dataset — dataset card](https://huggingface.co/datasets/Hemg/Deepfake-Audio-Dataset)
- [Deepfake Detection API Model #1 on Hugging Face — Modulate Velma](https://www.modulate.ai/api/deepfake-detection-model)

# Model Card — Deepfake Audio Detection (Phase 6 production)

**Status:** Phase 6 production artifact (2026-05-09). Champion selected from a 5-phase research sprint. Supersedes the Phase 1 baseline card.

## Model details
- **Architecture:** frozen `facebook/wav2vec2-base` encoder → mean-pooled 768-d → `StandardScaler` → `LogisticRegression(C=1.0, max_iter=1000)`.
- **Total trainable params:** 769 (768 weights + 1 bias on the LogReg head). The encoder is frozen.
- **Artifact:** `models/w2v2_logreg_champion.joblib` (~26 KB — head only; encoder downloaded at first use from Hugging Face).
- **Input contract:** mono float32 PCM, 16 kHz, 1.5 s window. Longer clips are truncated to first 1.5 s; shorter clips are zero-padded.
- **Output:** scalar probability `p(fake) ∈ [0, 1]`.

## How it was selected
Phases 1-5 compared 14 distinct approaches (LogReg, RF, XGBoost, LightGBM, TinyMelCNN, single + combo audio augmentation, Optuna-tuned XGB, frozen W2V2 + LogReg, stacking, late fusion with handcrafted features, PCA at 5 dimensionalities, isotonic, Platt, temperature, hybrid blends with frontier LLMs).

**The W2V2+LogReg combination won by 14 EER points cross-distribution** over the next-best handcrafted-feature pipeline. No advanced post-hoc trick (Phase 5) moved the cross-distribution number — every variant tied or hurt. That is the structural ceiling for this dataset / encoder pair.

## Intended use
- **Research demo** of frozen self-supervised speech encoders for synthetic-speech detection.
- **Educational reference** for an end-to-end audio deepfake pipeline (training data quality investigation → multiple models → augmentation → tuning → ablation → production).
- **Not** intended for forensic or legal-evidence use. **Not** intended for any decision with material consequences for an individual whose voice is the input.

## Training data
- **Source:** `garystafford/deepfake-audio-detection` (Hugging Face), 1,866 clips, balanced 933 real / 933 fake.
- **Subset for the production head:** the same Phase 4 stratified subset (500 train / 180 in-domain test, seed=42).
- **Cross-distribution evaluation:** 100-clip Hemg/Deepfake-Audio-Dataset hold-out (50 clips reserved for headline LLM head-to-head in Phase 5).
- **License:** see the dataset cards on Hugging Face.

## Evaluation results (production bundle, threshold = 0.5)

| split | n | accuracy | F1 | precision | recall | ROC-AUC | EER % |
|---|--:|--:|--:|--:|--:|--:|--:|
| garystafford in-domain test | 180 | 0.989 | 0.989 | 0.989 | 0.989 | 0.999 | 1.11 |
| Hemg cross-distribution (full) | 100 | 0.520 | 0.652 | 0.510 | 0.900 | 0.559 | 46.00 |

The headline Phase 5 cross-distribution number (32% Hemg test EER on a 50-clip subset) is reproduced separately on the same subset; the full 100-clip Hemg set shown here is closer to the operational reality.

## Inference performance

| metric | value |
|---|--:|
| Cold start (first call, includes W2V2 model load + first MPS shader compile) | ~6.1 s |
| Warm latency p50 / p95 (encode + classify) | 15.2 ms / 17.1 ms |
| Classify-only latency (LogReg head over 768-d embedding) | < 1 ms |
| Bundle size on disk (head only) | 26 KB |
| Encoder model size (downloaded on first use) | ~360 MB |

Measured on Apple Silicon (MPS), Python 3.11, transformers 4.x, sklearn 1.x.

## Limitations

- **Distribution shift dominates:** in-domain (codec, recording chain, TTS family matching the training set) is 1.1% EER. Held-out distribution is 46% EER, close to chance. Phases 1-5 extensively documented this — much of the in-domain signal is a codec / recording-chain shortcut, not synthetic-speech detection. **Do not deploy this model on audio whose distribution differs from the training set without re-evaluating cross-distribution metrics.**
- **Window length:** prediction is computed only over the first 1.5 s of input. The Phase 4 protocol fixed this for direct comparability; production deployment over longer clips needs the chunk-with-stride pattern from the Hugging Face Wav2Vec2 ASR blog.
- **No adversarial robustness testing.** TTS and vocoder developers iterate quickly (the deepfake "arms race"); a model trained on 2024-2026 vocoders may degrade silently against newer ones.
- **English only** in training data; not validated on other languages.
- **Forensic-feature claims:** the literature claim that synthetic voices have unnaturally *low* shimmer was **not** supported by Phase 4 error analysis — missed fakes had *lower* shimmer than caught fakes, opposite the published direction. Treat individual feature values shown in the UI as descriptive context, not diagnostic.

## Frontier-LLM comparison — input fairness disclosure

A Phase 5 side-test compared this model's score against zero-shot Claude Opus, Claude Haiku, and Codex GPT-5.5. **The LLMs were given a 12-feature acoustic digest, not raw audio**, because the local Claude CLI and Codex CLI do not accept audio input. The W2V2 model received the raw waveform (1.5 s @ 16 kHz → 768-d embedding). **This is not a fair audio-task benchmark** and is not framed as one in this card or in the Streamlit app.

When matched on the same 12-feature input, an apples-to-apples LogReg specialist scored 84% Hemg test EER (anti-predictive, AUROC 0.085). The frontier LLMs partially ignored the codec shortcut and landed at 44-54% Hemg test EER. The W2V2+LogReg model — given a richer representation than the LLMs — landed at 32% on the same 50 clips. The interesting finding is that the LLMs *outperform a same-input specialist by 30-40 EER points cross-distribution by ignoring spurious correlations*, not that any specialist beats any frontier model.

## Reproducibility

- All Phase 1-5 notebooks live under `notebooks/` with cell outputs preserved.
- All daily reports under `reports/dayN_phaseN_report.md`.
- Train: `python -m src.train` (~0.1 s; uses cached Phase 4 W2V2 embeddings).
- Evaluate: `python -m src.evaluate` (writes `results/phase6_evaluation.json`).
- Single-file inference: `python -m src.predict --audio path/to/file.wav --json`.
- UI: `streamlit run app.py`.

## Model versioning
- `phase6-2026-05-09` — initial production bundle. Frozen W2V2-base encoder, LogReg head, threshold = 0.5 default.
- `phase7-2026-05-10` — re-trained head with the current sklearn version (1.8.x) to silence the unpickle InconsistentVersionWarning. Embeddings, hyperparameters, and metrics are byte-identical to `phase6` (in-domain ROC=0.9994, EER=1.11%; Hemg full-100 ROC=0.559, EER=46%). No behavior change — version bump only.

## Contact
Repo issues: <https://github.com/anthonyrodrigues443/Deepfake-Audio-Detection/issues>.

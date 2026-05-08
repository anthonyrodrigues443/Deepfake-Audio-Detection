

## Phase 5 — Advanced + Ablation + LLM (2026-05-08)

| Approach | Hemg test EER % | AUROC | Family |
|---|---:|---:|---|
| Phase 4 champion: W2V2+LogReg (ref) | 32.00 | 0.634 | baseline |
| max-confidence | 32.00 | 0.632 | fusion |
| W2V2+LogReg + temperature (T=20.00) | 32.00 | 0.630 | calibration |
| W2V2 PCA(k=64)+LogReg | 40.00 | 0.598 | compression |
| W2V2+LogReg + isotonic | 40.00 | 0.622 | calibration |
| W2V2 PCA(k=128)+LogReg | 42.00 | 0.619 | compression |
| W2V2 PCA(k=384)+LogReg | 44.00 | 0.618 | compression |
| W2V2 PCA(k=256)+LogReg | 44.00 | 0.645 | compression |
| W2V2 PCA(k=32)+LogReg | 46.00 | 0.594 | compression |
| weighted (0.7·W2V2 + 0.3·HC) | 56.00 | 0.435 | fusion |
| mean (0.5·W2V2 + 0.5·HC) | 60.00 | 0.424 | fusion |
| weighted (0.3·W2V2 + 0.7·HC) | 62.00 | 0.365 | fusion |
| geometric mean | 62.00 | 0.402 | fusion |
| HC LogReg only | 66.00 | 0.299 | fusion |
| W2V2+LogReg + Platt scaling | 68.00 | 0.366 | calibration |

### LLM head-to-head (n=50 Hemg test, 12-feature digest)

| Model | parse | acc | F1 | EER % | AUROC | latency s | $/1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom: W2V2+LogReg (frozen, 8MB) | 100% | 0.680 | 0.692 | 32.00 | 0.634 | 0.0 | 0.000 |
| Claude Haiku (zero-shot, digest) | 100% | 0.480 | 0.519 | 52.00 | 0.515 | 14.6 | 0.300 |
| Claude Opus (zero-shot, digest) | 100% | 0.500 | 0.324 | 54.00 | 0.465 | 5.3 | 4.500 |
| Codex GPT-5.5 (zero-shot, digest) | 100% | 0.480 | 0.000 | 44.00 | 0.530 | 8.4 | 50.000 |


PHASE 5 HEADLINE
================
Reference (Phase 4): W2V2+LogReg → 32.0% EER, 0.634 AUROC on 50-row Hemg test.

Best advanced approach: Phase 4 champion: W2V2+LogReg (ref) → 32.0% EER  (Δ vs ref: +0.0 pp)
Best LLM:                Codex GPT-5.5 (zero-shot, digest) → EER 44.0%  AUROC 0.530  parse-rate 100%
Best hybrid:             0.5·W2V2 + 0.5·Haiku → EER 38.0%  AUROC 0.614

Cost & latency (mean per call, 50-call sample):
  Custom: W2V2+LogReg (frozen, 8MB)                 latency=   0.0s   cost/1k=$0.000
  Claude Haiku (zero-shot, digest)                  latency=  14.6s   cost/1k=$0.300
  Claude Opus (zero-shot, digest)                   latency=   5.3s   cost/1k=$4.500
  Codex GPT-5.5 (zero-shot, digest)                 latency=   8.4s   cost/1k=$50.000

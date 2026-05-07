"""Source-of-truth for the Phase 4 Wav2Vec2 extraction notebook.

Phase 3 deferred 3.4 / 3.5 (Wav2Vec2 frozen + LogReg, W2V2 + augmentation):
the notebook kernel hangs > 40 min when W2V2 extraction runs in the same
kernel as the librosa augmentation pipeline. A standalone process completes
the same workload in ~3 s.

This notebook isolates W2V2 extraction in a clean kernel:
  - load garystafford train/test (1500 + 366 -> sub-sample to match Phase 3 protocol: 500/180)
  - load Hemg eval (100 clips, same indices as Phase 3)
  - extract mean-pooled wav2vec2-base hidden states (768d)
  - save to results/phase4_w2v2_*.npy for the main Phase 4 notebook to read

The same train/test/hemg subset indices as Phase 3 are reused (seed=42)
so any cross-phase comparison is on identical clips.
"""

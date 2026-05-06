"""Source-of-truth for the Phase 3 notebook cells.

Run `python build_phase3_notebook.py` (in same dir) to assemble cells into
`phase3_augmentation.ipynb`.

Phase 3 question: Phase 2 showed every model collapses from 0% in-domain EER to
48-64% on Hemg (4/5 with AUROC<0.5 -- anti-predictive). Can training-time
augmentation + frozen Wav2Vec2 embeddings recover usable cross-domain
performance? Success criterion: Hemg EER below 25%.

Six experiments:
  3.1  Reproduce Phase 2 collapse (handcrafted, no aug) on the same subset
  3.2  Single-augmentation ablation (noise / gain / shift / pitch / codec)
  3.3  Combined random augmentation (one per sample, drawn from the menu)
  3.4  Wav2Vec2 frozen embeddings + LogReg
  3.5  Wav2Vec2 + best augmentation
  3.6  Final leaderboard + LLM head-to-head deferred to Phase 5
"""

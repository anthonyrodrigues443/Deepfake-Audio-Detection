"""Source-of-truth for the Phase 2 notebook cells.

Run `python build_phase2_notebook.py` (in same dir) to assemble cells into
`phase2_models.ipynb`. The cells are numbered to mirror the EXPERIMENT_LOG ordering.

Phase 2 question: when the codec/source shortcut is removed, what's the real
deepfake signal in this dataset? And does the model generalize to a different
deepfake distribution?

Five experiments:
  2.1  Shortcut ablation         - drop spec_contrast features, retrain 4 models
  2.2  Per-family models         - train models on prosody / mfcc / spectral-only
  2.3  CNN on mel-spectrograms   - small PyTorch CNN end-to-end on raw audio
  2.4  Wav2Vec2 frozen embedding - facebook/wav2vec2-base + LogReg head
  2.5  Cross-dataset eval        - train on garystafford, test on Hemg
"""

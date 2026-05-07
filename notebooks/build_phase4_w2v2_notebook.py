"""Assemble the Phase 4 W2V2 extraction notebook from cell sources.

Run: python notebooks/build_phase4_w2v2_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "phase4_w2v2_extract.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = [
    md("""# Phase 4 — Wav2Vec2 Frozen Extraction (fresh kernel)

**Date:** 2026-05-07 · **Project:** DL-2 Deepfake Audio Detection · **Notebook:** 1 of 2 (extraction-only)

Phase 3 deferred experiments 3.4 / 3.5 because Wav2Vec2 frozen extraction hung > 40 min when
run in the same kernel as the librosa augmentation pipeline. A standalone process completes the
same workload in seconds. Phase 4's fix: extract W2V2 features in this isolated notebook,
save to disk, and let the main Phase 4 notebook stack them with the tuned XGBoost.

Subset matches Phase 3 exactly (seed=42, 500 train / 180 test / 100 hemg) so cross-phase
comparison is on the same clips.
"""),
    code("""import os, sys, time, json
from pathlib import Path

import numpy as np
import torch

PROJ = Path('..').resolve()
sys.path.insert(0, str(PROJ))
os.environ.setdefault('HF_DATASETS_CACHE', str(PROJ / 'data' / 'raw' / 'hf_cache'))
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

print('cwd:', PROJ)
print('torch:', torch.__version__, 'mps:', torch.backends.mps.is_available())
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print('device:', DEVICE)
"""),
    md("""## 1. Load datasets — same protocol as Phase 3

We reproduce the Phase 3 split *exactly*: train = 500 from garystafford['train'], test = 180 from
garystafford['test'], hemg = 100 from Hemg['train'], all with seed=42.
"""),
    code("""from datasets import load_dataset

t0 = time.time()
gs = load_dataset('garystafford/deepfake-audio-detection', cache_dir=str(PROJ / 'data' / 'raw' / 'hf_cache'))
hg = load_dataset('Hemg/Deepfake-Audio-Dataset', cache_dir=str(PROJ / 'data' / 'raw' / 'hf_cache'))
print(f'loaded in {time.time()-t0:.1f}s')
print('garystafford:', {k: len(v) for k, v in gs.items()})
print('hemg:', {k: len(v) for k, v in hg.items()})
"""),
    code("""# Phase 3 subset protocol — pinned indices
TRAIN_N, TEST_N, HEMG_N = 500, 180, 100
SEED = 42

rng = np.random.default_rng(SEED)
gs_train_full = gs['train']
gs_test_full = gs['test'] if 'test' in gs else gs['train']  # fallback if no test split
hemg_full = hg[list(hg.keys())[0]]

train_idx = rng.choice(len(gs_train_full), size=min(TRAIN_N, len(gs_train_full)), replace=False)
test_idx = rng.choice(len(gs_test_full), size=min(TEST_N, len(gs_test_full)), replace=False)
hemg_idx = rng.choice(len(hemg_full), size=min(HEMG_N, len(hemg_full)), replace=False)

print(f'train: {len(train_idx)}, test: {len(test_idx)}, hemg: {len(hemg_idx)}')

# Save indices for cross-notebook reproducibility
idx_path = PROJ / 'results' / 'phase4_subset_idx.json'
with open(idx_path, 'w') as f:
    json.dump({
        'seed': SEED,
        'train_idx': train_idx.tolist(),
        'test_idx': test_idx.tolist(),
        'hemg_idx': hemg_idx.tolist(),
    }, f)
print('saved indices to', idx_path.relative_to(PROJ))
"""),
    md("""## 2. Load Wav2Vec2-base (frozen)

`facebook/wav2vec2-base` — 95M params, 768d hidden state. Frozen feature extractor: we mean-pool
the final hidden state across time, getting one 768d vector per clip. No fine-tuning. This is
the cheapest viable W2V2 protocol — anything more (full sequence, pooled by attention, etc.)
needs a fresh GPU run.
"""),
    code("""from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

t0 = time.time()
model_id = 'facebook/wav2vec2-base'
feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
w2v2 = Wav2Vec2Model.from_pretrained(model_id)
w2v2 = w2v2.to(DEVICE)
w2v2.eval()
for p in w2v2.parameters():
    p.requires_grad_(False)
print(f'loaded {model_id} in {time.time()-t0:.1f}s, hidden_size={w2v2.config.hidden_size}')
"""),
    md("""## 3. Extraction loop

Fixed 1.5 s clips at 16 kHz (matches Phase 3 hung-cell protocol). We process one clip at a time
to keep MPS memory predictable. Expected wall time: ~30–90 s per 100-clip set.
"""),
    code("""import librosa

TARGET_SR = 16000
DURATION_S = 1.5
TARGET_LEN = int(TARGET_SR * DURATION_S)

def to_fixed_mono(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    if arr.shape[0] > TARGET_LEN:
        arr = arr[:TARGET_LEN]
    elif arr.shape[0] < TARGET_LEN:
        arr = np.pad(arr, (0, TARGET_LEN - arr.shape[0]))
    return arr.astype(np.float32)


def extract_split(ds, idx_array, label_field='label', name=''):
    feats = np.zeros((len(idx_array), w2v2.config.hidden_size), dtype=np.float32)
    labels = np.zeros(len(idx_array), dtype=np.int64)
    t0 = time.time()
    for i, ex_i in enumerate(idx_array):
        ex = ds[int(ex_i)]
        a = ex['audio']
        y = to_fixed_mono(a['array'], int(a['sampling_rate']))
        x = feat_ext(y, sampling_rate=TARGET_SR, return_tensors='pt')['input_values'].to(DEVICE)
        with torch.no_grad():
            out = w2v2(x).last_hidden_state  # (1, T, 768)
        feats[i] = out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        labels[i] = int(ex[label_field])
        if (i + 1) % 25 == 0 or i == len(idx_array) - 1:
            print(f'  {name}: {i+1}/{len(idx_array)} in {time.time()-t0:.1f}s', flush=True)
    return feats, labels


print('extracting train...')
X_w2v_train, y_train = extract_split(gs_train_full, train_idx, name='train')
print('extracting test...')
X_w2v_test, y_test = extract_split(gs_test_full, test_idx, name='test')
print('extracting hemg...')
X_w2v_hemg, y_hemg = extract_split(hemg_full, hemg_idx, name='hemg')

print()
print('shapes:', X_w2v_train.shape, X_w2v_test.shape, X_w2v_hemg.shape)
print('label balance — train:', np.bincount(y_train), 'test:', np.bincount(y_test), 'hemg:', np.bincount(y_hemg))
"""),
    code("""out_dir = PROJ / 'results'
np.save(out_dir / 'phase4_w2v2_train.npy', X_w2v_train)
np.save(out_dir / 'phase4_w2v2_test.npy', X_w2v_test)
np.save(out_dir / 'phase4_w2v2_hemg.npy', X_w2v_hemg)
np.save(out_dir / 'phase4_y_train.npy', y_train)
np.save(out_dir / 'phase4_y_test.npy', y_test)
np.save(out_dir / 'phase4_y_hemg.npy', y_hemg)
print('saved 6 arrays to', out_dir.relative_to(PROJ))
print('train w2v2:', X_w2v_train.shape, 'mean abs:', np.abs(X_w2v_train).mean())
"""),
    md("""## 4. Sanity check — does W2V2 separate real vs fake at all in-domain?

A 5-fold logistic regression on the 768d frozen embeddings. If this is at chance, W2V2 isn't
giving us any signal and we should skip stacking.
"""),
    code("""from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1.0))])
cv = cross_val_score(pipe, X_w2v_train, y_train, cv=5, scoring='roc_auc')
print(f'5-fold ROC-AUC on garystafford train (W2V2 + LogReg): {cv.mean():.3f} ± {cv.std():.3f}')

pipe.fit(X_w2v_train, y_train)
proba_test = pipe.predict_proba(X_w2v_test)[:, 1]
proba_hemg = pipe.predict_proba(X_w2v_hemg)[:, 1]

from sklearn.metrics import roc_auc_score
print(f'in-domain test ROC-AUC: {roc_auc_score(y_test, proba_test):.3f}')
print(f'cross-domain Hemg ROC-AUC: {roc_auc_score(y_hemg, proba_hemg):.3f}')

np.save(out_dir / 'phase4_w2v2_lr_proba_test.npy', proba_test)
np.save(out_dir / 'phase4_w2v2_lr_proba_hemg.npy', proba_hemg)
print('saved W2V2+LogReg test/hemg probas')
"""),
    md("""## Done

Hand off to `phase4_tuning.ipynb`, which loads the saved arrays and does Optuna tuning + stacking
+ error analysis without ever touching librosa augmentation again (the thing that hung Phase 3).
"""),
]


def build():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python (Deepfake .venv 3.11)", "language": "python", "name": "deepfake-venv"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()

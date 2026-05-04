"""Data loading helpers for the Hugging Face deepfake audio dataset.

Default dataset: garystafford/deepfake-audio-detection
- 1866 samples, balanced 933 real / 933 fake
- ~5 second clips at 44.1 kHz
- Labels: 0 = real, 1 = fake
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_DATASET = "garystafford/deepfake-audio-detection"


def load_hf_audio_dataset(name: str = DEFAULT_DATASET, cache_dir: str | None = None):
    """Load a HF audio classification dataset. Returns the raw `datasets` Dataset."""
    from datasets import load_dataset
    if cache_dir:
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    ds = load_dataset(name)
    return ds


def to_arrays(ds_split, n: int | None = None):
    """Convert a HF audio split into (audio_arrays, sample_rates, labels)."""
    arrays, sample_rates, labels = [], [], []
    for i, ex in enumerate(ds_split):
        if n is not None and i >= n:
            break
        a = ex["audio"]
        arrays.append(np.asarray(a["array"], dtype=np.float32))
        sample_rates.append(int(a["sampling_rate"]))
        labels.append(int(ex["label"]))
    return arrays, sample_rates, np.asarray(labels, dtype=np.int64)

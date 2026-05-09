"""Audio loading + preprocessing for the deepfake-audio production pipeline.

Single source of truth for: file -> mono float32 array at 16 kHz, fixed 1.5 s window.
Both train.py and predict.py call into here so the inference contract matches what
the W2V2 LogReg head was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
DURATION_S = 1.5
TARGET_LEN = int(TARGET_SR * DURATION_S)


@dataclass(frozen=True)
class AudioConfig:
    target_sr: int = TARGET_SR
    duration_s: float = DURATION_S

    @property
    def target_len(self) -> int:
        return int(self.target_sr * self.duration_s)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = str(path)
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False)
    return np.asarray(y, dtype=np.float32), int(sr)


def to_fixed_mono(y: np.ndarray, sr: int, cfg: AudioConfig | None = None) -> np.ndarray:
    """Mono -> resample to 16 kHz -> trim/pad to 1.5 s. Same contract as Phase 4 W2V2."""
    cfg = cfg or AudioConfig()
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    if sr != cfg.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sr)
    n = cfg.target_len
    if y.shape[0] > n:
        y = y[:n]
    elif y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return y.astype(np.float32)


def load_and_preprocess(path: str | Path, cfg: AudioConfig | None = None) -> np.ndarray:
    y, sr = load_audio(path)
    return to_fixed_mono(y, sr, cfg)

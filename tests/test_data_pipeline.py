"""Tests for src/data_pipeline.py — the production audio I/O contract.

The W2V2+LogReg head was trained on 1.5 s @ 16 kHz mono float32 windows.
Any drift in this contract silently corrupts predictions, so the tests pin
the exact shape, dtype, sample rate, and pad/truncate behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data_pipeline import (
    AudioConfig,
    load_and_preprocess,
    load_audio,
    to_fixed_mono,
)


def _sine(sr: int, dur_s: float, f0: float = 220.0, channels: int = 1, seed: int = 0) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    rng = np.random.default_rng(seed)
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.02 * rng.standard_normal(t.shape)
    y = y.astype(np.float32)
    if channels == 2:
        y = np.stack([y, y * 0.9], axis=-1)
    return y


def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.target_sr == 16000
    assert cfg.duration_s == 1.5
    assert cfg.target_len == 24000


def test_audio_config_is_frozen():
    """AudioConfig is the contract — if it mutates, train/predict drift apart."""
    cfg = AudioConfig()
    with pytest.raises(Exception):
        cfg.target_sr = 22050  # type: ignore[misc]


def test_to_fixed_mono_pads_short_clip():
    cfg = AudioConfig()
    y = _sine(sr=16000, dur_s=0.5)  # 8000 samples, needs to grow to 24000
    out = to_fixed_mono(y, sr=16000, cfg=cfg)
    assert out.shape == (cfg.target_len,)
    assert out.dtype == np.float32
    # First 8000 samples are the signal; the rest is zero-pad
    assert np.allclose(out[:8000], y)
    assert np.all(out[8000:] == 0)


def test_to_fixed_mono_truncates_long_clip():
    cfg = AudioConfig()
    y = _sine(sr=16000, dur_s=4.0)  # 64000 samples, needs to shrink to 24000
    out = to_fixed_mono(y, sr=16000, cfg=cfg)
    assert out.shape == (cfg.target_len,)
    assert np.allclose(out, y[: cfg.target_len])


def test_to_fixed_mono_resamples_to_16k():
    cfg = AudioConfig()
    y = _sine(sr=44100, dur_s=1.5)
    out = to_fixed_mono(y, sr=44100, cfg=cfg)
    assert out.shape == (cfg.target_len,)
    assert out.dtype == np.float32
    # Resampled, so values won't equal the original, but they should be bounded
    assert np.isfinite(out).all()
    assert out.max() <= 1.5 and out.min() >= -1.5


def test_to_fixed_mono_collapses_stereo():
    cfg = AudioConfig()
    y = _sine(sr=16000, dur_s=1.5, channels=2)
    assert y.ndim == 2
    out = to_fixed_mono(y, sr=16000, cfg=cfg)
    assert out.ndim == 1
    assert out.shape == (cfg.target_len,)
    # Mean of two scaled copies of the same sine
    expected = (y[:, 0] + y[:, 1]) / 2.0
    assert np.allclose(out, expected, atol=1e-5)


def test_load_audio_roundtrip(tmp_path: Path):
    sr = 16000
    y = _sine(sr=sr, dur_s=1.5)
    p = tmp_path / "test.wav"
    sf.write(p, y, sr)
    y2, sr2 = load_audio(p)
    assert sr2 == sr
    assert y2.dtype == np.float32
    assert y2.shape == y.shape
    assert np.allclose(y2, y, atol=1e-4)


def test_load_and_preprocess_end_to_end(tmp_path: Path):
    p = tmp_path / "noisy.wav"
    sf.write(p, _sine(sr=22050, dur_s=2.0), 22050)
    out = load_and_preprocess(p)
    cfg = AudioConfig()
    assert out.shape == (cfg.target_len,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_to_fixed_mono_exact_length_is_passthrough():
    cfg = AudioConfig()
    y = _sine(sr=16000, dur_s=1.5)
    assert len(y) == cfg.target_len
    out = to_fixed_mono(y, sr=16000, cfg=cfg)
    assert np.allclose(out, y)

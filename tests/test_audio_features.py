"""Tests for src/audio_features.py — feature extractor must be deterministic and finite."""
import numpy as np

from src.audio_features import (FeatureConfig, extract_features,
                                  feature_names, preprocess)


def _synthetic_clip(sr=16000, dur_s=2.0, f0=200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur_s)) / sr
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(t.shape)
    return y.astype(np.float32), sr


def test_preprocess_shape_and_dtype():
    cfg = FeatureConfig(target_sr=16000, max_duration_s=4.0)
    y, sr = _synthetic_clip(sr=44100, dur_s=2.0)
    out = preprocess(y, sr, cfg)
    assert out.dtype == np.float32
    assert len(out) == int(cfg.target_sr * cfg.max_duration_s)


def test_extract_features_finite():
    cfg = FeatureConfig()
    y, sr = _synthetic_clip(sr=16000, dur_s=2.0, f0=180.0, seed=42)
    feats = extract_features(y, sr, cfg)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Feature vector must contain no NaN/Inf"
    assert feats.ndim == 1
    assert len(feats) == len(feature_names(cfg)), "Feature count must match feature_names()"


def test_extract_features_deterministic():
    cfg = FeatureConfig()
    y, sr = _synthetic_clip(seed=7)
    a = extract_features(y, sr, cfg)
    b = extract_features(y, sr, cfg)
    assert np.allclose(a, b, equal_nan=False)


def test_jitter_low_for_pure_tone():
    """A pure sine wave should have very low jitter (clean periodicity)."""
    cfg = FeatureConfig()
    y, sr = _synthetic_clip(sr=16000, dur_s=2.0, f0=180.0, seed=0)
    feats = extract_features(y, sr, cfg)
    names = feature_names(cfg)
    jitter_idx = names.index("jitter_local")
    # Pure tone with small noise: jitter should be small (< 0.05 typical)
    assert feats[jitter_idx] < 0.20, f"Jitter for near-pure-tone should be small, got {feats[jitter_idx]}"

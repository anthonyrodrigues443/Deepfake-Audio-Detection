"""Audio augmentations for cross-domain robustness.

Phase 2 finding: handcrafted models hit ~0% in-domain EER on garystafford but
collapse to 48-64% on the Hemg deepfake dataset (4/5 with AUROC<0.5 — anti-
predictive). The codec/source-mismatch fingerprint that says "fake" on the
training distribution says "real" on the held-out one.

Augmentations here perturb the surface acoustic conditions during training so
the classifier can't rely on the fixed fingerprint:

- gaussian_noise        : additive white Gaussian noise at a chosen SNR
- gain                  : random ±dB gain (level mismatch)
- time_shift            : random circular shift up to max_shift_s seconds
- pitch_shift           : ±2 semitones (changes harmonic content slightly)
- codec_simulation      : downsample-upsample roundtrip — kills high-freq energy,
                          which is exactly what a lossy codec does
- random_aug            : pick one of the above with given prob and params

These transforms operate on raw waveforms, not features, so the downstream
feature extractor (`src.audio_features.extract_features`) sees a perturbed
signal and produces a perturbed feature vector. That's the point: the same
clip yields multiple feature vectors, breaking the per-clip codec fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # librosa is optional at import time so unit tests can run without audio deps
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


@dataclass
class AugmentationConfig:
    snr_db_low: float = 10.0
    snr_db_high: float = 30.0
    gain_db_low: float = -6.0
    gain_db_high: float = 6.0
    max_shift_s: float = 0.4
    pitch_semitones_max: float = 2.0
    codec_target_sr_low: int = 6000   # downsampled "transcode" target floor
    codec_target_sr_high: int = 11025
    p_apply: float = 1.0  # probability that the random_aug applies any aug


def gaussian_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise at the requested signal-to-noise ratio (dB)."""
    if y.size == 0:
        return y
    sig_power = float(np.mean(y ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(size=y.shape).astype(y.dtype) * np.sqrt(noise_power)
    return (y + noise).astype(y.dtype)


def gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a constant gain in dB."""
    factor = 10.0 ** (gain_db / 20.0)
    return (y * factor).astype(y.dtype)


def time_shift(y: np.ndarray, sr: int, max_shift_s: float, rng: np.random.Generator) -> np.ndarray:
    """Circular shift the waveform by up to ±max_shift_s seconds."""
    n = y.shape[0]
    if n == 0:
        return y
    max_samples = int(max_shift_s * sr)
    if max_samples <= 0:
        return y
    shift = int(rng.integers(-max_samples, max_samples + 1))
    return np.roll(y, shift)


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """librosa pitch shift. Slow (~50ms/clip at 16k) but useful for prosody robustness."""
    if librosa is None:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(y.dtype)


def codec_simulation(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Downsample to target_sr and upsample back — drops high-frequency content.

    This isn't a real MP3/OPUS encode, but it produces the dominant artifact of
    lossy speech codecs (high-band attenuation) at a fraction of the cost.
    Chosen target_sr below 8k mimics narrowband telephony codecs (G.722, GSM);
    target_sr around 11k mimics low-bitrate MP3.
    """
    if librosa is None or target_sr >= sr:
        return y
    down = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    up = librosa.resample(down, orig_sr=target_sr, target_sr=sr)
    if up.shape[0] < y.shape[0]:
        up = np.pad(up, (0, y.shape[0] - up.shape[0]))
    return up[: y.shape[0]].astype(y.dtype)


def apply_one(name: str, y: np.ndarray, sr: int, cfg: AugmentationConfig,
              rng: np.random.Generator) -> np.ndarray:
    """Apply a single named augmentation with random parameters from cfg."""
    if name == "noise":
        snr = float(rng.uniform(cfg.snr_db_low, cfg.snr_db_high))
        return gaussian_noise(y, snr, rng)
    if name == "gain":
        g = float(rng.uniform(cfg.gain_db_low, cfg.gain_db_high))
        return gain(y, g)
    if name == "shift":
        return time_shift(y, sr, cfg.max_shift_s, rng)
    if name == "pitch":
        s = float(rng.uniform(-cfg.pitch_semitones_max, cfg.pitch_semitones_max))
        return pitch_shift(y, sr, s)
    if name == "codec":
        target = int(rng.integers(cfg.codec_target_sr_low, cfg.codec_target_sr_high + 1))
        return codec_simulation(y, sr, target)
    raise ValueError(f"unknown augmentation: {name}")


def random_aug(y: np.ndarray, sr: int, names: tuple[str, ...] = ("noise", "gain", "shift", "codec"),
               cfg: Optional[AugmentationConfig] = None,
               rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, str]:
    """Pick one augmentation uniformly from `names` and apply it.

    Returns (augmented_waveform, applied_name). If cfg.p_apply rolls "skip",
    returns (y, "none").
    """
    cfg = cfg or AugmentationConfig()
    rng = rng or np.random.default_rng()
    if rng.random() > cfg.p_apply:
        return y, "none"
    name = str(rng.choice(names))
    return apply_one(name, y, sr, cfg, rng), name

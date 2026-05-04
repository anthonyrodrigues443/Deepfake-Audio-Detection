"""Audio feature extraction for deepfake detection.

Three families:
- short-term spectral: MFCC (+ delta, delta2), LFCC, spectral centroid/bandwidth/rolloff/flatness/contrast
- long-term: zero-crossing rate, RMS energy, chroma
- prosody / forensic: F0 statistics, jitter, shimmer (the "deepfakes forget to breathe / have unnaturally smooth pitch" features)

Each extractor returns a 1-D numpy array of summary statistics (mean + std).
The full feature vector aggregates them into a single fixed-length vector,
suitable for classical ML baselines (LogReg/RF/XGB).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


@dataclass
class FeatureConfig:
    target_sr: int = 16000
    max_duration_s: float = 4.0
    n_mfcc: int = 20
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    f0_fmin: float = 50.0
    f0_fmax: float = 500.0


def _summarize(x: np.ndarray) -> np.ndarray:
    """Return [mean, std, min, max] per feature dim.

    x shape: (n_features, n_frames) -> returns (n_features * 4,)
    """
    if x.size == 0:
        return np.zeros(4, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return np.concatenate([
        np.nanmean(x, axis=1),
        np.nanstd(x, axis=1),
        np.nanmin(x, axis=1),
        np.nanmax(x, axis=1),
    ]).astype(np.float32)


def preprocess(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    """Mono, resample to cfg.target_sr, trim/pad to max_duration_s."""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != cfg.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sr)
    target_len = int(cfg.target_sr * cfg.max_duration_s)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def mfcc_features(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """MFCC + delta + delta2, summarized."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=cfg.target_sr, n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([_summarize(mfcc), _summarize(d1), _summarize(d2)])


def spectral_features(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Spectral centroid, bandwidth, rolloff, flatness, contrast, ZCR, RMS."""
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length))
    centroid = librosa.feature.spectral_centroid(S=S, sr=cfg.target_sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=cfg.target_sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=cfg.target_sr)
    flatness = librosa.feature.spectral_flatness(S=S)
    contrast = librosa.feature.spectral_contrast(S=S, sr=cfg.target_sr)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=cfg.hop_length)
    rms = librosa.feature.rms(S=S, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
    return np.concatenate([
        _summarize(centroid), _summarize(bandwidth), _summarize(rolloff),
        _summarize(flatness), _summarize(contrast), _summarize(zcr), _summarize(rms),
    ])


def prosody_features(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """F0 statistics + jitter (cycle-to-cycle period variation) + shimmer (cycle-to-cycle amplitude variation).

    The forensic-audio claim: human voices have natural F0 micro-variations.
    Neural vocoders smooth these out, so synthetic voices have unnaturally low jitter/shimmer.
    """
    # Use librosa.yin (autocorrelation-based, ~10x faster than pyin which is
    # Bayesian and Viterbi-decoded). For Phase 1 baselines, yin's accuracy is
    # plenty — Phase 2+ can revisit with pyin if jitter signal looks promising.
    try:
        f0 = librosa.yin(
            y, fmin=cfg.f0_fmin, fmax=cfg.f0_fmax, sr=cfg.target_sr,
            frame_length=cfg.n_fft * 4, hop_length=cfg.hop_length,
        )
        # yin returns f0 for every frame; we treat f0 inside [fmin, fmax*0.95]
        # as "voiced" (no probability mask available). Outside-range frames are
        # mostly noise / silence — discard them.
        voiced_mask = (f0 >= cfg.f0_fmin * 1.05) & (f0 <= cfg.f0_fmax * 0.95)
        voiced_prob = voiced_mask.astype(np.float32)
    except Exception:
        f0 = np.full(1, np.nan)
        voiced_mask = np.zeros(1, dtype=bool)
        voiced_prob = np.zeros(1)

    f0_voiced = f0[voiced_mask] if f0 is not None else np.array([])
    if f0_voiced.size < 3:
        return np.zeros(11, dtype=np.float32)

    # Jitter: relative period perturbation (consecutive period diffs / mean period)
    periods = 1.0 / f0_voiced
    period_diffs = np.abs(np.diff(periods))
    jitter_local = np.mean(period_diffs) / np.mean(periods) if np.mean(periods) > 0 else 0.0

    # Shimmer: relative amplitude perturbation across voiced frames
    rms = librosa.feature.rms(y=y, hop_length=cfg.hop_length)[0]
    voiced_idx = np.where(voiced_mask)[0] if voiced_mask is not None else np.array([], dtype=int)
    voiced_idx = voiced_idx[voiced_idx < len(rms)]
    if len(voiced_idx) >= 3:
        amps = rms[voiced_idx]
        amp_diffs = np.abs(np.diff(amps))
        shimmer_local = np.mean(amp_diffs) / np.mean(amps) if np.mean(amps) > 0 else 0.0
    else:
        shimmer_local = 0.0

    voicing_ratio = float(voiced_mask.mean()) if voiced_mask.size else 0.0

    return np.array([
        np.mean(f0_voiced), np.std(f0_voiced),
        np.min(f0_voiced), np.max(f0_voiced),
        np.median(f0_voiced),
        np.percentile(f0_voiced, 25), np.percentile(f0_voiced, 75),
        jitter_local, shimmer_local,
        voicing_ratio,
        len(f0_voiced) / max(len(f0), 1),  # voiced frame fraction
    ], dtype=np.float32)


def extract_features(y: np.ndarray, sr: int, cfg: FeatureConfig | None = None) -> np.ndarray:
    """Extract the full feature vector. Returns (D,) float32 array."""
    cfg = cfg or FeatureConfig()
    y = preprocess(y, sr, cfg)
    return np.concatenate([
        mfcc_features(y, cfg),
        spectral_features(y, cfg),
        prosody_features(y, cfg),
    ]).astype(np.float32)


def feature_names(cfg: FeatureConfig | None = None) -> list[str]:
    """Names for each dim of the extract_features() output."""
    cfg = cfg or FeatureConfig()
    stats = ["mean", "std", "min", "max"]
    names = []
    # MFCC + delta + delta2
    for tag in ("mfcc", "mfcc_d1", "mfcc_d2"):
        for s in stats:
            for i in range(cfg.n_mfcc):
                names.append(f"{tag}_{i}_{s}")
    # Spectral
    for tag in ("centroid", "bandwidth", "rolloff", "flatness"):
        for s in stats:
            names.append(f"spec_{tag}_{s}")
    # Spectral contrast (default 7 bands)
    for s in stats:
        for i in range(7):
            names.append(f"spec_contrast{i}_{s}")
    for tag in ("zcr", "rms"):
        for s in stats:
            names.append(f"spec_{tag}_{s}")
    # Prosody
    names += [
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_median", "f0_p25", "f0_p75",
        "jitter_local", "shimmer_local", "voicing_ratio", "voiced_frac",
    ]
    return names

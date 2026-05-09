"""End-to-end inference: audio file -> {real|fake, probability, latency, features}.

Loads the joblib bundle once. Subsequent calls reuse the same encoder + head.

CLI:
    python -m src.predict --audio path/to/file.wav
    python -m src.predict --audio file.wav --threshold 0.5 --json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.data_pipeline import AudioConfig, load_and_preprocess
from src.w2v2_encoder import embed

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJ / "models" / "w2v2_logreg_champion.joblib"
DEFAULT_THRESHOLD = 0.5

_bundle_cache: dict = {"path": None, "bundle": None}


@dataclass
class PredictionResult:
    label: str
    fake_probability: float
    threshold: float
    latency_ms: float
    encode_ms: float
    classify_ms: float
    audio_path: str
    encoder_id: str
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict:
    path = Path(path)
    if _bundle_cache["path"] == path and _bundle_cache["bundle"] is not None:
        return _bundle_cache["bundle"]
    bundle = joblib.load(path)
    _bundle_cache["path"] = path
    _bundle_cache["bundle"] = bundle
    return bundle


def predict_array(
    y: np.ndarray,
    sr: int = 16000,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    threshold: float = DEFAULT_THRESHOLD,
) -> PredictionResult:
    """Run inference on a 1-D float32 waveform already at 16 kHz, fixed-length."""
    bundle = load_bundle(bundle_path)
    head = bundle["head"]

    t_enc0 = time.perf_counter()
    emb = embed(y, sampling_rate=sr).reshape(1, -1)
    t_enc = (time.perf_counter() - t_enc0) * 1000

    t_cls0 = time.perf_counter()
    p_fake = float(head.predict_proba(emb)[0, 1])
    t_cls = (time.perf_counter() - t_cls0) * 1000

    label = "fake" if p_fake >= threshold else "real"
    return PredictionResult(
        label=label,
        fake_probability=p_fake,
        threshold=threshold,
        latency_ms=t_enc + t_cls,
        encode_ms=t_enc,
        classify_ms=t_cls,
        audio_path="",
        encoder_id=bundle["encoder_id"],
        model_version=bundle["version"],
    )


def predict_file(
    audio_path: str | Path,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    threshold: float = DEFAULT_THRESHOLD,
) -> PredictionResult:
    cfg = AudioConfig()
    y = load_and_preprocess(audio_path, cfg)
    res = predict_array(y, sr=cfg.target_sr, bundle_path=bundle_path, threshold=threshold)
    res.audio_path = str(audio_path)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Deepfake audio detector — single-file inference")
    ap.add_argument("--audio", required=True, type=Path, help="Path to .wav/.flac/.mp3")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}")
        return 2
    if not args.bundle.exists():
        print(f"error: bundle not found: {args.bundle}. Run `python -m src.train` first.")
        return 2

    res = predict_file(args.audio, args.bundle, args.threshold)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"file:        {res.audio_path}")
        print(f"prediction:  {res.label.upper()}  (fake_prob={res.fake_probability:.4f} @ thr={res.threshold:.2f})")
        print(f"latency:     {res.latency_ms:.1f} ms  (encode {res.encode_ms:.1f} + classify {res.classify_ms:.1f})")
        print(f"model:       {res.encoder_id}  +  LogReg head  ({res.model_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen Wav2Vec2-base encoder with mean-pooled hidden states (768-d).

Singleton — encoder loads once per process. The Phase 4 protocol (1.5 s @ 16 kHz,
mean-pool last_hidden_state) is reproduced here so train + serve use identical
embeddings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

_MODEL_ID = "facebook/wav2vec2-base"
_HIDDEN_SIZE = 768

_state: dict = {"feat_ext": None, "model": None, "device": None}


def _select_device() -> str:
    forced = os.environ.get("DEEPFAKE_W2V2_DEVICE")
    if forced:
        return forced
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_encoder():
    if _state["model"] is None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        import torch

        device = _select_device()
        feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_ID)
        model = Wav2Vec2Model.from_pretrained(_MODEL_ID)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _state["feat_ext"] = feat_ext
        _state["model"] = model
        _state["device"] = device
    return _state["feat_ext"], _state["model"], _state["device"]


def embed(y: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """1-D float32 waveform at sampling_rate -> (768,) mean-pooled embedding."""
    import torch

    feat_ext, model, device = get_encoder()
    inputs = feat_ext(y, sampling_rate=sampling_rate, return_tensors="pt")
    x = inputs["input_values"].to(device)
    with torch.no_grad():
        out = model(x).last_hidden_state  # (1, T, 768)
    return out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def embed_batch(ys: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """Batched version — ys shape (B, N) at sampling_rate -> (B, 768)."""
    feats = np.zeros((len(ys), _HIDDEN_SIZE), dtype=np.float32)
    for i, y in enumerate(ys):
        feats[i] = embed(y, sampling_rate=sampling_rate)
    return feats


HIDDEN_SIZE = _HIDDEN_SIZE
MODEL_ID = _MODEL_ID

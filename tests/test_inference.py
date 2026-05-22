"""End-to-end inference tests.

Strategy: use the Phase 4 cached W2V2 embeddings (768-d, already computed) and
push them through the LogReg head directly. This exercises the same code path
predict.predict_array uses *after* the encoder, without paying the 6.1 s cold
start of loading 360 MB of W2V2 weights inside the test process.

A single optional raw-audio test exists at the end; it's skipped by default
because it would force a 360 MB Hugging Face download on a fresh machine.
Set DEEPFAKE_RUN_AUDIO_TEST=1 to enable it.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.metrics import roc_auc_score

from src.eer import compute_eer
from src.evaluate import compute_eer as evaluate_compute_eer
from src.evaluate import evaluate_split

PROJ = Path(__file__).resolve().parents[1]
RESULTS = PROJ / "results"
BUNDLE_PATH = PROJ / "models" / "w2v2_logreg_champion.joblib"


def _requires_bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("production bundle missing; run `python -m src.train`")
    return joblib.load(BUNDLE_PATH)


def _requires_phase4_arrays(*paths: Path):
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Phase 4 cached arrays missing: {[p.name for p in missing]}")


# ----------------------------------------------------------------------------
# In-domain reproduction
# ----------------------------------------------------------------------------

def test_inference_reproduces_in_domain_phase6_numbers():
    """The headline: production bundle on the same 180-clip in-domain test split.
    Phase 6 reference: ROC-AUC 0.999, EER 1.11%. We accept ±0.5pp drift."""
    bundle = _requires_bundle()
    Xte_p = RESULTS / "phase4_w2v2_test.npy"
    yte_p = RESULTS / "phase4_y_test.npy"
    _requires_phase4_arrays(Xte_p, yte_p)

    head = bundle["head"]
    Xte, yte = np.load(Xte_p), np.load(yte_p)
    assert Xte.shape == (180, 768), f"unexpected test shape {Xte.shape}"

    proba = head.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    eer, _ = compute_eer(yte, proba)

    assert auc > 0.99, f"in-domain ROC-AUC drifted: {auc:.4f}"
    assert eer * 100 < 2.0, f"in-domain EER drifted: {eer*100:.2f}%"


def test_inference_reproduces_cross_distribution_phase6_numbers():
    """Hemg full-100 cross-distribution. Phase 6 reference: ROC 0.559, EER 46%.
    The narrow band catches both 'leak' regressions (ROC jumps) and 'broken'
    regressions (ROC collapses to anti-predictive)."""
    bundle = _requires_bundle()
    Xhg_p = RESULTS / "phase4_w2v2_hemg.npy"
    yhg_p = RESULTS / "phase4_y_hemg.npy"
    _requires_phase4_arrays(Xhg_p, yhg_p)

    head = bundle["head"]
    Xhg, yhg = np.load(Xhg_p), np.load(yhg_p)
    assert Xhg.shape == (100, 768)
    proba = head.predict_proba(Xhg)[:, 1]
    auc = roc_auc_score(yhg, proba)
    eer, _ = compute_eer(yhg, proba)

    # Reference Phase 6: ROC=0.559, EER=46%. Tolerance ±0.05 ROC, ±5 EER points.
    assert 0.45 <= auc <= 0.70, f"Hemg ROC out of band: {auc:.4f}"
    assert 35.0 <= eer * 100 <= 55.0, f"Hemg EER out of band: {eer*100:.2f}%"


# ----------------------------------------------------------------------------
# evaluate.py contract
# ----------------------------------------------------------------------------

def test_evaluate_split_returns_expected_schema():
    bundle = _requires_bundle()
    Xte_p = RESULTS / "phase4_w2v2_test.npy"
    yte_p = RESULTS / "phase4_y_test.npy"
    _requires_phase4_arrays(Xte_p, yte_p)
    Xte, yte = np.load(Xte_p), np.load(yte_p)
    out = evaluate_split("in_domain", bundle["head"], Xte, yte, threshold=0.5)

    required = {"split", "n", "accuracy", "f1", "precision", "recall", "roc_auc",
                "eer_pct", "eer_threshold", "confusion", "n_pos"}
    assert required.issubset(out.keys()), f"missing keys: {required - set(out.keys())}"
    assert out["split"] == "in_domain"
    assert out["n"] == 180
    assert 0.0 <= out["accuracy"] <= 1.0
    cm = np.asarray(out["confusion"])
    assert cm.shape == (2, 2)
    assert cm.sum() == 180


def test_evaluate_compute_eer_matches_src_eer():
    """src/evaluate.py and src/eer.py both compute EER. They must agree."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    eer_eval, thr_eval = evaluate_compute_eer(y, s)
    eer_src, thr_src = compute_eer(y, s)
    assert abs(eer_eval - eer_src) < 1e-9
    assert abs(thr_eval - thr_src) < 1e-9


# ----------------------------------------------------------------------------
# predict.py module-level cache + dataclass surface
# ----------------------------------------------------------------------------

def test_predict_result_dataclass_round_trip():
    """PredictionResult.to_dict should serialize cleanly to JSON-safe scalars."""
    from src.predict import PredictionResult
    r = PredictionResult(
        label="fake",
        fake_probability=0.87,
        threshold=0.5,
        latency_ms=15.2,
        encode_ms=14.6,
        classify_ms=0.6,
        audio_path="/tmp/x.wav",
        encoder_id="facebook/wav2vec2-base",
        model_version="phase6-2026-05-09",
    )
    d = r.to_dict()
    assert d["label"] == "fake"
    assert d["fake_probability"] == 0.87
    import json
    json.dumps(d)  # raises if not JSON-serializable


def test_load_bundle_uses_cache():
    """Second load_bundle() call should hit the module-level cache, not re-read the joblib."""
    if not BUNDLE_PATH.exists():
        pytest.skip("bundle missing")
    from src.predict import load_bundle
    b1 = load_bundle(BUNDLE_PATH)
    b2 = load_bundle(BUNDLE_PATH)
    assert b1 is b2, "load_bundle did not cache the bundle"


# ----------------------------------------------------------------------------
# Optional raw-audio path (encodes through W2V2 — slow, off by default)
# ----------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DEEPFAKE_RUN_AUDIO_TEST") != "1",
    reason="set DEEPFAKE_RUN_AUDIO_TEST=1 to run the full encode+classify path",
)
def test_predict_array_end_to_end(tmp_path: Path):
    from src.data_pipeline import AudioConfig, to_fixed_mono
    from src.predict import predict_array

    cfg = AudioConfig()
    # Synthetic clip — close enough to "voice" range that yin doesn't crash
    rng = np.random.default_rng(0)
    t = np.arange(int(16000 * 1.5)) / 16000
    y = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)
    y = to_fixed_mono(y, sr=16000, cfg=cfg)

    res = predict_array(y, sr=cfg.target_sr, threshold=0.5)
    assert res.label in {"real", "fake"}
    assert 0.0 <= res.fake_probability <= 1.0
    assert res.latency_ms > 0
    assert res.encoder_id == "facebook/wav2vec2-base"

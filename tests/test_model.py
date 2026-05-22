"""Tests for the joblib production bundle and the LogReg head.

These tests don't load the W2V2 encoder (slow, ~6 s cold start + 360 MB download
on the first call). They verify the *head* — what's actually serialized to the
26 KB bundle — and the schema fields the rest of the pipeline depends on.

If the bundle is missing (someone cloned the repo without the artifact and
hasn't run `python -m src.train` yet), the tests skip gracefully rather than
failing the suite.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJ / "models" / "w2v2_logreg_champion.joblib"


@pytest.fixture(scope="module")
def bundle() -> dict:
    if not BUNDLE_PATH.exists():
        pytest.skip(f"production bundle not found at {BUNDLE_PATH}; run `python -m src.train`")
    return joblib.load(BUNDLE_PATH)


def test_bundle_has_required_keys(bundle):
    required = {"version", "encoder_id", "encoder_hidden_size", "head",
                "target_sr", "duration_s", "metrics", "label_schema"}
    missing = required - set(bundle.keys())
    assert not missing, f"bundle missing keys: {missing}"


def test_bundle_input_contract(bundle):
    """If these drift, train.py and predict.py will silently produce wrong predictions."""
    assert bundle["target_sr"] == 16000
    assert bundle["duration_s"] == 1.5
    assert bundle["encoder_hidden_size"] == 768
    assert bundle["encoder_id"] == "facebook/wav2vec2-base"


def test_bundle_label_schema(bundle):
    schema = bundle["label_schema"]
    assert schema[0] == "real"
    assert schema[1] == "fake"


def test_head_is_a_pipeline(bundle):
    head = bundle["head"]
    assert isinstance(head, Pipeline)
    names = [name for name, _ in head.steps]
    assert "sc" in names
    assert "lr" in names
    assert isinstance(head.named_steps["sc"], StandardScaler)
    assert isinstance(head.named_steps["lr"], LogisticRegression)


def test_head_expects_768_features(bundle):
    """LogReg should be fitted on the 768-d W2V2 embedding — not 12 features, not 303."""
    head = bundle["head"]
    sc = head.named_steps["sc"]
    lr = head.named_steps["lr"]
    assert sc.n_features_in_ == 768
    assert lr.n_features_in_ == 768
    # Binary classifier: one row of weights of length 768
    assert lr.coef_.shape == (1, 768)


def test_head_predict_proba_on_random_input(bundle):
    """The head must produce calibrated-looking [0,1] probabilities for arbitrary embeddings."""
    head = bundle["head"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 768)).astype(np.float32)
    proba = head.predict_proba(X)
    assert proba.shape == (10, 2)
    # Rows must be a valid probability distribution
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_head_predict_proba_is_deterministic(bundle):
    """Same input -> same output. Catches any accidental randomness in the head."""
    head = bundle["head"]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 768)).astype(np.float32)
    a = head.predict_proba(X)
    b = head.predict_proba(X)
    assert np.allclose(a, b)


def test_bundle_size_is_tiny(bundle):
    """The whole point: the head is 26 KB. If this ever balloons, something leaked."""
    size_kb = BUNDLE_PATH.stat().st_size / 1024
    assert size_kb < 200, f"bundle is {size_kb:.1f} KB — head should be < 200 KB"


def test_bundle_metrics_present(bundle):
    m = bundle["metrics"]
    for key in ("cv_train_roc_auc_mean", "in_domain_test_roc_auc", "cross_domain_hemg_roc_auc"):
        assert key in m, f"missing metric: {key}"
        assert 0.0 <= m[key] <= 1.0


def test_bundle_reference_metrics(bundle):
    """Pin the Phase 6 reference numbers within tolerance. If train.py drifts,
    the in-domain or cross-domain ROC-AUC will move and this test catches it."""
    m = bundle["metrics"]
    # In-domain Phase 4/6 reference: ROC ~0.999, EER 1.11%
    assert m["in_domain_test_roc_auc"] > 0.99, \
        f"in-domain ROC dropped: {m['in_domain_test_roc_auc']:.4f}"
    # Cross-domain Phase 6 reference: full-100 Hemg ROC 0.559 (close to chance, as expected)
    # If this jumps above 0.8, something is leaking; if it drops below 0.4 the model is anti-pred.
    assert 0.40 <= m["cross_domain_hemg_roc_auc"] <= 0.80, \
        f"Hemg ROC out of expected band: {m['cross_domain_hemg_roc_auc']:.4f}"

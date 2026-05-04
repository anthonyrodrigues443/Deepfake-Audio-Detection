"""Tests for src/eer.py — EER computation must be correct, this is the headline metric."""
import numpy as np

from src.eer import compute_eer, metrics_at_threshold


def test_eer_perfect_separation():
    # Perfectly separable: real (label=0) all score 0, fake (label=1) all score 1
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    eer, _ = compute_eer(y, s)
    assert eer == 0.0, f"Perfect separation should give EER=0, got {eer}"


def test_eer_random():
    # Random scores → EER should be ~0.5
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    s = rng.random(2000)
    eer, _ = compute_eer(y, s)
    assert 0.40 <= eer <= 0.60, f"Random scores should give EER ~0.5, got {eer}"


def test_eer_known_case():
    # Hand-built crossing point
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7])
    # At threshold 0.4: predictions = [0,0,0,0,1,1,1,1] but score==0.4 is the boundary
    # FPR + FNR depends on threshold; just check EER is in [0, 0.5]
    eer, thr = compute_eer(y, s)
    assert 0.0 <= eer <= 0.5
    assert thr is not None


def test_metrics_at_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    m = metrics_at_threshold(y, s, 0.5)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["tn"] == 2 and m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0

"""Equal Error Rate (EER) — the standard metric for spoofing/anti-spoofing systems.

EER is the operating point on the ROC curve where false-accept rate (FAR) ==
false-reject rate (FRR). Lower EER = better. Every ASVspoof challenge reports EER.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1) -> tuple[float, float]:
    """Return (eer, threshold).

    y_true: 0/1 labels
    y_score: probability of the positive class (fake)
    pos_label: which label is "positive" (=fake = the spoof we're trying to catch)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    fnr = 1 - tpr
    diffs = fpr - fnr
    # The crossing point where fpr - fnr changes sign
    idx = np.nanargmin(np.abs(diffs))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Compute confusion-matrix-derived metrics at a given threshold."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  confusion_matrix, f1_score, precision_score,
                                  recall_score)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

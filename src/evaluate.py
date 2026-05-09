"""Evaluation suite — runs the production bundle on the cached Phase 4 splits
and emits both in-domain and cross-distribution metrics.

This is the gate: if metrics differ from the Phase 4/5 reference numbers by
more than tolerance, the bundle is broken (wrong head, wrong encoder, etc).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJ = Path(__file__).resolve().parents[1]
RESULTS = PROJ / "results"
MODELS = PROJ / "models"


def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    diffs = np.abs(fpr - fnr)
    idx = int(np.argmin(diffs))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thr[idx])


def evaluate_split(name: str, head, X, y, threshold: float = 0.5) -> dict:
    p = head.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    eer, eer_thr = compute_eer(y, p)
    return {
        "split": name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
        "roc_auc": float(roc_auc_score(y, p)),
        "eer_pct": float(eer * 100),
        "eer_threshold": eer_thr,
        "confusion": confusion_matrix(y, yhat).tolist(),
        "n_pos": int(y.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=MODELS / "w2v2_logreg_champion.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=RESULTS / "phase6_evaluation.json")
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    head = bundle["head"]

    Xte = np.load(RESULTS / "phase4_w2v2_test.npy")
    yte = np.load(RESULTS / "phase4_y_test.npy")
    Xhg = np.load(RESULTS / "phase4_w2v2_hemg.npy")
    yhg = np.load(RESULTS / "phase4_y_hemg.npy")

    results = {
        "model_version": bundle["version"],
        "encoder_id": bundle["encoder_id"],
        "threshold": args.threshold,
        "splits": [
            evaluate_split("garystafford_test_in_domain", head, Xte, yte, args.threshold),
            evaluate_split("hemg_full_cross_distribution", head, Xhg, yhg, args.threshold),
        ],
    }

    print(f"{'split':<32}  {'n':>4}  {'acc':>6}  {'f1':>6}  {'roc':>6}  {'eer%':>6}")
    for s in results["splits"]:
        print(
            f"{s['split']:<32}  {s['n']:>4}  {s['accuracy']:>6.3f}  "
            f"{s['f1']:>6.3f}  {s['roc_auc']:>6.3f}  {s['eer_pct']:>6.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

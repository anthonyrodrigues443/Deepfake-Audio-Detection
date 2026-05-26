"""Reproducible training entry point — fits the W2V2+LogReg champion head.

Reads Phase 4's saved 768-d Wav2Vec2 embeddings (train split) + labels, fits
StandardScaler -> LogisticRegression(C=1.0, max_iter=1000), and writes a single
joblib bundle that predict.py can load directly.

If you want to re-extract embeddings from scratch (i.e. without the Phase 4
.npy files), set --extract-from-hf — but that requires the garystafford dataset
to be downloadable and is the slow path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[1]
RESULTS = PROJ / "results"
MODELS = PROJ / "models"


def load_phase4_embeddings():
    paths = {
        "X_train": RESULTS / "phase4_w2v2_train.npy",
        "X_test": RESULTS / "phase4_w2v2_test.npy",
        "X_hemg": RESULTS / "phase4_w2v2_hemg.npy",
        "y_train": RESULTS / "phase4_y_train.npy",
        "y_test": RESULTS / "phase4_y_test.npy",
        "y_hemg": RESULTS / "phase4_y_hemg.npy",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 4 embeddings missing: %s. Run notebooks/phase4_w2v2_extract.ipynb first."
            % ", ".join(missing)
        )
    return {k: np.load(p) for k, p in paths.items()}


def fit_head(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MODELS / "w2v2_logreg_champion.joblib")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for ROC-AUC sanity check")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    arrays = load_phase4_embeddings()
    Xtr, ytr = arrays["X_train"], arrays["y_train"]
    Xte, yte = arrays["X_test"], arrays["y_test"]
    Xhg, yhg = arrays["X_hemg"], arrays["y_hemg"]
    print(f"loaded embeddings: train={Xtr.shape}  test={Xte.shape}  hemg={Xhg.shape}")

    pipe = Pipeline(
        [("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42))]
    )
    cv_scores = cross_val_score(pipe, Xtr, ytr, cv=args.cv, scoring="roc_auc")
    print(f"{args.cv}-fold CV ROC-AUC on train: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    head = fit_head(Xtr, ytr)
    p_test = head.predict_proba(Xte)[:, 1]
    p_hemg = head.predict_proba(Xhg)[:, 1]
    auc_test = roc_auc_score(yte, p_test)
    auc_hemg = roc_auc_score(yhg, p_hemg)
    print(f"holdout in-domain test ROC-AUC: {auc_test:.4f}")
    print(f"holdout cross-domain Hemg ROC-AUC: {auc_hemg:.4f}")

    bundle = {
        "version": "phase7-2026-05-10",
        "encoder_id": "facebook/wav2vec2-base",
        "encoder_hidden_size": 768,
        "head": head,
        "target_sr": 16000,
        "duration_s": 1.5,
        "metrics": {
            "cv_train_roc_auc_mean": float(cv_scores.mean()),
            "cv_train_roc_auc_std": float(cv_scores.std()),
            "in_domain_test_roc_auc": float(auc_test),
            "cross_domain_hemg_roc_auc": float(auc_hemg),
        },
        "label_schema": {0: "real", 1: "fake"},
    }
    joblib.dump(bundle, args.out)
    sz_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({sz_kb:.1f} KB) in {time.time()-t0:.1f}s")

    summary_path = MODELS / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {k: v for k, v in bundle.items() if k != "head"},
            indent=2,
            default=str,
        )
    )
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

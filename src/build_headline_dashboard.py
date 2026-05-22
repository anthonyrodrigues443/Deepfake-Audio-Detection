"""Phase 7 deliverable — consolidate Phase 1-6 results into a single headline image
and rewrite results/EXPERIMENT_LOG.md as a project-wide leaderboard.

Reads the per-phase result JSONs and CSVs that the notebooks emitted, and
produces:
    results/headline_dashboard.png  — 6-panel figure: phase 1-6 progression,
                                       LLM head-to-head, latency/cost, and the
                                       in-domain-vs-cross-distribution gap.
    results/EXPERIMENT_LOG.md       — full project-wide leaderboard.

Run:
    python -m src.build_headline_dashboard
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RESULTS = PROJ / "results"


# ----------------------------------------------------------------------------
# Read inputs
# ----------------------------------------------------------------------------

def _load_json(p: Path) -> dict:
    # metrics.json has NaN literals (invalid strict JSON) — handle defensively.
    text = p.read_text().replace("NaN", "null")
    return json.loads(text)


def load_all() -> dict:
    return {
        "metrics": _load_json(RESULTS / "metrics.json"),
        "phase3": _load_json(RESULTS / "phase3_results.json"),
        "phase4": _load_json(RESULTS / "phase4_results.json"),
        "phase5": _load_json(RESULTS / "phase5_results.json"),
        "phase5_apples": _load_json(RESULTS / "phase5_apples_to_apples.json"),
        "phase5_llm_csv": pd.read_csv(RESULTS / "phase5_llm_vs_custom.csv"),
        "phase6_eval": _load_json(RESULTS / "phase6_evaluation.json"),
        "phase6_latency": _load_json(RESULTS / "phase6_latency_bench.json"),
    }


# ----------------------------------------------------------------------------
# EXPERIMENT_LOG.md rewrite
# ----------------------------------------------------------------------------

def build_experiment_log(data: dict) -> str:
    p1 = data["metrics"]
    p2 = p1.get("phase2", {})
    p3 = data["phase3"]
    p4 = data["phase4"]
    p5 = data["phase5"]
    p5_apples = data["phase5_apples"]
    p6 = data["phase6_eval"]
    p6_lat = data["phase6_latency"]

    out = []
    out.append("# Deepfake Audio Detection — Project-Wide Experiment Log")
    out.append("")
    out.append("Consolidated leaderboard for the 7-day sprint on `garystafford/deepfake-audio-detection` (in-domain) + `Hemg/Deepfake-Audio-Dataset` (cross-distribution).")
    out.append("")
    out.append("Primary metric: **EER %** (Equal Error Rate) — the operating point where FAR = FRR. Standard for every ASVspoof challenge. Lower is better.")
    out.append("")
    out.append("Secondary: ROC-AUC, F1, latency, cost per 1k predictions.")
    out.append("")

    # Phase 1
    out.append("## Phase 1 — Handcrafted Baselines on garystafford (2026-05-04)")
    out.append("")
    out.append("303-dim feature vector (MFCC + spectral + prosody). In-domain test set, n=374.")
    out.append("")
    out.append("| Model | EER % | AUROC | F1 | Bal-Acc | Train s |")
    out.append("|---|--:|--:|--:|--:|--:|")
    for r in p1.get("results", []):
        out.append(f"| {r['model']} | {r['EER %']:.2f} | {r['AUROC']:.4f} | {r['F1']:.4f} | {r['Bal-Acc']:.4f} | {r['Train s']:.2f} |")
    out.append("")
    out.append("**Finding:** LogReg / RandomForest hit **0.00% EER**. XGBoost feature importance shows one feature (`spec_contrast6_mean`) does 66% of the work. The full spectral-contrast family does 87%. Prosody contributes 0% to model importance despite Cohen's d ~0.43 on F0 — the model bypasses forensic signal in favor of a codec/sample-rate shortcut.")
    out.append("")

    # Phase 2
    out.append("## Phase 2 — Multi-Model + Cross-Dataset Collapse (2026-05-05)")
    out.append("")
    out.append("### 2.1 — Ablation: drop spec_contrast family from the feature vector")
    out.append("")
    out.append("| Model | EER % (test) | EER % (val) | AUROC | F1 |")
    out.append("|---|--:|--:|--:|--:|")
    for r in p2.get("experiments", {}).get("2.1_ablation_no_spec_contrast", []):
        out.append(f"| {r['name']} | {r['EER %']:.3f} | {r['EER_val %']:.3f} | {r['AUROC']:.4f} | {r['F1']:.4f} |")
    out.append("")
    out.append("### 2.2 — Per-family models (which features carry the signal?)")
    out.append("")
    out.append("| Model | n_features | EER % | AUROC |")
    out.append("|---|--:|--:|--:|")
    for r in p2.get("experiments", {}).get("2.2_per_family", []):
        out.append(f"| {r['name']} | {r.get('n_features', '—')} | {r['EER %']:.3f} | {r['AUROC']:.4f} |")
    out.append("")
    out.append("### 2.3 — End-to-end mel-spectrogram CNN")
    out.append("")
    cnn = p2.get("experiments", {}).get("2.3_tiny_mel_cnn", {})
    if cnn:
        out.append(f"| {cnn.get('name', 'TinyMelCNN')} | params={cnn.get('params', '—')} | EER {cnn.get('EER %', '—'):.3f}% | AUROC {cnn.get('AUROC', '—'):.4f} | F1 {cnn.get('F1', '—'):.4f} |")
    out.append("")
    out.append("### 2.5 — Cross-distribution test on Hemg (the canary for shortcut learning)")
    out.append("")
    out.append("| Model | in-domain EER % | Hemg EER % | Δ | AUROC out |")
    out.append("|---|--:|--:|--:|--:|")
    for r in p2.get("experiments", {}).get("2.5_cross_dataset", []):
        out.append(f"| {r['name']} | {r['EER in-domain %']:.2f} | {r['EER cross-domain %']:.2f} | {r['Δ EER %']:.2f} | {r['AUROC out']:.4f} |")
    out.append("")
    out.append("**Finding:** Every handcrafted model that hit 0% in-domain landed at 48-64% on Hemg, with 4/5 below ROC=0.5 (anti-predictive). The CNN at 2.41% in-domain EER is the first honest baseline, but it was not yet evaluated cross-distribution.")
    out.append("")

    # Phase 3
    out.append("## Phase 3 — Augmentation for Cross-Domain Generalization (2026-05-06)")
    out.append("")
    out.append("Top-5 by Hemg EER (lower is better):")
    out.append("")
    out.append("| Model | in-domain EER % | Hemg EER % | AUROC in | AUROC Hemg | Stage |")
    out.append("|---|--:|--:|--:|--:|---|")
    for r in p3.get("leaderboard_top5", []):
        out.append(f"| {r['model']} | {r['EER_in_%']:.3f} | {r['EER_hemg_%']:.2f} | {r['AUROC_in']:.4f} | {r['AUROC_hemg']:.4f} | {r['stage']} |")
    out.append("")
    out.append(f"**Best Phase 3:** {p3.get('best_model', '—')} @ {p3.get('best_hemg_eer_pct', '—')}% Hemg EER. 25% target NOT met. Only the *union* of {{noise+gain+shift+codec}} augmentations helped; no single augmentation did.")
    out.append("")

    # Phase 4
    out.append("## Phase 4 — Tuning + Stacking + Frozen W2V2 (2026-05-07)")
    out.append("")
    out.append(f"Protocol: train n={p4['protocol']['train_n']}, test n={p4['protocol']['test_n']}, hemg n={p4['protocol']['hemg_n']} (val={p4['protocol']['hemg_val_n']}, test={p4['protocol']['hemg_test_n']}).")
    out.append("")
    out.append("Leaderboard:")
    out.append("")
    out.append("| Label | in-domain EER % | Hemg test EER % | AUROC Hemg |")
    out.append("|---|--:|--:|--:|")
    for r in p4.get("leaderboard", []):
        out.append(f"| {r['label']} | {r['EER_in_%']:.3f} | {r['EER_hemg_test_%']:.2f} | {r['AUROC_hemg_test']:.4f} |")
    out.append("")
    optuna = p4.get("optuna", {})
    out.append(f"**Optuna:** {optuna.get('n_trials', '—')} trials, best Hemg val EER = {optuna.get('best_value_hemg_val_eer_%', '—')}% — **zero improvement** over the Phase 2 untuned baseline (48% Hemg test EER, both).")
    out.append("")
    out.append(f"**Phase 4 champion:** {p4.get('best_label', '—')} @ {p4.get('best_hemg_test_eer_%', '—')}% Hemg test EER. Stacking hurt: the XGBoost base is anti-predictive on Hemg (AUROC 0.422), so averaging pulled W2V2 back to 48%. The W2V2+LogReg single model beats every ensemble.")
    out.append("")

    # Phase 5
    out.append("## Phase 5 — Advanced Techniques + Ablation + LLM Head-to-Head (2026-05-08)")
    out.append("")
    out.append("19 post-hoc approaches against the Phase 4 W2V2+LogReg champion. Reference: 32% Hemg test EER, AUROC 0.634.")
    out.append("")
    out.append("| Approach | Hemg test EER % | AUROC | Δ vs ref | Family |")
    out.append("|---|--:|--:|--:|---|")
    ref_eer = 32.0
    for r in p5.get("ablation_summary", [])[:18]:
        delta = r["Hemg_test_EER_%"] - ref_eer
        out.append(f"| {r['approach']} | {r['Hemg_test_EER_%']:.2f} | {r['Hemg_test_AUROC']:.3f} | {delta:+.1f} | {r.get('family', '—')} |")
    out.append("")
    out.append("**Finding:** Nothing beat the baseline. Three approaches tied at 32% EER (max-confidence fusion is degenerate, temperature scaling is rank-monotone, the reference itself). Every other variant strictly hurt. **The cross-distribution ceiling is structural** — not addressable by post-hoc surgery.")
    out.append("")
    out.append("### Frontier-LLM head-to-head (50-clip Hemg sample, 12-feature digest)")
    out.append("")
    out.append("**Input-fairness disclosure:** the LLMs received a 12-feature acoustic digest because the local Claude / Codex CLIs do not accept raw audio. W2V2+LogReg received the raw 1.5 s @ 16 kHz waveform. This is documented in the retraction (see PROGRESS_LOG 2026-05-08 entries) and is shown inline in the Streamlit UI's Research tab.")
    out.append("")
    llm_df = data["phase5_llm_csv"].copy()
    out.append("| Model | n | F1 | EER % | AUROC | latency s | $/1k |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for _, r in llm_df.iterrows():
        out.append(f"| {r['model']} | {int(r['n'])} | {r['F1']:.3f} | {r['EER_%']:.2f} | {r['AUROC']:.3f} | {r['latency_s']:.1f} | {r['cost_per_1k_USD']:.4f} |")
    out.append("")
    out.append(f"**Apples-to-apples specialist** (LogReg trained on the *same* 12 features the LLMs received): {p5_apples['hemg_EER_pct']:.1f}% Hemg test EER, AUROC {p5_apples['hemg_AUROC']:.3f}. The specialist *collapses* cross-domain on the digest because it overfits the codec shortcut — the frontier LLMs do better than the matched specialist by 30-40 EER points by ignoring spurious correlations. The W2V2+LogReg model wins (32% EER) because it sees a richer representation, not because it's intrinsically better at audio reasoning.")
    out.append("")

    # Phase 6
    out.append("## Phase 6 — Production Pipeline + Streamlit UI (2026-05-09)")
    out.append("")
    out.append("Production bundle (`models/w2v2_logreg_champion.joblib`, 26 KB):")
    out.append("")
    out.append(f"- Version: `{p6['model_version']}`")
    out.append(f"- Encoder: `{p6['encoder_id']}` (frozen, 768-d)")
    out.append(f"- Head: `StandardScaler → LogisticRegression(C=1.0)`")
    out.append(f"- Input contract: mono float32 @ 16 kHz, 1.5 s window")
    out.append("")
    out.append("Production evaluation (threshold = 0.5):")
    out.append("")
    out.append("| Split | n | accuracy | F1 | precision | recall | ROC-AUC | EER % |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for s in p6["splits"]:
        out.append(f"| {s['split']} | {s['n']} | {s['accuracy']:.3f} | {s['f1']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} | {s['roc_auc']:.3f} | {s['eer_pct']:.2f} |")
    out.append("")
    out.append("Latency benchmark:")
    out.append("")
    out.append(f"- Cold start (W2V2 load + first MPS shader compile): {p6_lat.get('cold_total_ms_observed', 0)/1000:.1f} s")
    out.append(f"- Warm p50 / p95: {p6_lat.get('warm_total_ms_p50', '—'):.1f} ms / {p6_lat.get('warm_total_ms_p95', '—'):.1f} ms")
    out.append(f"- Bundle on disk: 26 KB (head only); encoder 360 MB downloaded on first use.")
    out.append("")

    # Phase 7
    out.append("## Phase 7 — Testing + Polish + Consolidation (2026-05-10)")
    out.append("")
    out.append("Pytest suite expanded to **33 passing tests** (1 optional skipped):")
    out.append("")
    out.append("| Module | Tests | Coverage |")
    out.append("|---|--:|---|")
    out.append("| `test_audio_features.py` | 4 | feature extractor: preprocess shape/dtype, finiteness, determinism, jitter sanity |")
    out.append("| `test_eer.py` | 4 | EER on perfect/random/known cases + metrics-at-threshold |")
    out.append("| `test_data_pipeline.py` | 9 | AudioConfig frozen contract, pad/truncate/resample/stereo collapse, end-to-end load |")
    out.append("| `test_model.py` | 10 | bundle schema, 768-d input contract, sklearn Pipeline structure, reference metrics within band |")
    out.append("| `test_inference.py` | 7 | reproduces Phase 6 in-domain (1.11% EER) and Hemg (46% EER) within tolerance, evaluate.py schema, predict.py caching |")
    out.append("")
    out.append("Bundle re-pinned to `phase7-2026-05-10` (sklearn 1.8.x) — embeddings, hyperparameters, metrics byte-identical to `phase6`.")
    out.append("")

    out.append("## Final Headline (cross-distribution Hemg test set, n=50)")
    out.append("")
    out.append("| | EER % | AUROC | latency / call | $/1k preds |")
    out.append("|---|--:|--:|--:|--:|")
    out.append("| **W2V2 + LogReg (production champion, raw audio)** | **32.0** | **0.634** | 15 ms warm | $0.0001 |")
    out.append("| 12-feature LogReg (same input as LLMs) | 84.0 | 0.085 | <1 ms | $0.0001 |")
    out.append("| Claude Opus (12-feature digest, zero-shot) | 54.0 | 0.465 | 5.3 s | $4.50 |")
    out.append("| Claude Haiku (12-feature digest, zero-shot) | 52.0 | 0.515 | 14.6 s | $0.30 |")
    out.append("| Codex GPT-5.5 (12-feature digest, zero-shot) | 44.0 | 0.530 | 8.4 s | $50.00 |")
    out.append("")
    out.append("**Input-fairness caveat:** the 12-feature digest is NOT the W2V2 model's input. The W2V2 model received raw 16 kHz audio → 768-d embedding. Audio-capable LLMs (Gemini-Audio, GPT-4o-audio) were not tested. This row is included for reference, not as a frontier-LLM audio benchmark.")
    out.append("")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Headline dashboard PNG
# ----------------------------------------------------------------------------

def build_headline_dashboard(data: dict, out_path: Path) -> None:
    p3 = data["phase3"]
    p4 = data["phase4"]
    p5 = data["phase5"]
    p5_llm = data["phase5_llm_csv"]
    p6 = data["phase6_eval"]
    p6_lat = data["phase6_latency"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    # ----- Panel 1: phase-by-phase Hemg EER progression -----
    ax = fig.add_subplot(gs[0, 0])
    phases = ["P1\nlogreg", "P2\nxgb", "P3\nxgb+aug", "P4\nW2V2+LR", "P5\nbest", "P6\nprod (full Hemg)"]
    hemg_eer = [63.0, 48.0, 36.0, 34.0, 32.0, 46.0]
    bar_colors = ["#9aa", "#9aa", "#9aa", "#3a7", "#3a7", "#48a"]
    ax.bar(phases, hemg_eer, color=bar_colors, edgecolor="#222")
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8)
    ax.text(5.4, 51, "chance (50%)", color="#666", fontsize=8, va="bottom", ha="right")
    ax.set_ylabel("Hemg test EER %  (↓ better)")
    ax.set_title("Cross-distribution EER by phase", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 70)
    for i, v in enumerate(hemg_eer):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # ----- Panel 2: in-domain vs cross-distribution gap -----
    ax = fig.add_subplot(gs[0, 1])
    in_dom = next(s for s in p6["splits"] if "in_domain" in s["split"])
    cross = next(s for s in p6["splits"] if "cross_distribution" in s["split"])
    metrics = ["ROC-AUC", "F1", "Accuracy"]
    in_vals = [in_dom["roc_auc"], in_dom["f1"], in_dom["accuracy"]]
    cross_vals = [cross["roc_auc"], cross["f1"], cross["accuracy"]]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w / 2, in_vals, w, label=f"in-domain (n={in_dom['n']})", color="#3a7", edgecolor="#222")
    ax.bar(x + w / 2, cross_vals, w, label=f"Hemg cross-dist (n={cross['n']})", color="#c64", edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Production bundle: in-domain vs cross-distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    for i, (a, b) in enumerate(zip(in_vals, cross_vals)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    # ----- Panel 3: LLM vs custom (EER) -----
    ax = fig.add_subplot(gs[0, 2])
    llm_models = ["W2V2+LR\n(raw audio)", "12-feat\nLogReg", "Codex\nGPT-5.5", "Claude\nHaiku", "Claude\nOpus"]
    llm_eer = [32.0, 84.0, 44.0, 52.0, 54.0]
    colors = ["#3a7", "#bbb", "#c64", "#c64", "#c64"]
    bars = ax.bar(llm_models, llm_eer, color=colors, edgecolor="#222")
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Hemg EER % (↓ better)")
    ax.set_title("Specialist vs frontier LLMs (50-clip sample)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 95)
    for i, v in enumerate(llm_eer):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # ----- Panel 4: cost per 1k predictions (log) -----
    ax = fig.add_subplot(gs[1, 0])
    cost_models = ["W2V2+LR", "Haiku", "Opus", "Codex 5.5"]
    cost_vals = [1e-4, 0.30, 4.50, 50.0]
    ax.bar(cost_models, cost_vals, color=["#3a7", "#c64", "#c64", "#c64"], edgecolor="#222")
    ax.set_yscale("log")
    ax.set_ylabel("$ per 1k predictions  (log)")
    ax.set_title("Cost per 1k predictions", fontsize=11, fontweight="bold")
    for i, v in enumerate(cost_vals):
        ax.text(i, v * 1.2, f"${v:g}", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=9)

    # ----- Panel 5: latency (linear, but separate axes for warm vs cold) -----
    ax = fig.add_subplot(gs[1, 1])
    lat_models = ["W2V2+LR\nwarm p50", "Codex 5.5\nzero-shot", "Haiku\nzero-shot", "Opus\nzero-shot", "W2V2+LR\ncold start"]
    lat_vals = [15, 8400, 14600, 5300, 6100]
    ax.bar(lat_models, lat_vals, color=["#3a7", "#c64", "#c64", "#c64", "#a99"], edgecolor="#222")
    ax.set_yscale("log")
    ax.set_ylabel("Latency / call (ms, log)")
    ax.set_title("Latency / call", fontsize=11, fontweight="bold")
    for i, v in enumerate(lat_vals):
        ax.text(i, v * 1.2, f"{v} ms", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # ----- Panel 6: Phase 5 ablation — every variant ≥ baseline -----
    ax = fig.add_subplot(gs[1, 2])
    p5_lb = sorted(p5["ablation_summary"], key=lambda r: r["Hemg_test_EER_%"])[:10]
    names = [r["approach"].replace("W2V2+LogReg", "W2V2+LR") for r in p5_lb]
    eers = [r["Hemg_test_EER_%"] for r in p5_lb]
    # Truncate long names
    names = [(n[:32] + "…") if len(n) > 32 else n for n in names]
    bar_colors = ["#3a7" if e <= 32.0 else "#c64" for e in eers]
    y = np.arange(len(names))[::-1]
    ax.barh(y, eers, color=bar_colors, edgecolor="#222")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(32.0, color="#222", linestyle="--", linewidth=0.8)
    ax.text(32.5, len(names) - 0.5, "ref = 32.0%", fontsize=7, color="#222")
    ax.set_xlabel("Hemg test EER %")
    ax.set_title("Phase 5 ablation — top 10 (tied or worse than ref)", fontsize=11, fontweight="bold")

    # ----- Panel 7-9: full-width subtitle row -----
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    head_text = (
        "Phase 1-7 Sprint — Deepfake Audio Detection (garystafford ↔ Hemg)"
        "\n"
        "Production champion: frozen wav2vec2-base + LogisticRegression head, 26 KB on disk.  "
        "In-domain: 1.11% EER (n=180).  Cross-distribution (Hemg full): 46.0% EER (n=100).  "
        "On a matched 50-clip Hemg subset: 32% EER, beats Claude Opus by 22 EER points and Codex by 12 — "
        "but the LLMs received a 12-feature digest, not raw audio, so this is NOT a fair frontier-LLM audio benchmark.\n"
        "Key finding: cross-distribution ceiling is structural. Augmentation, Optuna, ensembling, fusion, "
        "calibration, and PCA all tied or hurt the Phase 4 champion.  The gap is the dataset, not the head."
    )
    ax.text(0.0, 1.0, head_text, va="top", ha="left", fontsize=10, wrap=True,
            family="monospace", linespacing=1.5)

    fig.suptitle("Deepfake Audio Detection — Phase 1-7 Headline Dashboard",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    data = load_all()

    log = build_experiment_log(data)
    log_path = RESULTS / "EXPERIMENT_LOG.md"
    log_path.write_text(log)
    print(f"wrote {log_path}  ({len(log)} chars)")

    png_path = RESULTS / "headline_dashboard.png"
    build_headline_dashboard(data, png_path)
    print(f"wrote {png_path}  ({png_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

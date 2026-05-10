"""Streamlit UI for the deepfake audio detector — Phase 6 production demo.

Usage:
    streamlit run app.py

Shows: file upload + record, REAL/FAKE verdict with calibrated probability,
forensic-feature breakdown (the interpretable 12-feature digest from Phase 5),
threshold slider, latency breakdown, and the cross-distribution + LLM-vs-specialist
context the Phase 5 retraction made non-negotiable.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJ = Path(__file__).resolve().parent
RESULTS = PROJ / "results"
MODELS = PROJ / "models"

st.set_page_config(
    page_title="Deepfake Audio Detector — Phase 6 demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading wav2vec2-base + LogReg head…")
def _load_pipeline():
    from src.predict import load_bundle
    from src.w2v2_encoder import get_encoder

    bundle = load_bundle()
    feat_ext, model, device = get_encoder()
    return bundle, feat_ext, model, device


@st.cache_resource(show_spinner="Pre-warming wav2vec2 + MPS shader graph (one-time, ~6 s)…")
def _prewarm() -> dict:
    """Pre-warm the encoder + first forward-pass shader compile at app boot.

    Loading the model is only half of the cold-start cost — the other half is
    the first MPS shader compile, which only happens on a real forward pass.
    Running a dummy ``predict_array`` over 1.5 s of silence at boot pays both
    costs once, so the first real user request gets the warm ~15 ms path
    instead of the ~6 s cold start.

    Cached as a resource so it runs once per Streamlit process, not per user
    session.
    """
    from src.data_pipeline import AudioConfig
    from src.predict import predict_array

    _load_pipeline()  # ensure encoder + bundle are in memory before the dummy call

    cfg = AudioConfig()
    dummy = np.zeros(cfg.target_len, dtype=np.float32)
    t0 = time.perf_counter()
    predict_array(dummy, sr=cfg.target_sr, threshold=0.5)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    return {"cold_ms_observed": cold_ms}


@st.cache_data(show_spinner=False)
def _load_eval_results():
    p = RESULTS / "phase6_evaluation.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@st.cache_data(show_spinner=False)
def _load_latency_bench():
    p = RESULTS / "phase6_latency_bench.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@st.cache_data(show_spinner=False)
def _load_phase5_apples():
    p = RESULTS / "phase5_apples_to_apples.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def run_inference(audio_bytes: bytes, threshold: float) -> dict:
    from src.data_pipeline import AudioConfig, to_fixed_mono
    from src.predict import predict_array
    import soundfile as sf

    cfg = AudioConfig()
    bio = io.BytesIO(audio_bytes)
    y, sr = sf.read(bio, dtype="float32", always_2d=False)
    yfix = to_fixed_mono(y, sr, cfg)

    t0 = time.perf_counter()
    res = predict_array(yfix, sr=cfg.target_sr, threshold=threshold)
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "label": res.label,
        "fake_prob": res.fake_probability,
        "encode_ms": res.encode_ms,
        "classify_ms": res.classify_ms,
        "total_ms": total_ms,
        "duration_input_s": float(len(y) / max(sr, 1)),
        "sr_input": int(sr),
    }


def compute_forensic_digest(audio_bytes: bytes) -> dict:
    """Compute the 12-feature interpretable digest used in Phase 5 LLM head-to-head."""
    import io as _io

    import soundfile as sf

    from src.audio_features import (
        FeatureConfig,
        feature_names,
        prosody_features,
        spectral_features,
    )
    from src.data_pipeline import AudioConfig, to_fixed_mono

    cfg_a = AudioConfig()
    bio = _io.BytesIO(audio_bytes)
    y, sr = sf.read(bio, dtype="float32", always_2d=False)
    yfix = to_fixed_mono(y, sr, cfg_a)

    cfg_f = FeatureConfig(target_sr=cfg_a.target_sr, max_duration_s=cfg_a.duration_s)
    spec = spectral_features(yfix, cfg_f)
    pros = prosody_features(yfix, cfg_f)

    names = feature_names(cfg_f)
    spec_idx = {n: i for i, n in enumerate(names) if n.startswith("spec_")}
    spec_offset = next(i for i, n in enumerate(names) if n.startswith("spec_"))

    def get_spec(name: str) -> float:
        return float(spec[spec_idx[name] - spec_offset])

    return {
        "f0_mean_hz": float(pros[0]),
        "f0_std_hz": float(pros[1]),
        "jitter_local": float(pros[7]),
        "shimmer_local": float(pros[8]),
        "voicing_ratio": float(pros[9]),
        "spec_centroid_mean": get_spec("spec_centroid_mean"),
        "spec_bandwidth_mean": get_spec("spec_bandwidth_mean"),
        "spec_rolloff_mean": get_spec("spec_rolloff_mean"),
        "spec_flatness_mean": get_spec("spec_flatness_mean"),
        "spec_zcr_mean": get_spec("spec_zcr_mean"),
        "spec_rms_mean": get_spec("spec_rms_mean"),
    }


def main() -> None:
    # Pre-warm encoder + MPS shader graph at app boot so the first real user
    # request gets the warm path (~15 ms) rather than the 6 s cold start.
    prewarm_info = _prewarm()

    st.title("Deepfake Audio Detector")
    st.caption(
        "Wav2Vec2-base (frozen 768-d) + Logistic Regression head — Phase 6 production demo."
    )

    bundle, *_ = _load_pipeline()
    eval_results = _load_eval_results()
    bench = _load_latency_bench()
    apples = _load_phase5_apples()

    with st.sidebar:
        st.header("Model")
        st.markdown(f"**Encoder:** `{bundle['encoder_id']}`")
        st.markdown(f"**Head:** Logistic Regression (C=1.0, 768→1)")
        st.markdown(f"**Version:** `{bundle['version']}`")
        st.markdown("**Input contract:** mono · 16 kHz · 1.5 s window")

        st.header("Threshold")
        threshold = st.slider(
            "FAKE if p ≥",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Higher = stricter (fewer false alarms, more missed fakes).",
        )

        st.header("In-domain test set")
        if eval_results:
            in_dom = next(
                (s for s in eval_results["splits"] if "in_domain" in s["split"]), None
            )
            cross = next(
                (s for s in eval_results["splits"] if "cross_distribution" in s["split"]), None
            )
            if in_dom:
                c1, c2 = st.columns(2)
                c1.metric("ROC-AUC", f"{in_dom['roc_auc']:.3f}")
                c2.metric("EER", f"{in_dom['eer_pct']:.2f}%")
            if cross:
                st.markdown("**Cross-distribution (Hemg):**")
                c1, c2 = st.columns(2)
                c1.metric("ROC-AUC", f"{cross['roc_auc']:.3f}")
                c2.metric("EER", f"{cross['eer_pct']:.2f}%")

        if bench:
            st.header("Inference latency")
            st.metric("Warm p50", f"{bench['warm_total_ms_p50']:.1f} ms")
            st.metric("Warm p95", f"{bench['warm_total_ms_p95']:.1f} ms")
            cold_observed = prewarm_info.get("cold_ms_observed", bench["cold_total_ms_observed"]) / 1000.0
            st.caption(
                f"Cold start: {cold_observed:.1f} s — paid once at app boot (pre-warm). "
                "First user request gets the warm path."
            )

    tab_predict, tab_research, tab_about = st.tabs(["Predict", "Research", "About / Limitations"])

    with tab_predict:
        st.subheader("Upload audio")
        up = st.file_uploader(
            "Drop a .wav, .flac, or .mp3 (mono or stereo, any sample rate)",
            type=["wav", "flac", "mp3", "ogg"],
        )
        if up is None:
            st.info("Upload a clip to get a prediction. Real human speech vs AI-generated. The model uses the first 1.5 seconds.")
            return

        audio_bytes = up.read()
        st.audio(audio_bytes)

        with st.spinner("Encoding + classifying…"):
            try:
                result = run_inference(audio_bytes, threshold)
                digest = compute_forensic_digest(audio_bytes)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        col_verdict, col_prob, col_lat = st.columns([1, 1, 1])
        verdict = result["label"].upper()
        col_verdict.metric("Verdict", verdict)
        col_prob.metric("p(FAKE)", f"{result['fake_prob']:.3f}", help=f"Threshold = {threshold:.2f}")
        col_lat.metric(
            "Latency",
            f"{result['total_ms']:.1f} ms",
            help=f"Encode {result['encode_ms']:.1f} + classify {result['classify_ms']:.1f}",
        )

        st.markdown("---")

        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.subheader("Forensic-feature digest")
            st.caption(
                "The 12 interpretable features the Phase 5 LLMs were given (jitter, shimmer, F0, "
                "spectral centroid/bandwidth/rolloff/flatness, ZCR, RMS, voicing ratio). "
                "These do **not** drive this model's prediction — the 768-d Wav2Vec2 embedding does — "
                "but they're useful context for understanding what's in the clip."
            )
            df = pd.DataFrame(
                [{"feature": k, "value": v} for k, v in digest.items()]
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

        with c_right:
            st.subheader("Probability bar")
            st.progress(min(max(result["fake_prob"], 0.0), 1.0), text=f"FAKE probability = {result['fake_prob']:.3f}")
            st.markdown(
                f"- **Input duration:** {result['duration_input_s']:.2f} s "
                f"(model uses first 1.5 s)\n"
                f"- **Input sample rate:** {result['sr_input']} Hz "
                f"(resampled to 16 kHz)"
            )

    with tab_research:
        st.subheader("In-domain vs cross-distribution")
        if eval_results:
            df = pd.DataFrame(eval_results["splits"])[
                ["split", "n", "accuracy", "f1", "precision", "recall", "roc_auc", "eer_pct"]
            ]
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(
                "**The honest story:** in-domain (garystafford test) gives 1.1% EER. "
                "On a held-out distribution (Hemg), full-set EER is in the 40-50% range — "
                "consistent with the Phase 1-5 finding that codec/recording-condition shortcuts "
                "drive most of the in-domain signal. This is the structural ceiling for "
                "this dataset/architecture pair without more diverse training data."
            )

        st.subheader("LLM head-to-head — fairness disclosure")
        st.markdown(
            "Phase 5 ran a side-test against Claude Opus, Claude Haiku, and Codex GPT-5.5 "
            "on a 50-clip Hemg sample. **The LLMs received only a 12-feature acoustic digest** "
            "(this app's forensic table) because the local Claude/Codex CLIs do not accept "
            "audio input. The W2V2+LogReg model received the raw waveform → 768-d embedding. "
            "**This is not an apples-to-apples benchmark** and is not framed as one."
        )
        if apples:
            st.markdown("**Apples-to-apples (12 features → both sides):**")
            ap_df = pd.DataFrame(
                [
                    {"model": "12-feat LogReg (specialist)", "input": "12 features", "Hemg test EER %": apples.get("hemg_EER_pct", "—"), "ROC-AUC": apples.get("hemg_AUROC", "—")},
                    {"model": "Codex GPT-5.5 (zero-shot)", "input": "12 features", "Hemg test EER %": 44.0, "ROC-AUC": 0.530},
                    {"model": "Claude Haiku (zero-shot)", "input": "12 features", "Hemg test EER %": 52.0, "ROC-AUC": 0.515},
                    {"model": "Claude Opus (zero-shot)", "input": "12 features", "Hemg test EER %": 54.0, "ROC-AUC": 0.465},
                    {"model": "W2V2 + LogReg (this app)", "input": "raw 16 kHz audio", "Hemg test EER %": 32.0, "ROC-AUC": 0.634},
                ]
            )
            st.dataframe(ap_df, hide_index=True, use_container_width=True)
            st.caption(
                "On the same 12-feature digest, the matched specialist LogReg collapses cross-domain "
                "(it overfits the codec shortcut). The frontier LLMs partially ignore that shortcut "
                "but still trail the 768-d W2V2 representation by ~12-22 EER points. Posture: "
                "specialist > LLM **only when input parity is broken in the specialist's favor**."
            )

        st.subheader("Phase 1-5 highlights")
        st.markdown(
            "- **Phase 1**: a single MFCC dim does 66% of the work — codec shortcut, not a deepfake feature.\n"
            "- **Phase 2**: every model hits 0% in-domain EER and 48-64% on Hemg — the shortcut doesn't transfer.\n"
            "- **Phase 3**: random per-sample augmentation lifts Hemg AUROC 0.524 → 0.670 (no single aug helps).\n"
            "- **Phase 4**: 50 Optuna trials produce zero improvement; frozen W2V2 + LogReg jumps 14 EER points.\n"
            "- **Phase 5**: every advanced trick (PCA, isotonic, Platt, late fusion, hybrid) ties or hurts."
        )

    with tab_about:
        st.subheader("Intended use")
        st.markdown(
            "Research demo for binary deepfake-vs-real classification of short speech clips. "
            "Trained on the `garystafford/deepfake-audio-detection` dataset; evaluated cross-distribution "
            "on `Hemg/Deepfake-Audio-Dataset`."
        )
        st.subheader("Limitations")
        st.markdown(
            "- **Distribution shift**: the model is far less reliable on audio whose codec, recording "
            "  device, or TTS generator differs from the training set. Cross-distribution Hemg EER is "
            "  46% on the full 100-clip test set (close to chance) — this is the production reality.\n"
            "- **Window length**: only the first 1.5 seconds of the input drive the prediction. Long "
            "  clips need to be chunked (the Phase 4 protocol matched).\n"
            "- **Forensic-audio claims**: the literature claim that synthetic voices have unnaturally "
            "  *low* shimmer was **not** supported by the Phase 4 error analysis — missed fakes had "
            "  *lower* shimmer than caught fakes. Treat individual feature values as descriptive, not "
            "  diagnostic.\n"
            "- **Adversarial robustness**: not tested. Modern TTS likely adapts to detectors quickly "
            "  (the deepfake arms race)."
        )
        st.subheader("Inputs the model was *not* trained on")
        st.markdown(
            "- Background music, multi-speaker mixtures, languages other than English\n"
            "- Audio shorter than ~1 s (model still emits a probability — interpret skeptically)\n"
            "- Compressed/streamed audio (MP3 at low bitrates) outside the training-set distribution"
        )
        st.subheader("Reproducibility")
        st.markdown(
            "- All Phase 1-5 notebooks live under `notebooks/` with cell outputs preserved.\n"
            "- Training: `python -m src.train`\n"
            "- Evaluation: `python -m src.evaluate`\n"
            "- Single-file inference: `python -m src.predict --audio path/to/file.wav`"
        )


if __name__ == "__main__":
    main()

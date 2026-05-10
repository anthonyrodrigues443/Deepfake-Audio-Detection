"""A/B benchmark of cold-start with vs. without the app-boot pre-warm hook.

Runs two scenarios in fresh Python subprocesses (so module-level caches
don't leak between runs):

  Scenario A — no pre-warm (the demo's behaviour before commit 2a52de7):
      Fresh process. The first user request pays the full cold start
      (model load + first MPS shader compile). Subsequent requests are warm.

  Scenario B — with pre-warm (current behaviour):
      Fresh process. App boot runs a dummy ``predict_array`` over 1.5 s of
      silence. The pre-warm cost is paid once, before any user request
      arrives; the first real user request lands on the warm path.

For each scenario we measure end-to-end ``predict_array`` latency (encode +
classify) on a dummy 1.5 s 16 kHz silence buffer, so the measurement isolates
the inference path from any audio-loading or feature-extraction overhead.

Usage:
    .venv/bin/python scripts/bench_prewarm.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SENTINEL = "BENCH_JSON="


_INNER_TEMPLATE = r"""
import json, time
import numpy as np
from src.data_pipeline import AudioConfig
from src.predict import predict_array

cfg = AudioConfig()
dummy = np.zeros(cfg.target_len, dtype=np.float32)

scenario = {scenario!r}

prewarm_ms = None
if scenario == "with_prewarm":
    t = time.perf_counter()
    predict_array(dummy, sr=cfg.target_sr, threshold=0.5)
    prewarm_ms = (time.perf_counter() - t) * 1000.0

# First "user" call. Under no_prewarm this is the cold path; under
# with_prewarm the prewarm above already paid the model load + shader compile.
t0 = time.perf_counter()
predict_array(dummy, sr=cfg.target_sr, threshold=0.5)
first_user_ms = (time.perf_counter() - t0) * 1000.0

# Steady-state 30-call warm bench
warm_ms = []
for _ in range(30):
    t = time.perf_counter()
    predict_array(dummy, sr=cfg.target_sr, threshold=0.5)
    warm_ms.append((time.perf_counter() - t) * 1000.0)

print({sentinel!r} + json.dumps({{
    "scenario": scenario,
    "first_user_ms": first_user_ms,
    "prewarm_ms": prewarm_ms,
    "warm_ms_samples": warm_ms,
}}))
"""


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _run_scenario(scenario: str) -> dict:
    code = _INNER_TEMPLATE.format(scenario=scenario, sentinel=SENTINEL)
    cp = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJ),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        cp.check_returncode()
    for line in cp.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise RuntimeError(f"sentinel {SENTINEL!r} not found in stdout")


def _summarize(raw: dict) -> dict:
    w = raw["warm_ms_samples"]
    out = {
        "scenario": raw["scenario"],
        "first_user_ms": round(raw["first_user_ms"], 1),
        "warm_p50_ms": round(_percentile(w, 50), 2),
        "warm_p95_ms": round(_percentile(w, 95), 2),
        "warm_mean_ms": round(sum(w) / len(w), 2),
        "warm_min_ms": round(min(w), 2),
        "warm_n": len(w),
    }
    if raw.get("prewarm_ms") is not None:
        out["prewarm_ms"] = round(raw["prewarm_ms"], 1)
    return out


def main() -> None:
    print("Running Scenario A (no pre-warm) in fresh subprocess...", flush=True)
    a = _summarize(_run_scenario("no_prewarm"))

    print("Running Scenario B (with pre-warm) in fresh subprocess...", flush=True)
    b = _summarize(_run_scenario("with_prewarm"))

    speedup = a["first_user_ms"] / b["first_user_ms"] if b["first_user_ms"] > 0 else float("inf")
    bench = {
        "method": "fresh-subprocess A/B; dummy 1.5 s silence; predict_array end-to-end timing",
        "device_env": "mps (apple silicon, M-series)",
        "n_warm": a["warm_n"],
        "scenarios": {"no_prewarm": a, "with_prewarm": b},
        "first_user_speedup": round(speedup, 1),
        "first_user_delta_ms": round(a["first_user_ms"] - b["first_user_ms"], 0),
    }

    out_path = PROJ / "results" / "phase6_latency_bench_prewarm.json"
    out_path.write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))
    print(f"\nWrote {out_path.relative_to(PROJ)}")


if __name__ == "__main__":
    main()

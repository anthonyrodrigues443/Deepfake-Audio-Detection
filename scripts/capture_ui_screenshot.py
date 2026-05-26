"""Capture results/ui_screenshot.png of the running Streamlit app.

Assumes Streamlit is already serving on http://localhost:8501 (started separately).
Uploads results/_demo_clip.wav, waits for the prediction to render, then takes a
full-page screenshot. Writes results/ui_screenshot.png.

Run:
    streamlit run app.py --server.headless true --server.port 8501 &
    python scripts/capture_ui_screenshot.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJ = Path(__file__).resolve().parents[1]
DEMO_CLIP = PROJ / "results" / "_demo_clip.wav"
OUT_PATH = PROJ / "results" / "ui_screenshot.png"
TAB_OUT = PROJ / "results" / "ui_screenshot_research.png"


def capture(url: str, demo_clip: Path, out_path: Path, tab_out: Path) -> int:
    if not demo_clip.exists():
        print(f"missing demo clip: {demo_clip}", file=sys.stderr)
        return 2

    # Allow override to a pre-installed Chromium (e.g. ~/Library/Caches/ms-playwright/chromium-1217)
    import os
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    launch_kwargs = {"headless": True}
    if exe:
        launch_kwargs["executable_path"] = exe

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1700},
                                  device_scale_factor=2)
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        # Streamlit takes a moment after networkidle to finish the first render
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=30000)
        page.wait_for_selector('text=Deepfake Audio Detector', timeout=30000)
        time.sleep(2.0)  # let metric cards finalize values

        # Upload the demo clip into the file_uploader
        file_input = page.locator('input[type="file"]')
        file_input.set_input_files(str(demo_clip))

        # Wait for the predict-tab output to appear (Verdict metric label)
        page.wait_for_selector('text=Verdict', timeout=90000)
        page.wait_for_selector('text=Forensic-feature digest', timeout=30000)
        time.sleep(1.5)

        page.screenshot(path=str(out_path), full_page=True)
        print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

        # Now click the Research tab and capture a second view
        try:
            research_tab = page.locator('button[role="tab"]:has-text("Research")').first
            research_tab.click()
            page.wait_for_selector('text=In-domain vs cross-distribution', timeout=15000)
            time.sleep(1.0)
            page.screenshot(path=str(tab_out), full_page=True)
            print(f"wrote {tab_out}  ({tab_out.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"research tab capture skipped: {e}", file=sys.stderr)

        browser.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--clip", type=Path, default=DEMO_CLIP)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--tab-out", type=Path, default=TAB_OUT)
    args = ap.parse_args()
    return capture(args.url, args.clip, args.out, args.tab_out)


if __name__ == "__main__":
    raise SystemExit(main())

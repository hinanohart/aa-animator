#!/usr/bin/env python3
# Copyright 2026 Hinano Hart <hinanohart@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""v0.2 multi-style comparison runner.

Generates 3 images × 4 styles = 12 MP4s, computes metrics, sends to Discord,
and writes /tmp/aa_v02_comparison/results.json.

Usage:
    python3 run_v02_comparison.py [--cols N] [--fps N] [--duration S]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure src is importable when run from scripts/
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aa_animator_v2.metrics_v02 import compute_metrics
from aa_animator_v2.style_a_gallery import generate_style_a
from aa_animator_v2.style_b_boo_inspired import generate_style_b
from aa_animator_v2.style_c_dog_shape import generate_style_c
from aa_animator_v2.style_d_all import generate_style_d

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OUT_DIR = Path("/tmp/aa_v02_comparison")
_DISCORD_HOOK = Path.home() / ".claude/hooks/discord-send-file.sh"

_IMAGES: list[tuple[str, str]] = [
    ("/tmp/aa_illustration_tests/generated_ghost.png", "ghost"),
    ("/tmp/aa_illustration_tests/generated_monster.png", "monster"),
    ("/tmp/aa_illustration_tests/local_character_cutout.png", "cutout"),
]

_STYLES: list[tuple[str, str]] = [
    ("A", "gallery"),
    ("B", "boo-inspired motion"),
    ("C", "DoG+6D shape"),
    ("D", "all-in-one"),
]

_GENERATORS = {
    "A": generate_style_a,
    "B": generate_style_b,
    "C": generate_style_c,
    "D": generate_style_d,
}


# ---------------------------------------------------------------------------
# Discord sender
# ---------------------------------------------------------------------------

def _send_discord(file_path: Path, label: str) -> bool:
    """Send a file to Discord via the local hook script.

    Args:
        file_path: Path to file to send.
        label: Human-readable label for Discord message.

    Returns:
        True on success, False on failure.
    """
    if not _DISCORD_HOOK.exists():
        print(f"[discord] hook not found: {_DISCORD_HOOK}", file=sys.stderr)
        return False

    size_mb = file_path.stat().st_size / 1024 / 1024
    if size_mb > 8.0:
        print(f"[discord] skip {file_path.name} — size {size_mb:.1f}MB > 8MB", file=sys.stderr)
        return False

    result = subprocess.run(
        [str(_DISCORD_HOOK), str(file_path), f"aa-animator v0.2 {label}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        print(f"[discord] sent: {file_path.name}", file=sys.stderr)
        return True
    else:
        print(f"[discord] error ({result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="aa-animator v0.2 multi-style comparison")
    parser.add_argument("--cols", type=int, default=80, help="character canvas width")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0, help="clip length in seconds")
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--no-discord", action="store_true", help="skip Discord upload")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check input images exist
    for img_path, img_name in _IMAGES:
        if not Path(img_path).exists():
            print(f"[runner] WARNING: image not found, skipping: {img_path}", file=sys.stderr)

    results: list[dict] = []
    discord_sent: int = 0
    discord_failed: int = 0

    print(f"[runner] output dir: {_OUT_DIR}", file=sys.stderr)
    print("[runner] generating 3 images × 4 styles = up to 12 MP4s", file=sys.stderr)

    for style_id, style_label in _STYLES:
        gen_fn = _GENERATORS[style_id]
        for img_path_str, img_name in _IMAGES:
            img_path = Path(img_path_str)
            if not img_path.exists():
                results.append({
                    "style": style_id,
                    "image": img_name,
                    "status": "skipped_missing_input",
                    "output_path": None,
                    "metrics": {},
                })
                continue

            out_name = f"style{style_id}_{img_name}.mp4"
            out_path = _OUT_DIR / out_name
            label = f"style {style_id} ({style_label}) - {img_name}"

            print(f"\n[runner] generating: {label}", file=sys.stderr)
            t0 = time.time()
            try:
                info = gen_fn(
                    img_path,
                    out_path,
                    cols=args.cols,
                    fps=args.fps,
                    duration=args.duration,
                    canvas_size=args.canvas_size,
                )
                elapsed = time.time() - t0
                print(f"[runner] done in {elapsed:.1f}s → {out_path.name}", file=sys.stderr)

                # Compute metrics
                metrics = compute_metrics(out_path)
                print(
                    f"[runner] metrics: KL={metrics['palette_kl_divergence']:.3f} "
                    f"top={metrics['top_glyph_ratio']:.2%} "
                    f"heavy={metrics['outline_heavy_ratio']:.2%} "
                    f"hamming={metrics['hamming_distance_mean']:.2%} "
                    f"swing={metrics['silhouette_swing']:.3f} "
                    f"size={metrics['file_size_mb']:.1f}MB",
                    file=sys.stderr,
                )

                result_entry: dict = {
                    "style": style_id,
                    "style_label": style_label,
                    "image": img_name,
                    "status": "ok",
                    "output_path": str(out_path),
                    "elapsed_s": round(elapsed, 1),
                    "canvas_w": info["canvas_w"],
                    "canvas_h": info["canvas_h"],
                    "n_frames": info["n_frames"],
                    "metrics": metrics,
                    "discord_sent": False,
                }

                # Discord send
                if not args.no_discord:
                    sent = _send_discord(out_path, f"style {style_id} ({style_label}) - {img_name}")
                    result_entry["discord_sent"] = sent
                    if sent:
                        discord_sent += 1
                    else:
                        discord_failed += 1
                    time.sleep(0.5)  # rate limit courtesy

                results.append(result_entry)

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"[runner] FAILED: {label} — {exc}", file=sys.stderr)
                results.append({
                    "style": style_id,
                    "style_label": style_label,
                    "image": img_name,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_s": round(elapsed, 1),
                    "output_path": None,
                    "metrics": {},
                    "discord_sent": False,
                })

    # Write results.json
    results_path = _OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": {
                    "cols": args.cols,
                    "fps": args.fps,
                    "duration": args.duration,
                    "canvas_size": args.canvas_size,
                },
                "discord_sent": discord_sent,
                "discord_failed": discord_failed,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[runner] results written: {results_path}", file=sys.stderr)

    # Summary table
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    skip = [r for r in results if r["status"] == "skipped_missing_input"]
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  Generated: {len(ok)}/12   Errors: {len(err)}   Skipped: {len(skip)}", file=sys.stderr)
    print(f"  Discord: sent={discord_sent}  failed={discord_failed}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    if ok:
        print(f"\n  {'Style':<8} {'Image':<12} {'KL':>6} {'Top%':>7} {'Heavy%':>8} {'Hmng%':>7} {'Swing':>7} {'MB':>5}", file=sys.stderr)
        print(f"  {'-'*8} {'-'*12} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*5}", file=sys.stderr)
        for r in ok:
            m = r["metrics"]
            print(
                f"  {r['style']:<8} {r['image']:<12} "
                f"{m.get('palette_kl_divergence', -1):>6.3f} "
                f"{m.get('top_glyph_ratio', -1):>7.1%} "
                f"{m.get('outline_heavy_ratio', -1):>8.1%} "
                f"{m.get('hamming_distance_mean', -1):>7.1%} "
                f"{m.get('silhouette_swing', -1):>7.3f} "
                f"{m.get('file_size_mb', 0):>5.1f}",
                file=sys.stderr,
            )

    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())

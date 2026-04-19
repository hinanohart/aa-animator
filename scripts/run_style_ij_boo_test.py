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
"""v0.6 style_i + style_j boo test runner.

Generates 3 images × 2 styles = 6 MP4s in /tmp/aa_v06_boo/,
computes metrics, sends to Discord, writes results.json.

Usage:
    python3 run_style_ij_boo_test.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aa_animator_v2.metrics_v02 import compute_metrics
from aa_animator_v2.style_i_long_boo import generate_style_i
from aa_animator_v2.style_j_slime_boo import generate_style_j

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OUT_DIR = Path("/tmp/aa_v06_boo")
_DISCORD_HOOK = Path.home() / ".claude/hooks/discord-send-file.sh"

_IMAGES: list[tuple[str, str]] = [
    ("/tmp/aa_illustration_tests/generated_ghost.png", "ghost"),
    ("/tmp/aa_illustration_tests/generated_monster.png", "monster"),
    ("/tmp/aa_illustration_tests/local_character_cutout.png", "cutout"),
]

_STYLES: list[tuple[str, str, object]] = [
    ("I", "long+boo", generate_style_i),
    ("J", "slime+boo", generate_style_j),
]


# ---------------------------------------------------------------------------
# Discord sender (same pattern as run_style_g_test.py)
# ---------------------------------------------------------------------------


def _send_discord(file_path: Path, label: str) -> bool:
    if not _DISCORD_HOOK.exists():
        print(f"  [discord] hook not found: {_DISCORD_HOOK}", file=sys.stderr)
        return False
    size_mb = file_path.stat().st_size / 1024 / 1024
    if size_mb > 8.0:
        print(f"  [discord] skip {file_path.name} — {size_mb:.1f}MB > 8MB", file=sys.stderr)
        return False
    result = subprocess.run(
        [str(_DISCORD_HOOK), str(file_path), label],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(
            f"  [discord] FAILED ({result.returncode}): {result.stderr.strip()[:120]}",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="aa-animator v0.6 style_i + style_j boo runner")
    parser.add_argument("--dry-run", action="store_true", help="skip Discord upload")
    parser.add_argument("--cols", type=int, default=80)
    parser.add_argument("--canvas-size", type=int, default=512)
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    discord_ok = 0
    discord_fail = 0
    total = len(_IMAGES) * len(_STYLES)
    done = 0

    print(f"[runner] output dir: {_OUT_DIR}", file=sys.stderr)
    print(
        f"[runner] generating {len(_IMAGES)} images × {len(_STYLES)} styles = {total} MP4s",
        file=sys.stderr,
    )

    for style_id, style_label, gen_fn in _STYLES:
        style_fps = 24 if style_id == "I" else 30
        style_dur = 4.0 if style_id == "I" else 5.0
        for img_path_str, img_name in _IMAGES:
            img_path = Path(img_path_str)
            done += 1
            out_name = f"style{style_id}_{img_name}.mp4"
            out_path = _OUT_DIR / out_name
            discord_label = f"aa-animator v0.6 style_{style_id.lower()} {style_label} - {img_name}"

            print(f"\n[{done}/{total}] {discord_label}", file=sys.stderr)

            if not img_path.exists():
                print(f"  SKIP — image not found: {img_path}", file=sys.stderr)
                results.append(
                    {
                        "style": style_id,
                        "image": img_name,
                        "status": "skipped_missing_input",
                        "output_path": None,
                    }
                )
                continue

            t0 = time.time()
            try:
                info = gen_fn(
                    img_path,
                    out_path,
                    cols=args.cols,
                    canvas_size=args.canvas_size,
                    fps=style_fps,
                    duration=style_dur,
                )
                elapsed = time.time() - t0
                size_mb = out_path.stat().st_size / 1_048_576

                metrics = compute_metrics(out_path)

                print(
                    f"  OK  {out_name}  {size_mb:.2f}MB  {info['n_frames']}f  {elapsed:.1f}s  "
                    f"outline_ring_cells={info.get('outline_ring_cells', '?')}  "
                    f"blue_glow_cells={info.get('blue_glow_cells', '?')}  "
                    f"KL={metrics['palette_kl_divergence']:.3f}  "
                    f"heavy={metrics['outline_heavy_ratio']:.1%}",
                    file=sys.stderr,
                )

                rec: dict = {
                    "style": style_id,
                    "style_label": style_label,
                    "image": img_name,
                    "status": "ok",
                    "output_path": str(out_path),
                    "elapsed_s": round(elapsed, 1),
                    "size_mb": round(size_mb, 3),
                    "n_frames": info["n_frames"],
                    "canvas_w": info["canvas_w"],
                    "canvas_h": info["canvas_h"],
                    "outline_ring_cells": info.get("outline_ring_cells", 0),
                    "blue_glow_cells": info.get("blue_glow_cells", 0),
                    "metrics": metrics,
                    "discord_sent": False,
                }

                if not args.dry_run:
                    sent = _send_discord(out_path, discord_label)
                    rec["discord_sent"] = sent
                    if sent:
                        discord_ok += 1
                        print("  [discord] sent", file=sys.stderr)
                    else:
                        discord_fail += 1
                    time.sleep(0.5)

                results.append(rec)

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  FAIL  {exc}", file=sys.stderr)
                results.append(
                    {
                        "style": style_id,
                        "style_label": style_label,
                        "image": img_name,
                        "status": "error",
                        "error": str(exc),
                        "elapsed_s": round(elapsed, 1),
                        "output_path": None,
                        "discord_sent": False,
                    }
                )

    # Write results.json
    results_path = _OUT_DIR / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "discord_sent": discord_ok,
                "discord_failed": discord_fail,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n[runner] results written: {results_path}", file=sys.stderr)

    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") == "error"]
    skip = [r for r in results if r.get("status") == "skipped_missing_input"]

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(
        f"  Generated: {len(ok)}/{total}   Errors: {len(err)}   Skipped: {len(skip)}",
        file=sys.stderr,
    )
    if not args.dry_run:
        print(f"  Discord: sent={discord_ok}  failed={discord_fail}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    if ok:
        print(
            f"\n  {'Style':<8} {'Image':<12} {'MB':>5} {'Ring':>6} {'Blue':>6} {'Heavy%':>8} {'KL':>6}",
            file=sys.stderr,
        )
        print(
            f"  {'-' * 8} {'-' * 12} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 6}",
            file=sys.stderr,
        )
        for r in ok:
            m = r["metrics"]
            print(
                f"  {r['style']:<8} {r['image']:<12} "
                f"{r['size_mb']:>5.2f} "
                f"{r['outline_ring_cells']:>6} "
                f"{r['blue_glow_cells']:>6} "
                f"{m.get('outline_heavy_ratio', -1):>8.1%} "
                f"{m.get('palette_kl_divergence', -1):>6.3f}",
                file=sys.stderr,
            )

    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())

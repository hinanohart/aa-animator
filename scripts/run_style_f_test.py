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
"""Style F dynamic lighting animation test runner.

Generates 3 images × 5 patterns = 15 MP4s in /tmp/aa_v04_lighting/,
then sends all 15 to Discord with labelled messages.

Usage:
    python3 run_style_f_test.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aa_animator_v2.style_f_lighting import generate_style_f, VALID_PATTERNS  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OUT_DIR = Path("/tmp/aa_v04_lighting")
_DISCORD_HOOK = Path.home() / ".claude/hooks/discord-send-file.sh"

_IMAGES: list[tuple[str, str]] = [
    ("/tmp/aa_illustration_tests/generated_ghost.png", "ghost"),
    ("/tmp/aa_illustration_tests/generated_monster.png", "monster"),
    ("/tmp/aa_illustration_tests/local_character_cutout.png", "cutout"),
]

_PATTERNS: list[str] = list(VALID_PATTERNS)


# ---------------------------------------------------------------------------
# Discord sender
# ---------------------------------------------------------------------------

def _send_discord(file_path: Path, label: str) -> bool:
    """Send a file to Discord via the local hook script.

    Args:
        file_path: Path to file to send.
        label: Human-readable message attached to the upload.

    Returns:
        True on success, False on failure.
    """
    if not _DISCORD_HOOK.exists():
        print(f"  [discord] hook not found: {_DISCORD_HOOK}")
        return False
    result = subprocess.run(
        [str(_DISCORD_HOOK), str(file_path), label],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"  [discord] FAILED ({result.returncode}): {result.stderr.strip()[:120]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Style F dynamic lighting test runner")
    parser.add_argument("--dry-run", action="store_true", help="Skip Discord upload")
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--rows", type=int, default=41)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--light-intensity", type=float, default=0.5)
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    discord_ok = 0
    discord_fail = 0

    total = len(_IMAGES) * len(_PATTERNS)
    done = 0

    for img_path, img_name in _IMAGES:
        if not Path(img_path).exists():
            print(f"[SKIP] image not found: {img_path}")
            continue

        for pattern in _PATTERNS:
            done += 1
            out_name = f"styleF_{pattern}_{img_name}.mp4"
            out_path = _OUT_DIR / out_name
            label = f"aa-animator v0.4 Style F lighting:{pattern} - {img_name}"

            print(f"[{done}/{total}] {label}")
            t0 = time.time()

            try:
                info = generate_style_f(
                    img_path,
                    out_path,
                    pattern=pattern,
                    cols=args.cols,
                    rows=args.rows,
                    fps=args.fps,
                    duration=args.duration,
                    sigma=args.sigma,
                    light_intensity=args.light_intensity,
                )
                elapsed = time.time() - t0
                size_mb = out_path.stat().st_size / 1_048_576
                print(
                    f"  OK  {out_name}  {size_mb:.2f} MB  "
                    f"{info['n_frames']}f  {elapsed:.1f}s  "
                    f"max_delta={info['max_delta_per_frame']:.4f}  "
                    f"smooth={info['smooth_verified']}  "
                    f"invariant={info['char_grid_invariant']}"
                )
                rec = {
                    "output": str(out_path),
                    "pattern": pattern,
                    "image": img_name,
                    "size_mb": round(size_mb, 3),
                    "n_frames": info["n_frames"],
                    "canvas_w": info["canvas_w"],
                    "canvas_h": info["canvas_h"],
                    "char_grid_invariant": info["char_grid_invariant"],
                    "smooth_verified": info["smooth_verified"],
                    "max_delta_per_frame": info["max_delta_per_frame"],
                    "elapsed_s": round(elapsed, 2),
                    "discord": None,
                }

                if not args.dry_run:
                    ok = _send_discord(out_path, label)
                    rec["discord"] = ok
                    if ok:
                        discord_ok += 1
                        print("  [discord] sent")
                    else:
                        discord_fail += 1
                else:
                    print("  [discord] dry-run, skipped")

                results.append(rec)

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  FAIL  {exc}")
                results.append({
                    "output": str(out_path),
                    "pattern": pattern,
                    "image": img_name,
                    "error": str(exc),
                    "elapsed_s": round(elapsed, 2),
                })

    results_path = _OUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {results_path}")

    ok_count = sum(1 for r in results if "error" not in r)
    print(f"Generated: {ok_count}/{len(results)} MP4s")
    if not args.dry_run:
        print(f"Discord:   {discord_ok} sent, {discord_fail} failed")


if __name__ == "__main__":
    main()

"""
aa-animator v0.1 validation script — dynamic AMP generalization test.

Runs n=3 images through the full v0.1 pipeline (Mode C: braille + black bg)
and measures flicker_std with dynamic AMP_PX enabled.

Usage:
    python scripts/validate_v0_1.py [--image-dir DIR] [--out-dir DIR]

Outputs:
    scripts/v0_1_validation_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

MASK_DIR: Path | None = None  # set after args parsed

# ── CLI ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="aa-animator v0.1 validation")
_parser.add_argument(
    "--image-dir",
    default=str(Path.home() / "test_images"),
    help="Directory containing the 3 test images",
)
_parser.add_argument(
    "--out-dir",
    default="/tmp/aa_v01_validation",
    help="Directory for temporary output files",
)
_parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=1000,
    help="Bootstrap resamples for CI computation",
)
_args = _parser.parse_args()

IMAGE_DIR = Path(_args.image_dir)
OUT_DIR = Path(_args.out_dir)
N_BOOTSTRAP = _args.n_bootstrap
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR = OUT_DIR / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    "bike_art_base.jpg",
    "bench_rgb.jpg",
    "cliffhanger_base.png",
]

FLICKER_STD_THRESH = 0.01
NCHARS = 11  # " ·~ox+=*%$@"


def _frame_to_charseq(pil_img: Image.Image) -> np.ndarray:
    g = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    return (g * (NCHARS - 1)).astype(np.int32).clip(0, NCHARS - 1).flatten()


def _flicker_std(frames: list[Image.Image]) -> float:
    seqs = [_frame_to_charseq(f) for f in frames]
    dists = [float(np.mean(seqs[i] != seqs[i + 1])) for i in range(len(seqs) - 1)]
    return float(np.std(dists))


def _bootstrap_ci(values: list[float], n: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=np.float64)
    stats = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return lo, hi


def run_image(image_name: str) -> dict:
    image_path = IMAGE_DIR / image_name
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    t0 = time.perf_counter()

    from aa_animator_v2.pipeline import AAAnimator

    animator = AAAnimator(mode="braille", bg="black", fps=30, cols=100, n_frames=30, glow=True)
    animator.load_image(image_path)
    assert animator._img_np is not None
    img_pil = Image.fromarray((animator._img_np * 255).astype(np.uint8))

    # Detect mask source: attempt rembg; track which path was taken
    mask_source = "otsu"
    try:
        from rembg import remove as _rembg_remove  # type: ignore[import-not-found]
        rgba = _rembg_remove(img_pil)
        alpha = np.array(rgba, dtype=np.float32)[:, :, 3]
        thresh = animator._otsu_threshold(alpha.astype(np.uint8))
        candidate = alpha > thresh
        if candidate.mean() >= 0.05:
            mask_source = "rembg"
    except Exception:
        pass

    animator._fg_mask = animator.segment_subject(img_pil)
    animator._depth = animator.estimate_depth(img_pil)

    # Save mask PNG artefact
    if animator._fg_mask is not None and MASK_DIR is not None:
        stem = Path(image_name).stem
        mask_png = MASK_DIR / f"{stem}_mask.png"
        mask_u8 = (animator._fg_mask.astype(np.uint8) * 255)
        Image.fromarray(mask_u8).save(str(mask_png))

    fg_coverage = float(animator._fg_mask.mean()) if animator._fg_mask is not None else 0.3

    from aa_animator_v2.parallax import dynamic_amp_px, orbit_displacement, warp_mask

    amp = dynamic_amp_px(fg_coverage)

    warped_masks = None
    if animator._fg_mask is not None and animator._depth is not None:
        warped_masks = []
        for t in range(30):
            dx, dy = orbit_displacement(t, 30, amp)
            warped_masks.append(warp_mask(animator._fg_mask, animator._depth, dx, dy))

    raw_frames = animator.generate_frames()
    pil_frames = animator.render_frames(raw_frames, warped_masks=warped_masks)

    std = _flicker_std(pil_frames)
    elapsed = time.perf_counter() - t0

    return {
        "fg_coverage": round(fg_coverage, 4),
        "amp_px_used": amp,
        "flicker_std": round(std, 6),
        "pass": bool(std <= FLICKER_STD_THRESH),
        "elapsed_s": round(elapsed, 2),
        "mask_source": mask_source,
        "canvas_cols": animator.cols,
        "canvas_rows": animator._rows,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
print("aa-animator v0.1 validation (dynamic AMP, n=3 images)")
print("=" * 60)

results: dict[str, dict] = {}
flicker_stds: list[float] = []

for img_name in TEST_IMAGES:
    print(f"\nProcessing: {img_name} ...")
    r = run_image(img_name)
    results[img_name] = r
    if "error" not in r:
        flicker_stds.append(r["flicker_std"])
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  fg_coverage={r['fg_coverage']:.3f}  amp_px={r['amp_px_used']}  "
            f"flicker_std={r['flicker_std']:.4f}  [{status}]  ({r['elapsed_s']:.1f}s)"
        )
    else:
        print(f"  ERROR: {r['error']}")

# Bootstrap CI
ci_lo, ci_hi = (float("nan"), float("nan"))
if len(flicker_stds) >= 2:
    ci_lo, ci_hi = _bootstrap_ci(flicker_stds, n=N_BOOTSTRAP)

consistent_pass = all(r.get("pass", False) for r in results.values() if "error" not in r)

output = {
    "version": "0.1.0-dev",
    "images": results,
    "bootstrap_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
    "consistent_pass": consistent_pass,
}

out_json = Path("scripts/v0_1_validation_results.json")
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 60)
print(f"Bootstrap 95% CI on flicker_std: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Threshold: {FLICKER_STD_THRESH}")
print(f"Consistent PASS (all images): {consistent_pass}")
print(f"Results written: {out_json}")

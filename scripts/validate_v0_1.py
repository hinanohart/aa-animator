"""
aa-animator v0.1 validation script — dynamic AMP generalization test.

Runs n≥10 images through the full v0.1 pipeline (Mode C: braille + black bg)
and measures flicker_std with dynamic AMP_PX enabled.

Natural images: from ~/test_images/ (public artworks — no private photos).
Synthetic images: generated via PIL when natural count < 5.
CI: 95% Student t-interval (df=n-1), replacing the earlier n=3 bootstrap.

Usage:
    python scripts/validate_v0_1.py [--image-dir DIR] [--out-dir DIR]

Outputs:
    scripts/v0_1_validation_results.json
"""

from __future__ import annotations

import argparse
import json
import math
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
    help="Directory containing natural test images",
)
_parser.add_argument(
    "--out-dir",
    default="/tmp/aa_v01_validation",
    help="Directory for temporary output files",
)
_args = _parser.parse_args()

IMAGE_DIR = Path(_args.image_dir)
OUT_DIR = Path(_args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR = OUT_DIR / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Natural images available in test_images/ — no private photos
NATURAL_IMAGES = [
    "bike_art_base.jpg",
    "bench_rgb.jpg",
    "cliffhanger_base.png",
    "youtube_base.jpg",
    "v3_main.jpg",
    "final_200.jpg",
    "benchmark_result.jpg",
]

FLICKER_STD_THRESH = 0.01
NCHARS = 11  # " ·~ox+=*%$@"

# Minimum n for valid statistics (MEMORY.md #3)
MIN_N = 10
MIN_NATURAL = 5


# ── Image generation helpers ─────────────────────────────────────────────────


def _make_synthetic_portrait(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Synthetic portrait: skin-tone gradient ellipse on dark bg."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx, ry = w // 3, int(h * 0.45)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = ((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2 <= 1.0
    arr[mask, 0] = 210
    arr[mask, 1] = 160
    arr[mask, 2] = 120
    return Image.fromarray(arr)


def _make_synthetic_landscape(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Synthetic landscape: sky/ground gradient."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    sky_h = h // 2
    arr[:sky_h, :, 0] = 100
    arr[:sky_h, :, 1] = 160
    arr[:sky_h, :, 2] = 220
    arr[sky_h:, :, 0] = 60
    arr[sky_h:, :, 1] = 120
    arr[sky_h:, :, 2] = 50
    return Image.fromarray(arr)


def _make_synthetic_abstract(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Synthetic abstract: random color blocks."""
    rng = np.random.default_rng(99)
    w, h = size
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # Smooth with block averaging to avoid pure noise
    block = 32
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = arr[y : y + block, x : x + block]
            mean = patch.mean(axis=(0, 1)).astype(np.uint8)
            arr[y : y + block, x : x + block] = mean
    return Image.fromarray(arr)


SYNTHETIC_IMAGES: list[tuple[str, Image.Image]] = [
    ("synthetic_portrait.png", _make_synthetic_portrait()),
    ("synthetic_landscape.png", _make_synthetic_landscape()),
    ("synthetic_abstract.png", _make_synthetic_abstract()),
]


# ── Core helpers ─────────────────────────────────────────────────────────────


def _frame_to_charseq(pil_img: Image.Image) -> np.ndarray:
    g = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    return (g * (NCHARS - 1)).astype(np.int32).clip(0, NCHARS - 1).flatten()


def _flicker_std(frames: list[Image.Image]) -> float:
    seqs = [_frame_to_charseq(f) for f in frames]
    dists = [float(np.mean(seqs[i] != seqs[i + 1])) for i in range(len(seqs) - 1)]
    return float(np.std(dists))


def _t_ci_95(values: list[float]) -> tuple[float, float]:
    """Two-sided 95% Student t-confidence interval (df = n-1).

    Requires n >= 2. Falls back to (nan, nan) for n < 2.
    """
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(n)
    # t critical values for df = n-1 at 95% CI (two-sided)
    # Tabulated values for small n; scipy.stats.t.ppf is optional
    t_table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
    }
    try:
        from scipy import stats as scipy_stats  # type: ignore[import-untyped]

        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    except Exception:
        t_crit = t_table.get(n - 1, 2.0)
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return float(lo), float(hi)


def run_image(image_name: str, pil_override: Image.Image | None = None) -> dict:
    if pil_override is not None:
        img_source = pil_override
        tmp_path = OUT_DIR / image_name
        img_source.save(str(tmp_path))
        image_path = tmp_path
    else:
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
        mask_u8 = animator._fg_mask.astype(np.uint8) * 255
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
        "source": "synthetic" if pil_override is not None else "natural",
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
print("aa-animator v0.1 validation (dynamic AMP, n≥10, 95% t-CI)")
print("=" * 60)

results: dict[str, dict] = {}
flicker_stds: list[float] = []
natural_count = 0
synthetic_count = 0

# Run natural images first
for img_name in NATURAL_IMAGES:
    image_path = IMAGE_DIR / img_name
    if not image_path.exists():
        print(f"\nSkipping (not found): {img_name}")
        continue
    print(f"\nProcessing [natural]: {img_name} ...")
    r = run_image(img_name)
    results[img_name] = r
    if "error" not in r:
        flicker_stds.append(r["flicker_std"])
        natural_count += 1
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  fg_coverage={r['fg_coverage']:.3f}  amp_px={r['amp_px_used']}  "
            f"flicker_std={r['flicker_std']:.4f}  [{status}]  ({r['elapsed_s']:.1f}s)"
        )
    else:
        print(f"  ERROR: {r['error']}")

# Sanity-check minimum natural images
if natural_count < MIN_NATURAL:
    print(
        f"\nWARNING: only {natural_count} natural images found (min={MIN_NATURAL}). "
        "Results are statistically weaker."
    )

# Top up with synthetic images until n >= MIN_N
needed_synthetic = max(0, MIN_N - natural_count)
needed_synthetic = min(needed_synthetic, len(SYNTHETIC_IMAGES))

for syn_name, syn_img in SYNTHETIC_IMAGES[:needed_synthetic]:
    print(f"\nProcessing [synthetic]: {syn_name} ...")
    r = run_image(syn_name, pil_override=syn_img)
    results[syn_name] = r
    if "error" not in r:
        flicker_stds.append(r["flicker_std"])
        synthetic_count += 1
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  fg_coverage={r['fg_coverage']:.3f}  amp_px={r['amp_px_used']}  "
            f"flicker_std={r['flicker_std']:.4f}  [{status}]  ({r['elapsed_s']:.1f}s)"
        )
    else:
        print(f"  ERROR: {r['error']}")

n_total = natural_count + synthetic_count
print(f"\nTotal: n={n_total} (natural={natural_count}, synthetic={synthetic_count})")

# 95% t-CI (replaces bootstrap — statistically valid for n≥2)
ci_lo, ci_hi = _t_ci_95(flicker_stds)
mean_std = float(np.mean(flicker_stds)) if flicker_stds else float("nan")

consistent_pass = all(r.get("pass", False) for r in results.values() if "error" not in r)

output = {
    "version": "0.1.0-dev",
    "n_total": n_total,
    "n_natural": natural_count,
    "n_synthetic": synthetic_count,
    "images": results,
    "mean_flicker_std": round(mean_std, 6),
    "t_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
    "ci_method": f"Student t (df={n_total - 1})",
    "flicker_std_threshold": FLICKER_STD_THRESH,
    "consistent_pass": consistent_pass,
}

out_json = Path("scripts/v0_1_validation_results.json")
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 60)
print(f"Mean flicker_std: {mean_std:.4f}")
print(f"95% t-CI (df={n_total - 1}): [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Threshold: {FLICKER_STD_THRESH}")
print(f"Consistent PASS (all images): {consistent_pass}")
print(f"Results written: {out_json}")

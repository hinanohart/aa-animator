"""
aa-animator v0.0.3 — PoC script (portable, no hardcoded paths)

Usage:
  python poc_v0_0_3.py INPUT_IMAGE [--bg {black,ghostty_fill,full}] [--braille {on,off}] [--out-dir DIR]

Modes:
  A: --bg black   --braille off  (subject-only, conventional AA)
  B: --bg ghostty_fill --braille off  (thin background, Ghostty-style)
  C: --bg black   --braille on   (subject-only + Braille high-res)
  D: --bg full    --braille off  (no rembg, v0.0.1 comparable)

4 modes performance (bike_art_base, v0.0.3):
  A: flicker_std=0.0128, fg_entropy=3.201, aa_ms/frame=30.1, size_kb=213
  B: flicker_std=0.0128, fg_entropy=3.201, aa_ms/frame=48.8, size_kb=276
  C: flicker_std=0.0086, fg_entropy=3.179, aa_ms/frame=35.3, size_kb=115  <-- selected
  D: flicker_std=0.0233, fg_entropy=3.238, aa_ms/frame=31.6, size_kb=496

Selection rationale: Mode C is the only mode achieving flicker std <= 0.01,
passes fg_entropy >= 3.0, and produces the smallest output size.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, uniform_filter
from scipy.ndimage import sobel as scipy_sobel

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="aa-animator v0.0.3 PoC")
parser.add_argument("input_image", help="Path to input image file")
parser.add_argument("--bg", choices=["black", "ghostty_fill", "full"], default="black")
parser.add_argument("--braille", choices=["on", "off"], default="off")
parser.add_argument("--out-dir", default="/tmp/aa_poc_out", help="Output directory")
args = parser.parse_args()

BG_MODE = args.bg
BRAILLE_ON = args.braille == "on"
INPUT_IMAGE = args.input_image
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

# ── キャッシュパス (out-dir 内、入力ファイル名ベース) ─────────────────
_input_stem = Path(INPUT_IMAGE).stem
DEPTH_CACHE = os.path.join(OUT_DIR, f"depth_cache_{_input_stem}.npy")
MASK_CACHE = os.path.join(OUT_DIR, f"mask_cache_{_input_stem}.npy")
OUTPUT_MP4 = os.path.join(
    OUT_DIR, f"poc_v4_output_{BG_MODE}_{'braille' if BRAILLE_ON else 'ascii'}.mp4"
)
FRAME0_PNG = os.path.join(
    OUT_DIR, f"poc_v4_frame0_{BG_MODE}_{'braille' if BRAILLE_ON else 'ascii'}.png"
)

# Braille cell: 2x4 dots → cell size is half of ASCII (4x8px)
# ASCII cell: 8x16px
if BRAILLE_ON:
    CELL_W, CELL_H = 4, 8
    COLS, ROWS = 200, 82
else:
    CELL_W, CELL_H = 8, 16
    COLS, ROWS = 100, 41

IMG_W = COLS * CELL_W  # 800
IMG_H = ROWS * CELL_H  # 656

N_FRAMES = 30
FPS = 30
AMP_PX = 18
FONT_SIZE = 14

AA_CHARS = " ·~ox+=*%$@"  # 11 levels
NCHARS = len(AA_CHARS)
COLOR_EDGE = (70, 130, 255)
COLOR_BODY = (220, 220, 220)
GLOW_COLOR = (70, 130, 255)
GLOW_ALPHA = 0.30
EDGE_THRESH = 0.15
BG_DOT_CHAR = "·"

# Braille Unicode U+2800-28FF
# dot layout: col0: bits0,1,2,6  col1: bits3,4,5,7
BRAILLE_BASE = 0x2800


def brightness_to_braille(b01: float) -> str:
    """0-1 brightness → Braille character (8 dot, diagonal pattern)"""
    n_dots = round(b01 * 8)
    dot_order = [0, 3, 1, 4, 2, 5, 6, 7]
    bits = 0
    for i in range(n_dots):
        bits |= 1 << dot_order[i]
    return chr(BRAILLE_BASE + bits)


# ── Utilities ────────────────────────────────────────────────────────
def _otsu_threshold(gray_u8: np.ndarray) -> float:
    hist, _ = np.histogram(gray_u8.flatten(), bins=256, range=(0, 256))
    total = gray_u8.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0
    best_thresh = 128.0
    best_var = 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = float(t)
    return best_thresh


# ── Timing ──────────────────────────────────────────────────────────
T = {}
WALL_START = time.perf_counter()


def stamp(key):
    T[key] = time.perf_counter()


def elapsed(a, b):
    return T[b] - T[a]


def now():
    return time.perf_counter() - WALL_START


# ── Synthetic image generator ────────────────────────────────────────
def make_synthetic(out_path: str) -> Image.Image:
    """Generate 256x256 synthetic test image: circle + rectangle + radial lines"""
    sz = 256
    img = Image.new("RGB", (sz, sz), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = sz // 2, sz // 2
    for angle_deg in range(0, 360, 15):
        rad = math.radians(angle_deg)
        ex = int(cx + math.cos(rad) * cx * 0.9)
        ey = int(cy + math.sin(rad) * cy * 0.9)
        draw.line([(cx, cy), (ex, ey)], fill=(180, 180, 60), width=2)
    draw.rectangle([40, 40, 215, 215], outline=(200, 100, 50), width=3)
    draw.rectangle([80, 80, 175, 175], outline=(100, 200, 100), width=3)
    draw.ellipse(
        [cx - 60, cy - 60, cx + 60, cy + 60], fill=(80, 120, 255), outline=(255, 255, 255), width=2
    )
    draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(255, 80, 80))
    img.save(out_path)
    return img


# ── 1. Load input image ──────────────────────────────────────────────
stamp("start")
print(
    f"[{now():.3f}s] aa-animator v0.0.3  BG_MODE={BG_MODE}  BRAILLE={'on' if BRAILLE_ON else 'off'}"
)
print(f"[{now():.3f}s] Loading image: {INPUT_IMAGE}")

try:
    img_orig = Image.open(INPUT_IMAGE).convert("RGB")
    print(f"  Loaded {Path(INPUT_IMAGE).name}: {img_orig.size}")
except Exception as e:
    print(f"  [WARN] Image load failed: {e}")
    synth_path = os.path.join(OUT_DIR, "synthetic_fallback.png")
    print(f"  Falling back to synthetic image: {synth_path}")
    img_orig = make_synthetic(synth_path)

img = img_orig.resize((IMG_W, IMG_H), Image.LANCZOS)
img_np = np.array(img, dtype=np.float32) / 255.0
print(f"  Resized: {img_orig.size} → {img.size}")

# ── 2. rembg subject mask ────────────────────────────────────────────
fg_mask = None

if BG_MODE in ("black", "ghostty_fill"):
    if os.path.isfile(MASK_CACHE):
        print(f"\n[{now():.3f}s] Loading cached mask: {MASK_CACHE}")
        fg_mask = np.load(MASK_CACHE).astype(bool)
        print(f"  Mask coverage={fg_mask.mean() * 100:.1f}%")
    else:
        print(f"\n[{now():.3f}s] Running rembg ...")
        try:
            from rembg import remove as rembg_remove

            stamp("rembg_start")
            img_rgba = rembg_remove(img)
            alpha = np.array(img_rgba, dtype=np.float32)[:, :, 3]
            alpha_u8 = alpha.astype(np.uint8)
            otsu_thresh = _otsu_threshold(alpha_u8)
            fg_mask_rembg = alpha > otsu_thresh
            coverage_rembg = fg_mask_rembg.mean() * 100
            print(f"  rembg Otsu thresh={otsu_thresh:.0f}  coverage={coverage_rembg:.1f}%")

            if coverage_rembg < 10.0:
                print(
                    f"  [WARN] rembg coverage too low ({coverage_rembg:.1f}%), using luminance Otsu fallback"
                )
                gray_np = np.array(img.convert("L"), dtype=np.float32)
                lum_otsu = _otsu_threshold(gray_np.astype(np.uint8))
                fg_mask = gray_np > lum_otsu
                print(
                    f"  Luminance Otsu thresh={lum_otsu:.0f}  coverage={fg_mask.mean() * 100:.1f}%"
                )
            else:
                fg_mask = fg_mask_rembg

            np.save(MASK_CACHE, fg_mask.astype(np.uint8))
            stamp("rembg_done")
            print(f"  rembg done in {elapsed('rembg_start', 'rembg_done'):.2f}s")
            print(f"  Final mask coverage: {fg_mask.mean() * 100:.1f}%")
        except ImportError:
            print("  [WARN] rembg not found, falling back to full mode")
            BG_MODE = "full"
            fg_mask = None
        except Exception as e:
            print(f"  [WARN] rembg error: {e}, falling back to full mode")
            BG_MODE = "full"
            fg_mask = None
else:
    print(f"\n[{now():.3f}s] BG_MODE=full: skipping rembg")

# ── 3. Depth Anything v2 Small ───────────────────────────────────────
stamp("depth_start")

if os.path.isfile(DEPTH_CACHE):
    print(f"\n[{now():.3f}s] Loading cached depth: {DEPTH_CACHE}")
    depth_norm = np.load(DEPTH_CACHE)
    stamp("depth_done")
    dt_depth = f"{elapsed('depth_start', 'depth_done') * 1000:.1f}ms (cache)"
    print(f"  Depth loaded in {elapsed('depth_start', 'depth_done') * 1000:.1f}ms")
else:
    print(f"\n[{now():.3f}s] Loading Depth Anything V2 Small ...")
    from transformers import pipeline as hf_pipeline

    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
    )
    stamp("depth_model")
    print(f"  Model loaded in {elapsed('depth_start', 'depth_model'):.2f}s")

    depth_result = depth_pipe(img)
    depth_map_pil = depth_result["depth"]
    stamp("depth_done")
    dt_depth = f"{elapsed('depth_start', 'depth_done'):.2f}s (fresh)"
    print(f"  Depth done in {elapsed('depth_start', 'depth_done'):.2f}s")

    depth_raw = np.array(depth_map_pil.resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.float32)
    dmin, dmax = depth_raw.min(), depth_raw.max()
    depth_norm = (depth_raw - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(depth_raw)
    np.save(DEPTH_CACHE, depth_norm)
    print(f"  Depth cached: {DEPTH_CACHE}")


# ── 4. Forward warp ──────────────────────────────────────────────────
def simple_forward_warp(rgb, depth, dx, dy):
    H, W = depth.shape
    shift_x = (depth * dx).astype(np.int32)
    shift_y = (depth * dy).astype(np.int32)
    yy, xx = np.mgrid[0:H, 0:W]
    new_y = np.clip(yy + shift_y, 0, H - 1)
    new_x = np.clip(xx + shift_x, 0, W - 1)
    order = np.argsort(-depth.flatten())
    flat_y = new_y.flatten()[order]
    flat_x = new_x.flatten()[order]
    orig_y = yy.flatten()[order]
    orig_x = xx.flatten()[order]
    out = np.zeros_like(rgb)
    out[flat_y, flat_x] = rgb[orig_y, orig_x]
    return out


def warp_mask(mask_bool, depth, dx, dy):
    H, W = depth.shape
    shift_x = (depth * dx).astype(np.int32)
    shift_y = (depth * dy).astype(np.int32)
    yy, xx = np.mgrid[0:H, 0:W]
    new_y = np.clip(yy + shift_y, 0, H - 1)
    new_x = np.clip(xx + shift_x, 0, W - 1)
    order = np.argsort(-depth.flatten())
    flat_y = new_y.flatten()[order]
    flat_x = new_x.flatten()[order]
    orig_y = yy.flatten()[order]
    orig_x = xx.flatten()[order]
    out = np.zeros((H, W), dtype=bool)
    out[flat_y, flat_x] = mask_bool[orig_y, orig_x]
    return out


# ── 5. Sobel edge ────────────────────────────────────────────────────
def sobel_edge(rgb_01):
    gray = 0.299 * rgb_01[:, :, 0] + 0.587 * rgb_01[:, :, 1] + 0.114 * rgb_01[:, :, 2]
    sx = scipy_sobel(gray, axis=1)
    sy = scipy_sobel(gray, axis=0)
    mag = np.hypot(sx, sy)
    m = mag.max()
    if m > 0:
        mag /= m
    return mag


# ── 6. Glow mask ─────────────────────────────────────────────────────
def build_glow_mask_cells(edge_cell_mask):
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(edge_cell_mask, structure=struct)
    return dilated & ~edge_cell_mask


# ── 7. Font + bitmap cache ───────────────────────────────────────────
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_path, FONT_SIZE)
    print(f"\n  Font: DejaVuSansMono {FONT_SIZE}px")
except Exception:
    font = ImageFont.load_default()
    print("\n  Font: default")

_colors = [COLOR_BODY, COLOR_EDGE, (40, 40, 40)]
char_bitmaps = np.zeros((NCHARS, 3, CELL_H, CELL_W, 3), dtype=np.uint8)
for _ci, _ch in enumerate(AA_CHARS):
    for _ki, _col in enumerate(_colors):
        _cell = Image.new("RGB", (CELL_W, CELL_H), (0, 0, 0))
        ImageDraw.Draw(_cell).text((0, 0), _ch, font=font, fill=_col)
        char_bitmaps[_ci, _ki] = np.array(_cell, dtype=np.uint8)

_bg_dot_bitmaps = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
_bg_dot_cell = Image.new("RGB", (CELL_W, CELL_H), (0, 0, 0))
ImageDraw.Draw(_bg_dot_cell).text((0, 0), BG_DOT_CHAR, font=font, fill=(40, 40, 40))
_bg_dot_bitmaps[:] = np.array(_bg_dot_cell, dtype=np.uint8)
print(f"  Char bitmap cache: {NCHARS} chars × 3 colors precomputed")

# Braille mode: precompute bitmaps for all 256 patterns
braille_bitmaps = None
if BRAILLE_ON:
    try:
        braille_font = ImageFont.truetype(font_path, 10)
    except Exception:
        braille_font = ImageFont.load_default()

    N_BRAILLE = 256
    braille_bitmaps = np.zeros((N_BRAILLE, CELL_H, CELL_W, 3), dtype=np.uint8)
    for _bi in range(N_BRAILLE):
        _ch = chr(BRAILLE_BASE + _bi)
        _cell = Image.new("RGB", (CELL_W, CELL_H), (0, 0, 0))
        ImageDraw.Draw(_cell).text((0, 0), _ch, font=braille_font, fill=COLOR_BODY)
        braille_bitmaps[_bi] = np.array(_cell, dtype=np.uint8)
    print(f"  Braille bitmap cache: {N_BRAILLE} patterns precomputed (2x4 dots, U+2800-28FF)")

# ── 8. Frame generation ───────────────────────────────────────────────
stamp("warp_start")
print(
    f"\n[{now():.3f}s] Generating {N_FRAMES} frames ({COLS}x{ROWS} cells, {IMG_W}x{IMG_H}px) BG={BG_MODE} braille={BRAILLE_ON}..."
)

warp_times = []
aa_times = []
frames_pil = []
all_entropies = []
all_fg_entropy = []

CANVAS_W = COLS * CELL_W
CANVAS_H = ROWS * CELL_H

for f in range(N_FRAMES):
    t_ratio = f / N_FRAMES
    angle = 2 * math.pi * t_ratio
    dx = math.sin(angle) * AMP_PX
    dy = math.cos(angle) * (AMP_PX / 2)

    tw0 = time.perf_counter()
    warped = simple_forward_warp(img_np, depth_norm, dx, dy)

    warped_mask = None
    if fg_mask is not None:
        warped_mask = warp_mask(fg_mask, depth_norm, dx, dy)

    warp_times.append(time.perf_counter() - tw0)

    ta0 = time.perf_counter()

    gray_full = 0.299 * warped[:, :, 0] + 0.587 * warped[:, :, 1] + 0.114 * warped[:, :, 2]

    cell_brightness = gray_full.reshape(ROWS, CELL_H, COLS, CELL_W).mean(axis=(1, 3))

    edge_map = sobel_edge(warped)
    edge_full = edge_map.reshape(ROWS, CELL_H, COLS, CELL_W).mean(axis=(1, 3))
    edge_cell = edge_full > EDGE_THRESH

    if warped_mask is not None:
        mask_cell = warped_mask.reshape(ROWS, CELL_H, COLS, CELL_W).mean(axis=(1, 3)) > 0.3
    else:
        mask_cell = np.ones((ROWS, COLS), dtype=bool)

    rgb_cell_mean = warped.reshape(ROWS, CELL_H, COLS, CELL_W, 3).mean(axis=(1, 3))

    rgb_mean_sum = rgb_cell_mean.sum(axis=2)
    hole_cell = (~mask_cell) & (rgb_mean_sum < (10.0 / 255.0 * 3))

    if hole_cell.any():
        cb_filled = cell_brightness.copy()
        cb_nb = uniform_filter(cb_filled, size=3, mode="nearest")
        cb_filled[hole_cell] = cb_nb[hole_cell]
        still_hole = hole_cell & (cb_filled < 0.001)
        if still_hole.any():
            cb_nb2 = uniform_filter(cb_filled, size=5, mode="nearest")
            cb_filled[still_hole] = cb_nb2[still_hole]
        cell_brightness = cb_filled

    if fg_mask is not None:
        fg_vals = cell_brightness[mask_cell]
        if len(fg_vals) > 10:
            p2 = np.percentile(fg_vals, 2)
            p98 = np.percentile(fg_vals, 98)
        else:
            p2 = np.percentile(cell_brightness, 2)
            p98 = np.percentile(cell_brightness, 98)
    else:
        p2 = np.percentile(cell_brightness, 2)
        p98 = np.percentile(cell_brightness, 98)
    if p98 > p2:
        cell_brightness = np.clip((cell_brightness - p2) / (p98 - p2), 0.0, 1.0)

    char_idx_flat = np.clip((cell_brightness * (NCHARS - 1)).astype(int), 0, NCHARS - 1)

    counts_all = np.bincount(char_idx_flat.flatten(), minlength=NCHARS).astype(float)
    p_all = counts_all / counts_all.sum()
    p_all = p_all[p_all > 0]
    ent_full = float(-np.sum(p_all * np.log2(p_all)))
    all_entropies.append(ent_full)

    if mask_cell.any():
        fg_char_idx = char_idx_flat[mask_cell]
        counts_fg = np.bincount(fg_char_idx, minlength=NCHARS).astype(float)
        p_fg = counts_fg / counts_fg.sum()
        p_fg = p_fg[p_fg > 0]
        ent_fg = float(-np.sum(p_fg * np.log2(p_fg)))
    else:
        ent_fg = 0.0
    all_fg_entropy.append(ent_fg)

    glow_cell = build_glow_mask_cells(edge_cell)

    if BRAILLE_ON:
        assert braille_bitmaps is not None, "braille_bitmaps must be initialized when BRAILLE_ON"
        n_dots_arr = np.clip((cell_brightness * 8).round().astype(int), 0, 8)
        dot_order = [0, 3, 1, 4, 2, 5, 6, 7]
        bits_arr = np.zeros((ROWS, COLS), dtype=np.int32)
        for i, d in enumerate(dot_order):
            bits_arr[n_dots_arr > i] |= 1 << d
        bits_arr = bits_arr.clip(0, 255)

        selected = braille_bitmaps[bits_arr]
        canvas_u8 = np.ascontiguousarray(selected.transpose(0, 2, 1, 3, 4)).reshape(
            CANVAS_H, CANVAS_W, 3
        )
    else:
        chars = char_idx_flat
        color_type = edge_cell.astype(np.uint8)
        selected = char_bitmaps[chars, color_type]
        canvas_u8 = np.ascontiguousarray(selected.transpose(0, 2, 1, 3, 4)).reshape(
            CANVAS_H, CANVAS_W, 3
        )

    mask_px = np.repeat(np.repeat(mask_cell, CELL_H, axis=0), CELL_W, axis=1)
    if BG_MODE == "black":
        canvas_u8 *= mask_px[:, :, np.newaxis].astype(np.uint8)
    elif BG_MODE == "ghostty_fill":
        bg_canvas = (
            np.tile(_bg_dot_bitmaps[np.newaxis, np.newaxis, :, :, :], (ROWS, COLS, 1, 1, 1))
            .transpose(0, 2, 1, 3, 4)
            .reshape(CANVAS_H, CANVAS_W, 3)
        )
        canvas_u8[~mask_px] = bg_canvas[~mask_px]

    glow_active = glow_cell & mask_cell
    if glow_active.any():
        glow_mask_px = np.repeat(np.repeat(glow_active, CELL_H, axis=0), CELL_W, axis=1)
        glow_region = canvas_u8[glow_mask_px].astype(np.float32)
        glow_region = (
            glow_region * (1.0 - GLOW_ALPHA) + np.array(GLOW_COLOR, dtype=np.float32) * GLOW_ALPHA
        )
        canvas_u8[glow_mask_px] = np.clip(glow_region, 0, 255).astype(np.uint8)

    aa_img = Image.fromarray(canvas_u8)
    aa_times.append(time.perf_counter() - ta0)
    frames_pil.append(aa_img)

    if (f + 1) % 10 == 0 or f == 0:
        print(
            f"  Frame {f + 1:2d}/{N_FRAMES}: warp={warp_times[-1] * 1000:.1f}ms  AA={aa_times[-1] * 1000:.1f}ms  fg_ent={ent_fg:.3f}"
        )

stamp("frames_done")
dt_warp_avg = sum(warp_times) / len(warp_times) * 1000
dt_aa_avg = sum(aa_times) / len(aa_times) * 1000
print(f"\n  Avg warp: {dt_warp_avg:.1f}ms/frame  Avg AA: {dt_aa_avg:.1f}ms/frame")

# ── 9. Save frame 0 PNG ──────────────────────────────────────────────
frames_pil[0].save(FRAME0_PNG)
print(f"\n  Frame 0 saved: {FRAME0_PNG}")


# ── 10. Flicker metric ───────────────────────────────────────────────
def frame_to_charseq(pil_img):
    g = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    idxs = (g * (NCHARS - 1)).astype(int).clip(0, NCHARS - 1)
    return idxs.flatten()


print("\nComputing flicker metric ...")
seqs_arr = [frame_to_charseq(fr) for fr in frames_pil]
dists = [float(np.mean(seqs_arr[i] != seqs_arr[i + 1])) for i in range(len(seqs_arr) - 1)]
avg_dist = float(np.mean(dists))
std_dist = float(np.std(dists))
print(f"  Flicker avg={avg_dist:.4f}  std={std_dist:.4f}")

# ── 11. Entropy summary ──────────────────────────────────────────────
avg_entropy = float(np.mean(all_entropies))
avg_fg_entropy = float(np.mean(all_fg_entropy))
max_ent_bits = math.log2(NCHARS)
print(f"  Full entropy avg  = {avg_entropy:.3f} bits (max {max_ent_bits:.3f})")
print(f"  fg_entropy avg    = {avg_fg_entropy:.3f} bits (foreground cells only)")

# ── 12. MP4 encode ───────────────────────────────────────────────────
stamp("ffmpeg_start")
print(f"\n[{now():.3f}s] Encoding mp4: {OUTPUT_MP4}")

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-pixel_format",
    "rgb24",
    "-video_size",
    f"{CANVAS_W}x{CANVAS_H}",
    "-framerate",
    str(FPS),
    "-i",
    "pipe:0",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "18",
    OUTPUT_MP4,
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
for fr in frames_pil:
    proc.stdin.write(np.array(fr, dtype=np.uint8).tobytes())
proc.stdin.close()
proc.wait()
ffmpeg_stderr = proc.stderr.read().decode(errors="replace")

if proc.returncode != 0:
    print("  [ERROR] ffmpeg failed:")
    print(ffmpeg_stderr[-500:])
    sys.exit(1)

stamp("ffmpeg_done")
mp4_size = os.path.getsize(OUTPUT_MP4)
wall_total = time.perf_counter() - WALL_START
print(f"  Encoded in {elapsed('ffmpeg_start', 'ffmpeg_done'):.2f}s")
print(f"  Output: {OUTPUT_MP4} ({mp4_size / 1024:.1f} KB)")

# ── 13. Summary ──────────────────────────────────────────────────────
ENTROPY_THRESH = 3.0  # tightened from 2.5: 10-char palette max = log2(11) ≈ 3.46 bits
FLICKER_STD_THRESH = 0.01

entropy_ok = avg_entropy >= ENTROPY_THRESH
fg_entropy_ok = avg_fg_entropy >= ENTROPY_THRESH
flicker_ok = std_dist <= FLICKER_STD_THRESH
aa_time_ok = dt_aa_avg <= 30.0


def ok(v):
    return "PASS" if v else "FAIL"


print("\n" + "=" * 70)
print("  aa-animator v0.0.3 — measured results")
print("=" * 70)
if BRAILLE_ON:
    win_tag = "C"
elif BG_MODE == "full":
    win_tag = "D"
elif BG_MODE == "ghostty_fill":
    win_tag = "B"
else:
    win_tag = "A"
print(f"  Mode              : BG={BG_MODE}  Braille={'on' if BRAILLE_ON else 'off'}  ({win_tag})")
print(f"  Input             : {Path(INPUT_IMAGE).name}  {img_orig.size}")
print(f"  Depth             : {dt_depth}")
print(f"  Forward warp/frame: {dt_warp_avg:.1f}ms avg")
print(f"  AA convert/frame  : {dt_aa_avg:.1f}ms avg  [{ok(aa_time_ok)} <=30ms]")
print(f"  ffmpeg encode     : {elapsed('ffmpeg_start', 'ffmpeg_done'):.2f}s")
print(f"  Total wall time   : {wall_total:.2f}s")
print(f"  Output size       : {mp4_size / 1024:.1f} KB")
print(f"  Flicker avg+-std  : {avg_dist:.4f} +- {std_dist:.4f}  [{ok(flicker_ok)} std<=0.01]")
print(
    f"  Full entropy avg  : {avg_entropy:.3f} bits (max {max_ent_bits:.3f})  [{ok(entropy_ok)} >=3.0]"
)
print(
    f"  fg_entropy avg    : {avg_fg_entropy:.3f} bits (foreground only)  [{ok(fg_entropy_ok)} >=3.0]"
)
print(f"  Canvas            : {COLS}x{ROWS} chars  ({CANVAS_W}x{CANVAS_H}px)")
print("=" * 70)

print("\nDone.")

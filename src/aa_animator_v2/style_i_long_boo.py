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
"""Style I: long_cinematic base + Ghostty-boo post-processing.

Motion design (ported/extended from aa_animator.py gallery presets):
  - pan    : slow horizontal pan  (aa_animator.py:225-228 pan effect)
  - zoom   : gentle zoom-in over clip (aa_animator.py:264-267 zoom effect)
  - vignette: radial edge darkening (aa_animator.py:346-355 _vignette)

Ghostty-boo reinforcement (via _boo_postprocess):
  - outline_ring : Sobel edges → '@' glyph, bright outline colour
  - blue_glow    : light-glyph cells → blue (80,140,255) tint

Charset : CHARS_GHOSTTY = " ·~ox+=*%$@" (11-level Ghostty ramp)
Canvas  : 96 cols × 48 rows (wide 2:1, non-square, Ghostty boo proportion)
FPS     : 24
Duration: 4.0 s
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aa_animator_v2._boo_postprocess import (
    CELL_H,
    CELL_W,
    FONT_RATIO,
    apply_blue_glow,
    apply_outline_ring,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)
_FONT_SIZE: int = 12

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]

# Pan / zoom tuning — ported from aa_animator.py pan/zoom concepts
_PAN_AMP_FRAC: float = 0.08    # max pan as fraction of canvas width
_ZOOM_MIN: float = 1.00        # zoom at t=0
_ZOOM_MAX: float = 1.12        # zoom at t=1 (gentle push-in)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Motion: pan + zoom  (aa_animator.py:225-228, 264-267)
# ---------------------------------------------------------------------------

def _apply_pan_zoom(
    img: Image.Image,
    t: float,
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """Apply slow pan + progressive zoom to RGBA source image.

    Pan  : horizontal drift of ±_PAN_AMP_FRAC × canvas_w using sin(t*2π).
    Zoom : linear interpolation from _ZOOM_MIN to _ZOOM_MAX over [0, 1].

    Args:
        img: Source RGBA image.
        t: Normalised time in [0, 1].
        canvas_w: Canvas pixel width.
        canvas_h: Canvas pixel height.

    Returns:
        RGB PIL Image at (canvas_w, canvas_h).
    """
    zoom = _ZOOM_MIN + (_ZOOM_MAX - _ZOOM_MIN) * t
    pan_x = int(_PAN_AMP_FRAC * canvas_w * math.sin(t * math.pi * 2.0))

    cw, ch = img.size
    new_w = max(1, int(cw * zoom))
    new_h = max(1, int(ch * zoom))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (*_BG_COLOR, 255))
    x = (canvas_w - new_w) // 2 + pan_x
    y = (canvas_h - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Vignette (aa_animator.py:346-355)
# ---------------------------------------------------------------------------

def _apply_vignette(img: Image.Image) -> Image.Image:
    """Radial darkening — d²×0.85 attenuation (aa_animator.py _vignette port).

    Args:
        img: RGB PIL Image.

    Returns:
        Vignette-applied RGB PIL Image.
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    d /= d.max()
    mask = np.clip(1.0 - d ** 2 * 0.85, 0.15, 1.0)
    arr *= mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# DensityAA renderer with Ghostty charset
# ---------------------------------------------------------------------------

def _render_density_aa(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame as Ghostty-palette density AA.

    Args:
        frame_img: RGB PIL Image (canvas size).
        cols: Character canvas width in cells.
        font: Pre-loaded monospace font.

    Returns:
        RGB PIL Image of the rendered AA frame.
    """
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * FONT_RATIO))
    small = frame_img.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)

    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx = np.clip((lum / 255.0 * n).astype(int), 0, n)

    out = Image.new("RGB", (cols * CELL_W, rows * CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for y in range(rows):
        for x in range(cols):
            if lum[y, x] < 6:
                continue
            ch = charset[idx[y, x]]
            if ch == " ":
                continue
            r, g, b = arr[y, x, 0], arr[y, x, 1], arr[y, x, 2]
            boost = max(1.0, 1.6 - lum[y, x] / 350)
            color = (
                min(255, int(r * boost)),
                min(255, int(g * boost)),
                min(255, int(b * boost)),
            )
            d.text((x * CELL_W, y * CELL_H), ch, fill=color, font=font)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_style_i(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 96,
    fps: int = 24,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style I (long_cinematic + Ghostty-boo) animation MP4.

    Applies pan + zoom + vignette motion (long_cinematic base), then
    reinforces Ghostty-boo aesthetics via outline_ring and blue_glow
    post-processing.

    Args:
        input_path: Path to input image.
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells (default 96, wide 2:1 aspect).
        fps: Output video frame rate (default 24, matches long_cinematic).
        duration: Clip length in seconds.
        canvas_size: Pixel canvas size for compositing.

    Returns:
        Dict with output_path, n_frames, canvas_w, canvas_h,
        outline_ring_cells, blue_glow_cells.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    # Scale source to fit canvas, maintain aspect ratio
    img_src.thumbnail((canvas_size, canvas_size), Image.LANCZOS)

    # Wide canvas: 2:1 aspect (Ghostty boo proportion, 96×48 cells → ~672×672 px, but pixel canvas is rectangular)
    canvas_w_px = canvas_size
    canvas_h_px = canvas_size // 2

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    # Probe output pixel dimensions
    probe_rgb = _apply_pan_zoom(img_src, 0.0, canvas_w_px, canvas_h_px)
    probe_aa = _render_density_aa(probe_rgb, cols, font)
    out_w, out_h = probe_aa.size
    # Ensure even dimensions for yuv420p
    out_w -= out_w % 2
    out_h -= out_h % 2

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{out_w}x{out_h}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-tune", "animation", "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    outline_cells_total = 0
    blue_cells_total = 0

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb = _apply_pan_zoom(img_src, t, canvas_w_px, canvas_h_px)
        frame_rgb = _apply_vignette(frame_rgb)
        aa_frame = _render_density_aa(frame_rgb, cols, font)

        # Count '@' cells before post-processing (baseline)
        arr_before = np.array(aa_frame)

        # Post-processing: outline ring then blue glow
        aa_frame = apply_outline_ring(aa_frame, frame_rgb, cols, font)
        aa_frame = apply_blue_glow(aa_frame, frame_rgb, cols)

        # Crop to even dimensions
        aa_frame = aa_frame.crop((0, 0, out_w, out_h))

        # Sample metrics on first frame only (cheap, representative)
        if i == 0:
            # Estimate outline ring cells by measuring '@' increase would be complex;
            # instead count cells with near-_OUTLINE_COLOR brightness
            arr_after = np.array(aa_frame)
            # Blue glow: cells with high blue channel relative to red
            blue_mask = (arr_after[:, :, 2].astype(int) - arr_after[:, :, 0].astype(int)) > 30
            blue_cells_total = int(blue_mask.sum())
            # Outline ring: cells that are bright and blue-white (240+)
            outline_mask = (arr_after[:, :, 0] > 200) & (arr_after[:, :, 2] > 220)
            outline_cells_total = int(outline_mask.sum())

        proc.stdin.write(np.array(aa_frame.convert("RGB"), dtype=np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-400:]}")

    print(f"[style-I long_boo] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": out_w,
        "canvas_h": out_h,
        "outline_ring_cells": outline_cells_total,
        "blue_glow_cells": blue_cells_total,
    }

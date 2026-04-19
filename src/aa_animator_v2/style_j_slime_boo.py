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
"""Style J: slime_dance base + blink keyframes + Ghostty-boo post-processing.

Motion design (self-designed, no boo frame data copied):
  - breathe  : squash & stretch anti-phase on x/y axes (aa_animator.py:209-213)
  - blink    : cubic ease-in/out blink curve applied to upper-region cells,
               darken them at blink_timings [0.18, 0.50, 0.82] (normalised t).
               Ported from aa_animator.py:317-324 _blink_curve concept.

Ghostty-boo reinforcement (via _boo_postprocess):
  - outline_ring : Sobel edges → '@' glyph, bright outline colour
  - blue_glow    : light-glyph cells → blue (80,140,255) tint

Cell grid: FIXED positions (no pixel-space deformation; threshold modulation only)
Charset : " ·~ox+=*%$@"
FPS     : 30
Duration: 5.0 s
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

# Breathe (slime mass swing) tuning
_BREATHE_FREQ: float = 2.0  # cycles per second
_BREATHE_X_AMP: float = 0.08
_BREATHE_Y_AMP: float = 0.10

# Blink tuning — ported from aa_animator.py:317-324 _blink_curve concept
_BLINK_TIMINGS: list[float] = [0.18, 0.50, 0.82]  # normalised t for blink peaks
_BLINK_HALF_WIDTH: float = 0.04  # half-width of blink window in normalised t
_BLINK_DARKEN: float = 0.25  # factor: eye region brightness during blink (0=black, 1=unchanged)

# Eye region: upper quarter of canvas, centre third horizontally
_EYE_REGION_TOP: float = 0.15  # start row fraction
_EYE_REGION_BOT: float = 0.42  # end row fraction
_EYE_REGION_LEFT: float = 0.25  # start col fraction
_EYE_REGION_RIGHT: float = 0.75  # end col fraction


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Blink curve (aa_animator.py:317-324 _blink_curve port)
# ---------------------------------------------------------------------------


def _blink_intensity(t: float) -> float:
    """Return blink darkening intensity in [0, 1] at normalised time t.

    Uses cubic ease-in/out within each blink window. Returns 0 outside any
    window (no darkening), 1 at the peak of a blink.

    Args:
        t: Normalised time in [0, 1].

    Returns:
        Blink intensity in [0, 1].
    """
    best = 0.0
    for bt in _BLINK_TIMINGS:
        dt = abs(t - bt)
        if dt < _BLINK_HALF_WIDTH:
            x = dt / _BLINK_HALF_WIDTH  # 0 at peak, 1 at edge
            # Cubic ease: smooth rise to peak (x inverted: 1 at peak)
            intensity = 1.0 - (x * x * (3.0 - 2.0 * x))
            best = max(best, intensity)
    return best


# ---------------------------------------------------------------------------
# Motion: breathe (slime dance)
# ---------------------------------------------------------------------------


def _apply_breathe(
    img: Image.Image,
    t: float,
    canvas_size: int,
    fps: float,
) -> Image.Image:
    """Apply breathe (mass swing) to RGBA source image.

    Squash and stretch in anti-phase on x/y axes, creating a gelatinous
    slime-dance oscillation.  Cell grid positions are NOT deformed; only
    the composited pixel canvas is scaled.

    Args:
        img: Source RGBA image.
        t: Normalised time in [0, 1].
        canvas_size: Square canvas pixel size.
        fps: Frames per second (for frequency calculation).

    Returns:
        RGB PIL Image at (canvas_size, canvas_size).
    """
    freq_rad = t * math.pi * 2.0 * _BREATHE_FREQ
    scale_x = 1.0 + _BREATHE_X_AMP * math.sin(freq_rad)
    scale_y = 1.0 - _BREATHE_Y_AMP * math.sin(freq_rad)  # anti-phase

    cw, ch = img.size
    new_w = max(1, int(cw * scale_x))
    new_h = max(1, int(ch * scale_y))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Blink post-process (cell-level threshold modulation)
# ---------------------------------------------------------------------------


def _apply_blink(
    rendered: Image.Image,
    t: float,
    rows: int,
    cols: int,
) -> Image.Image:
    """Darken eye-region cells during blink keyframes.

    Modulates cell brightness in the upper-centre region of the rendered
    AA frame.  The cell grid positions stay fixed; only per-cell colour
    is dimmed.  Intensity is driven by _blink_intensity(t).

    Args:
        rendered: DensityAA-rendered RGB PIL Image.
        t: Normalised time in [0, 1].
        rows: Cell grid row count.
        cols: Cell grid column count.

    Returns:
        Blink-modified RGB PIL Image.
    """
    intensity = _blink_intensity(t)
    if intensity < 0.01:
        return rendered

    # Row/col bounds for eye region
    r_start = max(0, int(_EYE_REGION_TOP * rows))
    r_end = min(rows, int(_EYE_REGION_BOT * rows))
    c_start = max(0, int(_EYE_REGION_LEFT * cols))
    c_end = min(cols, int(_EYE_REGION_RIGHT * cols))

    if r_start >= r_end or c_start >= c_end:
        return rendered

    # Compute pixel bounding box for eye region
    y0_px = r_start * CELL_H
    y1_px = r_end * CELL_H
    x0_px = c_start * CELL_W
    x1_px = c_end * CELL_W

    out = rendered.copy()
    arr = np.array(out, dtype=np.float32)
    region = arr[y0_px:y1_px, x0_px:x1_px]
    # Darken by lerping toward black based on blink intensity and _BLINK_DARKEN
    darken_factor = 1.0 - intensity * (1.0 - _BLINK_DARKEN)
    region *= darken_factor
    arr[y0_px:y1_px, x0_px:x1_px] = region

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# DensityAA renderer
# ---------------------------------------------------------------------------


def _render_density_aa(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame as Ghostty-palette density AA.

    Args:
        frame_img: RGB PIL Image.
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


def generate_style_j(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 80,
    fps: int = 30,
    duration: float = 5.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style J (slime_dance + blink + Ghostty-boo) animation MP4.

    Applies breathe (slime mass swing) motion + blink keyframes at three
    timings, then reinforces Ghostty-boo aesthetics via outline_ring and
    blue_glow post-processing.  Cell grid positions are fixed (no pixel
    deformation); only threshold/brightness modulation is applied per cell.

    Args:
        input_path: Path to input image.
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells.
        fps: Output video frame rate (default 30).
        duration: Clip length in seconds (default 5.0).
        canvas_size: Square compositing canvas pixel size.

    Returns:
        Dict with output_path, n_frames, canvas_w, canvas_h,
        blink_timings, outline_ring_cells, blue_glow_cells.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    img_src.thumbnail((canvas_size, canvas_size), Image.LANCZOS)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    # Derive cell grid dimensions from a test render
    probe_rgb = _apply_breathe(img_src, 0.0, canvas_size, fps)
    probe_aa = _render_density_aa(probe_rgb, cols, font)
    out_w, out_h = probe_aa.size
    out_w -= out_w % 2
    out_h -= out_h % 2

    # Cell row count (for blink region calculation)
    rows = max(1, int(canvas_size * cols / canvas_size * FONT_RATIO))

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{out_w}x{out_h}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-tune",
        "animation",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
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
        frame_rgb = _apply_breathe(img_src, t, canvas_size, fps)
        aa_frame = _render_density_aa(frame_rgb, cols, font)

        # Blink keyframe modulation (cell-level, fixed grid)
        aa_frame = _apply_blink(aa_frame, t, rows, cols)

        # Ghostty-boo post-processing
        aa_frame = apply_outline_ring(aa_frame, frame_rgb, cols, font)
        aa_frame = apply_blue_glow(aa_frame, frame_rgb, cols)

        # Crop to even dimensions
        aa_frame = aa_frame.crop((0, 0, out_w, out_h))

        if i == 0:
            arr_after = np.array(aa_frame)
            blue_mask = (arr_after[:, :, 2].astype(int) - arr_after[:, :, 0].astype(int)) > 30
            blue_cells_total = int(blue_mask.sum())
            outline_mask = (arr_after[:, :, 0] > 200) & (arr_after[:, :, 2] > 220)
            outline_cells_total = int(outline_mask.sum())

        proc.stdin.write(np.array(aa_frame.convert("RGB"), dtype=np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-400:]}")

    print(f"[style-J slime_boo] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": out_w,
        "canvas_h": out_h,
        "blink_timings": _BLINK_TIMINGS,
        "outline_ring_cells": outline_cells_total,
        "blue_glow_cells": blue_cells_total,
    }

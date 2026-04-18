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
"""Style B: Style A + boo-inspired self-designed motion template.

Motion design (NO data copied from boo frames):
  - Silhouette mass swing: scale_x/scale_y anti-phase squash
    (heavier squash than Style A breathe, inspired by aa_animator.py:209-213)
  - Head bob + glow pulse with phase offset (bob leads glow by pi/4)
  - Centre pos micro-drift: ±2 cell sway via sin
  - Glow brightness modulated in sync with mass swing

All motion is synthesised from first principles using sin/cos.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants (same ramp / palette as Style A)
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_EDGE_THRESH: float = 0.15
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)

_CELL_W: int = 7
_CELL_H: int = 14
_FONT_SIZE: int = 12
_FONT_RATIO: float = 0.50

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]

# Boo-motion tuning knobs (self-designed, not extracted from any boo frame data)
_MASS_SWING_FREQ: float = 3.0   # cycles per second
_MASS_X_AMP: float = 0.10       # max horizontal squash fraction
_MASS_Y_AMP: float = 0.12       # max vertical stretch fraction
_SWAY_CELLS: float = 2.0        # ±cell sway amplitude
_BOB_PX: float = 14.0           # vertical bob pixels
_GLOW_ALPHA_BASE: float = 0.20
_GLOW_ALPHA_SWING: float = 0.20  # glow oscillates 0.20..0.40


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
# Boo-inspired motion (self-designed)
# ---------------------------------------------------------------------------

def _apply_boo_motion(img: Image.Image, t: float, canvas_size: int) -> tuple[Image.Image, float]:
    """Apply boo-inspired motion template.

    Motion layers (all synthesised, no boo data):
      1. Silhouette mass swing — anti-phase squash/stretch on x/y axes.
      2. Head bob — vertical oscillation with slight landing squash.
      3. Centre sway — horizontal micro-drift ±_SWAY_CELLS * cell width.
      4. Glow amplitude — returned as float for renderer to use.

    Args:
        img: Source RGBA image.
        t: Normalised time [0, 1].
        canvas_size: Square canvas pixel size.

    Returns:
        Tuple of (composited RGB Image, glow_alpha float).
    """
    freq_rad = t * math.pi * 2.0 * _MASS_SWING_FREQ

    # 1. Silhouette mass swing (anti-phase on x vs y)
    scale_x = 1.0 + _MASS_X_AMP * math.sin(freq_rad)
    scale_y = 1.0 - _MASS_Y_AMP * math.sin(freq_rad)  # anti-phase

    # 2. Head bob (phase offset pi/4 vs swing)
    bob_phase = math.sin(freq_rad - math.pi / 4)
    offset_y = -_BOB_PX * max(0.0, bob_phase)
    # Landing squash
    if bob_phase < -0.65:
        scale_x *= 1.05
        scale_y *= 0.95

    # 3. Centre sway — slow drift, distinct from swing frequency
    sway_px = _SWAY_CELLS * _CELL_W * math.sin(t * math.pi * 2.0 * 1.3)
    offset_x = sway_px

    # 4. Glow alpha: pulsates in sync with mass swing (phase +pi/2 for smooth feel)
    glow_alpha = _GLOW_ALPHA_BASE + _GLOW_ALPHA_SWING * (0.5 + 0.5 * math.sin(freq_rad + math.pi / 2))

    # Apply transform
    cw, ch = img.size
    new_w = max(1, int(cw * scale_x))
    new_h = max(1, int(ch * scale_y))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2 + int(offset_x)
    y = (canvas_size - new_h) // 2 + int(offset_y)
    canvas.paste(resized, (x, y), resized)

    return canvas.convert("RGB"), glow_alpha


# ---------------------------------------------------------------------------
# DensityAA render (same as style_a_gallery — kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _render_density_aa(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small = frame_img.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)

    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx = np.clip((lum / 255.0 * n).astype(int), 0, n)

    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
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
            d.text((x * _CELL_W, y * _CELL_H), ch, fill=color, font=font)

    return out


def _apply_edge_glow(
    rendered: Image.Image,
    frame_img: Image.Image,
    cols: int,
    glow_alpha: float,
) -> Image.Image:
    from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]

    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small_gray = np.array(frame_img.convert("L").resize((cols, rows), Image.LANCZOS)).astype(np.float32)

    sx = scipy_sobel(small_gray, axis=1)
    sy = scipy_sobel(small_gray, axis=0)
    mag = np.hypot(sx, sy)
    if mag.max() > 0:
        mag /= mag.max()
    edge_cell = mag > _EDGE_THRESH

    glow_cell = binary_dilation(edge_cell, structure=np.ones((3, 3), dtype=bool)) & ~edge_cell
    if not glow_cell.any():
        return rendered

    canvas_arr = np.array(rendered, dtype=np.float32)
    for ry in range(rows):
        for cx in range(cols):
            if not glow_cell[ry, cx]:
                continue
            y0, y1 = ry * _CELL_H, (ry + 1) * _CELL_H
            x0, x1 = cx * _CELL_W, (cx + 1) * _CELL_W
            region = canvas_arr[y0:y1, x0:x1]
            region[:] = (
                region * (1.0 - glow_alpha)
                + np.array(_GLOW_COLOR, dtype=np.float32) * glow_alpha
            )

    return Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_style_b(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 80,
    fps: int = 30,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style B (boo-inspired motion) animation MP4.

    Args:
        input_path: Path to input image.
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells.
        fps: Output video frame rate.
        duration: Clip length in seconds.
        canvas_size: Square canvas pixel size.

    Returns:
        Dict with output_path, n_frames, canvas_w, canvas_h.
    """
    import subprocess
    import sys

    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    img_src.thumbnail((canvas_size, canvas_size), Image.LANCZOS)
    canvas_base = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    ox = (canvas_size - img_src.width) // 2
    oy = (canvas_size - img_src.height) // 2
    canvas_base.paste(img_src, (ox, oy), img_src)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    probe_rgb, _ = _apply_boo_motion(canvas_base, 0.0, canvas_size)
    probe = _render_density_aa(probe_rgb, cols, font)
    canvas_w, canvas_h = probe.size

    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb, glow_alpha = _apply_boo_motion(canvas_base, t, canvas_size)
        aa_frame = _render_density_aa(frame_rgb, cols, font)
        aa_frame = _apply_edge_glow(aa_frame, frame_rgb, cols, glow_alpha)
        frames.append(aa_frame)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{canvas_w}x{canvas_h}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-tune", "animation", "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-400:]}")

    print(f"[style-B] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }

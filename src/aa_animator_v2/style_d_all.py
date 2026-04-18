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
"""Style D: All-in-one — A + B + C combined for comparison.

Composition:
  - Motion: Style B boo-inspired motion (mass swing + sway + dynamic glow).
  - Edge render: Style C DoG + Sobel direction glyphs on edge cells.
  - Body render: Style A DensityAA with colour preservation on body cells.
  - Glow: dynamic alpha from Style B, applied to Style C edge neighbourhood.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]
from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants (unified)
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_EDGE_GLYPHS: list[str] = ["|", "/", "-", "\\"]
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)

_CELL_W: int = 7
_CELL_H: int = 14
_FONT_SIZE: int = 12
_FONT_RATIO: float = 0.50

_DOG_THRESH: float = 0.04
_SOBEL_FRACTION: float = 0.65

# Boo-motion knobs (from Style B)
_MASS_SWING_FREQ: float = 3.0
_MASS_X_AMP: float = 0.10
_MASS_Y_AMP: float = 0.12
_SWAY_CELLS: float = 2.0
_BOB_PX: float = 14.0
_GLOW_ALPHA_BASE: float = 0.20
_GLOW_ALPHA_SWING: float = 0.20

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]


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
# Style B motion
# ---------------------------------------------------------------------------

def _apply_boo_motion(img: Image.Image, t: float, canvas_size: int) -> tuple[Image.Image, float]:
    freq_rad = t * math.pi * 2.0 * _MASS_SWING_FREQ
    scale_x = 1.0 + _MASS_X_AMP * math.sin(freq_rad)
    scale_y = 1.0 - _MASS_Y_AMP * math.sin(freq_rad)
    bob_phase = math.sin(freq_rad - math.pi / 4)
    offset_y = -_BOB_PX * max(0.0, bob_phase)
    if bob_phase < -0.65:
        scale_x *= 1.05
        scale_y *= 0.95
    sway_px = _SWAY_CELLS * _CELL_W * math.sin(t * math.pi * 2.0 * 1.3)
    glow_alpha = _GLOW_ALPHA_BASE + _GLOW_ALPHA_SWING * (
        0.5 + 0.5 * math.sin(freq_rad + math.pi / 2)
    )
    cw, ch = img.size
    new_w = max(1, int(cw * scale_x))
    new_h = max(1, int(ch * scale_y))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2 + int(sway_px)
    y = (canvas_size - new_h) // 2 + int(offset_y)
    canvas.paste(resized, (x, y), resized)
    return canvas.convert("RGB"), glow_alpha


# ---------------------------------------------------------------------------
# DoG helper
# ---------------------------------------------------------------------------

def _dog_edge(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 1.6) -> np.ndarray:
    g1 = gaussian_filter(gray, sigma=sigma1)
    g2 = gaussian_filter(gray, sigma=sigma2)
    dog = np.abs(g1 - g2)
    if dog.max() > 0:
        dog /= dog.max()
    return dog


def _sobel_direction_glyph(dx: float, dy: float) -> str:
    angle = math.atan2(dy, dx) * 180.0 / math.pi
    angle = angle % 180.0
    if angle < 22.5 or angle >= 157.5:
        return "|"
    elif angle < 67.5:
        return "/"
    elif angle < 112.5:
        return "-"
    else:
        return "\\"


# ---------------------------------------------------------------------------
# Combined render: A body + C edge
# ---------------------------------------------------------------------------

def _render_combined(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    glow_alpha: float,
) -> Image.Image:
    """Render Style D: DensityAA body + DoG directional edge glyphs + dynamic glow."""
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small = frame_img.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    arr_01 = arr / 255.0
    lum = _srgb_luma(arr_01)

    dog = _dog_edge(lum)
    sx = scipy_sobel(lum, axis=1)
    sy = scipy_sobel(lum, axis=0)
    sobel_mag = np.hypot(sx, sy)
    sobel_max = sobel_mag.max()
    sobel_norm = sobel_mag / sobel_max if sobel_max > 0 else sobel_mag

    edge_mask = (dog > _DOG_THRESH) & (sobel_norm > _SOBEL_FRACTION * sobel_norm.max() + 1e-8)

    # Density ramp for body chars
    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    body_idx = np.clip((lum * n).astype(int), 0, n)

    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for ry in range(rows):
        for cx in range(cols):
            if lum[ry, cx] < 0.02:
                continue
            px, py = cx * _CELL_W, ry * _CELL_H

            if edge_mask[ry, cx]:
                # Style C: directional edge glyph in blue
                ch = _sobel_direction_glyph(float(sx[ry, cx]), float(sy[ry, cx]))
                color = _GLOW_COLOR
            else:
                # Style A: density AA with colour preservation
                ch = charset[body_idx[ry, cx]]
                if ch == " ":
                    continue
                r = arr[ry, cx, 0]
                g = arr[ry, cx, 1]
                b = arr[ry, cx, 2]
                boost = max(1.0, 1.6 - lum[ry, cx] * 255.0 / 350.0)
                color = (
                    min(255, int(r * boost)),
                    min(255, int(g * boost)),
                    min(255, int(b * boost)),
                )

            d.text((px, py), ch, fill=color, font=font)

    # Dynamic glow (Style B alpha) on edge-adjacent body cells
    glow_cell = binary_dilation(edge_mask, structure=np.ones((3, 3), dtype=bool)) & ~edge_mask
    if glow_cell.any():
        canvas_arr = np.array(out, dtype=np.float32)
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
        out = Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_style_d(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 80,
    fps: int = 30,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style D (all-in-one: A+B+C) animation MP4.

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

    probe_rgb, probe_glow = _apply_boo_motion(canvas_base, 0.0, canvas_size)
    probe = _render_combined(probe_rgb, cols, font, probe_glow)
    canvas_w, canvas_h = probe.size

    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb, glow_alpha = _apply_boo_motion(canvas_base, t, canvas_size)
        aa_frame = _render_combined(frame_rgb, cols, font, glow_alpha)
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

    print(f"[style-D] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }

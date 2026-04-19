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
"""Style F: dynamic lighting AA animation.

A 2-D light source moves smoothly through cell-grid space.  Each cell's
brightness is a Gaussian falloff from the light position — a **continuous
float**, never a boolean.  Characters and positions are fixed; only colour
and glyph luminance change between frames.

Lighting patterns
-----------------
horizontal_sweep  : light sweeps left↔right (sinusoidal, 0.5 Hz)
vertical_drop     : light drops top→bottom and loops (CRT scan, gradient)
orbit             : light traces a circular path around centre
lissajous         : 2-D Lissajous curve (a=3, b=2)
spotlight_trail   : light moves on orbit + past trail decays (temporal echo)

No boolean masks anywhere — every value is a smooth float in [0, 1].
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants — identical palette/style to style_e_signal.py
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_STATIC_COLOR: tuple[int, int, int] = (215, 215, 215)
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)

_CELL_W: int = 7
_CELL_H: int = 14
_FONT_SIZE: int = 12

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]

VALID_PATTERNS = (
    "horizontal_sweep",
    "vertical_drop",
    "orbit",
    "lissajous",
    "spotlight_trail",
)


# ---------------------------------------------------------------------------
# Font loader
# ---------------------------------------------------------------------------


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (H, W, 3) float32 [0-255]."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Static base — built once, frame-invariant
# ---------------------------------------------------------------------------


def compute_base_brightness(
    image: Image.Image,
    cols: int = 100,
    rows: int = 41,
) -> np.ndarray:
    """Compute per-cell base brightness from source image (frame-invariant).

    Args:
        image: Source image (any PIL mode; converted to RGB internally).
        cols: Grid width in cells.
        rows: Grid height in cells.

    Returns:
        Float32 ndarray (rows, cols) in [0, 1] representing cell luma.
    """
    rgb = image.convert("RGB")
    small = rgb.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)
    return (lum / 255.0).astype(np.float32)


def _build_char_grid(
    base_brightness: np.ndarray,
) -> list[list[str]]:
    """Build static char grid from base brightness array.

    Args:
        base_brightness: Float32 (rows, cols) in [0, 1].

    Returns:
        rows × cols list[list[str]] — space for near-black cells.
    """
    rows, cols = base_brightness.shape
    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx = np.clip((base_brightness * n).astype(int), 0, n)
    char_grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            ch = charset[idx[r, c]] if base_brightness[r, c] >= (6.0 / 255.0) else " "
            row.append(ch)
        char_grid.append(row)
    return char_grid


# ---------------------------------------------------------------------------
# Light-source position functions — return (Lx, Ly) in cell coordinates
# ---------------------------------------------------------------------------


def light_horizontal_sweep(t: float, rows: int, cols: int) -> tuple[float, float]:
    """Light sweeps left↔right sinusoidally at 0.5 Hz, y fixed at centre.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        (Lx, Ly) light source position in cell coordinates.
    """
    freq = 0.5
    lx = (cols / 2.0) + (cols / 2.0) * math.sin(2.0 * math.pi * freq * t)
    ly = rows / 2.0
    return lx, ly


def light_vertical_drop(t: float, rows: int, cols: int) -> tuple[float, float]:
    """Light drops top→bottom linearly at 10 cells/sec, loops.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        (Lx, Ly) light source position in cell coordinates.
    """
    speed = 10.0
    ly = (t * speed) % rows
    lx = cols / 2.0
    return lx, ly


def light_orbit(t: float, rows: int, cols: int) -> tuple[float, float]:
    """Light traces circular orbit with radius = min(rows, cols) / 3.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        (Lx, Ly) light source position in cell coordinates.
    """
    freq = 0.5
    r = min(rows, cols) / 3.0
    cx, cy = cols / 2.0, rows / 2.0
    lx = cx + r * math.cos(2.0 * math.pi * freq * t)
    ly = cy + r * math.sin(2.0 * math.pi * freq * t)
    return lx, ly


def light_lissajous(t: float, rows: int, cols: int) -> tuple[float, float]:
    """Light traces Lissajous curve (a=3, b=2, phi=pi/2).

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        (Lx, Ly) light source position in cell coordinates.
    """
    rx = cols / 3.0
    ry = rows / 3.0
    cx, cy = cols / 2.0, rows / 2.0
    a, b = 3.0, 2.0
    phi = math.pi / 2.0
    freq = 0.25
    lx = cx + rx * math.sin(a * 2.0 * math.pi * freq * t + phi)
    ly = cy + ry * math.sin(b * 2.0 * math.pi * freq * t)
    return lx, ly


# ---------------------------------------------------------------------------
# Intensity computation — pure float, no booleans
# ---------------------------------------------------------------------------


def compute_lit_intensity(
    base_brightness: np.ndarray,
    lx: float,
    ly: float,
    sigma: float = 8.0,
    light_intensity: float = 0.5,
) -> np.ndarray:
    """Compute per-cell lit intensity as float in [0, 1].

    lit[r,c] = clip(base[r,c] + light_intensity * gaussian_falloff(dist), 0, 1)

    No thresholding, no boolean mask — all values are continuous floats.

    Args:
        base_brightness: Float32 (rows, cols) in [0, 1] — frame-invariant.
        lx: Light source x position in cell coordinates.
        ly: Light source y position in cell coordinates.
        sigma: Gaussian radius in cells (larger = softer falloff).
        light_intensity: Peak additive brightness from the light source.

    Returns:
        Float32 ndarray (rows, cols) in [0, 1].
    """
    rows, cols = base_brightness.shape
    xs = np.arange(cols, dtype=np.float32)
    ys = np.arange(rows, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist_sq = (xx - lx) ** 2 + (yy - ly) ** 2
    falloff = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
    lit = base_brightness + light_intensity * falloff
    return np.clip(lit, 0.0, 1.0)


def compute_trail_intensity(
    base_brightness: np.ndarray,
    trail_grid: np.ndarray,
) -> np.ndarray:
    """Combine base brightness with accumulated trail (spotlight_trail pattern).

    Args:
        base_brightness: Float32 (rows, cols) in [0, 1].
        trail_grid: Float32 (rows, cols) accumulated light intensity in [0, 1].

    Returns:
        Float32 ndarray (rows, cols) in [0, 1].
    """
    return np.clip(base_brightness + trail_grid, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Glyph quantisation — smooth re-quantise per frame
# ---------------------------------------------------------------------------


def quantize_to_glyph(lit_value: float, palette: str = _GHOSTTY_CHARS) -> str:
    """Map float intensity to a Ghostty palette character.

    Args:
        lit_value: Float in [0, 1].
        palette: Character ramp (darkest first, brightest last).

    Returns:
        Single character from palette.
    """
    n = len(palette) - 1
    idx = int(np.clip(lit_value * n, 0, n))
    return palette[idx]


def _recompute_char_grid(
    char_grid_static: list[list[str]],
    lit_intensity: np.ndarray,
    palette: str = _GHOSTTY_CHARS,
) -> list[list[str]]:
    """Re-quantise char grid using per-cell lit intensity (smooth, per-frame).

    Cells that are space in the static grid stay space — background is never lit.
    Non-space cells have their glyph updated to reflect current lit intensity.

    Args:
        char_grid_static: Static base char grid (space = background, never changed).
        lit_intensity: Float32 (rows, cols) in [0, 1].
        palette: Ghostty character ramp.

    Returns:
        New rows × cols list[list[str]] for this frame — char_grid_static is NOT mutated.
    """
    rows = len(char_grid_static)
    cols = len(char_grid_static[0]) if rows > 0 else 0
    n = len(palette) - 1
    frame_grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            if char_grid_static[r][c] == " ":
                row.append(" ")
            else:
                idx = int(np.clip(lit_intensity[r, c] * n, 0, n))
                row.append(palette[idx])
        frame_grid.append(row)
    return frame_grid


# ---------------------------------------------------------------------------
# Color interpolation — continuous lerp, no boolean switch
# ---------------------------------------------------------------------------


def lerp_color(
    white: tuple[int, int, int],
    blue: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Linear interpolate between two RGB colours.

    Args:
        white: Low-intensity (dim) colour.
        blue: High-intensity (bright/lit) colour.
        alpha: Blend factor in [0, 1] — 0.0 → white, 1.0 → blue.

    Returns:
        Interpolated (R, G, B) tuple.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    return (
        int(white[0] * (1.0 - a) + blue[0] * a),
        int(white[1] * (1.0 - a) + blue[1] * a),
        int(white[2] * (1.0 - a) + blue[2] * a),
    )


# ---------------------------------------------------------------------------
# Frame renderer — static cell positions, smooth per-frame colour
# ---------------------------------------------------------------------------


def render_frame(
    char_grid: list[list[str]],
    color_grid: np.ndarray,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame from char grid and per-cell colour (float lit intensity).

    Args:
        char_grid: rows × cols str grid — space cells are never drawn.
        color_grid: Float32 (rows, cols) in [0, 1] — per-cell lit intensity,
            used to interpolate colour from STATIC_COLOR to GLOW_COLOR.
        font: Pre-loaded monospace font.

    Returns:
        RGB PIL Image.
    """
    rows = len(char_grid)
    cols = len(char_grid[0]) if rows > 0 else 0
    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for r in range(rows):
        for c in range(cols):
            ch = char_grid[r][c]
            if ch == " ":
                continue
            color = lerp_color(_STATIC_COLOR, _GLOW_COLOR, float(color_grid[r, c]))
            d.text((c * _CELL_W, r * _CELL_H), ch, fill=color, font=font)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_style_f(
    input_path: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "orbit",
    cols: int = 100,
    rows: int = 41,
    fps: int = 30,
    duration: float = 4.0,
    sigma: float = 8.0,
    light_intensity: float = 0.5,
    trail_decay: float = 0.85,
) -> dict:
    """Generate Style F dynamic lighting animation MP4.

    The light source moves smoothly through 2-D cell space; per-cell
    brightness and colour are continuous floats — no boolean masks.

    Args:
        input_path: Source image path.
        output_path: Destination .mp4 path.
        pattern: One of VALID_PATTERNS.
        cols: Cell grid width.
        rows: Cell grid height.
        fps: Output frame rate.
        duration: Clip length in seconds.
        sigma: Gaussian light radius in cells.
        light_intensity: Peak additive brightness from light source (0-1).
        trail_decay: Per-frame decay multiplier for spotlight_trail (0-1).

    Returns:
        Dict with keys: output_path, n_frames, canvas_w, canvas_h,
        pattern, char_grid_invariant (True), smooth_verified (True).
    """
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"pattern must be one of {VALID_PATTERNS}, got {pattern!r}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path)
    base_brightness = compute_base_brightness(img, cols=cols, rows=rows)
    char_grid_static = _build_char_grid(base_brightness)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)
    canvas_w = cols * _CELL_W
    canvas_h = rows * _CELL_H

    # Light position functions
    _light_fns = {
        "horizontal_sweep": light_horizontal_sweep,
        "vertical_drop": light_vertical_drop,
        "orbit": light_orbit,
        "lissajous": light_lissajous,
        "spotlight_trail": light_orbit,  # uses orbit trajectory + temporal trail
    }
    light_fn = _light_fns[pattern]

    # Accumulation grid for spotlight_trail (zeroed at start)
    trail_grid = np.zeros((rows, cols), dtype=np.float32)

    # Smoothness verification: track per-frame max delta
    prev_lit: np.ndarray | None = None
    max_delta_per_frame: list[float] = []

    # Snapshot static grid for invariance check
    import copy

    char_grid_copy = copy.deepcopy(char_grid_static)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{canvas_w}x{canvas_h}",
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

    for i in range(n_frames):
        t = i / fps

        lx, ly = light_fn(t, rows, cols)

        if pattern == "spotlight_trail":
            # Decay existing trail
            trail_grid *= trail_decay
            # Add current light pulse to trail
            xs = np.arange(cols, dtype=np.float32)
            ys = np.arange(rows, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            dist_sq = (xx - lx) ** 2 + (yy - ly) ** 2
            falloff = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
            trail_grid += light_intensity * falloff
            np.clip(trail_grid, 0.0, 1.0, out=trail_grid)
            lit = compute_trail_intensity(base_brightness, trail_grid)
        else:
            lit = compute_lit_intensity(base_brightness, lx, ly, sigma, light_intensity)

        # Re-quantise characters using lit intensity (smooth, per-frame)
        frame_char_grid = _recompute_char_grid(char_grid_static, lit)

        frame = render_frame(frame_char_grid, lit, font)
        proc.stdin.write(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes())

        # Track smoothness
        if prev_lit is not None:
            max_delta_per_frame.append(float(np.max(np.abs(lit - prev_lit))))
        prev_lit = lit.copy()

    proc.stdin.close()
    proc.wait()

    # Invariance check — static grid must not have been mutated
    assert char_grid_static == char_grid_copy, "char_grid_static was mutated — invariant violated"

    # Smoothness check: per-frame deltas must be < 0.50 (no boolean hard-flip).
    # Gaussian falloff at speed ~5 cells/frame with sigma=8 legitimately produces
    # ~0.19 delta at the light centre — that is visually smooth (continuous),
    # not a blink. The threshold catches only boolean on/off artefacts (delta≈1.0).
    if max_delta_per_frame:
        max_observed_delta = max(max_delta_per_frame)
        assert max_observed_delta < 0.50, (
            f"Intensity jump too large ({max_observed_delta:.4f}) — boolean mask detected"
        )
        smooth_verified = True
    else:
        smooth_verified = True

    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "pattern": pattern,
        "char_grid_invariant": True,
        "smooth_verified": smooth_verified,
        "max_delta_per_frame": max(max_delta_per_frame) if max_delta_per_frame else 0.0,
    }

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
"""Style G: 3D-aware dynamic lighting AA animation.

Character grid is built ONCE from the source image (static base layer).
Each frame the per-cell lit intensity changes via a 3D light source moving in
(Lx, Ly, Lz) space — with DA-V2 depth giving each cell a genuine z coordinate.

Lighting Patterns
-----------------
approach    : light source moves from far (z=0.9) toward the viewer (z=0), loops
orbit_3d    : light orbits on a 3D circle around the image centre
spiral      : light spirals in 3D (helical descent)
pendulum_3d : light swings back-and-forth on the z axis only
rim_light   : light fixed far behind the object (z=0.9) — edges glow as rim

Conventions
-----------
- depth 0 = foreground (close to viewer), depth 1 = background (far)
- Lz = 0 is at viewer plane; Lz = 1 is at the farthest background
- Z_SCALE normalises z distances to be comparable to cell-space x/y distances
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants — match Style E / Ghostty palette
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

VALID_PATTERNS = ("approach", "orbit_3d", "spiral", "pendulum_3d", "rim_light")

# Lighting parameters
_SIGMA: float = 10.0  # Gaussian falloff radius in cell units
_LIGHT_INTENSITY: float = 0.6
_RIM_SIGMA: float = 4.0  # Smaller sigma for rim_light edge sharpness


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
# Depth estimation — wraps pipeline._depth with cell-grid downsampling
# ---------------------------------------------------------------------------


def _estimate_depth_cell(
    image: Image.Image,
    cols: int,
    rows: int,
) -> np.ndarray:
    """Return per-cell depth map (rows, cols) in [0, 1].

    Uses DA-V2 Small via pipeline._depth.estimate_depth when available.
    Falls back to uniform 0.5 if transformers/torch are not installed.

    Args:
        image: Source PIL image (RGB).
        cols: Cell grid width.
        rows: Cell grid height.

    Returns:
        float32 ndarray of shape (rows, cols) with values in [0, 1].
        Convention: 0 = foreground (close), 1 = background (far).
    """
    from aa_animator_v2.pipeline._depth import estimate_depth as _da_estimate

    # estimate_depth takes (image, target_size=(w, h)) → (h, w) float32 [0,1]
    depth_cell = _da_estimate(image, target_size=(cols, rows))
    # DA-V2 convention: higher = closer. Invert so 0=near, 1=far.
    depth_cell = 1.0 - depth_cell
    return depth_cell.astype(np.float32)


# ---------------------------------------------------------------------------
# Base brightness + depth — built once per image
# ---------------------------------------------------------------------------


def compute_base_brightness_and_depth(
    image: Image.Image,
    cols: int = 100,
    rows: int = 41,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute static per-cell base brightness and depth.

    Args:
        image: Source image (any mode; converted to RGB internally).
        cols: Cell grid width.
        rows: Cell grid height.

    Returns:
        base: float32 ndarray (rows, cols) in [0, 1] — ITU-R luma fraction.
        depth_cell: float32 ndarray (rows, cols) in [0, 1] — 0=near, 1=far.
    """
    rgb = image.convert("RGB")
    small = rgb.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)
    base = (lum / 255.0).astype(np.float32)

    depth_cell = _estimate_depth_cell(rgb, cols, rows)
    return base, depth_cell


# ---------------------------------------------------------------------------
# Static char grid builder (uses base brightness quantised once)
# ---------------------------------------------------------------------------


def _build_char_grid(base: np.ndarray) -> list[list[str]]:
    """Build static char grid from base brightness.

    Args:
        base: float32 (rows, cols) in [0, 1].

    Returns:
        rows × cols list[list[str]].
    """
    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    rows, cols = base.shape
    idx = np.clip((base * n).astype(int), 0, n)
    char_grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            ch = charset[idx[r, c]] if base[r, c] >= (6.0 / 255.0) else " "
            row.append(ch)
        char_grid.append(row)
    return char_grid


# ---------------------------------------------------------------------------
# 3D light source trajectory functions
# ---------------------------------------------------------------------------


def light_approach(
    t: float,
    rows: int,
    cols: int,
    duration: float = 4.0,
) -> tuple[float, float, float]:
    """Light moves from far (z=0.9) toward viewer (z=0), then loops.

    Args:
        t: Time in seconds.
        rows: Grid height (for centre calculation).
        cols: Grid width (for centre calculation).
        duration: Clip duration for loop period.

    Returns:
        (Lx, Ly, Lz) in cell coordinates / depth units.
    """
    cx = cols / 2.0
    cy = rows / 2.0
    # z: 0.9 → 0 over duration, then loops
    phase = (t * 0.3) % 1.0
    lz = 0.9 - 0.9 * phase
    return cx, cy, lz


def light_orbit_3d(
    t: float,
    rows: int,
    cols: int,
    freq: float = 0.5,
) -> tuple[float, float, float]:
    """Light orbits on a 3D circle in the xz-plane, y fixed at centre.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.
        freq: Orbit frequency in Hz.

    Returns:
        (Lx, Ly, Lz) in cell coordinates / depth units.
    """
    cx = cols / 2.0
    cy = rows / 2.0
    r = cols * 0.4
    angle = 2.0 * math.pi * freq * t
    lx = cx + r * math.cos(angle)
    ly = cy
    lz = 0.5 + 0.3 * math.sin(angle)
    return lx, ly, lz


def light_spiral(
    t: float,
    rows: int,
    cols: int,
    duration: float = 4.0,
) -> tuple[float, float, float]:
    """Light follows a 3D helix — spiralling while descending in z.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.
        duration: Total clip duration for z ramp.

    Returns:
        (Lx, Ly, Lz) in cell coordinates / depth units.
    """
    cx = cols / 2.0
    cy = rows / 2.0
    r = cols * 0.3
    angular_speed = 1.5  # radians per second
    angle = angular_speed * t
    lx = cx + r * math.cos(angle)
    ly = cy + r * math.sin(angle) * 0.5
    # z descends from 0.9 to 0.0 over duration, then loops
    lz = 0.9 - 0.9 * ((t % duration) / duration)
    return lx, ly, lz


def light_pendulum_3d(
    t: float,
    rows: int,
    cols: int,
    freq: float = 0.4,
) -> tuple[float, float, float]:
    """Light swings on z axis (pendulum), x/y fixed at centre.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.
        freq: Pendulum frequency in Hz.

    Returns:
        (Lx, Ly, Lz) in cell coordinates / depth units.
    """
    cx = cols / 2.0
    cy = rows / 2.0
    lz = 0.5 + 0.4 * math.sin(2.0 * math.pi * freq * t)
    return cx, cy, lz


def light_rim_light(
    t: float,
    rows: int,
    cols: int,
    freq: float = 0.3,
) -> tuple[float, float, float]:
    """Rim light fixed far behind object (z=0.9), x/y oscillate slowly.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.
        freq: Oscillation frequency for x/y drift.

    Returns:
        (Lx, Ly, Lz) in cell coordinates / depth units.
    """
    cx = cols / 2.0
    cy = rows / 2.0
    drift_x = cols * 0.15 * math.sin(2.0 * math.pi * freq * t)
    drift_y = rows * 0.10 * math.cos(2.0 * math.pi * freq * t)
    lx = cx + drift_x
    ly = cy + drift_y
    lz = 0.9  # fixed far behind
    return lx, ly, lz


# ---------------------------------------------------------------------------
# Core 3D lighting computation
# ---------------------------------------------------------------------------


def compute_lit_intensity_3d(
    base: np.ndarray,
    depth_cell: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    sigma: float = _SIGMA,
    intensity: float = _LIGHT_INTENSITY,
    z_scale: float | None = None,
) -> np.ndarray:
    """Compute per-cell lit intensity for one frame.

    Each cell's 3D distance from the light source is computed using x/y cell
    coordinates and z from the depth map (scaled to cell-space).

    lit[r,c] = clamp(base[r,c] + intensity * exp(-dist_3d² / (2*sigma²)), 0, 1)

    Args:
        base: float32 (rows, cols) base brightness in [0, 1].
        depth_cell: float32 (rows, cols) depth in [0, 1]; 0=near, 1=far.
        lx: Light source x position in cell coordinates.
        ly: Light source y position in cell coordinates.
        lz: Light source z position in depth units [0, 1].
        sigma: Gaussian falloff radius in cell units.
        intensity: Peak light contribution added to base.
        z_scale: Scale factor for z distances. Defaults to cols/3.

    Returns:
        float32 ndarray (rows, cols) in [0, 1].
    """
    rows, cols = base.shape
    if z_scale is None:
        z_scale = cols / 3.0

    ys = np.arange(rows, dtype=np.float32)
    xs = np.arange(cols, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # (rows, cols)

    dx = xx - lx
    dy = yy - ly
    dz = (depth_cell - lz) * z_scale

    dist_sq = dx * dx + dy * dy + dz * dz
    falloff = np.exp(-dist_sq / (2.0 * sigma * sigma))

    lit = np.clip(base + intensity * falloff, 0.0, 1.0)
    return lit.astype(np.float32)


# ---------------------------------------------------------------------------
# Glyph quantisation — frame-varying character based on lit intensity
# ---------------------------------------------------------------------------


def quantize_to_glyph(lit: np.ndarray, palette: str = _GHOSTTY_CHARS) -> list[list[str]]:
    """Re-quantise lit intensity into glyph characters each frame.

    The character changes smoothly as lit intensity changes — this is the
    key mechanism for visible 3D lighting effect in the ASCII layer.

    Args:
        lit: float32 (rows, cols) in [0, 1] lit intensity.
        palette: Character ramp (dark → bright).

    Returns:
        rows × cols list[list[str]].
    """
    n = len(palette) - 1
    idx = np.clip((lit * n).astype(int), 0, n)
    rows, cols = lit.shape
    grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            # Background cells (very dark in lit) stay as space
            ch = palette[idx[r, c]] if lit[r, c] >= (6.0 / 255.0) else " "
            row.append(ch)
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Color interpolation
# ---------------------------------------------------------------------------


def lerp_color(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Linear interpolation between two RGB colors.

    Args:
        color_a: RGB tuple at alpha=0.
        color_b: RGB tuple at alpha=1.
        alpha: Blend factor in [0, 1].

    Returns:
        Interpolated RGB tuple.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (
        int(color_a[0] * (1.0 - alpha) + color_b[0] * alpha),
        int(color_a[1] * (1.0 - alpha) + color_b[1] * alpha),
        int(color_a[2] * (1.0 - alpha) + color_b[2] * alpha),
    )


# ---------------------------------------------------------------------------
# Frame renderer — static cell positions, dynamic chars and colors
# ---------------------------------------------------------------------------


def render_frame_g(
    char_grid: list[list[str]],
    color_grid: list[list[tuple[int, int, int]]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one Style G frame.

    Cell positions are always static; only char content and color change.

    Args:
        char_grid: rows × cols str grid for this frame.
        color_grid: rows × cols RGB color for each cell.
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
            color = color_grid[r][c]
            d.text((c * _CELL_W, r * _CELL_H), ch, fill=color, font=font)

    return out


def _compute_color_grid(
    lit: np.ndarray,
) -> list[list[tuple[int, int, int]]]:
    """Compute per-cell color from lit intensity by lerping static→glow.

    Args:
        lit: float32 (rows, cols) in [0, 1].

    Returns:
        rows × cols list[list[tuple[int,int,int]]].
    """
    rows, cols = lit.shape
    grid: list[list[tuple[int, int, int]]] = []
    for r in range(rows):
        row: list[tuple[int, int, int]] = []
        for c in range(cols):
            color = lerp_color(_STATIC_COLOR, _GLOW_COLOR, lit[r, c])
            row.append(color)
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_style_g(
    input_path: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "orbit_3d",
    cols: int = 100,
    rows: int = 41,
    fps: int = 30,
    duration: float = 4.0,
) -> dict:
    """Generate Style G 3D-aware lighting animation MP4.

    Character grid positions are static; per-cell glyph character and color
    update each frame based on a 3D light source trajectory and DA-V2 depth.

    Args:
        input_path: Source image path.
        output_path: Destination .mp4 path.
        pattern: One of "approach", "orbit_3d", "spiral", "pendulum_3d",
            "rim_light".
        cols: Cell grid width.
        rows: Cell grid height.
        fps: Output frame rate.
        duration: Clip length in seconds.

    Returns:
        Dict with keys: output_path, n_frames, canvas_w, canvas_h, pattern,
        depth_source, intensity_unique_values_ok (bool asserts passed).
    """
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"pattern must be one of {VALID_PATTERNS}, got {pattern!r}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert("RGB")

    # Build static base brightness and depth map (once)
    base, depth_cell = compute_base_brightness_and_depth(img, cols=cols, rows=rows)
    depth_source = "uniform_fallback" if np.allclose(depth_cell, 0.5) else "da_v2"

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)
    canvas_w = cols * _CELL_W
    canvas_h = rows * _CELL_H

    # Z_SCALE: normalise depth distances to cell-space x/y scale
    z_scale = cols / 3.0

    # Select lighting sigma: rim_light uses tighter beam
    sigma = _RIM_SIGMA if pattern == "rim_light" else _SIGMA

    # Map pattern name → trajectory function
    _light_fns = {
        "approach": lambda t: light_approach(t, rows, cols, duration),
        "orbit_3d": lambda t: light_orbit_3d(t, rows, cols),
        "spiral": lambda t: light_spiral(t, rows, cols, duration),
        "pendulum_3d": lambda t: light_pendulum_3d(t, rows, cols),
        "rim_light": lambda t: light_rim_light(t, rows, cols),
    }
    light_fn = _light_fns[pattern]

    # Generate frames and stream to ffmpeg
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

    prev_lit: np.ndarray | None = None
    lit_delta_max: float = 0.0
    # Collect per-cell values from the middle row across sampled frames to
    # verify the light field has at least 3 distinct float levels (no boolean).
    mid_row = rows // 2
    sampled_cell_values: list[float] = []

    for i in range(n_frames):
        t = i / fps

        lx, ly, lz = light_fn(t)
        lit = compute_lit_intensity_3d(
            base,
            depth_cell,
            lx,
            ly,
            lz,
            sigma=sigma,
            intensity=_LIGHT_INTENSITY,
            z_scale=z_scale,
        )

        char_grid_frame = quantize_to_glyph(lit)
        color_grid_frame = _compute_color_grid(lit)
        frame = render_frame_g(char_grid_frame, color_grid_frame, font)
        proc.stdin.write(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes())

        if prev_lit is not None:
            delta = float(np.abs(lit - prev_lit).max())
            lit_delta_max = max(lit_delta_max, delta)
        prev_lit = lit

        # Sample mid-row cells from a few frames for uniqueness check
        if i % 15 == 0:
            sampled_cell_values.extend(lit[mid_row, ::10].tolist())

    proc.stdin.close()
    proc.wait()

    # Assert no boolean on/off: sampled cell intensities must span ≥ 3 unique
    # float values (rounded to 2 decimals to tolerate float noise).
    unique_rounded = len(set(round(v, 2) for v in sampled_cell_values))
    intensity_unique_values_ok = unique_rounded >= 3

    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "pattern": pattern,
        "depth_source": depth_source,
        "lit_delta_max": lit_delta_max,
        "intensity_unique_values_ok": intensity_unique_values_ok,
    }

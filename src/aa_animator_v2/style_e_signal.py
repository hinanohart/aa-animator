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
"""Style E: signal-driven AA animation.

Character grid is built ONCE from the source image (static base layer).
Each frame only the glow mask changes — no PIL transform on the image.
Signal primitives drive which cells are lit per frame (CRT / LED panel VFX).

Signals
-------
jump   : horizontal band oscillates up/down
scan   : single scan line sweeps top→bottom, loops
wave   : radial wave expands from centre
pulse  : all non-space cells breathe together
combo  : jump + pulse superimposed
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants — match style_a_gallery.py Ghostty palette
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_STATIC_COLOR: tuple[int, int, int] = (215, 215, 215)
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

VALID_SIGNALS = ("jump", "scan", "wave", "pulse", "combo")


# ---------------------------------------------------------------------------
# Font loader
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (H, W, 3) float32 [0-255]."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Static base layer — built once, reused every frame
# ---------------------------------------------------------------------------

def render_static_base(
    image: Image.Image,
    cols: int = 100,
    rows: int = 41,
) -> tuple[list[list[str]], np.ndarray]:
    """Convert image to char grid and per-cell brightness (static, frame-invariant).

    Args:
        image: Source image (any mode; will be converted to RGB).
        cols: Grid width in cells.
        rows: Grid height in cells.

    Returns:
        char_grid: rows × cols list[list[str]], space for background cells.
        brightness_grid: float32 ndarray (rows, cols) in [0, 1], raw luma fraction.
    """
    rgb = image.convert("RGB")
    small = rgb.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)  # (rows, cols)

    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx = np.clip((lum / 255.0 * n).astype(int), 0, n)

    char_grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            # Background cells (very dark) stay as space
            ch = charset[idx[r, c]] if lum[r, c] >= 6 else " "
            row.append(ch)
        char_grid.append(row)

    brightness_grid = lum / 255.0  # normalise to [0, 1]
    return char_grid, brightness_grid


# ---------------------------------------------------------------------------
# Signal primitives — all return bool ndarray (rows, cols)
# ---------------------------------------------------------------------------

def signal_jump(t: float, rows: int, cols: int) -> np.ndarray:
    """Horizontal band oscillates vertically (2 Hz sine).

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        Boolean mask (rows, cols) — True where cells glow.
    """
    freq = 2.0
    band_half = 2.5  # half-height in cells
    y_center = (rows / 2.0) * (1.0 + 0.5 * math.sin(2.0 * math.pi * freq * t))
    ys = np.arange(rows, dtype=np.float32)
    mask_1d = np.abs(ys - y_center) < band_half
    return np.broadcast_to(mask_1d[:, None], (rows, cols)).copy()


def signal_scan(t: float, rows: int, cols: int) -> np.ndarray:
    """Scan line sweeps top→bottom, loops.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        Boolean mask (rows, cols).
    """
    scan_speed = 10.0  # cells per second
    y = (t * scan_speed) % rows
    ys = np.arange(rows, dtype=np.float32)
    # 2-cell wide band for visibility
    mask_1d = np.abs(ys - y) < 1.0
    return np.broadcast_to(mask_1d[:, None], (rows, cols)).copy()


def signal_wave(t: float, rows: int, cols: int) -> np.ndarray:
    """Radial wave expanding from image centre.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        Boolean mask (rows, cols).
    """
    wavelength = 5.0
    speed = 3.0
    threshold = 0.0

    cx, cy = cols / 2.0, rows / 2.0
    xs = np.arange(cols, dtype=np.float32)
    ys = np.arange(rows, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # (rows, cols)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phase = r / wavelength - t * speed
    return np.sin(2.0 * math.pi * phase) > threshold


def signal_pulse(t: float, rows: int, cols: int) -> np.ndarray:
    """All non-space cells breathe together (global alpha; returned as full-True mask).

    The actual alpha is embedded in the mask value here — callers use
    signal_pulse_alpha() for the float intensity and this for the cell selector.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        All-True boolean mask — alpha applied separately via pulse_alpha().
    """
    return np.ones((rows, cols), dtype=bool)


def pulse_alpha(t: float) -> float:
    """Return global glow intensity for pulse signal in [0, 1].

    Args:
        t: Time in seconds.

    Returns:
        Float intensity 0.0–1.0.
    """
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * 1.5 * t)


def signal_combo(t: float, rows: int, cols: int) -> np.ndarray:
    """jump + pulse: oscillating band mask with breathing intensity.

    Uses jump mask geometry (cells within band glow) while the caller
    applies pulse_alpha() to modulate brightness — the mask itself varies
    with the jump oscillation so frame-to-frame difference is detectable.

    Args:
        t: Time in seconds.
        rows: Grid height.
        cols: Grid width.

    Returns:
        Boolean mask (rows, cols) — jump band, alpha driven by pulse_alpha().
    """
    return signal_jump(t, rows, cols)


# ---------------------------------------------------------------------------
# Frame renderer — char_grid is static, glow_mask changes each frame
# ---------------------------------------------------------------------------

def render_frame(
    char_grid: list[list[str]],
    glow_mask: np.ndarray,
    brightness_grid: np.ndarray,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    glow_alpha: float = 1.0,
) -> Image.Image:
    """Render one frame from the static char grid and a dynamic glow mask.

    Args:
        char_grid: rows × cols str grid (space = background cell).
        glow_mask: bool ndarray (rows, cols) — glowing cells.
        brightness_grid: float ndarray (rows, cols) in [0, 1].
        font: Pre-loaded monospace font.
        glow_alpha: Overall glow intensity multiplier in [0, 1].

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
                continue  # background cell — never draw/glow
            if glow_mask[r, c]:
                # Blend glow_alpha towards pure glow colour
                alpha = glow_alpha
                color = (
                    int(_STATIC_COLOR[0] * (1.0 - alpha) + _GLOW_COLOR[0] * alpha),
                    int(_STATIC_COLOR[1] * (1.0 - alpha) + _GLOW_COLOR[1] * alpha),
                    int(_STATIC_COLOR[2] * (1.0 - alpha) + _GLOW_COLOR[2] * alpha),
                )
            else:
                color = _STATIC_COLOR
            d.text((c * _CELL_W, r * _CELL_H), ch, fill=color, font=font)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_style_e(
    input_path: str | Path,
    output_path: str | Path,
    *,
    signal: str = "jump",
    cols: int = 100,
    rows: int = 41,
    fps: int = 30,
    duration: float = 4.0,
) -> dict:
    """Generate Style E signal-driven animation MP4.

    Character grid is built once; only the glow mask changes per frame.

    Args:
        input_path: Source image path.
        output_path: Destination .mp4 path.
        signal: One of "jump", "scan", "wave", "pulse", "combo".
        cols: Cell grid width.
        rows: Cell grid height.
        fps: Output frame rate.
        duration: Clip length in seconds.

    Returns:
        Dict with keys: output_path, n_frames, canvas_w, canvas_h,
        signal, char_grid_invariant (True — assertions passed).
    """
    if signal not in VALID_SIGNALS:
        raise ValueError(f"signal must be one of {VALID_SIGNALS}, got {signal!r}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build static base (once)
    img = Image.open(input_path)
    char_grid, brightness_grid = render_static_base(img, cols=cols, rows=rows)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)
    canvas_w = cols * _CELL_W
    canvas_h = rows * _CELL_H

    # Select signal function
    _signal_fns = {
        "jump": signal_jump,
        "scan": signal_scan,
        "wave": signal_wave,
        "pulse": signal_pulse,
        "combo": signal_combo,
    }
    sig_fn = _signal_fns[signal]

    # Validate char_grid invariance: build mask for first and last frame,
    # assert char_grid itself is unchanged
    import copy
    char_grid_copy = copy.deepcopy(char_grid)

    # Generate frames and stream to ffmpeg
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
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    prev_mask: np.ndarray | None = None
    mask_varied = False

    for i in range(n_frames):
        t = i / fps  # absolute time in seconds

        glow_mask = sig_fn(t, rows, cols)
        alpha = pulse_alpha(t) if signal in ("pulse", "combo") else 1.0

        frame = render_frame(
            char_grid,
            glow_mask,
            brightness_grid,
            font,
            glow_alpha=alpha,
        )
        proc.stdin.write(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes())

        if prev_mask is not None and not np.array_equal(glow_mask, prev_mask):
            mask_varied = True
        prev_mask = glow_mask

    proc.stdin.close()
    proc.wait()

    # Assert char_grid was never mutated
    assert char_grid == char_grid_copy, "char_grid was mutated between frames — invariant violated"
    assert mask_varied or n_frames <= 1, "glow_mask never varied across frames — signal not working"

    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "signal": signal,
        "char_grid_invariant": True,
    }

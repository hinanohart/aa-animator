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
"""Shared boo-postprocess helpers for style_i (long_boo) and style_j (slime_boo).

Two transforms applied after DensityAA rendering:

1. apply_outline_ring  — Sobel edge detection on the source frame identifies
   silhouette-boundary cells; those cells are forced to '@' in the rendered
   output and brightened to (240, 240, 255) so the outline ring is visible
   regardless of source colour.

2. apply_blue_glow     — Cells whose rendered brightness maps to a *light*
   glyph (' ·~ox') receive a blue (80, 140, 255) colour tint. This mimics
   the Ghostty boo background-halo effect without copying any frame data.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Palette / tunables
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"

# Glyphs considered "light" — blue tint target
_LIGHT_GLYPHS: frozenset[str] = frozenset(" ·~ox")

# Blue tint colour (Ghostty halo blue)
_BLUE_TINT: tuple[int, int, int] = (80, 140, 255)
_BLUE_TINT_ALPHA: float = 0.45  # blend strength for blue glow

# Outline ring colour and edge detection threshold
_OUTLINE_COLOR: tuple[int, int, int] = (240, 240, 255)
_EDGE_THRESHOLD: float = 0.18   # Sobel magnitude threshold for edge cells

# Cell geometry — must match DensityAA geometry in style_i / style_j
CELL_W: int = 7
CELL_H: int = 14
FONT_RATIO: float = 0.50


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (H, W, 3) float32 [0–255]."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Public: apply_outline_ring
# ---------------------------------------------------------------------------

def apply_outline_ring(
    rendered: Image.Image,
    frame_img: Image.Image,
    cols: int,
    font,
    *,
    threshold: float = _EDGE_THRESHOLD,
) -> Image.Image:
    """Force silhouette-boundary cells to '@' with outline colour.

    Detects edge cells via Sobel magnitude on the source frame, then
    overwrites those cells in the rendered AA image with '@' drawn in
    _OUTLINE_COLOR.  The '@' glyph is the densest Ghostty char, giving a
    crisp, Ghostty-boo-style outline ring.

    Args:
        rendered: DensityAA-rendered RGB PIL Image.
        frame_img: Source RGB frame (canvas square) used for edge detection.
        cols: Cell grid width (same as used when rendering).
        font: Pre-loaded monospace font (same as used when rendering).
        threshold: Normalised Sobel magnitude threshold for edge cells.

    Returns:
        Modified RGB PIL Image with '@' outline ring applied.
    """
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * FONT_RATIO))

    gray = np.array(frame_img.convert("L").resize((cols, rows), Image.LANCZOS)).astype(np.float32)
    sx = scipy_sobel(gray, axis=1)
    sy = scipy_sobel(gray, axis=0)
    mag = np.hypot(sx, sy)
    if mag.max() > 0:
        mag /= mag.max()
    edge_mask = mag > threshold  # (rows, cols) bool

    if not edge_mask.any():
        return rendered

    out = rendered.copy()
    d = ImageDraw.Draw(out)
    for ry in range(rows):
        for cx in range(cols):
            if not edge_mask[ry, cx]:
                continue
            x0, y0 = cx * CELL_W, ry * CELL_H
            # Blank the cell background first so the @ stands out cleanly
            d.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], fill=(8, 8, 12))
            d.text((x0, y0), "@", fill=_OUTLINE_COLOR, font=font)

    return out


# ---------------------------------------------------------------------------
# Public: apply_blue_glow
# ---------------------------------------------------------------------------

def apply_blue_glow(
    rendered: Image.Image,
    frame_img: Image.Image,
    cols: int,
) -> Image.Image:
    """Tint light-glyph cells with Ghostty halo blue.

    Determines per-cell glyph by mapping cell luma → Ghostty char index.
    Cells mapping to a *light* glyph (' ·~ox') receive a blue
    (80, 140, 255) tint blended at _BLUE_TINT_ALPHA.  This creates the
    characteristic Ghostty boo aura around the dark subject.

    Args:
        rendered: DensityAA-rendered RGB PIL Image (after outline_ring).
        frame_img: Source RGB frame used to determine cell luma.
        cols: Cell grid width.

    Returns:
        Blue-glow-tinted RGB PIL Image.
    """
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * FONT_RATIO))

    small = np.array(frame_img.resize((cols, rows), Image.LANCZOS)).astype(np.float32)
    lum = _srgb_luma(small)

    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx_grid = np.clip((lum / 255.0 * n).astype(int), 0, n)

    canvas_arr = np.array(rendered, dtype=np.float32)
    blue = np.array(_BLUE_TINT, dtype=np.float32)
    alpha = _BLUE_TINT_ALPHA

    for ry in range(rows):
        for cx in range(cols):
            ch = charset[idx_grid[ry, cx]]
            if ch not in _LIGHT_GLYPHS:
                continue
            y0, y1 = ry * CELL_H, (ry + 1) * CELL_H
            x0, x1 = cx * CELL_W, (cx + 1) * CELL_W
            region = canvas_arr[y0:y1, x0:x1]
            region[:] = region * (1.0 - alpha) + blue * alpha

    return Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))

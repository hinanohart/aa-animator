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
"""AA rendering engine: converts pixel arrays to ASCII / Braille frame images.

Density ramp and color-mode logic ported from PoC (aa_animator.py,
aa_ghostty_v3.py, poc_v4.py) — original author: Hinano Hart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aa_animator_v2.dither import apply_bayer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ghostty 11-level density ramp (space → @)
AA_CHARS: str = " ·~ox+=*%$@"
NCHARS: int = len(AA_CHARS)

# Braille Unicode base
BRAILLE_BASE: int = 0x2800

# Dot order for sequential fill (top-left → bottom-right, zigzag)
# Braille bit layout: col0 → bits 0,1,2,6  col1 → bits 3,4,5,7
_BRAILLE_DOT_ORDER: list[int] = [0, 3, 1, 4, 2, 5, 6, 7]

# Ghostty-style colours
COLOR_BODY: tuple[int, int, int] = (220, 220, 220)
COLOR_EDGE: tuple[int, int, int] = (70, 130, 255)
GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
GLOW_ALPHA: float = 0.30
EDGE_THRESH: float = 0.15

ColorMode = Literal["color", "mono", "matrix", "cyber", "amber", "gradient", "invert"]
RenderMode = Literal["ascii", "braille"]
DitherMode = Literal["none", "bayer"]

_FONT_PATHS: list[str] = [
    # Linux (DejaVu)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    # macOS system fonts
    "/System/Library/Fonts/Monaco.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (…, 3) float32 array."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _apply_color_mode(rgb: np.ndarray, mode: ColorMode) -> np.ndarray:
    """Apply colour remapping to an (H, W, 3) float32 array in [0, 255]."""
    arr = np.asarray(rgb, dtype=np.float32)
    if mode == "color":
        return arr
    lum = _srgb_luma(arr)
    if mode == "mono":
        return np.stack([lum, lum, lum], axis=-1)
    if mode == "matrix":
        return np.stack([lum * 0.15, lum, lum * 0.3], axis=-1)
    if mode == "cyber":
        l01 = lum / 255.0
        r = 255 * (l01 ** 1.4) * 0.9
        g = 255 * l01 * 0.4
        b = 255 * (l01 ** 0.7)
        return np.stack([r, g, b], axis=-1)
    if mode == "amber":
        l01 = lum / 255.0
        return np.stack([255 * l01, 191 * l01, np.zeros_like(l01)], axis=-1)
    if mode == "gradient":
        l01 = lum / 255.0
        r = 50 + 200 * l01
        g = 200 - 100 * l01
        b = 255 - 50 * l01
        return np.stack([r, g, b], axis=-1)
    if mode == "invert":
        return 255.0 - arr
    return arr


# ---------------------------------------------------------------------------
# Braille utilities
# ---------------------------------------------------------------------------

def brightness_to_braille(brightness_01: float) -> str:
    """Map a 0-1 brightness value to a Braille character (U+2800-28FF).

    Uses 8-dot sequential fill in the canonical dot order so that brighter
    cells get more dots lit.

    Args:
        brightness_01: Luminance in [0.0, 1.0].

    Returns:
        A single Braille Unicode character.
    """
    n_dots = int(round(brightness_01 * 8))
    bits = 0
    for i in range(n_dots):
        bits |= 1 << _BRAILLE_DOT_ORDER[i]
    return chr(BRAILLE_BASE + bits)


def _build_braille_bitmaps(cell_w: int, cell_h: int, font_size: int) -> np.ndarray:
    """Pre-render all 256 Braille patterns to a uint8 numpy array.

    Returns:
        Array of shape (256, cell_h, cell_w, 3) uint8.
    """
    font = _load_font(font_size)
    bitmaps = np.zeros((256, cell_h, cell_w, 3), dtype=np.uint8)
    for i in range(256):
        cell = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
        ImageDraw.Draw(cell).text((0, 0), chr(BRAILLE_BASE + i), font=font, fill=COLOR_BODY)
        bitmaps[i] = np.array(cell, dtype=np.uint8)
    return bitmaps


def _brightness_to_braille_bits_vectorised(cell_brightness: np.ndarray) -> np.ndarray:
    """Vectorised conversion: (ROWS, COLS) float32 [0-1] → int32 bit patterns.

    Args:
        cell_brightness: Array of shape (ROWS, COLS) with values in [0, 1].

    Returns:
        Integer array of shape (ROWS, COLS) with values in [0, 255].
    """
    n_dots_arr = np.clip((cell_brightness * 8).round().astype(np.int32), 0, 8)
    bits_arr = np.zeros_like(n_dots_arr, dtype=np.int32)
    for i, d in enumerate(_BRAILLE_DOT_ORDER):
        bits_arr[n_dots_arr > i] |= 1 << d
    return bits_arr.clip(0, 255)


# ---------------------------------------------------------------------------
# FrameRenderer
# ---------------------------------------------------------------------------

class FrameRenderer:
    """Converts cell-level data to PIL Image frames.

    Args:
        mode: ``"ascii"`` or ``"braille"``.
        cell_w: Pixel width of one character cell.
        cell_h: Pixel height of one character cell.
        font_size: Font size passed to ImageFont.truetype.
        glow: Whether to apply the blue edge-glow effect.
    """

    def __init__(
        self,
        mode: RenderMode = "braille",
        cell_w: int = 4,
        cell_h: int = 8,
        font_size: int = 10,
        *,
        glow: bool = True,
        dither: DitherMode = "none",
    ) -> None:
        self.mode = mode
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.font_size = font_size
        self.glow = glow
        self.dither = dither

        self._font = _load_font(font_size)
        self._ascii_bitmaps = self._build_ascii_bitmaps()  # (NCHARS, 2, cell_h, cell_w, 3)
        self._braille_bitmaps: np.ndarray | None = None
        if mode == "braille":
            self._braille_bitmaps = _build_braille_bitmaps(cell_w, cell_h, font_size)

    # ------------------------------------------------------------------ build

    def _build_ascii_bitmaps(self) -> np.ndarray:
        """Pre-render ASCII chars in body and edge colours.

        Returns shape (NCHARS, 2, cell_h, cell_w, 3).
        Index [ci, 0] = body colour, [ci, 1] = edge colour.
        """
        bitmaps = np.zeros((NCHARS, 2, self.cell_h, self.cell_w, 3), dtype=np.uint8)
        colours = [COLOR_BODY, COLOR_EDGE]
        for ci, ch in enumerate(AA_CHARS):
            for ki, col in enumerate(colours):
                cell = Image.new("RGB", (self.cell_w, self.cell_h), (0, 0, 0))
                ImageDraw.Draw(cell).text((0, 0), ch, font=self._font, fill=col)
                bitmaps[ci, ki] = np.array(cell, dtype=np.uint8)
        return bitmaps

    # ------------------------------------------------------------------ public

    def render_frame(
        self,
        cell_brightness: np.ndarray,
        edge_cell: np.ndarray,
        mask_cell: np.ndarray | None = None,
        bg: Literal["black", "ghostty_fill"] = "black",
    ) -> Image.Image:
        """Render one frame image from cell arrays.

        Args:
            cell_brightness: (ROWS, COLS) float32 in [0, 1].
            edge_cell: (ROWS, COLS) bool — True where edge glow applies.
            mask_cell: (ROWS, COLS) bool — True for foreground pixels.
                       ``None`` means full canvas (no masking).
            bg: Background strategy.

        Returns:
            PIL RGBA / RGB Image of size (COLS * cell_w, ROWS * cell_h).
        """
        rows, cols = cell_brightness.shape
        canvas_w = cols * self.cell_w
        canvas_h = rows * self.cell_h

        if mask_cell is None:
            mask_cell = np.ones((rows, cols), dtype=bool)

        # Apply ordered dither before quantisation (operates on cell-level brightness)
        if self.dither == "bayer":
            cell_brightness = apply_bayer(cell_brightness)

        if self.mode == "braille":
            canvas_u8 = self._render_braille(cell_brightness, rows, cols, canvas_h, canvas_w)
        else:
            canvas_u8 = self._render_ascii(cell_brightness, edge_cell, rows, cols, canvas_h, canvas_w)

        # Background masking
        mask_px = np.repeat(np.repeat(mask_cell, self.cell_h, axis=0), self.cell_w, axis=1)
        if bg == "black":
            canvas_u8 *= mask_px[:, :, np.newaxis].astype(np.uint8)
        # ghostty_fill: foreground rendered, background left as rendered
        # (background cells just happen to be empty/black in braille mode — acceptable for Day 1)

        # Blue glow alpha-blend on edge neighbourhood
        if self.glow:
            from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]
            glow_cell = binary_dilation(edge_cell, structure=np.ones((3, 3), dtype=bool)) & ~edge_cell
            glow_active = glow_cell & mask_cell
            if glow_active.any():
                glow_px = np.repeat(np.repeat(glow_active, self.cell_h, axis=0), self.cell_w, axis=1)
                region = canvas_u8[glow_px].astype(np.float32)
                region = region * (1.0 - GLOW_ALPHA) + np.array(GLOW_COLOR, dtype=np.float32) * GLOW_ALPHA
                canvas_u8[glow_px] = np.clip(region, 0, 255).astype(np.uint8)

        return Image.fromarray(canvas_u8)

    # ------------------------------------------------------------------ private

    def _render_braille(
        self,
        cell_brightness: np.ndarray,
        rows: int,
        cols: int,
        canvas_h: int,
        canvas_w: int,
    ) -> np.ndarray:
        assert self._braille_bitmaps is not None
        bits_arr = _brightness_to_braille_bits_vectorised(cell_brightness)  # (ROWS, COLS)
        selected = self._braille_bitmaps[bits_arr]  # (ROWS, COLS, cell_h, cell_w, 3)
        return np.ascontiguousarray(
            selected.transpose(0, 2, 1, 3, 4)
        ).reshape(canvas_h, canvas_w, 3)

    def _render_ascii(
        self,
        cell_brightness: np.ndarray,
        edge_cell: np.ndarray,
        rows: int,
        cols: int,
        canvas_h: int,
        canvas_w: int,
    ) -> np.ndarray:
        char_idx = np.clip((cell_brightness * (NCHARS - 1)).astype(np.int32), 0, NCHARS - 1)
        color_type = edge_cell.astype(np.int32)  # 0 = body, 1 = edge
        selected = self._ascii_bitmaps[char_idx, color_type]  # (ROWS, COLS, cell_h, cell_w, 3)
        return np.ascontiguousarray(
            selected.transpose(0, 2, 1, 3, 4)
        ).reshape(canvas_h, canvas_w, 3)

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
"""Unit tests for _boo_postprocess shared helpers.

Key invariants verified:
1. apply_outline_ring: edge cells are replaced with '@' (outline colour).
2. apply_outline_ring: preserves non-edge regions (no full-canvas wipe).
3. apply_outline_ring: returns RGB PIL Image of same size.
4. apply_blue_glow: light-glyph cells receive blue tint (blue channel dominates).
5. apply_blue_glow: dark-glyph (heavy) cells are NOT blue-tinted.
6. apply_blue_glow: returns RGB PIL Image of same size.
7. Both functions are idempotent on background-only input (no crash).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from aa_animator_v2._boo_postprocess import (
    CELL_H,
    CELL_W,
    FONT_RATIO,
    apply_blue_glow,
    apply_outline_ring,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def font():
    return ImageFont.load_default()


@pytest.fixture()
def tiny_frame_rgb() -> Image.Image:
    """40×40 RGB image with a bright subject on dark background — creates edges."""
    img = Image.new("RGB", (40, 40), (8, 8, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 30, 30], fill=(220, 220, 220))
    return img


@pytest.fixture()
def dark_frame_rgb() -> Image.Image:
    """40×40 near-black RGB image — no edges."""
    return Image.new("RGB", (40, 40), (5, 5, 5))


@pytest.fixture()
def cols() -> int:
    return 10


@pytest.fixture()
def rendered_gray(cols) -> Image.Image:
    """Synthetic rendered AA frame: uniform mid-grey (maps to mid-density glyph)."""
    rows = max(1, int(40 * cols / 40 * FONT_RATIO))
    return Image.new("RGB", (cols * CELL_W, rows * CELL_H), (100, 100, 100))


@pytest.fixture()
def rendered_dark(cols) -> Image.Image:
    """Near-black rendered frame (maps to space / light glyphs)."""
    rows = max(1, int(40 * cols / 40 * FONT_RATIO))
    return Image.new("RGB", (cols * CELL_W, rows * CELL_H), (8, 8, 12))


# ---------------------------------------------------------------------------
# apply_outline_ring
# ---------------------------------------------------------------------------

class TestApplyOutlineRing:
    def test_output_same_size(self, rendered_gray, tiny_frame_rgb, cols, font):
        out = apply_outline_ring(rendered_gray, tiny_frame_rgb, cols, font)
        assert out.size == rendered_gray.size

    def test_output_mode_rgb(self, rendered_gray, tiny_frame_rgb, cols, font):
        out = apply_outline_ring(rendered_gray, tiny_frame_rgb, cols, font)
        assert out.mode == "RGB"

    def test_edge_cells_have_outline_colour(self, rendered_gray, tiny_frame_rgb, cols, font):
        """After outline ring, at least some pixels should match _OUTLINE_COLOR range."""
        out = apply_outline_ring(rendered_gray, tiny_frame_rgb, cols, font)
        arr = np.array(out)
        # _OUTLINE_COLOR = (240, 240, 255): look for bright bluish-white pixels
        bright_mask = (arr[:, :, 0] > 180) & (arr[:, :, 2] > 200)
        assert bright_mask.any(), "No outline-ring-coloured pixels found after apply_outline_ring"

    def test_no_crash_on_dark_frame(self, rendered_gray, dark_frame_rgb, cols, font):
        """Dark frame (no edges) must not crash — returns rendered unchanged or near-unchanged."""
        out = apply_outline_ring(rendered_gray, dark_frame_rgb, cols, font)
        assert out.size == rendered_gray.size
        assert out.mode == "RGB"

    def test_non_edge_region_preserved(self, tiny_frame_rgb, cols, font):
        """Background-only rendered frame stays mostly dark after outline ring."""
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        rendered_dark_local = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (8, 8, 12))
        out = apply_outline_ring(rendered_dark_local, tiny_frame_rgb, cols, font)
        arr = np.array(out)
        # Most pixels should still be dark (background), even with edge overlay
        dark_frac = (arr.mean(axis=2) < 50).mean()
        assert dark_frac > 0.5, f"Too many bright pixels after outline ring on dark rendered: {dark_frac:.2%}"


# ---------------------------------------------------------------------------
# apply_blue_glow
# ---------------------------------------------------------------------------

class TestApplyBlueGlow:
    def test_output_same_size(self, rendered_gray, tiny_frame_rgb, cols):
        out = apply_blue_glow(rendered_gray, tiny_frame_rgb, cols)
        assert out.size == rendered_gray.size

    def test_output_mode_rgb(self, rendered_gray, tiny_frame_rgb, cols):
        out = apply_blue_glow(rendered_gray, tiny_frame_rgb, cols)
        assert out.mode == "RGB"

    def test_light_glyph_cells_receive_blue_tint(self, dark_frame_rgb, cols):
        """Dark source → cells map to space/' '/'·' (light glyphs) → blue tint expected."""
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        # Rendered mid-grey (will change toward blue after glow)
        rendered_mid = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (80, 80, 80))
        # Very dark source → all cells map to light glyphs
        very_dark = Image.new("RGB", (40, 40), (10, 10, 10))
        out = apply_blue_glow(rendered_mid, very_dark, cols)
        arr = np.array(out).astype(float)
        # After blue tint blend, blue channel should exceed red channel
        blue_dominance = (arr[:, :, 2] - arr[:, :, 0]).mean()
        assert blue_dominance > 5.0, (
            f"Blue tint not applied to light-glyph cells: blue-red mean={blue_dominance:.2f}"
        )

    def test_heavy_glyph_cells_not_blue_tinted(self, cols):
        """Bright source → cells map to '@'/'$' (heavy glyphs) → no blue tint."""
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        # Bright white source → all cells map to heavy glyphs
        bright_src = Image.new("RGB", (40, 40), (240, 230, 200))
        rendered_white = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (200, 200, 200))
        out = apply_blue_glow(rendered_white, bright_src, cols)
        arr = np.array(out).astype(float)
        # Heavy glyphs should NOT receive blue tint: blue channel ≈ red channel
        blue_dominance = (arr[:, :, 2] - arr[:, :, 0]).mean()
        assert blue_dominance < 20.0, (
            f"Blue tint should not dominate heavy-glyph cells: blue-red mean={blue_dominance:.2f}"
        )

    def test_no_crash_on_background_only(self, dark_frame_rgb, cols):
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        bg_rendered = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (8, 8, 12))
        out = apply_blue_glow(bg_rendered, dark_frame_rgb, cols)
        assert out.mode == "RGB"
        assert out.size == bg_rendered.size

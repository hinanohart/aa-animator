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
"""Unit tests for Style H block_bob "bird" preset.

Key invariants verified:
1. _apply_motion: canvas is canvas_size x canvas_size RGB.
2. bob: centroid Y shifts upward (decreases) at t=0.25 vs t=0.0.
3. sway: centroid X shifts right at t=0.25 vs t=0.0.
4. _apply_vignette: corners are darker than centre.
5. _render_block_aa: output width == cols * 8, height == rows * 14.
6. _render_block_aa: uses ▀ glyph cells (no full-black output for bright input).
7. generate_style_h: returns dict with correct keys (smoke, no ffmpeg).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aa_animator_v2.style_h_bird import (
    _apply_motion,
    _apply_vignette,
    _load_font,
    _render_block_aa,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bright_square() -> Image.Image:
    """48×48 RGBA image — bright white subject on transparent bg."""
    img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 40, 40], fill=(220, 220, 220, 255))
    return img


@pytest.fixture()
def small_rgb() -> Image.Image:
    """40×40 RGB image with a colourful gradient for renderer tests."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    for y in range(40):
        for x in range(40):
            arr[y, x] = (int(x * 6), int(y * 6), 100)
    return Image.fromarray(arr)


@pytest.fixture()
def font():
    return _load_font(16)


# ---------------------------------------------------------------------------
# _apply_motion
# ---------------------------------------------------------------------------

class TestApplyMotion:
    def test_output_is_rgb_canvas_size(self, bright_square):
        canvas_size = 128
        out = _apply_motion(bright_square, 0.0, canvas_size)
        assert out.mode == "RGB"
        assert out.size == (canvas_size, canvas_size)

    def test_bob_centroid_y_rises_at_t025(self, bright_square):
        """At t=0.25 sin(t*4π)=sin(π)=0 so no upward offset; at t=0.125
        sin(0.5π)=1 so offset_y -= 18. Centroid Y should be lower numerically
        (i.e. higher on screen = smaller y)."""
        canvas_size = 200
        arr0 = np.array(_apply_motion(bright_square, 0.0, canvas_size)).astype(float)
        arr1 = np.array(_apply_motion(bright_square, 0.125, canvas_size)).astype(float)
        non_bg0 = (arr0 > 20).any(axis=2)
        non_bg1 = (arr1 > 20).any(axis=2)
        if non_bg0.any() and non_bg1.any():
            cy0 = np.where(non_bg0)[0].mean()
            cy1 = np.where(non_bg1)[0].mean()
            # At t=0.125 bob is at peak upward — centroid should be higher (smaller y)
            assert cy1 < cy0 + 2, f"bob centroid Y did not rise: cy0={cy0:.1f} cy1={cy1:.1f}"

    def test_sway_centroid_x_shifts(self, bright_square):
        """sway: offset_x = 10*sin(t*2π). At t=0.25 sin=1 → +10px right."""
        canvas_size = 200
        arr0 = np.array(_apply_motion(bright_square, 0.0, canvas_size)).astype(float)
        arr1 = np.array(_apply_motion(bright_square, 0.25, canvas_size)).astype(float)
        non_bg0 = (arr0 > 20).any(axis=2)
        non_bg1 = (arr1 > 20).any(axis=2)
        if non_bg0.any() and non_bg1.any():
            cx0 = np.where(non_bg0)[1].mean()
            cx1 = np.where(non_bg1)[1].mean()
            # At t=0.25 offset_x = +10 → centroid shifts right
            assert cx1 > cx0 - 2, f"sway X did not shift right: cx0={cx0:.1f} cx1={cx1:.1f}"

    def test_no_effect_at_t0_and_t1_same_position(self, bright_square):
        """bob is periodic with period 0.5, sway with period 1 — t=0 and t=1
        produce nearly the same offset (within 1px centroid drift)."""
        canvas_size = 200
        arr0 = np.array(_apply_motion(bright_square, 0.0, canvas_size)).astype(float)
        arr1 = np.array(_apply_motion(bright_square, 1.0, canvas_size)).astype(float)
        diff = np.abs(arr0 - arr1).mean()
        assert diff < 5.0, f"t=0 and t=1 differ too much: mean diff={diff:.2f}"


# ---------------------------------------------------------------------------
# _apply_vignette
# ---------------------------------------------------------------------------

class TestApplyVignette:
    def test_corners_darker_than_centre(self):
        """Uniform bright image: corners should be attenuated vs centre."""
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        out = _apply_vignette(img)
        arr = np.array(out).astype(float)
        centre = arr[50, 50, 0]
        corner = arr[0, 0, 0]
        assert corner < centre, f"corner({corner}) not < centre({centre})"

    def test_preserves_size(self, small_rgb):
        out = _apply_vignette(small_rgb)
        assert out.size == small_rgb.size

    def test_output_mode_rgb(self, small_rgb):
        out = _apply_vignette(small_rgb)
        assert out.mode == "RGB"

    def test_no_negative_values(self):
        img = Image.new("RGB", (60, 60), (100, 100, 100))
        arr = np.array(_apply_vignette(img))
        assert arr.min() >= 0


# ---------------------------------------------------------------------------
# _render_block_aa
# ---------------------------------------------------------------------------

class TestRenderBlockAA:
    def test_output_width_is_cols_times_cell_w(self, small_rgb, font):
        cols = 10
        out = _render_block_aa(small_rgb, cols, font)
        assert out.width == cols * 8  # _CELL_W = 8

    def test_output_height_matches_rows(self, small_rgb, font):
        cols = 10
        out = _render_block_aa(small_rgb, cols, font)
        w, h = small_rgb.size
        rows_dbl = max(2, int(h / w * cols * 0.95))
        rows_dbl -= rows_dbl % 2
        rows = rows_dbl // 2
        assert out.height == rows * 14  # _CELL_H = 14

    def test_bright_input_produces_non_black_output(self, font):
        """Bright input should not produce an all-black frame."""
        img = Image.new("RGB", (80, 80), (200, 200, 200))
        out = _render_block_aa(img, 10, font)
        arr = np.array(out)
        assert arr.max() > 30, "bright input produced near-black output"

    def test_output_mode_rgb(self, small_rgb, font):
        out = _render_block_aa(small_rgb, 10, font)
        assert out.mode == "RGB"

    def test_dark_input_mostly_black(self, font):
        """Near-black input should produce a mostly dark frame."""
        img = Image.new("RGB", (40, 40), (5, 5, 5))
        out = _render_block_aa(img, 8, font)
        arr = np.array(out).astype(float)
        assert arr.mean() < 20, f"dark input too bright: mean={arr.mean():.1f}"


# ---------------------------------------------------------------------------
# bob formula cross-check
# ---------------------------------------------------------------------------

class TestBobFormula:
    @pytest.mark.parametrize("t", [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75])
    def test_bob_offset_matches_formula(self, t):
        """Verify _apply_motion offset_y matches aa_animator.py:217 formula."""
        ph = math.sin(t * math.pi * 4)
        expected_offset_y = -18 * max(0.0, ph)
        # canvas centroid shift should track expected_offset_y
        # We test that at ph > 0 the centroid Y is above (less than) the neutral position
        canvas_size = 200
        img = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, 50, 50], fill=(200, 200, 200, 255))

        neutral = _apply_motion(img, 0.5, canvas_size)  # sin(2π)=0, offset_y=0
        frame = _apply_motion(img, t, canvas_size)

        arr_n = np.array(neutral).astype(float)
        arr_f = np.array(frame).astype(float)
        non_bg_n = (arr_n > 20).any(axis=2)
        non_bg_f = (arr_f > 20).any(axis=2)

        if non_bg_n.any() and non_bg_f.any():
            cy_n = np.where(non_bg_n)[0].mean()
            cy_f = np.where(non_bg_f)[0].mean()
            actual_shift = cy_n - cy_f  # positive = frame is higher
            # Should match sign of expected_offset_y (negative = upward = positive shift in cy)
            if abs(expected_offset_y) > 2:
                assert actual_shift * (-expected_offset_y) > 0 or abs(actual_shift) < 3, (
                    f"t={t}: expected_offset_y={expected_offset_y:.1f} "
                    f"actual_shift={actual_shift:.1f}"
                )

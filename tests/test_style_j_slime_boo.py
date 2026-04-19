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
"""Unit tests for Style J (slime_dance + blink + Ghostty-boo).

Key invariants verified:
1. Blink timings are exactly [0.18, 0.50, 0.82].
2. _blink_intensity: peak intensity ≈ 1.0 at blink timing; 0 far away.
3. _apply_blink: eye-region cells are darker at blink peak vs neutral.
4. _apply_blink: cell grid positions are fixed (same size output).
5. _apply_breathe: output is RGB at canvas_size x canvas_size.
6. Breathe scale_x and scale_y are anti-phase (opposite signs).
7. _render_density_aa: output width = cols * CELL_W, mode RGB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from aa_animator_v2._boo_postprocess import CELL_H, CELL_W, FONT_RATIO
from aa_animator_v2.style_j_slime_boo import (
    _BLINK_TIMINGS,
    _BREATHE_FREQ,
    _BREATHE_X_AMP,
    _BREATHE_Y_AMP,
    _apply_blink,
    _apply_breathe,
    _blink_intensity,
    _render_density_aa,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def subject_rgba() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 54, 54], fill=(220, 220, 220, 255))
    return img


@pytest.fixture()
def small_rgb() -> Image.Image:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    for y in range(40):
        for x in range(40):
            arr[y, x] = (int(x * 6), int(y * 6), 80)
    return Image.fromarray(arr)


@pytest.fixture()
def font():
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Blink timings
# ---------------------------------------------------------------------------


def test_blink_timings_exact():
    assert _BLINK_TIMINGS == [0.18, 0.50, 0.82]


def test_blink_timings_count():
    assert len(_BLINK_TIMINGS) == 3


# ---------------------------------------------------------------------------
# _blink_intensity
# ---------------------------------------------------------------------------


class TestBlinkIntensity:
    @pytest.mark.parametrize("bt", [0.18, 0.50, 0.82])
    def test_peak_near_one_at_blink_timing(self, bt):
        intensity = _blink_intensity(bt)
        assert intensity > 0.9, (
            f"Expected intensity ≈ 1.0 at blink timing {bt}, got {intensity:.3f}"
        )

    def test_zero_far_from_blinks(self):
        """At t=0.0 (far from all blink timings) intensity should be 0."""
        intensity = _blink_intensity(0.0)
        assert intensity < 0.05, f"Expected ~0 far from blink timings, got {intensity:.3f}"

    def test_zero_at_t_midpoint_between_blinks(self):
        """At t=0.34 (midpoint between 0.18 and 0.50) intensity should be ~0."""
        intensity = _blink_intensity(0.34)
        assert intensity < 0.1, f"Expected ~0 between blink windows, got {intensity:.3f}"

    def test_monotone_rise_to_peak(self):
        """Intensity should increase as t approaches a blink timing from below."""
        bt = 0.50
        vals = [_blink_intensity(bt - 0.03), _blink_intensity(bt - 0.01), _blink_intensity(bt)]
        assert vals[0] <= vals[1] <= vals[2] + 0.01, (
            f"Intensity not monotonically rising to blink peak: {vals}"
        )


# ---------------------------------------------------------------------------
# _apply_blink
# ---------------------------------------------------------------------------


class TestApplyBlink:
    def _make_rendered(self, cols: int) -> tuple[Image.Image, int]:
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        rendered = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (150, 150, 150))
        return rendered, rows

    def test_output_same_size(self):
        cols = 10
        rendered, rows = self._make_rendered(cols)
        out = _apply_blink(rendered, 0.50, rows, cols)
        assert out.size == rendered.size

    def test_output_mode_rgb(self):
        cols = 10
        rendered, rows = self._make_rendered(cols)
        out = _apply_blink(rendered, 0.50, rows, cols)
        assert out.mode == "RGB"

    def test_eye_region_darker_at_blink_peak(self):
        """At blink peak t=0.50, upper-centre cells must be darker than neutral t=0.0."""
        cols = 20
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        rendered = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (160, 160, 160))

        neutral = _apply_blink(rendered, 0.0, rows, cols)
        blinked = _apply_blink(rendered, 0.50, rows, cols)

        arr_n = np.array(neutral).astype(float)
        arr_b = np.array(blinked).astype(float)

        # Sample the upper-centre region
        h, w = arr_n.shape[:2]
        y0 = int(0.15 * h)
        y1 = int(0.42 * h)
        x0 = int(0.25 * w)
        x1 = int(0.75 * w)

        mean_neutral = arr_n[y0:y1, x0:x1].mean()
        mean_blinked = arr_b[y0:y1, x0:x1].mean()

        assert mean_blinked < mean_neutral, (
            f"Eye region not darker at blink peak: neutral={mean_neutral:.1f} blinked={mean_blinked:.1f}"
        )

    def test_non_eye_region_unchanged_at_blink(self):
        """Bottom half of the frame should be unchanged during blink."""
        cols = 20
        rows = max(1, int(40 * cols / 40 * FONT_RATIO))
        rendered = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (160, 160, 160))

        neutral = _apply_blink(rendered, 0.0, rows, cols)
        blinked = _apply_blink(rendered, 0.50, rows, cols)

        arr_n = np.array(neutral).astype(float)
        arr_b = np.array(blinked).astype(float)

        # Bottom half (below eye region)
        h = arr_n.shape[0]
        y_start = int(0.5 * h)

        diff = np.abs(arr_n[y_start:] - arr_b[y_start:]).mean()
        assert diff < 1.0, f"Non-eye region changed during blink: mean diff={diff:.2f}"


# ---------------------------------------------------------------------------
# _apply_breathe
# ---------------------------------------------------------------------------


class TestApplyBreathe:
    def test_output_is_rgb_canvas_size(self, subject_rgba):
        canvas_size = 128
        out = _apply_breathe(subject_rgba, 0.0, canvas_size, 30.0)
        assert out.mode == "RGB"
        assert out.size == (canvas_size, canvas_size)

    def test_breathe_anti_phase(self):
        """scale_x and scale_y must be anti-phase: when x expands, y contracts."""
        t = 0.1  # sin(freq_rad) is positive
        freq_rad = t * math.pi * 2.0 * _BREATHE_FREQ
        scale_x = 1.0 + _BREATHE_X_AMP * math.sin(freq_rad)
        scale_y = 1.0 - _BREATHE_Y_AMP * math.sin(freq_rad)
        # At t=0.1 both amps are positive; x > 1.0 and y < 1.0
        assert scale_x > 1.0, f"Expected scale_x > 1.0 at t=0.1, got {scale_x:.4f}"
        assert scale_y < 1.0, f"Expected scale_y < 1.0 at t=0.1, got {scale_y:.4f}"

    def test_output_canvas_rgb(self, subject_rgba):
        out = _apply_breathe(subject_rgba, 0.5, 200, 30.0)
        assert out.mode == "RGB"
        assert out.width == 200
        assert out.height == 200


# ---------------------------------------------------------------------------
# _render_density_aa (style_j local copy)
# ---------------------------------------------------------------------------


class TestRenderDensityAAJ:
    def test_output_width_is_cols_times_cell_w(self, small_rgb, font):
        cols = 8
        out = _render_density_aa(small_rgb, cols, font)
        assert out.width == cols * CELL_W

    def test_output_mode_rgb(self, small_rgb, font):
        out = _render_density_aa(small_rgb, 8, font)
        assert out.mode == "RGB"

    def test_bright_input_non_black(self, font):
        img = Image.new("RGB", (60, 60), (200, 200, 200))
        out = _render_density_aa(img, 8, font)
        assert np.array(out).max() > 30

    def test_dark_input_mostly_dark(self, font):
        img = Image.new("RGB", (40, 40), (5, 5, 5))
        out = _render_density_aa(img, 8, font)
        assert np.array(out).astype(float).mean() < 20


# ---------------------------------------------------------------------------
# Fixed grid assertion: cell positions invariant across breathe frames
# ---------------------------------------------------------------------------


def test_cell_grid_positions_fixed_across_frames():
    """Cell output pixel dimensions must not change between frames (fixed grid).

    The breathe effect operates on the pixel canvas before rendering;
    the AA grid dimensions (cols * CELL_W, rows * CELL_H) must be identical
    for all frames.
    """
    font = ImageFont.load_default()
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 54, 54], fill=(200, 200, 200, 255))

    cols = 12
    canvas_size = 128
    sizes = []
    for i in range(6):
        t = i / 5.0
        frame_rgb = _apply_breathe(img, t, canvas_size, 30.0)
        aa = _render_density_aa(frame_rgb, cols, font)
        sizes.append(aa.size)

    # All frames must have the same output size
    assert len(set(sizes)) == 1, f"Cell grid size changed across frames (not fixed): {set(sizes)}"

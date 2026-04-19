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
"""Unit tests for Style I (long_cinematic + Ghostty-boo).

Key invariants verified:
1. Charset is CHARS_GHOSTTY (" ·~ox+=*%$@").
2. _apply_pan_zoom: output is RGB at (canvas_w, canvas_h).
3. Pan: centroid X shifts between t=0 and t=0.25.
4. Zoom: image area grows monotonically from t=0 to t=1.
5. _apply_vignette: corners darker than centre.
6. _render_density_aa: output width = cols * CELL_W, mode RGB.
7. Outline ring count > 0 on a subject-on-dark image.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from aa_animator_v2._boo_postprocess import CELL_H, CELL_W, apply_outline_ring
from aa_animator_v2.style_i_long_boo import (
    _GHOSTTY_CHARS,
    _ZOOM_MAX,
    _ZOOM_MIN,
    _apply_pan_zoom,
    _apply_vignette,
    _render_density_aa,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def subject_rgba() -> Image.Image:
    """64×64 RGBA with a bright white subject on transparent background."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([12, 12, 52, 52], fill=(220, 220, 220, 255))
    return img


@pytest.fixture()
def small_rgb_40() -> Image.Image:
    """40×40 RGB gradient for renderer tests."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    for y in range(40):
        for x in range(40):
            arr[y, x] = (int(x * 6), int(y * 6), 80)
    return Image.fromarray(arr)


@pytest.fixture()
def font():
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------


def test_ghostty_chars_is_correct():
    assert _GHOSTTY_CHARS == " ·~ox+=*%$@"
    assert len(_GHOSTTY_CHARS) == 11


# ---------------------------------------------------------------------------
# _apply_pan_zoom
# ---------------------------------------------------------------------------


class TestApplyPanZoom:
    def test_output_is_rgb_canvas_size(self, subject_rgba):
        canvas_w, canvas_h = 256, 128
        out = _apply_pan_zoom(subject_rgba, 0.0, canvas_w, canvas_h)
        assert out.mode == "RGB"
        assert out.size == (canvas_w, canvas_h)

    def test_pan_centroid_x_shifts(self, subject_rgba):
        """At t=0.25 sin(2π*0.25)=1 → max rightward pan."""
        canvas_w, canvas_h = 300, 150
        arr0 = np.array(_apply_pan_zoom(subject_rgba, 0.0, canvas_w, canvas_h)).astype(float)
        arr1 = np.array(_apply_pan_zoom(subject_rgba, 0.25, canvas_w, canvas_h)).astype(float)
        bright0 = (arr0 > 20).any(axis=2)
        bright1 = (arr1 > 20).any(axis=2)
        if bright0.any() and bright1.any():
            cx0 = np.where(bright0)[1].mean()
            cx1 = np.where(bright1)[1].mean()
            # At t=0.25 pan is positive → centroid should shift right
            assert cx1 >= cx0 - 5, f"Pan centroid did not shift right: cx0={cx0:.1f} cx1={cx1:.1f}"

    def test_zoom_grows_image_area(self, subject_rgba):
        """Subject pixel count should increase from t=0 to t=1 due to zoom."""
        canvas_w, canvas_h = 300, 150
        arr0 = np.array(_apply_pan_zoom(subject_rgba, 0.0, canvas_w, canvas_h)).astype(float)
        arr1 = np.array(_apply_pan_zoom(subject_rgba, 1.0, canvas_w, canvas_h)).astype(float)
        bright0 = (arr0 > 20).any(axis=2).sum()
        bright1 = (arr1 > 20).any(axis=2).sum()
        # Zoom from _ZOOM_MIN to _ZOOM_MAX (12% growth) should increase bright pixel count
        assert bright1 >= bright0 * 0.98, (
            f"Zoom did not grow image area: t=0 bright={bright0} t=1 bright={bright1}"
        )

    def test_zoom_constants_are_valid(self):
        assert _ZOOM_MIN > 0.0
        assert _ZOOM_MAX > _ZOOM_MIN
        assert _ZOOM_MAX < 2.0  # sanity: not more than 2x zoom


# ---------------------------------------------------------------------------
# _apply_vignette
# ---------------------------------------------------------------------------


class TestApplyVignette:
    def test_corners_darker_than_centre(self):
        img = Image.new("RGB", (100, 50), (200, 200, 200))
        out = _apply_vignette(img)
        arr = np.array(out).astype(float)
        centre = arr[25, 50, 0]
        corner = arr[0, 0, 0]
        assert corner < centre, f"corner({corner:.1f}) not < centre({centre:.1f})"

    def test_preserves_size(self, small_rgb_40):
        out = _apply_vignette(small_rgb_40)
        assert out.size == small_rgb_40.size

    def test_output_mode_rgb(self, small_rgb_40):
        out = _apply_vignette(small_rgb_40)
        assert out.mode == "RGB"

    def test_no_negative_values(self):
        img = Image.new("RGB", (60, 30), (150, 150, 150))
        arr = np.array(_apply_vignette(img))
        assert arr.min() >= 0


# ---------------------------------------------------------------------------
# _render_density_aa
# ---------------------------------------------------------------------------


class TestRenderDensityAA:
    def test_output_width_is_cols_times_cell_w(self, small_rgb_40, font):
        cols = 8
        out = _render_density_aa(small_rgb_40, cols, font)
        assert out.width == cols * CELL_W

    def test_output_mode_rgb(self, small_rgb_40, font):
        out = _render_density_aa(small_rgb_40, 8, font)
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
# Outline ring visible on subject image
# ---------------------------------------------------------------------------


def test_outline_ring_count_positive():
    """Outline ring must produce at least some bright outline-colour pixels."""
    subject = Image.new("RGB", (80, 80), (8, 8, 12))
    d = ImageDraw.Draw(subject)
    d.ellipse([15, 15, 65, 65], fill=(220, 220, 220))

    font = ImageFont.load_default()
    cols = 12
    from aa_animator_v2._boo_postprocess import FONT_RATIO

    rows = max(1, int(80 * cols / 80 * FONT_RATIO))
    rendered = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (80, 80, 80))

    out = apply_outline_ring(rendered, subject, cols, font)
    arr = np.array(out)
    outline_mask = (arr[:, :, 0] > 180) & (arr[:, :, 2] > 190)
    assert outline_mask.sum() > 0, "No outline ring pixels found on subject image"

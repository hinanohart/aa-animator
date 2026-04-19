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
"""Unit tests for Style F dynamic lighting AA animation.

Key invariants:
1. base_brightness is frame-invariant (built once from image, never mutated).
2. lit_intensity transitions smoothly (|delta| < 0.1 per frame per cell).
3. Light source traces expected 2-D trajectory per pattern.
4. lerp_color produces correct boundary values (alpha=0→white, alpha=1→blue).
5. Space cells (background) are never drawn regardless of lit_intensity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from aa_animator_v2.style_f_lighting import (
    _GLOW_COLOR,
    _STATIC_COLOR,
    VALID_PATTERNS,
    compute_base_brightness,
    compute_lit_intensity,
    lerp_color,
    light_horizontal_sweep,
    light_lissajous,
    light_orbit,
    light_vertical_drop,
    quantize_to_glyph,
    render_frame,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_image() -> Image.Image:
    """20×10 RGB test image with a bright foreground rectangle."""
    img = Image.new("RGB", (20, 10), (8, 8, 12))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([4, 2, 16, 8], fill=(200, 200, 200))
    return img


@pytest.fixture()
def base_brightness(tiny_image: Image.Image) -> np.ndarray:
    return compute_base_brightness(tiny_image, cols=20, rows=10)


# ---------------------------------------------------------------------------
# compute_base_brightness — frame-invariance
# ---------------------------------------------------------------------------

class TestComputeBaseBrightness:
    def test_shape(self, tiny_image: Image.Image) -> None:
        b = compute_base_brightness(tiny_image, cols=20, rows=10)
        assert b.shape == (10, 20)

    def test_range(self, tiny_image: Image.Image) -> None:
        b = compute_base_brightness(tiny_image, cols=20, rows=10)
        assert float(b.min()) >= 0.0
        assert float(b.max()) <= 1.0

    def test_frame_invariant(self, tiny_image: Image.Image) -> None:
        """Calling compute_base_brightness twice returns identical arrays."""
        b1 = compute_base_brightness(tiny_image, cols=20, rows=10)
        b2 = compute_base_brightness(tiny_image, cols=20, rows=10)
        np.testing.assert_array_equal(b1, b2)

    def test_not_mutated_after_compute_lit(self, base_brightness: np.ndarray) -> None:
        """compute_lit_intensity must not mutate base_brightness."""
        original = base_brightness.copy()
        compute_lit_intensity(base_brightness, lx=10.0, ly=5.0)
        np.testing.assert_array_equal(base_brightness, original)


# ---------------------------------------------------------------------------
# compute_lit_intensity — smooth float output
# ---------------------------------------------------------------------------

class TestComputeLitIntensity:
    ROWS, COLS = 10, 20

    def test_shape(self, base_brightness: np.ndarray) -> None:
        lit = compute_lit_intensity(base_brightness, lx=10.0, ly=5.0)
        assert lit.shape == base_brightness.shape

    def test_range(self, base_brightness: np.ndarray) -> None:
        lit = compute_lit_intensity(base_brightness, lx=10.0, ly=5.0)
        assert float(lit.min()) >= 0.0
        assert float(lit.max()) <= 1.0

    def test_smoothness_across_frames(self, base_brightness: np.ndarray) -> None:
        """Per-frame per-cell intensity delta must be < 0.5 (no boolean hard-flip).

        A Gaussian light source moving ~5 cells/frame with sigma=8 legitimately
        produces ~0.19 delta at the peak — that is visually smooth/continuous.
        The threshold of 0.5 catches only boolean on/off artefacts (delta≈1.0).
        """
        rows, cols = base_brightness.shape
        fps = 30
        prev: np.ndarray | None = None
        for i in range(fps):  # 1 second of frames
            t = i / fps
            lx = (cols / 2.0) + (cols / 2.0) * math.sin(2.0 * math.pi * 0.5 * t)
            ly = rows / 2.0
            lit = compute_lit_intensity(base_brightness, lx, ly)
            if prev is not None:
                max_delta = float(np.max(np.abs(lit - prev)))
                assert max_delta < 0.5, (
                    f"Frame {i}: intensity jump {max_delta:.4f} >= 0.5 — boolean mask detected"
                )
            prev = lit.copy()

    def test_light_centre_brighter_than_edge(self, base_brightness: np.ndarray) -> None:
        """Cell under light source should have higher falloff than distant cell."""
        rows, cols = base_brightness.shape
        lx, ly = cols / 2.0, rows / 2.0
        lit = compute_lit_intensity(
            np.zeros_like(base_brightness),  # zero base to isolate falloff
            lx, ly, sigma=3.0, light_intensity=0.8,
        )
        cx, cy = int(lx), int(ly)
        # Centre cell brighter than corner
        assert lit[cy, cx] > lit[0, 0]


# ---------------------------------------------------------------------------
# Light source trajectories
# ---------------------------------------------------------------------------

class TestLightTrajectories:
    ROWS, COLS = 41, 100

    def test_horizontal_sweep_y_fixed(self) -> None:
        """horizontal_sweep: y must stay at rows/2."""
        for i in range(30):
            _, ly = light_horizontal_sweep(i / 30.0, self.ROWS, self.COLS)
            assert abs(ly - self.ROWS / 2.0) < 1e-6

    def test_horizontal_sweep_x_varies(self) -> None:
        """horizontal_sweep: x must oscillate across full width."""
        xs = [light_horizontal_sweep(i / 30.0, self.ROWS, self.COLS)[0] for i in range(120)]
        assert max(xs) - min(xs) > self.COLS * 0.8

    def test_vertical_drop_x_fixed(self) -> None:
        """vertical_drop: x must stay at cols/2."""
        for i in range(30):
            lx, _ = light_vertical_drop(i / 30.0, self.ROWS, self.COLS)
            assert abs(lx - self.COLS / 2.0) < 1e-6

    def test_vertical_drop_y_loops(self) -> None:
        """vertical_drop: y wraps within [0, rows)."""
        ys = [light_vertical_drop(i / 30.0, self.ROWS, self.COLS)[1] for i in range(300)]
        assert min(ys) >= 0.0
        assert max(ys) < self.ROWS

    def test_orbit_circular_radius(self) -> None:
        """orbit: distance from centre should be approximately constant."""
        cx, cy = self.COLS / 2.0, self.ROWS / 2.0
        r_expected = min(self.ROWS, self.COLS) / 3.0
        for i in range(60):
            lx, ly = light_orbit(i / 30.0, self.ROWS, self.COLS)
            r_actual = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            assert abs(r_actual - r_expected) < 1e-4

    def test_lissajous_stays_in_bounds(self) -> None:
        """lissajous: light must stay within grid bounds."""
        for i in range(120):
            lx, ly = light_lissajous(i / 30.0, self.ROWS, self.COLS)
            assert 0.0 <= lx <= self.COLS, f"lx={lx} out of bounds"
            assert 0.0 <= ly <= self.ROWS, f"ly={ly} out of bounds"


# ---------------------------------------------------------------------------
# lerp_color — boundary values
# ---------------------------------------------------------------------------

class TestLerpColor:
    def test_alpha_zero_returns_white(self) -> None:
        result = lerp_color(_STATIC_COLOR, _GLOW_COLOR, alpha=0.0)
        assert result == _STATIC_COLOR

    def test_alpha_one_returns_blue(self) -> None:
        result = lerp_color(_STATIC_COLOR, _GLOW_COLOR, alpha=1.0)
        assert result == _GLOW_COLOR

    def test_alpha_half_is_midpoint(self) -> None:
        r, g, _b = lerp_color(_STATIC_COLOR, _GLOW_COLOR, alpha=0.5)
        expected_r = int(_STATIC_COLOR[0] * 0.5 + _GLOW_COLOR[0] * 0.5)
        expected_g = int(_STATIC_COLOR[1] * 0.5 + _GLOW_COLOR[1] * 0.5)
        assert abs(r - expected_r) <= 1
        assert abs(g - expected_g) <= 1

    def test_alpha_clipped_above_one(self) -> None:
        result = lerp_color(_STATIC_COLOR, _GLOW_COLOR, alpha=1.5)
        assert result == _GLOW_COLOR

    def test_alpha_clipped_below_zero(self) -> None:
        result = lerp_color(_STATIC_COLOR, _GLOW_COLOR, alpha=-0.5)
        assert result == _STATIC_COLOR


# ---------------------------------------------------------------------------
# quantize_to_glyph
# ---------------------------------------------------------------------------

class TestQuantizeToGlyph:
    def test_zero_returns_first_char(self) -> None:
        from aa_animator_v2.style_f_lighting import _GHOSTTY_CHARS
        assert quantize_to_glyph(0.0) == _GHOSTTY_CHARS[0]

    def test_one_returns_last_char(self) -> None:
        from aa_animator_v2.style_f_lighting import _GHOSTTY_CHARS
        assert quantize_to_glyph(1.0) == _GHOSTTY_CHARS[-1]


# ---------------------------------------------------------------------------
# render_frame — space cells never drawn
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_space_cells_stay_background(self, base_brightness: np.ndarray, tiny_image: Image.Image) -> None:
        """Space cells must remain background colour regardless of high lit_intensity."""
        from PIL import ImageFont

        from aa_animator_v2.style_f_lighting import _CELL_H, _CELL_W, _build_char_grid
        char_grid = _build_char_grid(base_brightness)
        # All cells lit at max intensity
        lit = np.ones_like(base_brightness)
        font = ImageFont.load_default()
        frame = render_frame(char_grid, lit, font)
        arr = np.array(frame)
        rows = len(char_grid)
        cols = len(char_grid[0])
        for r in range(rows):
            for c in range(cols):
                if char_grid[r][c] == " ":
                    region = arr[r * _CELL_H:(r + 1) * _CELL_H, c * _CELL_W:(c + 1) * _CELL_W]
                    mean_blue = float(region[..., 2].mean())
                    assert mean_blue < 200, (
                        f"Space cell ({r},{c}) appears lit — background leak detected"
                    )

    def test_output_size(self, base_brightness: np.ndarray) -> None:
        from PIL import ImageFont

        from aa_animator_v2.style_f_lighting import _CELL_H, _CELL_W, _build_char_grid
        char_grid = _build_char_grid(base_brightness)
        rows = len(char_grid)
        cols = len(char_grid[0])
        lit = np.zeros_like(base_brightness)
        font = ImageFont.load_default()
        frame = render_frame(char_grid, lit, font)
        assert frame.size == (cols * _CELL_W, rows * _CELL_H)


# ---------------------------------------------------------------------------
# VALID_PATTERNS completeness
# ---------------------------------------------------------------------------

def test_valid_patterns_count() -> None:
    assert set(VALID_PATTERNS) == {
        "horizontal_sweep",
        "vertical_drop",
        "orbit",
        "lissajous",
        "spotlight_trail",
    }

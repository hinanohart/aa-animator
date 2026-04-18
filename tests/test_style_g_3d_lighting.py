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
"""Unit tests for Style G 3D-aware lighting AA animation.

Key invariants:
1. depth_cell shape is (rows, cols).
2. lit intensity is smooth across frames (|Δ| < 0.1 per cell per frame).
3. rim_light: cells with high depth (far background) are brighter than
   cells with low depth (near foreground) when light is at z=0.9.
4. approach: Lz decreases from start to mid-clip.
5. No boolean on/off: intensity must have ≥ 3 unique float values per animation.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aa_animator_v2.style_g_3d_lighting import (
    VALID_PATTERNS,
    _estimate_depth_cell,
    compute_base_brightness_and_depth,
    compute_lit_intensity_3d,
    lerp_color,
    light_approach,
    light_orbit_3d,
    light_pendulum_3d,
    light_rim_light,
    light_spiral,
    quantize_to_glyph,
    render_frame_g,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_image() -> Image.Image:
    """20×10 RGB test image with a bright subject region."""
    img = Image.new("RGB", (20, 10), (8, 8, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 2, 16, 8], fill=(200, 200, 200))
    return img


@pytest.fixture()
def base_and_depth(tiny_image: Image.Image):
    return compute_base_brightness_and_depth(tiny_image, cols=20, rows=10)


# ---------------------------------------------------------------------------
# depth shape
# ---------------------------------------------------------------------------

class TestDepthShape:
    def test_depth_cell_shape(self, tiny_image: Image.Image) -> None:
        """_estimate_depth_cell must return (rows, cols) shaped array."""
        rows, cols = 10, 20
        depth = _estimate_depth_cell(tiny_image.convert("RGB"), cols=cols, rows=rows)
        assert depth.shape == (rows, cols), (
            f"Expected depth shape ({rows}, {cols}), got {depth.shape}"
        )

    def test_depth_range(self, tiny_image: Image.Image) -> None:
        depth = _estimate_depth_cell(tiny_image.convert("RGB"), cols=20, rows=10)
        assert float(depth.min()) >= 0.0
        assert float(depth.max()) <= 1.0

    def test_base_and_depth_shapes_match(self, base_and_depth) -> None:
        base, depth_cell = base_and_depth
        assert base.shape == depth_cell.shape
        assert base.shape == (10, 20)

    def test_base_range(self, base_and_depth) -> None:
        base, _ = base_and_depth
        assert float(base.min()) >= 0.0
        assert float(base.max()) <= 1.0


# ---------------------------------------------------------------------------
# lit intensity smoothness
# ---------------------------------------------------------------------------

class TestLitSmoothness:
    ROWS, COLS = 10, 20

    def _make_base_depth(self):
        base = np.random.default_rng(42).uniform(0.1, 0.9, (self.ROWS, self.COLS)).astype(np.float32)
        depth = np.random.default_rng(7).uniform(0.0, 1.0, (self.ROWS, self.COLS)).astype(np.float32)
        return base, depth

    @pytest.mark.parametrize("pattern_fn,kwargs", [
        (light_approach, {}),
        (light_orbit_3d, {}),
        (light_spiral, {}),
        (light_pendulum_3d, {}),
        (light_rim_light, {}),
    ])
    def test_frame_to_frame_smoothness(self, pattern_fn, kwargs) -> None:
        """Max per-cell intensity delta between consecutive frames < 0.1."""
        base, depth = self._make_base_depth()
        fps = 30
        sigma = 4.0  # smaller sigma for speed in tests
        z_scale = self.COLS / 3.0

        prev_lit = None
        for i in range(30):
            t = i / fps
            lx, ly, lz = pattern_fn(t, self.ROWS, self.COLS, **kwargs)
            lit = compute_lit_intensity_3d(
                base, depth, lx, ly, lz,
                sigma=sigma,
                intensity=0.6,
                z_scale=z_scale,
            )
            if prev_lit is not None:
                delta = float(np.abs(lit - prev_lit).max())
                assert delta < 0.1, (
                    f"{pattern_fn.__name__}: max frame delta {delta:.4f} ≥ 0.1 "
                    f"at frame {i}"
                )
            prev_lit = lit


# ---------------------------------------------------------------------------
# rim_light: background cells (high depth) are brighter when Lz=0.9
# ---------------------------------------------------------------------------

class TestRimLightDepthEffect:
    def test_rim_light_bright_on_far_cells(self) -> None:
        """With light at z=0.9, far cells (depth≈1) are closer to light than near cells."""
        rows, cols = 10, 20
        # Uniform base so difference comes only from depth
        base = np.full((rows, cols), 0.3, dtype=np.float32)
        # Two extremes: near (depth=0) and far (depth=1)
        depth_near = np.zeros((rows, cols), dtype=np.float32)
        depth_far = np.ones((rows, cols), dtype=np.float32)

        lx, ly, lz = cols / 2.0, rows / 2.0, 0.9  # rim_light position
        z_scale = cols / 3.0
        sigma = 4.0

        lit_near = compute_lit_intensity_3d(
            base, depth_near, lx, ly, lz, sigma=sigma, z_scale=z_scale
        )
        lit_far = compute_lit_intensity_3d(
            base, depth_far, lx, ly, lz, sigma=sigma, z_scale=z_scale
        )

        # Far cells (depth=1, dz=0.1) should be closer to Lz=0.9 → brighter
        assert lit_far.mean() > lit_near.mean(), (
            "rim_light: far cells should be brighter than near cells "
            f"(far={lit_far.mean():.3f} near={lit_near.mean():.3f})"
        )


# ---------------------------------------------------------------------------
# approach: Lz decreases over time
# ---------------------------------------------------------------------------

class TestApproachPattern:
    def test_lz_decreases(self) -> None:
        """Lz from approach must decrease as t increases (light moves closer)."""
        rows, cols = 41, 100
        lz_values = [
            light_approach(t, rows, cols)[2]
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        ]
        # Not necessarily monotone every step (loops), but overall trend from
        # start: Lz(0)=0.9 down to Lz(3.0)=0.9-0.9*(3*0.3%1) — verify
        # the function returns decreasing values in the first half-period.
        first_half = lz_values[:4]  # t=0, 0.5, 1, 1.5
        for i in range(len(first_half) - 1):
            assert first_half[i] >= first_half[i + 1] - 1e-6, (
                f"approach Lz should decrease: Lz[{i}]={first_half[i]:.3f} "
                f"Lz[{i+1}]={first_half[i+1]:.3f}"
            )

    def test_lz_initial_value(self) -> None:
        """approach at t=0 must start at Lz≈0.9 (far)."""
        _, _, lz = light_approach(0.0, 41, 100)
        assert abs(lz - 0.9) < 1e-6, f"Expected Lz≈0.9 at t=0, got {lz}"


# ---------------------------------------------------------------------------
# boolean on/off prohibited: intensity must have ≥ 3 unique float values
# ---------------------------------------------------------------------------

class TestNoBooleanOnOff:
    def test_orbit_3d_intensity_not_binary(self) -> None:
        """orbit_3d must produce ≥ 3 unique mean intensity values across 12 frames."""
        rows, cols = 10, 20
        base = np.full((rows, cols), 0.4, dtype=np.float32)
        depth = np.full((rows, cols), 0.5, dtype=np.float32)
        fps = 30
        z_scale = cols / 3.0
        sigma = 4.0

        intensities = []
        for i in range(0, 120, 10):
            t = i / fps
            lx, ly, lz = light_orbit_3d(t, rows, cols)
            lit = compute_lit_intensity_3d(
                base, depth, lx, ly, lz, sigma=sigma, z_scale=z_scale
            )
            intensities.append(round(float(lit.mean()), 4))

        unique_count = len(set(intensities))
        assert unique_count >= 3, (
            f"orbit_3d intensity appears binary/static: only {unique_count} "
            f"unique values: {sorted(set(intensities))}"
        )


# ---------------------------------------------------------------------------
# quantize_to_glyph
# ---------------------------------------------------------------------------

class TestQuantizeToGlyph:
    def test_shape(self) -> None:
        lit = np.random.default_rng(0).uniform(0.0, 1.0, (10, 20)).astype(np.float32)
        grid = quantize_to_glyph(lit)
        assert len(grid) == 10
        assert all(len(row) == 20 for row in grid)

    def test_chars_in_palette(self) -> None:
        from aa_animator_v2.style_g_3d_lighting import _GHOSTTY_CHARS
        lit = np.random.default_rng(1).uniform(0.0, 1.0, (10, 20)).astype(np.float32)
        grid = quantize_to_glyph(lit)
        for row in grid:
            for ch in row:
                assert ch in _GHOSTTY_CHARS, f"Unexpected char {ch!r}"

    def test_very_dark_cells_are_space(self) -> None:
        """Cells with near-zero lit intensity must be space."""
        lit = np.zeros((5, 5), dtype=np.float32)
        grid = quantize_to_glyph(lit)
        for row in grid:
            for ch in row:
                assert ch == " ", f"Expected space for zero-lit cell, got {ch!r}"


# ---------------------------------------------------------------------------
# lerp_color
# ---------------------------------------------------------------------------

class TestLerpColor:
    def test_alpha_zero_gives_color_a(self) -> None:
        result = lerp_color((215, 215, 215), (70, 130, 255), 0.0)
        assert result == (215, 215, 215)

    def test_alpha_one_gives_color_b(self) -> None:
        result = lerp_color((215, 215, 215), (70, 130, 255), 1.0)
        assert result == (70, 130, 255)

    def test_midpoint(self) -> None:
        a = (0, 0, 0)
        b = (100, 100, 100)
        result = lerp_color(a, b, 0.5)
        assert result == (50, 50, 50)

    def test_clamps_alpha(self) -> None:
        # Should not raise even if alpha slightly out of range
        result_high = lerp_color((100, 100, 100), (200, 200, 200), 1.5)
        result_low = lerp_color((100, 100, 100), (200, 200, 200), -0.5)
        assert result_high == (200, 200, 200)
        assert result_low == (100, 100, 100)


# ---------------------------------------------------------------------------
# render_frame_g
# ---------------------------------------------------------------------------

class TestRenderFrameG:
    def test_output_size(self) -> None:
        from aa_animator_v2.style_g_3d_lighting import _CELL_W, _CELL_H
        from PIL import ImageFont
        rows, cols = 5, 10
        char_grid = [["@"] * cols for _ in range(rows)]
        color_grid = [[(215, 215, 215)] * cols for _ in range(rows)]
        font = ImageFont.load_default()
        frame = render_frame_g(char_grid, color_grid, font)
        assert frame.size == (cols * _CELL_W, rows * _CELL_H)

    def test_space_cells_not_drawn(self) -> None:
        """Space cells must remain background color."""
        from aa_animator_v2.style_g_3d_lighting import _BG_COLOR, _CELL_W, _CELL_H
        from PIL import ImageFont
        rows, cols = 3, 3
        char_grid = [[" "] * cols for _ in range(rows)]
        color_grid = [[(255, 0, 0)] * cols for _ in range(rows)]
        font = ImageFont.load_default()
        frame = render_frame_g(char_grid, color_grid, font)
        arr = np.array(frame)
        # Entire canvas should be background color
        bg = np.array(_BG_COLOR)
        mean_per_channel = arr.mean(axis=(0, 1))
        for ch_idx in range(3):
            assert abs(float(mean_per_channel[ch_idx]) - bg[ch_idx]) < 5.0, (
                "Space-only frame should be near background color"
            )


# ---------------------------------------------------------------------------
# VALID_PATTERNS completeness
# ---------------------------------------------------------------------------

def test_valid_patterns_count() -> None:
    assert set(VALID_PATTERNS) == {
        "approach", "orbit_3d", "spiral", "pendulum_3d", "rim_light"
    }


# ---------------------------------------------------------------------------
# compute_lit_intensity_3d — basic properties
# ---------------------------------------------------------------------------

class TestComputeLitIntensity3D:
    def test_output_shape(self) -> None:
        rows, cols = 10, 20
        base = np.full((rows, cols), 0.3, dtype=np.float32)
        depth = np.full((rows, cols), 0.5, dtype=np.float32)
        lit = compute_lit_intensity_3d(base, depth, cols / 2, rows / 2, 0.5)
        assert lit.shape == (rows, cols)

    def test_output_range(self) -> None:
        rows, cols = 10, 20
        base = np.random.default_rng(0).uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        depth = np.random.default_rng(1).uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
        lit = compute_lit_intensity_3d(base, depth, cols / 2, rows / 2, 0.5)
        assert float(lit.min()) >= 0.0
        assert float(lit.max()) <= 1.0

    def test_cell_near_light_is_brightest(self) -> None:
        """Cell closest to light source must be among the brightest."""
        rows, cols = 10, 20
        base = np.full((rows, cols), 0.0, dtype=np.float32)
        depth = np.full((rows, cols), 0.5, dtype=np.float32)
        lx, ly, lz = 5.0, 3.0, 0.5  # light at specific position
        lit = compute_lit_intensity_3d(base, depth, lx, ly, lz, sigma=3.0, z_scale=5.0)
        # Nearest cell
        r_near = int(round(ly))
        c_near = int(round(lx))
        center_val = lit[r_near, c_near]
        assert center_val >= lit.mean() * 1.5, (
            f"Cell nearest to light ({center_val:.3f}) should exceed mean ({lit.mean():.3f})"
        )

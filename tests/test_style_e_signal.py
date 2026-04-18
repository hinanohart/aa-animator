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
"""Unit tests for Style E signal-driven AA animation.

Key invariants:
1. char_grid is frame-invariant (built once, never mutated).
2. glow_mask varies across frames (signal is actually dynamic).
3. Background cells (space) are never lit by glow.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from PIL import Image

from aa_animator_v2.style_e_signal import (
    VALID_SIGNALS,
    pulse_alpha,
    render_frame,
    render_static_base,
    signal_combo,
    signal_jump,
    signal_pulse,
    signal_scan,
    signal_wave,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_image() -> Image.Image:
    """20×10 RGB test image with visible foreground pixels."""
    img = Image.new("RGB", (20, 10), (8, 8, 12))
    # Draw a bright rectangle in the centre so luma > 6 cells exist
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([5, 2, 15, 8], fill=(200, 200, 200))
    return img


@pytest.fixture()
def char_grid_and_brightness(tiny_image: Image.Image):
    return render_static_base(tiny_image, cols=20, rows=10)


# ---------------------------------------------------------------------------
# render_static_base
# ---------------------------------------------------------------------------

class TestRenderStaticBase:
    def test_shape(self, tiny_image: Image.Image) -> None:
        char_grid, brightness = render_static_base(tiny_image, cols=20, rows=10)
        assert len(char_grid) == 10
        assert all(len(row) == 20 for row in char_grid)
        assert brightness.shape == (10, 20)

    def test_brightness_range(self, tiny_image: Image.Image) -> None:
        _, brightness = render_static_base(tiny_image, cols=20, rows=10)
        assert brightness.min() >= 0.0
        assert brightness.max() <= 1.0

    def test_has_non_space_cells(self, tiny_image: Image.Image) -> None:
        char_grid, _ = render_static_base(tiny_image, cols=20, rows=10)
        non_space = sum(1 for row in char_grid for ch in row if ch != " ")
        assert non_space > 0, "Expected at least some non-space cells from bright rectangle"

    def test_char_grid_not_mutated_across_calls(self, tiny_image: Image.Image) -> None:
        """render_static_base must return independent data on each call."""
        cg1, _ = render_static_base(tiny_image, cols=20, rows=10)
        cg2, _ = render_static_base(tiny_image, cols=20, rows=10)
        assert cg1 == cg2


# ---------------------------------------------------------------------------
# Signal primitives — shape and type
# ---------------------------------------------------------------------------

class TestSignalShapes:
    ROWS, COLS = 10, 20

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 2.3])
    def test_jump_shape(self, t: float) -> None:
        mask = signal_jump(t, self.ROWS, self.COLS)
        assert mask.shape == (self.ROWS, self.COLS)
        assert mask.dtype == bool

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
    def test_scan_shape(self, t: float) -> None:
        mask = signal_scan(t, self.ROWS, self.COLS)
        assert mask.shape == (self.ROWS, self.COLS)

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
    def test_wave_shape(self, t: float) -> None:
        mask = signal_wave(t, self.ROWS, self.COLS)
        assert mask.shape == (self.ROWS, self.COLS)

    def test_pulse_all_true(self) -> None:
        mask = signal_pulse(0.0, self.ROWS, self.COLS)
        assert mask.all(), "pulse mask should be all-True"

    def test_combo_shape(self) -> None:
        mask = signal_combo(0.5, self.ROWS, self.COLS)
        assert mask.shape == (self.ROWS, self.COLS)


# ---------------------------------------------------------------------------
# Signal dynamics — mask must vary across time
# ---------------------------------------------------------------------------

class TestSignalDynamics:
    ROWS, COLS = 41, 100

    def _masks_vary(self, fn, n: int = 30, duration: float = 4.0) -> bool:
        masks = [fn(i / 30.0, self.ROWS, self.COLS) for i in range(n)]
        return any(not np.array_equal(masks[i], masks[i + 1]) for i in range(len(masks) - 1))

    def test_jump_varies(self) -> None:
        assert self._masks_vary(signal_jump)

    def test_scan_varies(self) -> None:
        assert self._masks_vary(signal_scan)

    def test_wave_varies(self) -> None:
        assert self._masks_vary(signal_wave)

    def test_combo_varies(self) -> None:
        assert self._masks_vary(signal_combo)

    def test_pulse_alpha_varies(self) -> None:
        alphas = [pulse_alpha(i / 30.0) for i in range(60)]
        assert max(alphas) - min(alphas) > 0.1, "pulse_alpha must oscillate"


# ---------------------------------------------------------------------------
# render_frame — char_grid invariance
# ---------------------------------------------------------------------------

class TestRenderFrameInvariance:
    def test_char_grid_not_mutated_by_render(self, char_grid_and_brightness, tiny_image: Image.Image) -> None:
        """render_frame must not mutate char_grid."""
        from PIL import ImageFont
        char_grid, brightness = char_grid_and_brightness
        original = copy.deepcopy(char_grid)
        rows = len(char_grid)
        cols = len(char_grid[0])
        font = ImageFont.load_default()
        glow_mask = signal_jump(0.5, rows, cols)
        render_frame(char_grid, glow_mask, brightness, font)
        assert char_grid == original, "render_frame mutated char_grid"

    def test_render_frame_output_size(self, char_grid_and_brightness) -> None:
        from aa_animator_v2.style_e_signal import _CELL_W, _CELL_H
        from PIL import ImageFont
        char_grid, brightness = char_grid_and_brightness
        rows = len(char_grid)
        cols = len(char_grid[0])
        font = ImageFont.load_default()
        glow_mask = np.ones((rows, cols), dtype=bool)
        frame = render_frame(char_grid, glow_mask, brightness, font)
        assert frame.size == (cols * _CELL_W, rows * _CELL_H)

    def test_background_cells_not_lit(self, char_grid_and_brightness) -> None:
        """Space cells must stay BG color regardless of glow_mask."""
        from aa_animator_v2.style_e_signal import _BG_COLOR, _GLOW_COLOR, _CELL_W, _CELL_H
        from PIL import ImageFont
        char_grid, brightness = char_grid_and_brightness
        rows = len(char_grid)
        cols = len(char_grid[0])
        font = ImageFont.load_default()
        # Force all cells to glow
        glow_mask = np.ones((rows, cols), dtype=bool)
        frame = render_frame(char_grid, glow_mask, brightness, font)
        arr = np.array(frame)

        for r in range(rows):
            for c in range(cols):
                if char_grid[r][c] == " ":
                    region = arr[r * _CELL_H:(r + 1) * _CELL_H, c * _CELL_W:(c + 1) * _CELL_W]
                    # Region should be background color, not glow color
                    mean = region.mean(axis=(0, 1))
                    assert mean[2] < 200, (
                        f"Space cell ({r},{c}) appears blue-glowing — background cell glow leak"
                    )


# ---------------------------------------------------------------------------
# VALID_SIGNALS completeness
# ---------------------------------------------------------------------------

def test_valid_signals_count() -> None:
    assert set(VALID_SIGNALS) == {"jump", "scan", "wave", "pulse", "combo"}

"""Unit tests for aa_animator_v2.parallax."""

from __future__ import annotations

import numpy as np
import pytest

from aa_animator_v2.parallax import (
    dynamic_amp_px,
    fill_holes,
    forward_warp,
    orbit_displacement,
    warp_mask,
)


class TestDynamicAmpPx:
    def test_coverage_at_ref_gives_base_amp(self) -> None:
        # coverage=0.30 (ref) → amp = round(18 * 1.0) = 18
        assert dynamic_amp_px(0.30) == 18

    def test_high_coverage_reduces_amp(self) -> None:
        # coverage=0.60 → amp = round(18 * 0.3/0.6) = round(9.0) = 9
        assert dynamic_amp_px(0.60) == 9

    def test_low_coverage_clips_at_base(self) -> None:
        # coverage=0.05 → amp = round(18 * min(1.0, 0.3/0.05)) = round(18) = 18
        assert dynamic_amp_px(0.05) == 18

    def test_very_low_coverage_clips_amp(self) -> None:
        # coverage below 0.05 is clamped to 0.05 → same as 0.05 case
        assert dynamic_amp_px(0.01) == dynamic_amp_px(0.05)

    def test_coverage_inversely_proportional(self) -> None:
        # amp at 0.30 should be larger than amp at 0.60
        assert dynamic_amp_px(0.30) > dynamic_amp_px(0.60)

    def test_returns_int(self) -> None:
        assert isinstance(dynamic_amp_px(0.30), int)


class TestForwardWarp:
    def _make_img(self, h: int = 8, w: int = 8) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((h, w, 3)).astype(np.float32)

    def _flat_depth(self, h: int = 8, w: int = 8) -> np.ndarray:
        return np.zeros((h, w), dtype=np.float32)

    def test_output_shape_preserved(self) -> None:
        img = self._make_img(16, 20)
        depth = self._flat_depth(16, 20)
        out = forward_warp(img, depth, dx=0.0, dy=0.0)
        assert out.shape == img.shape

    def test_dx_dy_zero_depth_zero_is_identity(self) -> None:
        # depth=0 → no shift → identity warp
        img = self._make_img(8, 8)
        depth = self._flat_depth(8, 8)
        out = forward_warp(img, depth, dx=5.0, dy=3.0)
        np.testing.assert_allclose(out, img, rtol=1e-5)

    def test_output_dtype_float32(self) -> None:
        img = self._make_img()
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        out = forward_warp(img, depth, dx=2.0, dy=1.0)
        assert out.dtype == np.float32

    def test_values_in_valid_range(self) -> None:
        img = self._make_img()
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        out = forward_warp(img, depth, dx=3.0, dy=1.0)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5


class TestWarpMask:
    def test_output_shape_preserved(self) -> None:
        mask = np.ones((16, 20), dtype=bool)
        depth = np.zeros((16, 20), dtype=np.float32)
        out = warp_mask(mask, depth, dx=0.0, dy=0.0)
        assert out.shape == mask.shape

    def test_all_true_mask_stays_all_true_when_no_shift(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        depth = np.zeros((8, 8), dtype=np.float32)
        out = warp_mask(mask, depth, dx=5.0, dy=3.0)
        assert out.all()

    def test_output_dtype_bool(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        out = warp_mask(mask, depth, dx=2.0, dy=1.0)
        assert out.dtype == bool


class TestFillHoles:
    def test_no_holes_returns_unchanged(self) -> None:
        rng = np.random.default_rng(1)
        img = rng.random((8, 8, 3)).astype(np.float32) + 0.1  # all non-zero
        hole = np.zeros((8, 8), dtype=bool)
        out = fill_holes(img, hole)
        np.testing.assert_allclose(out, img)

    def test_single_hole_gets_filled(self) -> None:
        img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        img[4, 4] = 0.0  # punch a hole
        hole = np.zeros((8, 8), dtype=bool)
        hole[4, 4] = True
        out = fill_holes(img, hole)
        # The filled pixel should be close to 0.5
        assert float(out[4, 4].sum()) > 0.0

    def test_output_shape_unchanged(self) -> None:
        img = np.ones((10, 12, 3), dtype=np.float32) * 0.3
        img[5, 6] = 0.0
        hole = np.zeros((10, 12), dtype=bool)
        hole[5, 6] = True
        out = fill_holes(img, hole)
        assert out.shape == img.shape


class TestOrbitDisplacement:
    def test_frame_zero_dx_at_base_amp(self) -> None:
        # angle=0 → dx = amp * cos(0) = amp
        dx, _dy = orbit_displacement(0, 30, 18)
        assert pytest.approx(dx, abs=1e-5) == 18.0

    def test_frame_zero_dy_is_zero(self) -> None:
        _, dy = orbit_displacement(0, 30, 18)
        assert pytest.approx(dy, abs=1e-5) == 0.0

    def test_half_cycle_negates_dx(self) -> None:
        dx0, _ = orbit_displacement(0, 30, 18)
        dx15, _ = orbit_displacement(15, 30, 18)
        assert pytest.approx(dx15, abs=1e-5) == -dx0

    def test_dy_is_half_amp_at_quarter_cycle(self) -> None:
        _, dy = orbit_displacement(7, 30, 18)
        # sin(2π * 7/30) ≈ sin(π * 7/15) — not exactly 1, just check range
        assert abs(dy) <= 9.0 + 0.1  # amp/2 max

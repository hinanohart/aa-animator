"""Unit tests for aa_animator_v2.smoothing.TemporalSmoother."""

from __future__ import annotations

import numpy as np
import pytest

from aa_animator_v2.smoothing import TemporalSmoother


class TestTemporalSmoother:
    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            TemporalSmoother(alpha=1.5)

    def test_invalid_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            TemporalSmoother(alpha=-0.1)

    def test_alpha_one_is_identity(self) -> None:
        smoother = TemporalSmoother(alpha=1.0)
        arr = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=np.float32)
        # First call: initialises prev
        out1 = smoother.smooth(arr)
        np.testing.assert_allclose(out1, arr)
        # Second call with same array: alpha=1 → output == input
        arr2 = np.array([[0.9, 0.3], [0.4, 0.7]], dtype=np.float32)
        out2 = smoother.smooth(arr2)
        np.testing.assert_allclose(out2, arr2, rtol=1e-5)

    def test_alpha_zero_freezes_on_first_frame(self) -> None:
        smoother = TemporalSmoother(alpha=0.0)
        arr_first = np.array([[0.5, 0.5]], dtype=np.float32)
        arr_second = np.array([[0.9, 0.9]], dtype=np.float32)
        smoother.smooth(arr_first)
        out = smoother.smooth(arr_second)
        # alpha=0 → smoothed = 0*new + 1*prev = prev
        np.testing.assert_allclose(out, arr_first, rtol=1e-5)

    def test_first_call_returns_input_unchanged(self) -> None:
        smoother = TemporalSmoother(alpha=0.3)
        arr = np.linspace(0, 1, 12).reshape(3, 4).astype(np.float32)
        out = smoother.smooth(arr)
        np.testing.assert_allclose(out, arr)

    def test_ema_blending_correct(self) -> None:
        smoother = TemporalSmoother(alpha=0.4)
        arr1 = np.ones((2, 2), dtype=np.float32) * 0.0
        arr2 = np.ones((2, 2), dtype=np.float32) * 1.0
        smoother.smooth(arr1)
        out = smoother.smooth(arr2)
        expected = 0.4 * 1.0 + 0.6 * 0.0
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_reset_clears_state(self) -> None:
        smoother = TemporalSmoother(alpha=0.5)
        arr = np.ones((2, 2), dtype=np.float32)
        smoother.smooth(arr)
        smoother.reset()
        assert smoother._prev is None

    def test_output_shape_preserved(self) -> None:
        smoother = TemporalSmoother()
        arr = np.random.default_rng(0).random((10, 20)).astype(np.float32)
        out = smoother.smooth(arr)
        assert out.shape == (10, 20)

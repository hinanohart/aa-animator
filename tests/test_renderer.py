"""Unit tests for aa_animator_v2.renderer."""

from __future__ import annotations

import numpy as np
import pytest

from aa_animator_v2.renderer import (
    BRAILLE_BASE,
    NCHARS,
    FrameRenderer,
    _brightness_to_braille_bits_vectorised,
    brightness_to_braille,
)


class TestBrailleLookup:
    def test_zero_brightness_gives_empty_braille(self) -> None:
        ch = brightness_to_braille(0.0)
        assert ch == chr(BRAILLE_BASE), f"expected U+2800, got {ch!r}"

    def test_full_brightness_gives_max_dots(self) -> None:
        ch = brightness_to_braille(1.0)
        # all 8 dots → bits 0,3,1,4,2,5,6,7 all set → 0xFF
        assert ch == chr(BRAILLE_BASE + 0xFF), f"expected U+28FF, got {ch!r}"

    def test_half_brightness_gives_four_dots(self) -> None:
        ch = brightness_to_braille(0.5)
        # round(0.5 * 8) = 4 → dots 0,3,1,4 → bits 0,1,3,4 → 0x01|0x02|0x08|0x10 = 0x1B
        expected_bits = 0x01 | 0x02 | 0x08 | 0x10
        assert ch == chr(BRAILLE_BASE + expected_bits), (
            f"expected {chr(BRAILLE_BASE + expected_bits)!r}, got {ch!r}"
        )

    def test_returns_single_char(self) -> None:
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert len(brightness_to_braille(v)) == 1

    def test_monotone_dot_count(self) -> None:
        """More brightness → more bits set (monotone)."""
        prev_popcount = -1
        for step in range(9):
            bv = step / 8
            ch = brightness_to_braille(bv)
            bits = ord(ch) - BRAILLE_BASE
            popcount = bin(bits).count("1")
            assert popcount >= prev_popcount, (
                f"dot count decreased at step {step}: prev={prev_popcount} cur={popcount}"
            )
            prev_popcount = popcount


class TestVectorisedBraille:
    def test_shape_preserved(self) -> None:
        arr = np.random.default_rng(0).random((10, 20), dtype=np.float32).astype(np.float32)
        result = _brightness_to_braille_bits_vectorised(arr)
        assert result.shape == (10, 20)

    def test_values_in_range(self) -> None:
        arr = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        result = _brightness_to_braille_bits_vectorised(arr)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_zero_gives_zero_bits(self) -> None:
        arr = np.zeros((5, 5), dtype=np.float32)
        result = _brightness_to_braille_bits_vectorised(arr)
        assert (result == 0).all()

    def test_one_gives_0xff(self) -> None:
        arr = np.ones((5, 5), dtype=np.float32)
        result = _brightness_to_braille_bits_vectorised(arr)
        assert (result == 0xFF).all()


class TestFrameRenderer:
    def _make_renderer(self, mode: str = "braille") -> FrameRenderer:
        return FrameRenderer(mode=mode, cell_w=4, cell_h=8, font_size=10, glow=False)  # type: ignore[arg-type]

    def test_instantiation_braille(self) -> None:
        r = self._make_renderer("braille")
        assert r.mode == "braille"
        assert r._braille_bitmaps is not None
        assert r._braille_bitmaps.shape == (256, 8, 4, 3)

    def test_instantiation_ascii(self) -> None:
        r = self._make_renderer("ascii")
        assert r.mode == "ascii"
        assert r._braille_bitmaps is None
        assert r._ascii_bitmaps.shape == (NCHARS, 2, 8, 4, 3)

    def test_render_frame_braille_output_size(self) -> None:
        r = self._make_renderer("braille")
        rows, cols = 10, 20
        brightness = np.full((rows, cols), 0.5, dtype=np.float32)
        edge = np.zeros((rows, cols), dtype=bool)
        img = r.render_frame(brightness, edge)
        assert img.size == (cols * 4, rows * 8)

    def test_render_frame_ascii_output_size(self) -> None:
        r = self._make_renderer("ascii")
        rows, cols = 5, 10
        brightness = np.random.default_rng(1).random((rows, cols)).astype(np.float32)
        edge = brightness > 0.5
        img = r.render_frame(brightness, edge)
        assert img.size == (cols * 4, rows * 8)

    def test_black_bg_masks_foreground(self) -> None:
        r = self._make_renderer("braille")
        rows, cols = 4, 8
        brightness = np.ones((rows, cols), dtype=np.float32)
        edge = np.zeros((rows, cols), dtype=bool)
        mask = np.zeros((rows, cols), dtype=bool)
        img = r.render_frame(brightness, edge, mask_cell=mask, bg="black")
        arr = np.array(img)
        # All pixels should be black (masked out)
        assert arr.sum() == 0, f"expected all zeros, got sum={arr.sum()}"

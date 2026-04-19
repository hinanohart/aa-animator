"""Unit tests for aa_animator_v2.dither (Bayer ordered dither)."""

from __future__ import annotations

import numpy as np

from aa_animator_v2.dither import apply_bayer


class TestApplyBayer:
    def test_output_shape_preserved(self) -> None:
        arr = np.full((8, 12), 0.5, dtype=np.float32)
        out = apply_bayer(arr)
        assert out.shape == arr.shape

    def test_output_dtype_float32(self) -> None:
        arr = np.full((4, 4), 0.5, dtype=np.float32)
        out = apply_bayer(arr)
        assert out.dtype == np.float32

    def test_output_clipped_to_unit_range(self) -> None:
        # Test with extremes: 0.0 and 1.0 inputs
        arr_lo = np.zeros((8, 8), dtype=np.float32)
        arr_hi = np.ones((8, 8), dtype=np.float32)
        assert float(apply_bayer(arr_lo).min()) >= 0.0
        assert float(apply_bayer(arr_hi).max()) <= 1.0

    def test_not_identity(self) -> None:
        # Bayer dither must change at least some values (uniform 0.5 gets perturbed)
        arr = np.full((8, 8), 0.5, dtype=np.float32)
        out = apply_bayer(arr)
        assert not np.allclose(out, arr), "apply_bayer should perturb values"

    def test_tiling_works_for_non_multiple_size(self) -> None:
        # 7x9 is not a multiple of 4 — tiling must not raise
        arr = np.full((7, 9), 0.3, dtype=np.float32)
        out = apply_bayer(arr)
        assert out.shape == (7, 9)

    def test_zero_strength_is_identity(self) -> None:
        arr = np.full((4, 4), 0.5, dtype=np.float32)
        out = apply_bayer(arr, strength=0.0)
        np.testing.assert_allclose(out, arr)

    def test_mean_distortion_near_zero(self) -> None:
        # The Bayer matrix is symmetric around 0 → mean perturbation ≈ 0
        arr = np.full((100, 100), 0.5, dtype=np.float32)
        out = apply_bayer(arr, strength=1.0 / 8.0)
        mean_diff = float(np.abs(out - arr).mean())
        # With 4x4 Bayer the mean absolute noise is 1/(8*16*2)*sum(|matrix|) ≈ 0.023
        assert mean_diff < 0.1


class TestDitherInRenderer:
    """Integration: FrameRenderer with dither='bayer' produces different output."""

    def test_bayer_dither_changes_quantised_bits(self) -> None:
        """Bayer dither perturbs cell brightness so that quantised bit patterns differ.

        Note: rendered pixel arrays may look identical on systems where the
        Braille font renders all dot patterns with the same bitmap (e.g. DejaVu
        fallback). We therefore test the intermediate bit-pattern level which is
        the observable effect of dithering.
        """
        from aa_animator_v2.dither import apply_bayer
        from aa_animator_v2.renderer import _brightness_to_braille_bits_vectorised

        rows, cols = 10, 20
        # Use a value near a quantisation boundary so Bayer noise crosses it.
        brightness = np.full((rows, cols), 3.5 / 8.0, dtype=np.float32)
        dithered = apply_bayer(brightness.copy())

        bits_none = _brightness_to_braille_bits_vectorised(brightness)
        bits_bayer = _brightness_to_braille_bits_vectorised(dithered)

        assert not np.array_equal(bits_none, bits_bayer), (
            "Bayer dither should produce different braille bit patterns near quantisation boundary"
        )

    def test_none_dither_is_default(self) -> None:
        from aa_animator_v2.renderer import FrameRenderer

        r = FrameRenderer(mode="braille", cell_w=4, cell_h=8, font_size=10, glow=False)
        assert r.dither == "none"


class TestAAAnimatorDither:
    """Smoke: AAAnimator accepts dither parameter and renders."""

    def test_animator_accepts_dither_bayer(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from PIL import Image

        from aa_animator_v2 import AAAnimator

        img = Image.new("RGB", (64, 64), color=(128, 64, 200))
        img_path = tmp_path / "dither_test.png"
        img.save(img_path)

        a = AAAnimator(mode="braille", cols=20, n_frames=2, glow=False, dither="bayer")  # type: ignore[arg-type]
        a.load_image(img_path)
        raw = a.generate_frames()
        pil_frames = a.render_frames(raw)

        assert len(pil_frames) == 2
        for frame in pil_frames:
            assert isinstance(frame, Image.Image)

    def test_animator_default_dither_is_none(self) -> None:
        from aa_animator_v2 import AAAnimator

        a = AAAnimator()
        assert a.dither == "none"

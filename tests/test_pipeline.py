"""Smoke tests for aa_animator_v2.pipeline.AAAnimator."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aa_animator_v2 import AAAnimator


class TestAAAnimatorInstantiation:
    def test_default_init(self) -> None:
        a = AAAnimator()
        assert a.mode == "braille"
        assert a.bg == "black"
        assert a.fps == 30
        assert a.cols == 100
        assert a.n_frames == 30

    def test_ascii_mode_init(self) -> None:
        a = AAAnimator(mode="ascii", bg="ghostty_fill", fps=24, cols=80)
        assert a.mode == "ascii"
        assert a.bg == "ghostty_fill"
        assert a.fps == 24
        assert a.cols == 80

    def test_repr_contains_state(self) -> None:
        a = AAAnimator()
        r = repr(a)
        assert "no image" in r
        assert "braille" in r


class TestAAAnimatorLoadImage:
    def test_load_missing_file_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        a = AAAnimator()
        with pytest.raises(FileNotFoundError):
            a.load_image("/nonexistent/path/image.png")

    def test_load_real_image(self, tmp_path) -> None:
        # Create a tiny synthetic PNG
        img = Image.new("RGB", (64, 64), color=(128, 64, 200))
        img_path = tmp_path / "test_input.png"
        img.save(img_path)

        a = AAAnimator(cols=20)
        a.load_image(img_path)
        assert a._img_np is not None
        assert a._img_np.dtype == np.float32
        assert a._img_np.min() >= 0.0
        assert a._img_np.max() <= 1.0

    def test_repr_after_load(self, tmp_path) -> None:
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        img_path = tmp_path / "tiny.png"
        img.save(img_path)

        a = AAAnimator(cols=10)
        a.load_image(img_path)
        r = repr(a)
        assert "no image" not in r
        assert "loaded" in r


class TestAAAnimatorEstimateDepth:
    def test_depth_returns_correct_shape(self, tmp_path) -> None:
        img_pil = Image.new("RGB", (64, 64), (100, 150, 200))
        a = AAAnimator(cols=16)
        a._img_h = 64
        a._img_w = 64
        depth = a.estimate_depth(img_pil)
        assert depth.shape == (64, 64)

    def test_depth_values_in_range(self, tmp_path) -> None:
        img_pil = Image.new("RGB", (32, 32), (200, 100, 50))
        a = AAAnimator(cols=8)
        a._img_h = 32
        a._img_w = 32
        depth = a.estimate_depth(img_pil)
        assert float(depth.min()) >= 0.0
        assert float(depth.max()) <= 1.0 + 1e-5

    def test_depth_is_float32(self, tmp_path) -> None:
        img_pil = Image.new("RGB", (32, 32), (50, 50, 50))
        a = AAAnimator(cols=8)
        a._img_h = 32
        a._img_w = 32
        depth = a.estimate_depth(img_pil)
        assert depth.dtype == np.float32


class TestAAAnimatorGenerateFrames:
    def test_generate_returns_correct_count(self, tmp_path) -> None:
        img = Image.new("RGB", (64, 64), (200, 100, 50))
        img_path = tmp_path / "gen_test.png"
        img.save(img_path)

        a = AAAnimator(cols=16, n_frames=5)
        a.load_image(img_path)
        frames = a.generate_frames()
        assert len(frames) == 5
        assert frames[0].shape == (a._img_np.shape[0], a._img_np.shape[1], 3)

    def test_generate_before_load_raises(self) -> None:
        a = AAAnimator()
        with pytest.raises(RuntimeError, match="load_image"):
            a.generate_frames()

    def test_generate_override_n_frames(self, tmp_path) -> None:
        img = Image.new("RGB", (32, 32), (0, 128, 255))
        img_path = tmp_path / "override.png"
        img.save(img_path)

        a = AAAnimator(cols=8, n_frames=10)
        a.load_image(img_path)
        frames = a.generate_frames(n_frames=3)
        assert len(frames) == 3


class TestAAAnimatorRenderFrames:
    def test_render_frames_produces_pil_images(self, tmp_path) -> None:
        img = Image.new("RGB", (64, 64), (180, 80, 40))
        img_path = tmp_path / "render_test.png"
        img.save(img_path)

        a = AAAnimator(mode="braille", cols=20, n_frames=2, glow=False)
        a.load_image(img_path)
        raw = a.generate_frames()
        pil_frames = a.render_frames(raw)

        assert len(pil_frames) == 2
        for frame in pil_frames:
            assert isinstance(frame, Image.Image)
            w, h = frame.size
            assert w == a.cols * a._cell_w
            assert h == a._rows * a._cell_h

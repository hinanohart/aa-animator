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
"""High-level :class:`AAAnimator` orchestrator.

Responsibilities are delegated to sibling modules:

* :mod:`._segmentation` — rembg / Otsu foreground masking
* :mod:`._depth`        — Depth Anything V2 Small inference + cache
* :mod:`._rendering`    — cell-level brightness stretch and hole fill
* :mod:`._encoding`     — ffmpeg MP4 subprocess
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

from aa_animator_v2.parallax import (
    dynamic_amp_px,
    fill_holes,
    forward_warp,
    orbit_displacement,
    warp_mask,
)
from aa_animator_v2.renderer import EDGE_THRESH, DitherMode, FrameRenderer, RenderMode
from aa_animator_v2.smoothing import TemporalSmoother

from ._depth import estimate_depth as _estimate_depth_impl
from ._encoding import export_mp4 as _export_mp4_impl
from ._rendering import fill_cell_holes, stretch_fg_brightness
from ._segmentation import segment_subject as _segment_subject_impl

BgMode = Literal["black", "ghostty_fill"]


class AAAnimator:
    """Still-image to animated ASCII art pipeline.

    Args:
        mode: Rendering mode — ``"braille"`` (default) or ``"ascii"``.
        bg: Background mode — ``"black"`` (subject-only) or ``"ghostty_fill"``.
        fps: Output video frame rate.
        cols: Character canvas width in cells.
        n_frames: Total animation frames.
        amp_px: Forward-warp parallax amplitude in pixels (stub for Day 2).
        glow: Enable Ghostty-style blue edge glow.
    """

    def __init__(
        self,
        mode: RenderMode = "braille",
        bg: BgMode = "black",
        fps: int = 30,
        cols: int = 100,
        n_frames: int = 30,
        amp_px: float = 18.0,
        *,
        glow: bool = True,
        dither: DitherMode = "none",
    ) -> None:
        self.mode = mode
        self.bg = bg
        self.fps = fps
        self.cols = cols
        self.n_frames = n_frames
        self.amp_px = amp_px
        self.glow = glow
        self.dither = dither

        # Derived geometry
        if mode == "braille":
            self._cell_w, self._cell_h = 4, 8
            self._font_size = 10
        else:
            self._cell_w, self._cell_h = 8, 16
            self._font_size = 14

        self._rows: int | None = None
        self._img_w: int | None = None
        self._img_h: int | None = None

        # State set by load_image / segment_subject
        self._img_np: np.ndarray | None = None      # (H, W, 3) float32 [0-1]
        self._fg_mask: np.ndarray | None = None     # (H, W) bool
        self._depth: np.ndarray | None = None       # (H, W) float32 [0-1]

        self._renderer: FrameRenderer | None = None
        self._smoother: TemporalSmoother = TemporalSmoother(alpha=0.3)
        self._edge_smoother: TemporalSmoother = TemporalSmoother(alpha=0.3)

    # ---------------------------------------------------------------------- repr

    def __repr__(self) -> str:
        state = "no image" if self._img_np is None else f"loaded {self._img_w}x{self._img_h}px"
        return (
            f"AAAnimator(mode={self.mode!r}, bg={self.bg!r}, fps={self.fps}, "
            f"cols={self.cols}, n_frames={self.n_frames}, state={state!r})"
        )

    # ---------------------------------------------------------------------- I/O

    def load_image(self, path: str | Path) -> None:
        """Load and resize an image to the renderer canvas dimensions.

        Args:
            path: Path to the input image file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            OSError: If the file cannot be opened as an image.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        # Derive canvas dimensions from cols
        rows = round(self.cols * 0.41)  # ~100:41 aspect ratio
        self._rows = rows
        self._img_w = self.cols * self._cell_w
        self._img_h = rows * self._cell_h

        img = Image.open(path).convert("RGB").resize(
            (self._img_w, self._img_h), Image.LANCZOS
        )
        self._img_np = np.array(img, dtype=np.float32) / 255.0
        self._fg_mask = None
        self._depth = None

    # ------------------------------------------------------------------ segment

    def segment_subject(self, image: Image.Image) -> np.ndarray:
        """Extract foreground mask via rembg (u2net).

        Falls back to luminance Otsu if rembg is unavailable or fails.

        Args:
            image: PIL RGB image at renderer resolution.

        Returns:
            Boolean mask array of shape (H, W).
        """
        return _segment_subject_impl(image)

    # ------------------------------------------------------------------- depth

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth map using Depth Anything V2 Small (Apache-2.0).

        See :func:`aa_animator_v2.pipeline._depth.estimate_depth` for
        caching and fallback behaviour.

        Args:
            image: PIL RGB image at renderer resolution.

        Returns:
            Array of shape (H, W) float32 with values in [0, 1].
            Higher values indicate closer (foreground) objects.
        """
        h = self._img_h or image.height
        w = self._img_w or image.width
        return _estimate_depth_impl(image, target_size=(w, h))

    # ------------------------------------------------------------------ frames

    def generate_frames(self, n_frames: int | None = None) -> list[np.ndarray]:
        """Generate parallax-warped pixel frames using depth-driven orbit motion.

        If depth has been estimated (``self._depth`` is not None), applies
        forward-warp parallax with dynamically scaled amplitude.  Falls back
        to repeating the base image when depth is unavailable.

        Args:
            n_frames: Override ``self.n_frames``.

        Returns:
            List of (H, W, 3) float32 arrays in [0, 1].

        Raises:
            RuntimeError: If :meth:`load_image` has not been called.
        """
        if self._img_np is None:
            raise RuntimeError("Call load_image() before generate_frames().")

        count = n_frames if n_frames is not None else self.n_frames

        if self._depth is None:
            return [self._img_np.copy() for _ in range(count)]

        # Dynamic AMP based on fg_coverage
        fg_coverage = float(self._fg_mask.mean()) if self._fg_mask is not None else 0.3
        amp = dynamic_amp_px(fg_coverage)

        frames: list[np.ndarray] = []
        for t in range(count):
            dx, dy = orbit_displacement(t, count, amp)
            warped = forward_warp(self._img_np, self._depth, dx, dy)

            # Pixel-level hole fill for scatter gaps
            hole = warped.sum(axis=2) < 1e-6
            if hole.any():
                warped = fill_holes(warped, hole)

            frames.append(warped)

        # Store last-used amp for external validation
        self._last_amp_px: int = amp
        self._last_fg_coverage: float = fg_coverage
        return frames

    def render_frames(
        self,
        frames: list[np.ndarray],
        warped_masks: list[np.ndarray] | None = None,
    ) -> list[Image.Image]:
        """Convert raw pixel frames to rendered AA images.

        Applies fg-only histogram stretch, temporal EMA smoothing, and
        cell-level uniform_filter hole-fill post-processing.

        Args:
            frames: List of (H, W, 3) float32 arrays in [0, 1].
            warped_masks: Optional per-frame (H, W) bool masks from
                ``warp_mask``.  Falls back to ``self._fg_mask`` when None.

        Returns:
            List of PIL Images ready for ffmpeg encoding.
        """
        if self._renderer is None:
            self._renderer = FrameRenderer(
                mode=self.mode,
                cell_w=self._cell_w,
                cell_h=self._cell_h,
                font_size=self._font_size,
                glow=self.glow,
                dither=self.dither,
            )

        rows = self._rows or 41
        cols = self.cols
        cell_h = self._cell_h
        cell_w = self._cell_w
        self._smoother.reset()
        self._edge_smoother.reset()
        rendered: list[Image.Image] = []

        for idx, frame_np in enumerate(frames):
            frame_mask = self._pick_frame_mask(warped_masks, idx)

            # Luminance → cell brightness
            gray = 0.299 * frame_np[:, :, 0] + 0.587 * frame_np[:, :, 1] + 0.114 * frame_np[:, :, 2]
            cell_brightness = gray.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))

            # Foreground mask at cell level
            mask_cell: np.ndarray | None = None
            if frame_mask is not None:
                mask_cell = (
                    frame_mask.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3)) > 0.3
                )

            # Cell-level hole fill post-process (handles residual gaps)
            if mask_cell is not None:
                cell_brightness = fill_cell_holes(
                    cell_brightness, frame_np, mask_cell, (rows, cell_h, cols, cell_w),
                )

            # Histogram stretch — fg cells only when mask available
            cell_brightness = stretch_fg_brightness(cell_brightness, mask_cell)

            # Temporal EMA smoothing
            cell_brightness = self._smoother.smooth(cell_brightness)

            # Sobel edge map → cell (with EMA smoothing to stabilise glow mask)
            edge_cell = self._compute_edge_cell(gray, rows, cell_h, cols, cell_w)

            img = self._renderer.render_frame(
                cell_brightness=cell_brightness,
                edge_cell=edge_cell,
                mask_cell=mask_cell,
                bg=self.bg,
            )
            rendered.append(img)

        return rendered

    # ------------------------------------------------------------------ helpers

    def _pick_frame_mask(
        self,
        warped_masks: list[np.ndarray] | None,
        idx: int,
    ) -> np.ndarray | None:
        """Return per-frame warped mask or fall back to the static fg mask."""
        if warped_masks is not None and idx < len(warped_masks):
            return warped_masks[idx]
        return self._fg_mask

    def _compute_edge_cell(
        self,
        gray: np.ndarray,
        rows: int,
        cell_h: int,
        cols: int,
        cell_w: int,
    ) -> np.ndarray:
        """Sobel magnitude → cell-averaged → EMA-smoothed → thresholded mask."""
        sx = scipy_sobel(gray, axis=1)
        sy = scipy_sobel(gray, axis=0)
        mag = np.hypot(sx, sy)
        if mag.max() > 0:
            mag /= mag.max()
        edge_cell_raw = mag.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))
        edge_cell_smoothed = self._edge_smoother.smooth(edge_cell_raw)
        return edge_cell_smoothed > EDGE_THRESH

    # ------------------------------------------------------------------ export

    def export_mp4(self, output_path: str | Path, frames: list[Image.Image]) -> None:
        """Encode rendered PIL frames to an MP4 via ffmpeg subprocess.

        Args:
            output_path: Destination .mp4 file path.
            frames: List of same-size PIL RGB Images.

        Raises:
            RuntimeError: If ffmpeg exits with a non-zero return code.
            FileNotFoundError: If ffmpeg is not found on PATH.
        """
        _export_mp4_impl(output_path, frames, fps=self.fps)

    # ------------------------------------------------------------------ private run helpers

    def _prepare_run(
        self, input_path: str | Path
    ) -> tuple[Image.Image, list[np.ndarray] | None]:
        """Load image, run segmentation + depth, compute warped masks.

        Shared setup for :meth:`animate`, :meth:`preview`, and :meth:`bake`.

        Args:
            input_path: Path to the input image.

        Returns:
            Tuple of (img_pil, warped_masks).
            ``warped_masks`` is ``None`` when fg_mask or depth is unavailable,
            otherwise a list of per-frame boolean mask arrays.
        """
        self.load_image(input_path)
        assert self._img_np is not None
        img_pil = Image.fromarray((self._img_np * 255).astype(np.uint8))

        if self.bg in ("black", "ghostty_fill"):
            self._fg_mask = self.segment_subject(img_pil)

        self._depth = self.estimate_depth(img_pil)

        warped_masks: list[np.ndarray] | None = None
        if self._fg_mask is not None and self._depth is not None:
            fg_coverage = float(self._fg_mask.mean())
            amp = dynamic_amp_px(fg_coverage)
            count = self.n_frames
            warped_masks = [
                warp_mask(self._fg_mask, self._depth, *orbit_displacement(t, count, amp))
                for t in range(count)
            ]

        return img_pil, warped_masks

    # ------------------------------------------------------------------ convenience

    def animate(self, input_path: str | Path, output_path: str | Path) -> None:
        """End-to-end: load → segment → depth → generate → render → export.

        Args:
            input_path: Path to the input image.
            output_path: Path for the output .mp4.
        """
        _img_pil, warped_masks = self._prepare_run(input_path)
        raw_frames = self.generate_frames()
        pil_frames = self.render_frames(raw_frames, warped_masks=warped_masks)
        self.export_mp4(output_path, pil_frames)
        print(f"[aa-animator] written: {output_path}", file=sys.stderr)

    def preview(self, input_path: str | Path, output_path: str | Path) -> None:
        """Render a single-frame PNG preview.

        Args:
            input_path: Path to the input image.
            output_path: Destination PNG path.
        """
        _img_pil, _warped_masks = self._prepare_run(input_path)
        raw_frames = self.generate_frames(n_frames=1)
        pil_frames = self.render_frames(raw_frames)
        output_path = Path(output_path)
        pil_frames[0].save(str(output_path))
        print(f"[aa-animator] preview written: {output_path}", file=sys.stderr)

    def bake(self, input_path: str | Path, out_dir: str | Path) -> None:
        """Render all animation frames as individual PNG files.

        Output files are named ``frame_NNNN.png`` inside *out_dir*.

        Args:
            input_path: Path to the input image.
            out_dir: Output directory (created if absent).
        """
        _img_pil, warped_masks = self._prepare_run(input_path)
        raw_frames = self.generate_frames()
        pil_frames = self.render_frames(raw_frames, warped_masks=warped_masks)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(pil_frames):
            frame.save(str(out_dir / f"frame_{i:04d}.png"))
        print(f"[aa-animator] baked {len(pil_frames)} frames to {out_dir}", file=sys.stderr)


__all__ = ["AAAnimator", "BgMode"]

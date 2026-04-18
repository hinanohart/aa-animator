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
"""Main pipeline: image → AA animation frames → MP4.

Day 1 scope: load_image, segment_subject (rembg), render_frames, export_mp4.
Day 2 scope: estimate_depth (DA-V2 Small), forward-warp parallax, pixel hole fill,
             temporal EMA smoothing, fg-only histogram stretch, ghostty_fill bg-dot,
             preview / bake CLI commands.
Day 3 scope: Bayer dither mode, ffmpeg optimisation flags, DRY _prepare_run helper.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

from aa_animator_v2.parallax import dynamic_amp_px, fill_holes, forward_warp, orbit_displacement, warp_mask
from aa_animator_v2.renderer import EDGE_THRESH, NCHARS, DitherMode, FrameRenderer, RenderMode
from aa_animator_v2.smoothing import TemporalSmoother

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical model ID for the Apache-2.0 Depth Anything V2 Small checkpoint.
# Defined once here to prevent drift between code and docs/legal-notes.md.
_DA_V2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

BgMode = Literal["black", "ghostty_fill"]


# ---------------------------------------------------------------------------
# AAAnimator
# ---------------------------------------------------------------------------


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
        self._depth: np.ndarray | None = None       # (H, W) float32 [0-1]  — stub

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
        rows = int(round(self.cols * 0.41))  # ~100:41 aspect ratio
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
        try:
            from rembg import remove as rembg_remove  # type: ignore[import-not-found]
        except ImportError:
            return self._otsu_fg_mask(image)

        try:
            rgba = rembg_remove(image)
            alpha = np.array(rgba, dtype=np.float32)[:, :, 3]
            thresh = self._otsu_threshold(alpha.astype(np.uint8))
            mask = alpha > thresh
            if mask.mean() < 0.05:  # coverage too low → fallback
                return self._otsu_fg_mask(image)
            return mask
        except Exception:
            return self._otsu_fg_mask(image)

    def _otsu_fg_mask(self, image: Image.Image) -> np.ndarray:
        gray = np.array(image.convert("L"), dtype=np.float32)
        thresh = self._otsu_threshold(gray.astype(np.uint8))
        return gray > thresh

    @staticmethod
    def _otsu_threshold(gray_u8: np.ndarray) -> float:
        hist, _ = np.histogram(gray_u8.flatten(), bins=256, range=(0, 256))
        total = gray_u8.size
        sum_total = float(np.dot(np.arange(256), hist))
        sum_bg = 0.0
        weight_bg = 0
        best_thresh = 128.0
        best_var = 0.0
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var > best_var:
                best_var = var
                best_thresh = float(t)
        return best_thresh

    # ------------------------------------------------------------------- depth

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth map using Depth Anything V2 Small (Apache-2.0).

        Model is loaded via ``transformers.pipeline`` and result is cached
        in ``~/.cache/aa_animator/depth/<input_hash>.npy`` to avoid repeated
        inference for the same image.

        Falls back to a uniform 0.5 map if transformers or the model are
        unavailable so the pipeline stays functional without the optional
        depth dependency.

        Args:
            image: PIL RGB image at renderer resolution.

        Returns:
            Array of shape (H, W) float32 with values in [0, 1].
            Higher values indicate closer (foreground) objects.
        """
        h = self._img_h or image.height
        w = self._img_w or image.width
        target_size = (w, h)

        # Cache key from image bytes
        img_bytes = np.array(image, dtype=np.uint8).tobytes()
        cache_key = hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()  # noqa: S324
        cache_dir = Path.home() / ".cache" / "aa_animator" / "depth"
        cache_path = cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            depth_raw = np.load(str(cache_path))
            return _normalize_depth(depth_raw, target_size)

        try:
            import torch  # type: ignore[import-not-found]
            from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            depth_pipe = hf_pipeline(
                task="depth-estimation",
                model=_DA_V2_MODEL_ID,
                device=device,
            )
            result = depth_pipe(image)
            depth_pil: Image.Image = result["depth"]
            depth_raw = np.array(depth_pil, dtype=np.float32)

            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), depth_raw)
            return _normalize_depth(depth_raw, target_size)

        except Exception:
            # Graceful fallback: uniform mid-depth keeps the pipeline running
            return np.full((h, w), 0.5, dtype=np.float32)

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
        if self._fg_mask is not None:
            fg_coverage = float(self._fg_mask.mean())
        else:
            fg_coverage = 0.3
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
        from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]

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
            # Per-frame warped mask (from parallax warp) or static fg_mask
            frame_mask: np.ndarray | None = None
            if warped_masks is not None and idx < len(warped_masks):
                frame_mask = warped_masks[idx]
            elif self._fg_mask is not None:
                frame_mask = self._fg_mask

            # Luminance → cell brightness
            gray = 0.299 * frame_np[:, :, 0] + 0.587 * frame_np[:, :, 1] + 0.114 * frame_np[:, :, 2]
            cell_brightness = (
                gray.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))
            )

            # Foreground mask at cell level
            mask_cell: np.ndarray | None = None
            if frame_mask is not None:
                mask_cell = (
                    frame_mask.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3)) > 0.3
                )

            # Cell-level hole fill post-process (handles residual gaps)
            if mask_cell is not None:
                rgb_cell = (
                    frame_np.reshape(rows, cell_h, cols, cell_w, 3).mean(axis=(1, 3))
                )
                rgb_sum = rgb_cell.sum(axis=2)
                hole_cell = (~mask_cell) & (rgb_sum < (10.0 / 255.0 * 3))
                if hole_cell.any():
                    cb_nb = uniform_filter(cell_brightness, size=3, mode="nearest")
                    cell_brightness[hole_cell] = cb_nb[hole_cell]
                    still_hole = hole_cell & (cell_brightness < 0.001)
                    if still_hole.any():
                        cb_nb2 = uniform_filter(cell_brightness, size=5, mode="nearest")
                        cell_brightness[still_hole] = cb_nb2[still_hole]

            # Histogram stretch — fg cells only when mask available
            cell_brightness = self._stretch_fg_brightness(cell_brightness, mask_cell)

            # Temporal EMA smoothing
            cell_brightness = self._smoother.smooth(cell_brightness)

            # Sobel edge map → cell (with EMA smoothing to stabilise glow mask)
            sx = scipy_sobel(gray, axis=1)
            sy = scipy_sobel(gray, axis=0)
            mag = np.hypot(sx, sy)
            if mag.max() > 0:
                mag /= mag.max()
            edge_cell_raw = mag.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))
            edge_cell_smoothed = self._edge_smoother.smooth(edge_cell_raw)
            edge_cell = edge_cell_smoothed > EDGE_THRESH

            img = self._renderer.render_frame(
                cell_brightness=cell_brightness,
                edge_cell=edge_cell,
                mask_cell=mask_cell,
                bg=self.bg,
            )
            rendered.append(img)

        return rendered

    # ------------------------------------------------------------------ helpers

    def _stretch_fg_brightness(
        self,
        cell_brightness: np.ndarray,
        mask_cell: np.ndarray | None,
    ) -> np.ndarray:
        """Histogram-stretch brightness using foreground cells only.

        When *mask_cell* is provided, percentile clipping is computed solely
        from foreground cells so that background values do not compress the
        useful dynamic range.

        Args:
            cell_brightness: (ROWS, COLS) float32 in [0, 1].
            mask_cell: (ROWS, COLS) bool or None.

        Returns:
            Stretched (ROWS, COLS) float32 in [0, 1].
        """
        if mask_cell is not None and mask_cell.any():
            fg_vals = cell_brightness[mask_cell]
            if len(fg_vals) > 10:
                p2 = float(np.percentile(fg_vals, 2))
                p98 = float(np.percentile(fg_vals, 98))
            else:
                p2 = float(np.percentile(cell_brightness, 2))
                p98 = float(np.percentile(cell_brightness, 98))
        else:
            p2 = float(np.percentile(cell_brightness, 2))
            p98 = float(np.percentile(cell_brightness, 98))

        if p98 > p2:
            stretched = np.clip((cell_brightness - p2) / (p98 - p2), 0.0, 1.0)
            # Re-apply only to mask cells; leave background intact
            if mask_cell is not None:
                out = cell_brightness.copy()
                out[mask_cell] = stretched[mask_cell]
                return out
            return stretched
        return cell_brightness

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
        if not frames:
            raise ValueError("frames list is empty.")

        output_path = Path(output_path)
        canvas_w, canvas_h = frames[0].size

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{canvas_w}x{canvas_h}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-g", "15",
            "-keyint_min", "15",
            "-tune", "animation",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.array(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-500:]}")

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
            warped_masks = []
            for t in range(count):
                dx, dy = orbit_displacement(t, count, amp)
                warped_masks.append(warp_mask(self._fg_mask, self._depth, dx, dy))

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_depth(depth_raw: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize and normalise a raw depth array to [0, 1].

    Args:
        depth_raw: 2-D float32 array from the depth model.
        target_size: ``(width, height)`` target dimensions.

    Returns:
        Normalised float32 array of shape ``(height, width)``.
    """
    w, h = target_size
    pil_depth = Image.fromarray(depth_raw).resize((w, h), Image.BILINEAR)
    arr = np.array(pil_depth, dtype=np.float32)
    dmin, dmax = arr.min(), arr.max()
    if dmax > dmin:
        return (arr - dmin) / (dmax - dmin)
    return np.zeros_like(arr)

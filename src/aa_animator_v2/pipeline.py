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
Depth estimation and parallax warp are stubs (Day 2).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

from aa_animator_v2.renderer import EDGE_THRESH, NCHARS, FrameRenderer, RenderMode

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
    ) -> None:
        self.mode = mode
        self.bg = bg
        self.fps = fps
        self.cols = cols
        self.n_frames = n_frames
        self.amp_px = amp_px
        self.glow = glow

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
        """STUB — returns uniform depth map.  Day 2 will integrate Depth Anything V2.

        Args:
            image: Unused in stub; accepted for API compatibility.

        Returns:
            Array of shape (H, W) filled with 0.5 (flat parallax).
        """
        h = self._img_h or image.height
        w = self._img_w or image.width
        return np.full((h, w), 0.5, dtype=np.float32)

    # ------------------------------------------------------------------ frames

    def generate_frames(self, n_frames: int | None = None) -> list[np.ndarray]:
        """STUB — returns repeated copies of the base image.  Day 2 adds parallax warp.

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
        return [self._img_np.copy() for _ in range(count)]

    def render_frames(self, frames: list[np.ndarray]) -> list[Image.Image]:
        """Convert raw pixel frames to rendered AA images.

        Args:
            frames: List of (H, W, 3) float32 arrays in [0, 1].

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
            )

        rows = self._rows or 41
        cols = self.cols
        cell_h = self._cell_h
        cell_w = self._cell_w
        rendered: list[Image.Image] = []

        for frame_np in frames:
            # Luminance → cell brightness
            gray = 0.299 * frame_np[:, :, 0] + 0.587 * frame_np[:, :, 1] + 0.114 * frame_np[:, :, 2]
            cell_brightness = (
                gray.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))
            )

            # Histogram stretch
            p2, p98 = np.percentile(cell_brightness, 2), np.percentile(cell_brightness, 98)
            if p98 > p2:
                cell_brightness = np.clip((cell_brightness - p2) / (p98 - p2), 0.0, 1.0)

            # Sobel edge map → cell
            sx = scipy_sobel(gray, axis=1)
            sy = scipy_sobel(gray, axis=0)
            mag = np.hypot(sx, sy)
            if mag.max() > 0:
                mag /= mag.max()
            edge_cell = (
                mag.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3)) > EDGE_THRESH
            )

            # Foreground mask at cell level
            mask_cell: np.ndarray | None = None
            if self._fg_mask is not None:
                mask_cell = (
                    self._fg_mask.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3)) > 0.3
                )

            img = self._renderer.render_frame(
                cell_brightness=cell_brightness,
                edge_cell=edge_cell,
                mask_cell=mask_cell,
                bg=self.bg,
            )
            rendered.append(img)

        return rendered

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
            "-pix_fmt", "yuv420p",
            "-crf", "18",
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

    # ------------------------------------------------------------------ convenience

    def animate(self, input_path: str | Path, output_path: str | Path) -> None:
        """End-to-end: load → segment → generate → render → export.

        Depth estimation and parallax warp are stubbed (Day 2).

        Args:
            input_path: Path to the input image.
            output_path: Path for the output .mp4.
        """
        self.load_image(input_path)
        assert self._img_np is not None
        img_pil = Image.fromarray((self._img_np * 255).astype(np.uint8))

        if self.bg in ("black", "ghostty_fill"):
            self._fg_mask = self.segment_subject(img_pil)

        raw_frames = self.generate_frames()
        pil_frames = self.render_frames(raw_frames)
        self.export_mp4(output_path, pil_frames)
        print(f"[aa-animator] written: {output_path}", file=sys.stderr)

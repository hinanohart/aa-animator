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
"""Forward-warp parallax engine for depth-driven animation.

Key design choices
------------------
* ``forward_warp`` and ``warp_mask`` use a depth-sorted scatter approach
  (deepest pixels written first so nearer pixels overwrite them).
* Dynamic AMP_PX scales inversely with fg_coverage so high-coverage subjects
  (bench 59 %) receive the same flicker budget as compact ones (bike 28 %).
* ``fill_holes`` applies two-pass pixel-level weighted fill (3x3 then 5x5)
  to repair scatter gaps before cell-level uniform_filter.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Dynamic amplitude
# ---------------------------------------------------------------------------

_BASE_AMP_PX: float = 18.0
_COVERAGE_REF: float = 0.30  # coverage at which amp = BASE


def dynamic_amp_px(fg_coverage: float) -> int:
    """Compute parallax amplitude scaled inversely with foreground coverage.

    Args:
        fg_coverage: Fraction of pixels classified as foreground in [0, 1].

    Returns:
        Integer pixel amplitude for the forward-warp orbit.
    """
    cov = max(0.05, min(1.0, fg_coverage))
    amp = _BASE_AMP_PX * max(0.1, min(1.0, _COVERAGE_REF / cov))
    return round(amp)


# ---------------------------------------------------------------------------
# Forward warp
# ---------------------------------------------------------------------------


def forward_warp(img: np.ndarray, depth: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Depth-sorted forward scatter warp for an RGB image.

    Pixels are written in ascending depth order (background first) so that
    foreground pixels (high depth) overwrite background pixels at the destination.

    Args:
        img: (H, W, 3) float32 source image in [0, 1].
        depth: (H, W) float32 depth map in [0, 1].  Higher = closer.
        dx: Horizontal displacement multiplier in pixels.
        dy: Vertical displacement multiplier in pixels.

    Returns:
        Warped (H, W, 3) float32 image.
    """
    h, w = depth.shape
    shift_x = (depth * dx).astype(np.int32)
    shift_y = (depth * dy).astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    new_y = np.clip(yy + shift_y, 0, h - 1)
    new_x = np.clip(xx + shift_x, 0, w - 1)
    # Write shallow (far) pixels first, deep (near) pixels last → near wins
    order = np.argsort(depth.flatten())
    flat_y = new_y.flatten()[order]
    flat_x = new_x.flatten()[order]
    orig_y = yy.flatten()[order]
    orig_x = xx.flatten()[order]
    out = np.zeros_like(img)
    out[flat_y, flat_x] = img[orig_y, orig_x]
    return out


def warp_mask(mask: np.ndarray, depth: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Forward-warp a boolean foreground mask using the same depth-sorted scatter.

    Args:
        mask: (H, W) bool foreground mask.
        depth: (H, W) float32 depth map in [0, 1].
        dx: Horizontal displacement multiplier in pixels.
        dy: Vertical displacement multiplier in pixels.

    Returns:
        Warped (H, W) bool mask.
    """
    h, w = depth.shape
    shift_x = (depth * dx).astype(np.int32)
    shift_y = (depth * dy).astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    new_y = np.clip(yy + shift_y, 0, h - 1)
    new_x = np.clip(xx + shift_x, 0, w - 1)
    order = np.argsort(depth.flatten())
    flat_y = new_y.flatten()[order]
    flat_x = new_x.flatten()[order]
    orig_y = yy.flatten()[order]
    orig_x = xx.flatten()[order]
    out = np.zeros((h, w), dtype=bool)
    out[flat_y, flat_x] = mask[orig_y, orig_x]
    return out


# ---------------------------------------------------------------------------
# Pixel-level hole fill
# ---------------------------------------------------------------------------


def fill_holes(img: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """Fill zero-valued scatter holes with weighted neighbour average.

    Two-pass strategy:
    * Pass 1: 3x3 neighbourhood mean for small gaps.
    * Pass 2: 5x5 neighbourhood mean for remaining large gaps.

    Args:
        img: (H, W, 3) float32 warped image — may contain zero pixels at holes.
        hole_mask: (H, W) bool — True where pixel is a hole to be filled.

    Returns:
        (H, W, 3) float32 image with holes filled.
    """
    if not hole_mask.any():
        return img

    out = img.copy()

    for kernel_size in (3, 5):
        remaining = hole_mask & (out.sum(axis=2) < 1e-6)
        if not remaining.any():
            break
        half = kernel_size // 2
        h, w = out.shape[:2]
        # Pad to handle borders
        padded = np.pad(out, ((half, half), (half, half), (0, 0)), mode="edge")
        # Build sum and count arrays
        sum_arr = np.zeros_like(out)
        count_arr = np.zeros((h, w), dtype=np.float32)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                neighbour = padded[dy : dy + h, dx : dx + w]
                is_filled = neighbour.sum(axis=2) > 1e-6
                count_arr += is_filled.astype(np.float32)
                sum_arr += neighbour * is_filled[:, :, np.newaxis]
        valid = count_arr > 0
        fill_candidates = remaining & valid
        if fill_candidates.any():
            out[fill_candidates] = sum_arr[fill_candidates] / count_arr[fill_candidates, np.newaxis]

    return out


# ---------------------------------------------------------------------------
# Orbit trajectory
# ---------------------------------------------------------------------------


def orbit_displacement(t: int, n_frames: int, amp_px: int) -> tuple[float, float]:
    """Circular parallax orbit displacement at frame *t*.

    Args:
        t: Frame index (0-based).
        n_frames: Total frame count.
        amp_px: Amplitude in pixels.

    Returns:
        ``(dx, dy)`` displacement tuple.
    """
    angle = 2 * math.pi * t / n_frames
    dx = amp_px * math.cos(angle)
    dy = amp_px / 2 * math.sin(angle)
    return dx, dy

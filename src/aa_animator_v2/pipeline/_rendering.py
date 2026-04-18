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
"""Cell-level rendering helpers.

Pure functions extracted from :class:`AAAnimator.render_frames` — foreground
histogram stretch and residual-hole post-processing via uniform filtering.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]


def stretch_fg_brightness(
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


def fill_cell_holes(
    cell_brightness: np.ndarray,
    frame_np: np.ndarray,
    mask_cell: np.ndarray,
    cell_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Fill residual gaps in cell brightness from neighbours (in-place on copy).

    Applies uniform_filter-based neighbour fill to cells that are outside the
    foreground mask and still near-black, then a second larger-radius pass for
    stubbornly dark cells.

    Args:
        cell_brightness: (ROWS, COLS) float32 brightness grid.
        frame_np: (H, W, 3) pixel frame used to detect near-black holes.
        mask_cell: (ROWS, COLS) bool foreground mask at cell resolution.
        cell_shape: ``(rows, cell_h, cols, cell_w)`` canvas geometry.

    Returns:
        Updated cell_brightness (copy) with residual gaps filled.
    """
    rows, cell_h, cols, cell_w = cell_shape
    rgb_cell = frame_np.reshape(rows, cell_h, cols, cell_w, 3).mean(axis=(1, 3))
    rgb_sum = rgb_cell.sum(axis=2)
    hole_cell = (~mask_cell) & (rgb_sum < (10.0 / 255.0 * 3))
    if not hole_cell.any():
        return cell_brightness

    out = cell_brightness.copy()
    cb_nb = uniform_filter(out, size=3, mode="nearest")
    out[hole_cell] = cb_nb[hole_cell]
    still_hole = hole_cell & (out < 0.001)
    if still_hole.any():
        cb_nb2 = uniform_filter(out, size=5, mode="nearest")
        out[still_hole] = cb_nb2[still_hole]
    return out

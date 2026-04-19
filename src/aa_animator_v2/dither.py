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
"""Ordered (Bayer) dithering for AA brightness quantisation.

Provides a 4x4 Bayer matrix dither that perturbs per-pixel brightness
before it is quantised into a character/dot count.  Applying dither at
the pixel level (before cell averaging) breaks up flat-tone banding that
is otherwise visible in large uniform regions.

Usage::

    from aa_animator_v2.dither import apply_bayer
    dithered = apply_bayer(brightness_01_hwc)  # returns same-shape array
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 4×4 Bayer matrix (normalised to [0, 1])
# ---------------------------------------------------------------------------

_BAYER_4X4_RAW: np.ndarray = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
)

# Scale to (-0.5, 0.5) range — symmetric so mean distortion is ~0
_BAYER_4X4: np.ndarray = (_BAYER_4X4_RAW / 16.0) - 0.5  # in [-0.5, 0.5)


def apply_bayer(
    brightness: np.ndarray,
    *,
    strength: float = 1.0 / 8.0,
) -> np.ndarray:
    """Apply 4×4 Bayer ordered dither to a brightness array.

    The dither matrix is tiled to match *brightness* shape and added
    after scaling by *strength*.  The result is clipped to [0, 1].

    Args:
        brightness: Float32 array of shape (H, W) with values in [0, 1].
            Typically a per-pixel luminance array before cell averaging.
        strength: Dither amplitude.  Default 1/8 adds ±1/16 noise —
            enough to break banding without visible grain.

    Returns:
        Dithered float32 array of the same shape, values clipped to [0, 1].
    """
    h, w = brightness.shape
    # Tile Bayer matrix to cover the full (H, W) extent
    tile_h = (h + 3) // 4
    tile_w = (w + 3) // 4
    tiled = np.tile(_BAYER_4X4, (tile_h, tile_w))[:h, :w]
    return np.clip(brightness + tiled * strength, 0.0, 1.0).astype(np.float32)

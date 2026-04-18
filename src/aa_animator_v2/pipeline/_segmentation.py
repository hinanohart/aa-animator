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
"""Foreground segmentation helpers.

Primary path: rembg (u2net).  Fallback: luminance Otsu thresholding when
rembg is unavailable, fails, or returns a very low-coverage mask.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def otsu_threshold(gray_u8: np.ndarray) -> float:
    """Compute the Otsu threshold (0-255) for an 8-bit grayscale array."""
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


def _otsu_fg_mask(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.float32)
    thresh = otsu_threshold(gray.astype(np.uint8))
    return gray > thresh


def segment_subject(image: Image.Image) -> np.ndarray:
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
        return _otsu_fg_mask(image)

    try:
        rgba = rembg_remove(image)
        alpha = np.array(rgba, dtype=np.float32)[:, :, 3]
        thresh = otsu_threshold(alpha.astype(np.uint8))
        mask = alpha > thresh
        if mask.mean() < 0.05:  # coverage too low → fallback
            return _otsu_fg_mask(image)
        return mask
    except Exception:
        return _otsu_fg_mask(image)

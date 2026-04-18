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
"""Temporal EMA smoothing for cell-level brightness.

Reduces frame-to-frame character flicker by blending current cell brightness
with an exponential moving average of previous frames.
"""

from __future__ import annotations

import numpy as np


class TemporalSmoother:
    """Exponential moving average (EMA) over cell brightness frames.

    Args:
        alpha: Blend coefficient.  ``alpha=1.0`` → identity (no smoothing).
               ``alpha=0.0`` → freeze on first frame.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear EMA state (call between animations)."""
        self._prev = None

    def smooth(self, cell_brightness: np.ndarray) -> np.ndarray:
        """Apply one EMA step and return smoothed brightness.

        Args:
            cell_brightness: (ROWS, COLS) float32 in [0, 1].

        Returns:
            Smoothed (ROWS, COLS) float32 array.
        """
        if self._prev is None:
            self._prev = cell_brightness.copy()
            return cell_brightness

        smoothed = self.alpha * cell_brightness + (1.0 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

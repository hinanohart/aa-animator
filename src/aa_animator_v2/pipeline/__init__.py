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
Day 4 scope: Split monolithic pipeline.py (623 lines) into focused sub-modules
             — _segmentation, _depth, _rendering, _encoding — orchestrated by
             AAAnimator in _animator.  Public API (``AAAnimator``, ``BgMode``,
             ``_DA_V2_MODEL_ID``, ``_normalize_depth``) is re-exported from
             this package root for backwards compatibility.
"""

from __future__ import annotations

from ._animator import AAAnimator, BgMode
from ._depth import _DA_V2_MODEL_ID, _normalize_depth

__all__ = ["_DA_V2_MODEL_ID", "AAAnimator", "BgMode", "_normalize_depth"]

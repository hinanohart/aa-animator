# SPDX-License-Identifier: MIT
# Copyright (c) 2026 aa-animator contributors
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

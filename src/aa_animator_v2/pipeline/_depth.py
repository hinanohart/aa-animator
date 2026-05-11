# SPDX-License-Identifier: MIT
# Copyright (c) 2026 aa-animator contributors
"""Depth-estimation helpers.

Wraps the Depth Anything V2 Small (Apache-2.0) HuggingFace checkpoint with
on-disk caching and a graceful uniform-midpoint fallback when torch /
transformers are unavailable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

# Canonical model ID for the Apache-2.0 Depth Anything V2 Small checkpoint.
# Defined once to prevent drift between code and docs/legal-notes.md.
_DA_V2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# HuggingFace commit SHA captured during the 2026-05-09 supply-chain
# audit. Pinning the revision blocks the case where the upstream
# repository silently re-uploads a new checkpoint with the same
# repo id (content drift), so deterministic behaviour and license
# claims survive the next push to ``main`` over there. Override
# explicitly when you want a newer or older checkpoint.
_DA_V2_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


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


def estimate_depth(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Estimate depth map using Depth Anything V2 Small (Apache-2.0).

    Model is loaded via ``transformers.pipeline`` and result is cached
    in ``~/.cache/aa_animator/depth/<input_hash>.npy`` to avoid repeated
    inference for the same image.

    Falls back to a uniform 0.5 map if transformers or the model are
    unavailable so the pipeline stays functional without the optional
    depth dependency.

    Args:
        image: PIL RGB image at renderer resolution.
        target_size: ``(width, height)`` target dimensions for the output.

    Returns:
        Array of shape (H, W) float32 with values in [0, 1].
        Higher values indicate closer (foreground) objects.
    """
    w, h = target_size

    # Cache key from image bytes
    img_bytes = np.array(image, dtype=np.uint8).tobytes()
    cache_key = hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()
    cache_dir = Path.home() / ".cache" / "aa_animator" / "depth"
    cache_path = cache_dir / f"{cache_key}.npy"

    if cache_path.exists():
        depth_raw = np.load(str(cache_path))
        return _normalize_depth(depth_raw, target_size)

    try:
        import torch  # type: ignore[import-not-found]
        from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=_DA_V2_MODEL_ID,
            revision=_DA_V2_MODEL_REVISION,
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

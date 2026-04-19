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
"""v0.2 evaluation metrics for multi-style comparison.

Metrics per MP4 (analyst spec):
  - palette_kl_divergence: KL-divergence of glyph distribution vs boo reference.
  - top_glyph_ratio: fraction of cells occupied by the single most common glyph.
  - outline_heavy_ratio: fraction of outline cells rendered with dense glyphs (@/$).
  - hamming_distance_mean: mean frame-to-frame Hamming distance as fraction.
  - silhouette_swing: relative range of silhouette mass centre across frames.

Boo reference distribution (from paper, not frame data):
  '$': 0.61, '@': 0.12, '%': 0.08, '*': 0.07, '=': 0.05, rest: 0.07
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Boo reference glyph distribution (from published analysis, not frame extraction)
_BOO_REF: dict[str, float] = {
    "$": 0.61,
    "@": 0.12,
    "%": 0.08,
    "*": 0.07,
    "=": 0.05,
    "other": 0.07,
}
_HEAVY_GLYPHS: frozenset[str] = frozenset(["@", "$"])
_GHOSTTY_CHARS: str = " ·~ox+=*%$@"


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL divergence D_KL(P||Q) with Laplace smoothing.

    Args:
        p: Empirical distribution dict.
        q: Reference distribution dict.

    Returns:
        KL divergence (nats).
    """
    all_keys = set(p.keys()) | set(q.keys())
    eps = 1e-9
    kl = 0.0
    for k in all_keys:
        pk = p.get(k, 0.0) + eps
        qk = q.get(k, 0.0) + eps
        kl += pk * np.log(pk / qk)
    return float(kl)


def _sample_frames(mp4_path: Path, n_samples: int = 8) -> list[np.ndarray]:
    """Extract evenly-spaced frames from MP4 using ffmpeg pipe.

    Args:
        mp4_path: Path to MP4 file.
        n_samples: Number of frames to sample.

    Returns:
        List of (H, W, 3) uint8 numpy arrays, may be empty on error.
    """
    import subprocess

    # Get frame count and dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets", "-show_entries",
        "stream=nb_read_packets,width,height",
        "-of", "csv=p=0",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
        parts = result.stdout.strip().split(",")
        if len(parts) < 3:
            return []
        w, h, total = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return []

    if total < 1:
        return []

    frames = []
    step = max(1, total // n_samples)
    for frame_idx in range(0, min(total, n_samples * step), step):
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(mp4_path),
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=15)
            raw = r.stdout
            if len(raw) == w * h * 3:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                frames.append(arr)
        except Exception:
            continue

    return frames


def _frame_to_glyph_grid(frame: np.ndarray, cell_w: int = 7, cell_h: int = 14) -> list[str]:
    """Approximate glyph identity per cell from rendered frame brightness.

    Maps mean cell brightness to Ghostty char ramp.

    Args:
        frame: (H, W, 3) uint8 array.
        cell_w: Cell width in pixels.
        cell_h: Cell height in pixels.

    Returns:
        List of glyph strings.
    """
    h, w = frame.shape[:2]
    rows = h // cell_h
    cols = w // cell_w
    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    glyphs = []
    for ry in range(rows):
        for cx in range(cols):
            cell = frame[ry * cell_h:(ry + 1) * cell_h, cx * cell_w:(cx + 1) * cell_w]
            lum = float(0.2126 * cell[:, :, 0].mean() + 0.7152 * cell[:, :, 1].mean()
                        + 0.0722 * cell[:, :, 2].mean())
            idx = int(np.clip(lum / 255.0 * n, 0, n))
            glyphs.append(charset[idx])
    return glyphs


def compute_metrics(mp4_path: str | Path) -> dict:
    """Compute v0.2 evaluation metrics for one MP4.

    Args:
        mp4_path: Path to MP4 file.

    Returns:
        Dict with keys: palette_kl_divergence, top_glyph_ratio,
        outline_heavy_ratio, hamming_distance_mean, silhouette_swing,
        file_size_mb, n_sample_frames.
    """
    mp4_path = Path(mp4_path)
    file_size_mb = mp4_path.stat().st_size / 1024 / 1024 if mp4_path.exists() else 0.0

    frames = _sample_frames(mp4_path, n_samples=8)
    if not frames:
        return {
            "palette_kl_divergence": -1.0,
            "top_glyph_ratio": -1.0,
            "outline_heavy_ratio": -1.0,
            "hamming_distance_mean": -1.0,
            "silhouette_swing": -1.0,
            "file_size_mb": file_size_mb,
            "n_sample_frames": 0,
        }

    # Glyph distribution across all sampled frames
    all_glyphs: list[str] = []
    glyph_grids: list[list[str]] = []
    for f in frames:
        g = _frame_to_glyph_grid(f)
        all_glyphs.extend(g)
        glyph_grids.append(g)

    total_cells = len(all_glyphs)
    from collections import Counter
    counts = Counter(all_glyphs)

    # Empirical distribution (collapse non-ghostty to 'other')
    ghostty_set = set(_GHOSTTY_CHARS)
    emp: dict[str, float] = {}
    for ch, cnt in counts.items():
        key = ch if ch in ghostty_set else "other"
        emp[key] = emp.get(key, 0.0) + cnt / total_cells

    kl = _kl_divergence(emp, _BOO_REF)
    top_glyph_ratio = max(counts.values()) / total_cells if total_cells > 0 else 0.0

    # Outline heavy ratio: fraction of cells mapped to @ or $
    heavy_count = sum(cnt for ch, cnt in counts.items() if ch in _HEAVY_GLYPHS)
    outline_heavy_ratio = heavy_count / total_cells if total_cells > 0 else 0.0

    # Hamming distance: mean cell change rate between consecutive frames
    hamming_vals: list[float] = []
    for i in range(1, len(glyph_grids)):
        prev, curr = glyph_grids[i - 1], glyph_grids[i]
        n_cells = min(len(prev), len(curr))
        if n_cells > 0:
            diff = sum(1 for a, b in zip(prev, curr, strict=False) if a != b)
            hamming_vals.append(diff / n_cells)
    hamming_mean = float(np.mean(hamming_vals)) if hamming_vals else 0.0

    # Silhouette mass swing: relative range of horizontal centre of mass
    # Computed from non-background pixels per frame
    centres: list[float] = []
    for f in frames:
        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        mask = gray > 20
        if mask.any():
            xs = np.where(mask)[1]
            centres.append(float(xs.mean() / f.shape[1]))
        else:
            centres.append(0.5)
    silhouette_swing = float(max(centres) - min(centres)) if len(centres) > 1 else 0.0

    return {
        "palette_kl_divergence": round(kl, 4),
        "top_glyph_ratio": round(top_glyph_ratio, 4),
        "outline_heavy_ratio": round(outline_heavy_ratio, 4),
        "hamming_distance_mean": round(hamming_mean, 4),
        "silhouette_swing": round(silhouette_swing, 4),
        "file_size_mb": round(file_size_mb, 2),
        "n_sample_frames": len(frames),
    }

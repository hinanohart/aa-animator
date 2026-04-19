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
"""Style C: Style A + DoG edge detection + 6D shape matching.

Rendering approach (architect spec):
  - DoG (sigma1=1.0, sigma2=1.6) to detect edges.
  - Sobel direction for oriented glyph selection: |  /  -  \\
  - 6D shape vector per cell (edge_mag, dxn, dyn, lum, r, g) + scipy cKDTree
    against 64 representative characters from the Ghostty palette.
  - Threshold: DoG > 0.04 AND Sobel_norm > 0.65 * max → edge glyph.
  - Body cells: 6D shape matching for glyph selection.
  - Edge cells: directional glyphs |/─\\ .

Motion: breathe + bob (same as Style A) so the difference is purely in the
render step.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_EDGE_GLYPHS: list[str] = ["|", "/", "-", "\\"]  # 0=vert, 1=diag+, 2=horiz, 3=diag-
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)

_CELL_W: int = 7
_CELL_H: int = 14
_FONT_SIZE: int = 12
_FONT_RATIO: float = 0.50

# DoG / Sobel thresholds (architect spec)
_DOG_THRESH: float = 0.04
_SOBEL_FRACTION: float = 0.65

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]

# 6D feature vector: (lum, r_norm, g_norm, b_norm, dog_norm, sobel_norm)
_FEATURE_DIM = 6


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Shape codebook: 64 chars from Ghostty palette + density extension
# ---------------------------------------------------------------------------


def _build_shape_codebook() -> tuple[list[str], np.ndarray]:
    """Build 64-character codebook from Ghostty palette with 6D feature vectors.

    Feature encoding: each character maps to a target feature position based on
    visual density (0–1) used as a proxy for lum, and zeros for colour/edge dims.
    KD-tree is queried with per-cell 6D vectors for nearest-char lookup.

    Returns:
        Tuple (chars list, ndarray shape (N, 6)).
    """
    # Expand palette to 64 slots by repeating density levels
    base = _GHOSTTY_CHARS  # 11 chars
    extended = (base * 6)[:64]  # 64 entries
    chars = list(extended)

    # Density proxy: uniform spacing 0..1 across 64 slots
    densities = np.linspace(0.0, 1.0, 64, dtype=np.float32)

    # 6D feature: (lum_proxy, 0, 0, 0, 0, density_proxy)
    # This allows KD-tree to prefer chars by luminance proximity
    codebook = np.zeros((64, _FEATURE_DIM), dtype=np.float32)
    codebook[:, 0] = densities  # lum
    codebook[:, 5] = densities  # sobel proxy (darker char → less edge weight)

    return chars, codebook


_CODEBOOK_CHARS, _CODEBOOK_VECS = _build_shape_codebook()
_KDTREE = cKDTree(_CODEBOOK_VECS)


# ---------------------------------------------------------------------------
# Frame motion (breathe + bob, same as Style A)
# ---------------------------------------------------------------------------


def _apply_effects(img: Image.Image, t: float, canvas_size: int) -> Image.Image:
    """Apply breathe + bob effects (identical to style_a_gallery)."""
    scale_x = 1.0 + 0.06 * (-math.sin(t * math.pi * 4))
    scale_y = 1.0 + 0.08 * math.sin(t * math.pi * 4)

    ph_bob = math.sin(t * math.pi * 4)
    offset_y = -18 * max(0.0, ph_bob)
    if ph_bob < -0.7:
        scale_x *= 1.06
        scale_y *= 0.94

    bright = 0.85 + 0.30 * (0.5 + 0.5 * math.sin(t * math.pi * 8))

    cw, ch = img.size
    new_w = max(1, int(cw * scale_x))
    new_h = max(1, int(ch * scale_y))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2 + int(offset_y)
    canvas.paste(resized, (x, y), resized)
    rgb = canvas.convert("RGB")

    if abs(bright - 1.0) > 0.01:
        from PIL import ImageEnhance

        rgb = ImageEnhance.Brightness(rgb).enhance(bright)

    return rgb


# ---------------------------------------------------------------------------
# DoG edge detection
# ---------------------------------------------------------------------------


def _dog_edge(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 1.6) -> np.ndarray:
    """Difference of Gaussians edge map normalised to [0, 1].

    Args:
        gray: (H, W) float32 image in [0, 1].
        sigma1: Fine-scale Gaussian sigma.
        sigma2: Coarse-scale Gaussian sigma.

    Returns:
        (H, W) float32 edge strength in [0, 1].
    """
    g1 = gaussian_filter(gray, sigma=sigma1)
    g2 = gaussian_filter(gray, sigma=sigma2)
    dog = np.abs(g1 - g2)
    if dog.max() > 0:
        dog /= dog.max()
    return dog


def _sobel_direction_glyph(dx: float, dy: float) -> str:
    """Map Sobel gradient direction to oriented glyph.

    Args:
        dx: x-gradient (Sobel).
        dy: y-gradient (Sobel).

    Returns:
        One of |, /, -, \\
    """
    angle = math.atan2(dy, dx) * 180.0 / math.pi
    angle = angle % 180.0
    if angle < 22.5 or angle >= 157.5:
        return "|"
    elif angle < 67.5:
        return "/"
    elif angle < 112.5:
        return "-"
    else:
        return "\\"


# ---------------------------------------------------------------------------
# DoG + 6D shape render
# ---------------------------------------------------------------------------


def _render_dog_shape(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render using DoG edge glyphs on edges, 6D shape-matched chars on body.

    Args:
        frame_img: RGB PIL Image.
        cols: Character canvas width in cells.
        font: Pre-loaded monospace font.

    Returns:
        RGB PIL Image.
    """
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small = frame_img.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)  # (rows, cols, 3) in [0,255]
    arr_01 = arr / 255.0

    lum = _srgb_luma(arr_01)  # (rows, cols)

    # DoG edge map at cell resolution
    dog = _dog_edge(lum, sigma1=1.0, sigma2=1.6)

    # Sobel direction map at cell resolution
    sx = scipy_sobel(lum, axis=1)
    sy = scipy_sobel(lum, axis=0)
    sobel_mag = np.hypot(sx, sy)
    sobel_max = sobel_mag.max()
    sobel_norm = sobel_mag / sobel_max if sobel_max > 0 else sobel_mag

    # Edge decision (architect spec)
    edge_mask = (dog > _DOG_THRESH) & (sobel_norm > _SOBEL_FRACTION * sobel_norm.max() + 1e-8)

    # 6D feature per cell for body chars
    feat = (
        np.stack(
            [
                lum,
                arr_01[..., 0],
                arr_01[..., 1],
                arr_01[..., 2],
                dog,
                sobel_norm,
            ],
            axis=-1,
        )
        .reshape(-1, _FEATURE_DIM)
        .astype(np.float32)
    )  # (rows*cols, 6)

    _, nn_idx = _KDTREE.query(feat)  # (rows*cols,)
    nn_idx = nn_idx.reshape(rows, cols)

    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for ry in range(rows):
        for cx in range(cols):
            if lum[ry, cx] < 0.02:
                continue

            px, py = cx * _CELL_W, ry * _CELL_H

            if edge_mask[ry, cx]:
                # Directional edge glyph in blue
                ch = _sobel_direction_glyph(float(sx[ry, cx]), float(sy[ry, cx]))
                color = _GLOW_COLOR
            else:
                # Body: 6D shape-matched char with preserved colour
                ch = _CODEBOOK_CHARS[nn_idx[ry, cx]]
                if ch == " ":
                    continue
                r = arr[ry, cx, 0]
                g = arr[ry, cx, 1]
                b = arr[ry, cx, 2]
                boost = max(1.0, 1.6 - lum[ry, cx] * 255.0 / 350.0)
                color = (
                    min(255, int(r * boost)),
                    min(255, int(g * boost)),
                    min(255, int(b * boost)),
                )

            d.text((px, py), ch, fill=color, font=font)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_style_c(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 80,
    fps: int = 30,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style C (DoG edge + 6D shape) animation MP4.

    Args:
        input_path: Path to input image.
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells.
        fps: Output video frame rate.
        duration: Clip length in seconds.
        canvas_size: Square canvas pixel size.

    Returns:
        Dict with output_path, n_frames, canvas_w, canvas_h.
    """
    import subprocess
    import sys

    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    img_src.thumbnail((canvas_size, canvas_size), Image.LANCZOS)
    canvas_base = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    ox = (canvas_size - img_src.width) // 2
    oy = (canvas_size - img_src.height) // 2
    canvas_base.paste(img_src, (ox, oy), img_src)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    probe_rgb = _apply_effects(canvas_base, 0.0, canvas_size)
    probe = _render_dog_shape(probe_rgb, cols, font)
    canvas_w, canvas_h = probe.size

    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb = _apply_effects(canvas_base, t, canvas_size)
        aa_frame = _render_dog_shape(frame_rgb, cols, font)
        frames.append(aa_frame)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{canvas_w}x{canvas_h}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-tune",
        "animation",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
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
        proc.stdin.write(np.array(frame.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-400:]}")

    print(f"[style-C] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }

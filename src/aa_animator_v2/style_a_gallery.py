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
"""Style A: DensityAA gallery port from aa_animator.py.

Ports the DensityAA class (aa_animator.py:383-416) with breathe + bob + pulse
effects (aa_animator.py:209-234) and Ghostty palette.
Produces colour-preserving density-ramp animation with blue glow on edges.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel as scipy_sobel  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants — Ghostty 11-level density ramp (same as renderer.py AA_CHARS)
# ---------------------------------------------------------------------------

_GHOSTTY_CHARS: str = " ·~ox+=*%$@"
_GLOW_COLOR: tuple[int, int, int] = (70, 130, 255)
_GLOW_ALPHA: float = 0.30
_EDGE_THRESH: float = 0.15
_BG_COLOR: tuple[int, int, int] = (8, 8, 12)

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]

# Cell geometry matching DensityAA in aa_animator.py
_CELL_W: int = 7
_CELL_H: int = 14
_FONT_SIZE: int = 12
_FONT_RATIO: float = 0.50


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (H, W, 3) float32 [0-255]."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Frame motion effects (breathe + bob + pulse from aa_animator.py:209-234)
# ---------------------------------------------------------------------------


def _apply_effects(img: Image.Image, t: float, canvas_size: int) -> Image.Image:
    """Apply breathe + bob + pulse effects to a PIL RGBA image.

    Args:
        img: Source RGBA image.
        t: Normalised time in [0, 1].
        canvas_size: Square canvas pixel size.

    Returns:
        RGB PIL Image at canvas_size x canvas_size.
    """
    scale = 1.0
    scale_x = scale_y = 1.0
    offset_y = 0.0
    bright = 1.0

    # breathe: squash & stretch (aa_animator.py:209-213)
    ph_breathe = math.sin(t * math.pi * 4)
    scale_x *= 1.0 + 0.06 * (-ph_breathe)
    scale_y *= 1.0 + 0.08 * ph_breathe

    # bob: bounce up, squash on landing (aa_animator.py:214-221)
    ph_bob = math.sin(t * math.pi * 4)
    offset_y -= 18 * max(0.0, ph_bob)
    if ph_bob < -0.7:
        scale_x *= 1.06
        scale_y *= 0.94

    # pulse: brightness flicker (aa_animator.py:233-234)
    bright *= 0.85 + 0.30 * (0.5 + 0.5 * math.sin(t * math.pi * 8))

    # Resize with squash/stretch
    cw, ch = img.size
    new_w = max(1, int(cw * scale * scale_x))
    new_h = max(1, int(ch * scale * scale_y))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Composite onto canvas
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2 + int(offset_y)
    canvas.paste(img_resized, (x, y), img_resized)
    rgb = canvas.convert("RGB")

    if abs(bright - 1.0) > 0.01:
        from PIL import ImageEnhance  # type: ignore

        rgb = ImageEnhance.Brightness(rgb).enhance(bright)

    return rgb


# ---------------------------------------------------------------------------
# DensityAA render (ported from aa_animator.py:383-416)
# ---------------------------------------------------------------------------


def _render_density_aa(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame as Ghostty-palette density AA.

    Preserves source image colour, boosts dark pixels, and uses the
    Ghostty 11-level char ramp.

    Args:
        frame_img: RGB PIL Image (canvas_size x canvas_size).
        cols: Character canvas width in cells.
        font: Pre-loaded monospace font.

    Returns:
        RGB PIL Image of the rendered AA frame.
    """
    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small = frame_img.resize((cols, rows), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    lum = _srgb_luma(arr)

    charset = _GHOSTTY_CHARS
    n = len(charset) - 1
    idx = np.clip((lum / 255.0 * n).astype(int), 0, n)

    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for y in range(rows):
        for x in range(cols):
            if lum[y, x] < 6:
                continue
            ch = charset[idx[y, x]]
            if ch == " ":
                continue
            r, g, b = arr[y, x, 0], arr[y, x, 1], arr[y, x, 2]
            # Dark pixel boost (aa_animator.py:408-414)
            boost = max(1.0, 1.6 - lum[y, x] / 350)
            color = (
                min(255, int(r * boost)),
                min(255, int(g * boost)),
                min(255, int(b * boost)),
            )
            d.text((x * _CELL_W, y * _CELL_H), ch, fill=color, font=font)

    return out


def _apply_edge_glow(
    rendered: Image.Image,
    frame_img: Image.Image,
    cols: int,
) -> Image.Image:
    """Alpha-blend Ghostty blue glow on edge-adjacent cells.

    Args:
        rendered: The AA-rendered frame image.
        frame_img: Source RGB frame (canvas_size x canvas_size).
        cols: Character canvas width.

    Returns:
        Glow-augmented RGB PIL Image.
    """
    from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]

    w, h = frame_img.size
    rows = max(1, int(h * cols / w * _FONT_RATIO))
    small_gray = np.array(frame_img.convert("L").resize((cols, rows), Image.LANCZOS)).astype(
        np.float32
    )

    sx = scipy_sobel(small_gray, axis=1)
    sy = scipy_sobel(small_gray, axis=0)
    mag = np.hypot(sx, sy)
    if mag.max() > 0:
        mag /= mag.max()
    edge_cell = mag > _EDGE_THRESH

    glow_cell = binary_dilation(edge_cell, structure=np.ones((3, 3), dtype=bool)) & ~edge_cell
    if not glow_cell.any():
        return rendered

    canvas_arr = np.array(rendered, dtype=np.float32)
    for ry in range(rows):
        for cx in range(cols):
            if not glow_cell[ry, cx]:
                continue
            y0, y1 = ry * _CELL_H, (ry + 1) * _CELL_H
            x0, x1 = cx * _CELL_W, (cx + 1) * _CELL_W
            region = canvas_arr[y0:y1, x0:x1]
            region[:] = (
                region * (1.0 - _GLOW_ALPHA) + np.array(_GLOW_COLOR, dtype=np.float32) * _GLOW_ALPHA
            )

    return Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_style_a(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 80,
    fps: int = 30,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style A (gallery DensityAA) animation MP4.

    Args:
        input_path: Path to input image (character/illustration recommended).
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells.
        fps: Output video frame rate.
        duration: Clip length in seconds.
        canvas_size: Square canvas pixel size for effect compositing.

    Returns:
        Dict with output_path, n_frames, canvas_w, canvas_h.
    """
    import subprocess
    import sys

    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    # Scale to canvas keeping aspect ratio, pad to square
    img_src.thumbnail((canvas_size, canvas_size), Image.LANCZOS)
    canvas_base = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    ox = (canvas_size - img_src.width) // 2
    oy = (canvas_size - img_src.height) // 2
    canvas_base.paste(img_src, (ox, oy), img_src)

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    # Derive output canvas dimensions from a test render
    probe = _render_density_aa(canvas_base.convert("RGB"), cols, font)
    canvas_w, canvas_h = probe.size

    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb = _apply_effects(canvas_base, t, canvas_size)
        aa_frame = _render_density_aa(frame_rgb, cols, font)
        aa_frame = _apply_edge_glow(aa_frame, frame_rgb, cols)
        frames.append(aa_frame)

    # Export via ffmpeg
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

    print(f"[style-A] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }

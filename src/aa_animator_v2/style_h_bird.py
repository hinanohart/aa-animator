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
"""Style H: block_bob preset from aa_animator.py — "bird" motion.

Complete port of the ``block_bob`` gallery preset
(aa_animator.py GALLERY_PRESETS line 669):

    ('block', ['bob', 'sway', 'vignette'], 'color', 'block_bob')

Motion effects (ported exactly from aa_animator.py:FrameEffects.render):
  bob      : upward bounce — offset_y -= 18 * max(0, sin(t*4π));
             landing squash: scale_x *= 1.06, scale_y *= 0.94 when sin < -0.7
  sway     : horizontal drift — offset_x += 10 * sin(t*2π)
  vignette : radial darkening mask (edge attenuation d²×0.85)

Rendering (ported from aa_animator.py:BlockAA):
  style    : block — half-block char ▀ with dual-colour per cell
  cell_w   : 8 px,  cell_h : 14 px,  font_size : 16 pt
  rows     : 2× pixel rows → halved → BlockAA rows
  color    : 'color' — source image colour preserved (no matrix/amber remap)
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants — matching aa_animator.py BlockAA geometry exactly
# ---------------------------------------------------------------------------

_BG_COLOR: tuple[int, int, int] = (8, 8, 12)
_CELL_W: int = 8
_CELL_H: int = 14
_FONT_SIZE: int = 16
_BLOCK_CHAR: str = "▀"

_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _srgb_luma(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from (H, W, 3) float32."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Motion effects: bob + sway (aa_animator.py:214-223)
# ---------------------------------------------------------------------------


def _apply_motion(img: Image.Image, t: float, canvas_size: int) -> Image.Image:
    """Apply bob + sway effects to RGBA source image.

    Ports aa_animator.py FrameEffects.render() for the bob and sway effects
    (lines 214-223). No breathe, no pulse — pure bob + sway only.

    Args:
        img: Source RGBA image.
        t: Normalised time in [0, 1].
        canvas_size: Square canvas size in pixels.

    Returns:
        RGB PIL Image composited onto canvas_size x canvas_size canvas.
    """
    scale_x = scale_y = 1.0
    offset_x = 0.0
    offset_y = 0.0

    # bob (aa_animator.py:214-221)
    ph = math.sin(t * math.pi * 4)
    offset_y -= 18 * max(0.0, ph)
    if ph < -0.7:
        scale_x *= 1.06
        scale_y *= 0.94

    # sway (aa_animator.py:223)
    offset_x += 10 * math.sin(t * math.pi * 2)

    cw, ch = img.size
    new_w = max(1, int(cw * scale_x))
    new_h = max(1, int(ch * scale_y))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*_BG_COLOR, 255))
    x = (canvas_size - new_w) // 2 + int(offset_x)
    y = (canvas_size - new_h) // 2 + int(offset_y)
    canvas.paste(img_resized, (x, y), img_resized)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Vignette (aa_animator.py:346-355)
# ---------------------------------------------------------------------------


def _apply_vignette(img: Image.Image) -> Image.Image:
    """Radial darkening — edges attenuated by d²×0.85.

    Ports aa_animator.py FrameEffects._vignette() exactly.

    Args:
        img: RGB PIL Image.

    Returns:
        Vignette-applied RGB PIL Image.
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    d /= d.max()
    mask = np.clip(1.0 - d**2 * 0.85, 0.15, 1.0)
    arr *= mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# BlockAA renderer (aa_animator.py:468-504)
# ---------------------------------------------------------------------------


def _render_block_aa(
    frame_img: Image.Image,
    cols: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render one frame as block ▀ AA with dual-colour per cell.

    Ports aa_animator.py BlockAA.render() (lines 473-504) exactly:
    - rows_dbl = floor(h/w * cols * 0.95) rounded down to even
    - rows = rows_dbl // 2
    - Each cell: top half = ▀ foreground colour, bottom half = background colour
    - color mode = 'color' (source image colour, no remapping)

    Args:
        frame_img: RGB PIL Image (canvas_size x canvas_size).
        cols: Character canvas width in cells.
        font: Pre-loaded font.

    Returns:
        RGB PIL Image of the rendered block AA frame.
    """
    w, h = frame_img.size
    rows_dbl = max(2, int(h / w * cols * 0.95))
    rows_dbl -= rows_dbl % 2
    rows = rows_dbl // 2

    small = frame_img.resize((cols, rows_dbl), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    # color mode: no remapping, use arr directly

    out = Image.new("RGB", (cols * _CELL_W, rows * _CELL_H), _BG_COLOR)
    d = ImageDraw.Draw(out)

    for ry in range(rows):
        for cx in range(cols):
            top = arr[ry * 2, cx]
            bot = arr[ry * 2 + 1, cx]
            tr, tg, tb = int(top[0]), int(top[1]), int(top[2])
            br, bg, bb = int(bot[0]), int(bot[1]), int(bot[2])
            bg_col = (max(0, br), max(0, bg), max(0, bb))
            fg_col = (min(255, tr), min(255, tg), min(255, tb))
            d.rectangle(
                [
                    cx * _CELL_W,
                    ry * _CELL_H,
                    (cx + 1) * _CELL_W,
                    (ry + 1) * _CELL_H,
                ],
                fill=bg_col,
            )
            d.text(
                (cx * _CELL_W, ry * _CELL_H - 2),
                _BLOCK_CHAR,
                fill=fg_col,
                font=font,
            )

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_style_h(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cols: int = 100,
    fps: int = 30,
    duration: float = 4.0,
    canvas_size: int = 512,
) -> dict:
    """Generate Style H (block_bob "bird" preset) animation MP4.

    Faithfully reproduces the ``block_bob`` gallery preset from
    aa_animator.py: block AA style with bob + sway + vignette effects,
    colour-preserving rendering.

    Args:
        input_path: Path to input image (any still image).
        output_path: Destination .mp4 path.
        cols: Character canvas width in cells (default 100 matches gallery).
        fps: Output video frame rate.
        duration: Clip length in seconds.
        canvas_size: Square compositing canvas size in pixels.

    Returns:
        Dict with keys: output_path, n_frames, canvas_w, canvas_h.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img_src = Image.open(input_path).convert("RGBA")
    # Scale to at most 480px (matching aa_animator.py:595-598)
    target = 480
    if max(img_src.size) > target:
        scale = target / max(img_src.size)
        img_src = img_src.resize(
            (int(img_src.width * scale), int(img_src.height * scale)),
            Image.LANCZOS,
        )
    # Canvas size = max(w, h) + 80  (aa_animator.py:601)
    sw, sh = img_src.size
    canvas_size = max(sw, sh) + 80

    font = _load_font(_FONT_SIZE)
    n_frames = int(duration * fps)

    # Probe output dimensions
    probe_rgb = _apply_motion(img_src, 0.0, canvas_size)
    probe_aa = _render_block_aa(probe_rgb, cols, font)
    canvas_w, canvas_h = probe_aa.size
    # Ensure even dimensions for yuv420p
    canvas_w -= canvas_w % 2
    canvas_h -= canvas_h % 2

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
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frame_rgb = _apply_motion(img_src, t, canvas_size)
        frame_rgb = _apply_vignette(frame_rgb)
        aa_frame = _render_block_aa(frame_rgb, cols, font)
        # Crop to even dimensions
        aa_frame = aa_frame.crop((0, 0, canvas_w, canvas_h))
        proc.stdin.write(np.array(aa_frame.convert("RGB"), dtype=np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-400:]}")

    print(f"[style-H bird] written: {output_path}", file=sys.stderr)
    return {
        "output_path": str(output_path),
        "n_frames": n_frames,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }

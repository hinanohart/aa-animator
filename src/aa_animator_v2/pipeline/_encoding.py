# SPDX-License-Identifier: MIT
# Copyright (c) 2026 aa-animator contributors
"""MP4 encoding via ffmpeg subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def export_mp4(output_path: str | Path, frames: list[Image.Image], fps: int) -> None:
    """Encode rendered PIL frames to an MP4 via ffmpeg subprocess.

    Args:
        output_path: Destination .mp4 file path.
        frames: List of same-size PIL RGB Images.
        fps: Output frame rate.

    Raises:
        ValueError: If *frames* is empty.
        RuntimeError: If ffmpeg exits with a non-zero return code.
        FileNotFoundError: If ffmpeg is not found on PATH.
    """
    if not frames:
        raise ValueError("frames list is empty.")

    output_path = Path(output_path)
    canvas_w, canvas_h = frames[0].size

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
        "-g",
        "15",
        "-keyint_min",
        "15",
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
        proc.stdin.write(np.array(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr[-500:]}")

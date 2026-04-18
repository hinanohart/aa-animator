"""E2E integration test: CLI animate subcommand → MP4.

Requires ffmpeg on PATH. Skips if ffmpeg is absent.
Skips rembg-dependent paths if rembg is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not found on PATH")
class TestCliAnimateE2E:
    """End-to-end test for `aa-animator animate` → MP4 output."""

    def _make_test_image(self, tmp_path: Path) -> Path:
        """Create a 64x64 synthetic PNG for fast testing."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(50, 200, 64, dtype=np.uint8).reshape(1, 64)
        img = Image.fromarray(arr)
        img_path = tmp_path / "e2e_input.png"
        img.save(str(img_path))
        return img_path

    def test_animate_produces_mp4(self, tmp_path: pytest.TempPathFactory) -> None:
        img_path = self._make_test_image(tmp_path)  # type: ignore[arg-type]
        out_path = tmp_path / "e2e_output.mp4"  # type: ignore[operator]

        result = subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "aa_animator_v2.cli",
                "animate", str(img_path),
                str(out_path),
                "--cols", "20",
                "--fps", "10",
                "--duration", "0.5",
                "--no-glow",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, (
            f"CLI exited with rc={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out_path.exists(), f"Output MP4 not created at {out_path}"
        assert out_path.stat().st_size > 0, "Output MP4 is empty"

    def test_animate_mp4_has_valid_duration(self, tmp_path: pytest.TempPathFactory) -> None:
        if not _has_ffprobe():
            pytest.skip("ffprobe not found on PATH")

        img_path = self._make_test_image(tmp_path)  # type: ignore[arg-type]
        out_path = tmp_path / "e2e_duration.mp4"  # type: ignore[operator]

        subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "aa_animator_v2.cli",
                "animate", str(img_path),
                str(out_path),
                "--cols", "20",
                "--fps", "10",
                "--duration", "1.0",
                "--no-glow",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

        ffprobe_path = shutil.which("ffprobe") or "ffprobe"
        probe = subprocess.run(  # noqa: S603
            [
                ffprobe_path, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert probe.returncode == 0, f"ffprobe failed: {probe.stderr}"
        duration = float(probe.stdout.strip())
        # Allow generous tolerance: 0.2s ≤ duration ≤ 5.0s
        assert 0.2 <= duration <= 5.0, f"Unexpected MP4 duration: {duration}s"

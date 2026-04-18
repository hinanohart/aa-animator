"""aa-animator — still image to animated ASCII art video.

Depth-parallax warp drives the motion; edge-aware rendering and blue-glow
contouring target Ghostty-grade visual quality.

Public API (v0.1+):
    animate(input_path, output_path, **kwargs) -> None
    preview(input_path, **kwargs) -> str

Version:
    aa_animator_v2.__version__
"""

from aa_animator_v2._version import __version__
from aa_animator_v2.pipeline import AAAnimator

__all__ = ["__version__", "AAAnimator"]

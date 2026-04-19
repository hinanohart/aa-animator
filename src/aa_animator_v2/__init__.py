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

__all__ = ["AAAnimator", "__version__"]

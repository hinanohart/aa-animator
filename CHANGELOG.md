# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] — 2026-04-18 (pre-release PoC, no PyPI release)

### Added

- Mini PoC: 100 × 41 cell canvas, 10 frames @ 30 fps
- Depth Anything v2 Small (Apache-2.0) for per-pixel depth estimation on CPU
- Forward-scatter parallax warp via `scipy.ndimage.map_coordinates` (backward sampling)
- 11-level density ramp AA renderer (`" ·~ox+=*%$@"`)
- Blue glow contour `(70, 130, 255)` matching Ghostty +boo aesthetic
- EMA temporal smoothing (α = 0.3) for flicker reduction
- ffmpeg pipe encode to MP4 (no intermediate PNG frames)

### Metrics (internal evaluation only)

- Flicker: **3.4% avg char edit distance** (threshold < 15%) — pass
- Pipeline time: ~10 s on CPU for 10 frames (DA-V2 model download included)
- SSIMULACRA2 vs Ghostty reference: not yet measured
- Subject boundary jitter: not applicable (no segmentation in PoC)

### Notes

- Internal evaluation only; no PyPI release
- `sniklaus/softmax-splatting` source code was not used or adapted
- DepthFlow (AGPL-3.0) was not imported; forward warp is self-implemented

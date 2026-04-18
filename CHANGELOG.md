# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3] — 2026-04-18 (pre-release PoC, no PyPI release)

### Added

- Braille mode: Unicode U+2800-28FF 2×4 dots (2× resolution, 8 bits/cell)
- fg_entropy metric: entropy measured on foreground cells only (rembg mask)
- rembg (u2net) subject mask with Otsu threshold + luminance fallback
- Hole-filling: 2-pass recursive fill + mask_cell/RGB_mean detection
- BG_MODE=ghostty_fill: low-density background characters for full-canvas feel
- Portable PoC script: `scripts/poc_v0_0_3.py` (sys.argv input, no hardcoded paths)

### Metrics (Mode C: --bg black --braille on, bike_art_base.jpg)

- Flicker std 0.0086 (threshold std <= 0.01) — PASS
- fg_entropy 3.179 bits (threshold >= 3.0 bits) — PASS
- Output size: 114.5 KB
- fg_entropy threshold raised from 2.5 to 3.0 bits (log2(11) ≈ 3.46, 3.0 is discriminating)

### Selected mode

Mode C selected: only mode achieving flicker std <= 0.01, fg_entropy pass, smallest output size.

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

- Flicker: flicker std 0.034 avg (v0.0.1, 10-frame mini PoC) — pass (threshold std <= 0.01 tightened in v0.0.3)
- Pipeline time: ~10 s on CPU for 10 frames (DA-V2 model download included)
- SSIMULACRA2 vs Ghostty reference: not yet measured
- Subject boundary jitter: not applicable (no segmentation in PoC)

### Notes

- Internal evaluation only; no PyPI release
- `sniklaus/softmax-splatting` source code was not used or adapted
- DepthFlow (AGPL-3.0) was not imported; forward warp is self-implemented

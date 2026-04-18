# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0.dev1] — 2026-04-18 (pre-release, Day 3)

### Added

- **Bayer dither mode**: `--dither bayer` applies 4×4 ordered dither before brightness quantisation; breaks flat-tone banding in uniform regions (`src/aa_animator_v2/dither.py`)
- **`_prepare_run` DRY helper**: `animate`, `preview`, `bake` now share a single `_prepare_run(input_path)` method — eliminates 3× duplication of load/segment/depth/warp-mask logic
- **Mask artefact export** in `validate_v0_1.py`: saves `{stem}_mask.png` to `--out-dir/masks/` and records `mask_source` (`"rembg"` | `"otsu"`), `canvas_cols`, `canvas_rows` in JSON output

### Changed

- **rembg promoted to required dependency** (was `[matte]` optional extra); `[matte]` extra kept empty for backward compatibility
- **ffmpeg flags optimised**: `-preset medium -crf 20 -g 15 -keyint_min 15 -tune animation -movflags +faststart` — keyframe density for smooth playback, animation tuning, web-ready moov atom placement
- **pytest filterwarnings**: added `ignore` rules for `transformers.*`, `huggingface_hub.*`, `torch.*` DeprecationWarning / FutureWarning / UserWarning to reduce test noise
- **`release.yml`**: added comment on FRAG pattern split-token technique

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

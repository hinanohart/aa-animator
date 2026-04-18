# aa-animator

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/aa-animator.svg)](https://pypi.org/project/aa-animator/)

**Turn any still photo into an animated ASCII art video — subject only, Ghostty-grade quality.**

One image in, one MP4 out. Depth-parallax warp drives the motion; edge-aware rendering and blue-glow contouring match Ghostty's visual standard. No hand-drawn frames, no fixed silhouette.

> **Recommended preset**: `style_i` (long_cinematic + Ghostty palette + outline ring + blue glow).
> After iterating through 10+ style variants (A–J), the `long_cinematic`-based pipeline delivered the most stable visual quality. Other styles (Braille, signal-driven, 3D lighting, bird/block-bob) are retained as experiments under `src/aa_animator_v2/style_*.py`.
>
> ```bash
> aa-animator animate photo.jpg -o out.mp4 --style i
> ```

---

## Quick Start

```bash
pip install aa-animator          # ~3 GB total install (torch + model weights)
aa-animator animate photo.jpg -o out.mp4
```

Open `out.mp4` in any player, or pipe into `ffplay` for an instant preview.

Expected output: a 4-second MP4 (120 frames at 30 fps) showing your subject rendered in Braille/density ASCII glyphs with a blue glow edge contour, orbiting through a ±8° parallax motion. First run downloads Depth Anything V2 Small weights (~99 MB) via HuggingFace Hub.

---

## Features

- **Depth-parallax motion** — Depth Anything v2 Small (Apache-2.0, ~99 MB) estimates per-pixel depth; a forward-warp orbit animates ±8° of camera motion across 120 frames at 30 fps
- **Ghostty-grade AA rendering** — 11-level density ramp, DoG+Sobel directional glyphs (`|/─\`), blue glow contour `(70, 130, 255)` matching the Ghostty +boo aesthetic
- **Temporal stability** — EMA smoothing (α = 0.3) and time-fixed Bayer 4×4 dithering keep flicker std below 0.01 (Mode C, bike_art_base, v0.0.3)
- **CPU-first design** — full pipeline runs on CPU; 120-frame clip completes in ~14 s on a modern CPU (no GPU required)
- **Multiple render styles** — `density`, `edge`, `block`, `braille`
- **Seven colour modes** — `color`, `mono`, `matrix`, `cyber`, `amber`, `gradient`, `invert`
- **Subject-only AA** (v0.2, `[matte]` extra) — BiRefNet or rembg isolates the foreground; background stays dark/transparent
- **i2v AI motion** (v0.4 optional) — Wan 2.1 I2V-14B-480P via external CLI for AI-driven motion on capable hardware

---

## How aa-animator compares

| Feature | **aa-animator** | ghosttime ★76 | Tortuise ★218 | chafa ★4608 |
|---|---|---|---|---|
| Input | Any still image | Fixed Ghostty boo data | Existing `.ply`/`.splat` | Any image |
| Output | Animated MP4 | Terminal animation | Terminal animation | Static AA |
| Motion source | Depth-parallax (AI) | Hand-drawn 235 frames | Gaussian Splat render | N/A |
| Subject-only | Yes (v0.2) | No (ghost silhouette fixed) | No | No |
| Blue glow aesthetic | Yes | Yes (Ghostty style) | No | No |
| PyPI installable | Yes | No | No (Rust binary) | No (C library) |
| Arbitrary subject | Yes | No | No | No |

---

## System Requirements

- Python 3.10–3.13
- ffmpeg on PATH
- **Installed size ~3 GB** — torch (~2.5 GB) + transformers + onnxruntime; Depth Anything V2 Small weights (~99 MB) downloaded at first run and cached by HuggingFace Hub
- CPU: ~14 s for a 120-frame clip (depth + mask cached after first run)
- GPU: CUDA-capable GPU reduces depth estimation to < 1 s

---

## Installation

### CPU (recommended for most users)

```bash
pip install aa-animator
```

### GPU acceleration

```bash
pip install aa-animator
# PyTorch with CUDA is pulled automatically if a CUDA wheel is available
# for your platform. For explicit control:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Subject-only extraction (v0.2+)

```bash
pip install "aa-animator[matte]"
```

### AI-driven i2v motion (v0.4+, requires 24 GB+ VRAM)

```bash
pip install "aa-animator[i2v]"
# External: install Wan 2.1 I2V separately and point --i2v-bin at its CLI
```

### Development

```bash
git clone https://github.com/hinanohart/aa-animator
cd aa-animator
pip install -e ".[dev]"
pre-commit install
```

---

## Usage

### CLI

```
aa-animator animate INPUT [OUTPUT]
  --style {density,edge,block,braille}         default: density
  --color {color,mono,matrix,cyber,amber,gradient,invert}  default: color
  --cols INT          terminal columns (character width)    default: 100
  --fps INT           output frame rate                     default: 30
  --duration FLOAT    clip length in seconds                default: 4.0
  --amp-deg FLOAT     parallax orbit amplitude in degrees   default: 8.0
  --ema FLOAT         temporal EMA smoothing coefficient    default: 0.3
  --glow / --no-glow  blue edge glow (Ghostty style)        default: --glow
  --depth-device {auto,cpu,cuda,mps}           default: auto
  --seed INT          RNG seed for reproducibility          default: 42
  --subject-only      foreground-only AA (requires [matte]) (v0.2)

aa-animator preview INPUT     # single-frame terminal preview
aa-animator bake INPUT OUTDIR # render gallery of style × color variants
```

### Examples

```bash
# Basic: default density style, colour, blue glow
aa-animator animate portrait.jpg

# Edge style, monochrome, no glow, 8-second clip
aa-animator animate cat.png cat_edge.mp4 --style edge --color mono --no-glow --duration 8

# Subject-only (requires pip install "aa-animator[matte]")
aa-animator animate portrait.jpg fg_only.mp4 --subject-only

# Force CPU, wider canvas, higher amplitude
aa-animator animate landscape.jpg wide.mp4 --cols 140 --amp-deg 12 --depth-device cpu

# Terminal preview (no video written)
aa-animator preview portrait.jpg --style braille --color cyber
```

---

## Motion template overlay (H7-B) — v0.2 preview

v0.2 will add `--ghostty-motion` which extracts parametric motion curves from Ghostty's 235 canonical frames:

- `eye_blink(t)` — cavity area time series from rows 16–22
- `mouth_open(t)` — boolean presence of open-mouth glyphs rows 19–22
- `outline_shimmer(t, θ)` — bold-char sliding along top arc and bottom skirt
- `skirt_wave(t, x)` — horizontal `+=*%` sweep mimicking walking motion

These curves are layered **on top of** parallax as per-character animation, producing semantic motion (blink, sway) that parallax alone cannot generate.

---

## Demo

*Demo GIF placeholder — will be replaced after v0.0.1 public Discord/X post (48-hour feedback window)*

Validation (internal evaluation, 2026-04-18, v0.1 Mode C):

**v0.1 flicker validation — n=10, 95% t-CI (df=9): [0.0027, 0.0073]**

- 9/10 images pass flicker_std <= 0.01 (7 natural + 3 synthetic images)
- Mean flicker_std: 0.0050 (well below 0.01 threshold)
- Single failure: `benchmark_result.jpg` at 0.0111 (low fg_coverage=21% triggers max amp)
- fg_entropy mean: 3.1+ bits (threshold >= 3.0)
- Pipeline time: ~4 s on CPU (depth + mask cached)
- Full audit trail: `docs/poc-validation.md`

> Note: The earlier n=3 bootstrap CI `[0.0048, 0.0088]` was statistically invalid
> (bootstrap on 3 points collapses to min/max). The n=10 t-CI above is the authoritative result.

---

## Evaluation metrics

| Metric | v0.1 n=10 (Mode C) | v0.1 target |
|---|---|---|
| Temporal flicker (flicker std, mean) | 0.0050 (t-CI [0.0027, 0.0073]) | <= 0.01 |
| Pass rate | 9/10 images | — |
| SSIMULACRA2 vs Ghostty reference | not yet measured | ±10% |
| Subject boundary jitter (IoU variance) | N/A | < 0.02 |
| Information density (fg_entropy) | 3.1+ bits | >= 3.0 bits |

---

## Development

```bash
# Run tests
pytest -v --cov=aa_animator_v2

# Lint + format
ruff check src tests
ruff format src tests

# Type check
mypy --strict src/aa_animator_v2

# Pre-commit hooks
pre-commit run --all-files
```

CI matrix: Python 3.10–3.13 × Ubuntu + macOS (see `.github/workflows/ci.yml`).

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

### Third-party attribution

This package uses the following components with their respective licenses:

| Component | License | Notes |
|---|---|---|
| Depth Anything V2 Small | Apache-2.0 | Downloaded at runtime via HuggingFace Hub; weights not bundled |
| PyTorch | BSD-3-Clause | Core tensor library |
| NumPy | BSD-3-Clause | Array operations |
| Pillow | HPND | Image I/O |
| OpenCV (headless) | Apache-2.0 | Video encode, edge detection |
| SciPy | BSD-3-Clause | Interpolation for warp |
| Transformers | Apache-2.0 | Model loading |
| HuggingFace Hub | Apache-2.0 | Weight download |
| tqdm | MIT/MPL-2.0 | Progress bars |
| rembg (matte extra) | MIT | Subject extraction via u2net |
| diffusers (i2v extra) | Apache-2.0 | i2v pipeline wrapper |
| accelerate (i2v extra) | Apache-2.0 | Multi-device support |

**Not included (AGPL / non-commercial):**

- DepthFlow — AGPL-3.0; may be called via external subprocess per FSF FAQ (not imported)
- RMBG-2.0 (BRIA) — non-commercial only; excluded entirely
- CogVideoX-5B — proprietary license; excluded entirely
- DA-V2 Base/Large — CC-BY-NC-4.0; only the Apache-2.0 Small variant is used

---

## Citation

The forward-warp motion in this project is conceptually informed by:

```
Niklaus, S., & Liu, F. (2020). Softmax Splatting for Video Frame Interpolation.
CVPR 2020. arXiv:2003.07360.
```

No source code from `sniklaus/softmax-splatting` was used or adapted (that repository carries no OSI license). The implementation derives from the paper's equations using PyTorch scatter operations.

---

## Limitations

- **Validation scale**: quality metrics (flicker std, fg_entropy) validated on n=10 internal test images (7 natural + 3 synthetic). Pass rate 9/10; results may vary with different subjects, lighting, or image resolution.
- **Render font**: Braille block output requires a terminal / video player with a Braille-capable font. In environments without one, glyph density may render as boxes.
- **Motion model**: parallax warp is a geometric approximation; it does not understand scene semantics (limbs, occlusion boundaries). Complex backgrounds may show tearing artifacts.
- **Subject extraction**: `--subject-only` uses rembg (u2net); results degrade on cluttered backgrounds or transparent subjects.
- **Install size**: ~3 GB due to PyTorch. A lightweight CPU-only variant is not yet available.
- **Platform**: tested on Linux (Ubuntu 22.04) and macOS 14. Windows is untested (WSL2 should work).
- **AI motion (i2v)**: v0.4 feature, not included in v0.1. Requires 24 GB+ VRAM.

Ghostty is the aesthetic inspiration; aa-animator is not affiliated with or endorsed by the Ghostty project.

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, commit style, and the personal-info leak check procedure. Please read the security policy in [SECURITY.md](SECURITY.md) before reporting vulnerabilities.

Feedback is welcome — if something breaks or looks wrong, please [open an issue](https://github.com/hinanohart/aa-animator/issues) or start a [Discussion](https://github.com/hinanohart/aa-animator/discussions).

---

*Built by [Hinano Hart](https://github.com/hinanohart)*

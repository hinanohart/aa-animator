# aa-animator

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/aa-animator.svg)](https://pypi.org/project/aa-animator/)

**Turn any still photo into an animated ASCII art video — subject only, Ghostty-grade quality.**

One image in, one MP4 out. Depth-parallax warp drives the motion; edge-aware rendering and blue-glow contouring match Ghostty's visual standard. No hand-drawn frames, no fixed silhouette.

---

## Quick Start

```bash
pip install aa-animator
aa-animator animate photo.jpg out.mp4
```

Open `out.mp4` in any player, or pipe into `ffplay` for an instant preview.

---

## Features

- **Depth-parallax motion** — Depth Anything v2 Small (Apache-2.0, ~99 MB) estimates per-pixel depth; a forward-warp orbit animates ±8° of camera motion across 120 frames at 30 fps
- **Ghostty-grade AA rendering** — 11-level density ramp, DoG+Sobel directional glyphs (`|/─\`), blue glow contour `(70, 130, 255)` matching the Ghostty +boo aesthetic
- **Temporal stability** — EMA smoothing (α = 0.3) and time-fixed Bayer 4×4 dithering keep flicker below 5% avg char edit distance
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

Mini PoC (internal evaluation, 2026-04-18):
- Canvas: 100 × 41 chars (matching Ghostty canonical size)
- Duration: 10 frames @ 30 fps
- Flicker: **3.4% avg char edit distance** (threshold < 15%)
- Pipeline time: ~10 s on CPU (DA-V2 model download included)

---

## Evaluation metrics

| Metric | Mini PoC (v0.0.1) | v0.1 target |
|---|---|---|
| Temporal flicker (char edit dist/frame) | 3.4% | < 15% |
| SSIMULACRA2 vs Ghostty reference | not yet measured | ±10% |
| Subject boundary jitter (IoU variance) | N/A (no seg in PoC) | < 0.02 |
| Information density (char entropy) | not yet measured | > 2.5 bits |

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

## Contributing

Issues and PRs welcome. Please read the security policy in [SECURITY.md](SECURITY.md) before reporting vulnerabilities.

---

*Built by [Hinano Hart](https://github.com/hinanohart)*

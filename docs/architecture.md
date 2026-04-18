# Architecture

## Pipeline overview

```
[Still image input]
        |
        v
[Depth estimation] — depth/depth_anything_v2.py
  Depth Anything v2 Small (Apache-2.0, ~99 MB)
  CPU: ~9.6 s first run (model download), ~200 ms subsequent
  GPU: <50 ms
        |
        v
[Parallax warp] — motion/parallax.py
  Forward-scatter warp via scatter accumulation
  ±8° orbit, 120 frames @ 30 fps (configurable)
  Produces raw warped frames (holes permitted in v0.1.0)
        |
        v
[AA rendering] — render/{density,edge,block,braille,glow}.py
  1. Sobel + DoG edge map → directional glyphs |/─\
  2. 6D shape vector (Alex Harri method) + 11-level density ramp
  3. Edge priority over density (threshold 0.65)
  4. Blue glow contour (70, 130, 255) via dilate + alpha blend
        |
        v
[Temporal smoothing] — temporal.py
  EMA α = 0.3 on per-cell character choice
  Time-fixed Bayer 4×4 dithering (seed-locked)
        |
        v
[Subject masking] — subject/base.py  (v0.2+, extras=[matte])
  Applied as alpha blend at AA render stage
  Order: depth (full image) → parallax → mask compositing
  Avoids foreground depth flattening (critic gate 1 fix)
        |
        v
[Video output] — io/video_out.py
  ffmpeg subprocess pipe (no intermediate PNG files)
  MP4 / asciinema .cast (v0.3+)
```

## Module responsibilities

| Module | Responsibility | v0.1 status |
|---|---|---|
| `cli.py` | argparse entry point, `animate`/`preview`/`bake` subcommands | skeleton |
| `pipeline.py` | orchestrator, wires modules together | skeleton |
| `config.py` | dataclass holding all run parameters | skeleton |
| `temporal.py` | EMA + Bayer dither | skeleton |
| `io/image_in.py` | PIL open + resize + normalise | skeleton |
| `io/video_out.py` | ffmpeg pipe writer | skeleton |
| `depth/depth_anything_v2.py` | DA-V2 Small wrapper, CPU/GPU auto-select | skeleton |
| `motion/parallax.py` | forward-scatter warp, orbit generation | skeleton |
| `render/charsets.py` | density ramps, directional glyph tables | skeleton |
| `render/color.py` | 7 colour mode mappers | skeleton |
| `render/density.py` | density-ramp renderer | skeleton |
| `render/edge.py` | DoG+Sobel edge renderer | skeleton |
| `render/glow.py` | blue glow contour compositor | skeleton |
| `subject/base.py` | abstract mask interface | v0.2 |
| `subject/birefnet.py` | BiRefNet (MIT) adapter | v0.2 |
| `subject/sam2.py` | SAM 2 (Apache) adapter | v0.2 |

## Why self-implemented forward warp (not DepthFlow)

DepthFlow uses AGPL-3.0. Importing it would create a copyleft propagation
risk for this Apache-2.0 package. FSF FAQ permits AGPL tools called via
`subprocess` (separate programs, file/pipe I/O only), but direct import
is not safe.

The self-implemented parallax warp uses PyTorch scatter operations derived
from the equations in Niklaus & Liu (CVPR 2020, arXiv:2003.07360). No
source code from `sniklaus/softmax-splatting` was used (that repository
carries no OSI license).

## Why forward warp v0.1.0 uses simple scatter (holes permitted)

Forward scatter maps source pixels to destination pixels according to
depth-derived displacement. When two source pixels map to the same
destination cell, the farther one is occluded. When no source pixel maps
to a destination cell (disocclusion), the cell is left empty (black).

v0.1.0 accepts these holes because:
1. Ghostty's own AA aesthetic embraces sparse, dark backgrounds
2. Hole-filling (iterative dilate + cv2.inpaint TELEA) adds ~200 lines
   and is deferred to v0.1.1 to avoid delaying the release

## Why Depth Anything v2 Small only (not Base/Large)

DA-V2 Base and Large are licensed CC-BY-NC-4.0 (non-commercial only)
per their HuggingFace model cards. Only the Small variant is Apache-2.0.
This project ships Apache-2.0 and must not depend on non-commercial weights.

## Depth estimation order (critic gate 1 fix)

Original proposal: `seg → depth → parallax`
Problem: segmenting first, then estimating depth only on the foreground,
causes the subject's depth map to appear flat (no background gradient to
anchor relative depth). This produces a "paper cutout" artefact when warped.

Fixed order: `depth (full image) → parallax → subject mask compositing`
The full-image depth map captures correct relative depth. The mask is
applied only at the final AA render stage as an alpha-blend gate.

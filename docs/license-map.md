# License map

This document catalogs all components considered for aa-animator,
classified by commercial distribution safety.

## GREEN — safe to bundle or depend on (Apache-2.0 compatible)

| Component | License | Usage |
|---|---|---|
| Depth Anything v2 **Small** | Apache-2.0 | Core depth model (weights downloaded at runtime) |
| BiRefNet | MIT | Subject extraction (extras=[matte], v0.2+) |
| SAM 2 tiny | Apache-2.0 | Interactive subject extraction (extras=[matte], v0.2+) |
| ViTMatte | MIT | Alpha matting refinement (v0.2+ optional) |
| rembg | MIT | Lightweight subject extraction (extras=[matte]) |
| MoGe | MIT | Geometry estimation alternative (v1.0+ research) |
| Wan 2.1 I2V-14B-480P (code + weights) | Apache-2.0 | i2v motion (extras=[i2v], v0.4+) |
| LTX-Video (code only) | Apache-2.0 | i2v alternative — code safe, weights GRAY |
| PyTorch | BSD-3-Clause | Tensor operations |
| NumPy | BSD-3-Clause | Array operations |
| Pillow | HPND | Image I/O |
| OpenCV headless | Apache-2.0 | Video encode, edge detection |
| SciPy | BSD-3-Clause | Interpolation |
| Transformers | Apache-2.0 | Model loading |
| HuggingFace Hub | Apache-2.0 | Weight download |
| tqdm | MIT/MPL-2.0 | Progress display |
| diffusers | Apache-2.0 | i2v pipeline |
| accelerate | Apache-2.0 | Multi-device support |
| hatchling | MIT | Build backend |

## GRAY — code OK but weights carry restrictions

| Component | Restriction | Decision |
|---|---|---|
| LTX-Video weights | OpenRAIL-M (use restrictions) | Do not bundle weights; user downloads separately |
| Wan 2.2 I2V-A14B | Apache-2.0 but 24-40 GB VRAM | Optional external CLI path only |

## RED — excluded entirely

| Component | License | Reason |
|---|---|---|
| **DepthFlow** | AGPL-3.0 | Copyleft propagation risk if imported; may only be called via `subprocess` per FSF FAQ |
| **Depth Anything v2 Base/Large** | CC-BY-NC-4.0 | Non-commercial only; contradicts Apache-2.0 distribution |
| **RMBG-2.0 (BRIA)** | BRIA non-commercial | Non-commercial only; explicitly not redistributable |
| **CogVideoX-5B** | Proprietary CogVideoX License | Registration required, MAU cap 1M, not OSI-compatible |
| **PIFuHD** | CC-BY-NC-4.0 | Non-commercial only |
| **DUSt3R** | CC-BY-NC-4.0 | Non-commercial only |
| **InsightFace** (LivePortrait dep.) | Non-commercial | Bundled with LivePortrait; poisons MIT claim |
| **SVD XT** | SVD Community (non-commercial) | Non-commercial only |
| **HunyuanVideo 1.5** | Hunyuan Community | EU/UK/KR excluded, 100M MAU cap |
| `sniklaus/softmax-splatting` | No license | No OSI license present; fork and adaptation prohibited |

## DepthFlow subprocess exception

DepthFlow may be called via `subprocess.run()` from aa-animator without
triggering AGPL propagation, per the FSF FAQ ("separate programs").
Conditions:

1. Only `subprocess.run()` — never `import depthflow`
2. Communication via pipe, stdout, or file I/O only
3. DepthFlow wheel must not be bundled in aa-animator's sdist or wheel
4. README must document the manual install step

This is the same precedent used by Apache Beam and MLflow calling GPL-build
ffmpeg binaries.

## Round G audit date

2026-04-18. Re-audit required before each PyPI release to catch
upstream license changes.

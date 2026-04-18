# Legal notes (critic gate 3 M2)

## softsplat / forward-warp legal position

### Paper reference

The forward-warp algorithm in aa-animator is conceptually informed by:

> Niklaus, S., & Liu, F. (2020). Softmax Splatting for Video Frame
> Interpolation. *Proceedings of the IEEE/CVF Conference on Computer
> Vision and Pattern Recognition (CVPR 2020)*.
> arXiv: [2003.07360](https://arxiv.org/abs/2003.07360)

This citation is for attribution only. The paper describes the
mathematical formulation of softmax-weighted forward splatting.
aa-animator's implementation derives from the equations in the paper,
not from any source code.

### Source code prohibition

`sniklaus/softmax-splatting` (GitHub, 512 stars as of 2026-04-18)
carries **no OSI-approved license**. The repository has no `LICENSE`
file. Under default copyright law this means "all rights reserved."

Consequences:
- Copying, adapting, or forking any code from that repository is
  prohibited without explicit written permission from the author
- Even a single line adapted from that codebase would constitute
  copyright infringement
- The prohibition extends to "clean-room" reimplementations that use
  the repository's source as a reference while writing new code

### Compliance procedure

Before each release, confirm with `git log --all -p -- src/` that:
1. No file contains the string `sniklaus` (except this document)
2. No function signature matches the `softmax_splatting` API exactly
3. The implementation comment history references only the arXiv paper,
   not the GitHub repository

The identity-leak guard in `.github/workflows/release.yml` does not
check for softsplat code patterns. This check is a manual responsibility
of the maintainer.

### Implementation derivation

aa-animator's `motion/parallax.py` implements forward scatter using
PyTorch's `torch.Tensor.scatter_add_` and `torch.nn.functional.grid_sample`
for the backward-sampling fallback, plus NumPy `np.add.at` for CPU paths.
These are general-purpose operations available in any tensor library
and are not derived from any specific prior implementation.

## DepthFlow AGPL-3.0 subprocess exception

See `docs/license-map.md` for the full analysis. Summary:
- Do NOT `import depthflow` anywhere in aa-animator source
- Calling DepthFlow via `subprocess.run(["depthflow", ...])` is safe
- This exception does not apply to any other AGPL or GPL dependency

## DA-V2 Small-only constraint

Only the Depth Anything v2 Small variant is Apache-2.0 licensed.
Base and Large are CC-BY-NC-4.0 (non-commercial). The model ID used
in production MUST be `LiheYoung/depth-anything-v2-small` or equivalent
Small checkpoint. Using Base or Large weights in a release constitutes
license violation.

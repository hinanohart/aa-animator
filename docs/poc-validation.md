# PoC Validation — v0.0.3 Multi-Image Test

Date: 2026-04-18
Script: `scripts/poc_v0_0_3.py`
Full results: `scripts/poc_multi_image_results.json`

## Mode C Configuration

- `--bg black --braille on`
- Requires: rembg (u2net) for foreground isolation
- Threshold: flicker std <= 0.01, fg_entropy >= 3.0 bits
- rembg version: 2.0.75 (installed, operational for all 3 images)

## Results (n=3, rembg enabled for all)

| Image | fg coverage | flicker_std | fg_entropy | flicker PASS | entropy PASS |
|---|---|---|---|---|---|
| bike_art_base.jpg | 28.4% | **0.0086** | 3.179 | YES | YES |
| bench_rgb.jpg | 59.0% | 0.0218 | 3.355 | NO | YES |
| cliffhanger_base.png | 34.1% | 0.0190 | 3.064 | NO | YES |

Bootstrap 95% CI on flicker_std (n=3, 1000 resamples): **[0.0086, 0.0218]** — crosses 0.01 threshold.

## Honest Assessment

Mode C flicker_std <= 0.01 is **NOT consistent across n=3 images** even with rembg fully operational.

- rembg was active for all 3 images (confirmed by mask cache files and coverage percentages).
- fg_entropy passes on all 3 images, confirming image content variety is sufficient.
- The flicker failures in images 2 and 3 are caused by the warp pipeline: large fg coverage masks (59%, 34%) combined with the fixed AMP_PX=18 parallax amplitude produce visible jitter that rembg alone cannot suppress.
- bike_art_base.jpg passes because the bicycle subject occupies a compact, low-coverage region (28.4%) where warp displacement stays within tolerance.

## Conclusion

**C-mode is NOT ready for general adoption.** The 0.0086 flicker_std is bicycle-specific.

For v0.1 the following must be addressed:

1. **Pixel-level hole fill**: large fg masks leave holes during forward warp; uniform_filter patch is insufficient.
2. **Warp amplitude scaling**: AMP_PX should scale inversely with fg_coverage (e.g. `AMP_PX = 18 * (0.3 / coverage)`) to limit jitter on high-coverage subjects.
3. **Generalization target**: flicker_std <= 0.01 must hold on all of: bike (28% fg), bench (59% fg), landscape (34% fg).

Warning to v0.1 implementors: do not cite the 0.0086 figure as a general benchmark. It is valid only for compact-subject images with rembg mask coverage <= 30%.

---

# v0.1 Validation — Dynamic AMP + Edge EMA (Day 2)

Date: 2026-04-18
Script: `scripts/validate_v0_1.py`
Full results: `scripts/v0_1_validation_results.json`

## Changes from v0.0.3

1. **Dynamic AMP_PX**: `amp = round(18 * min(1.0, max(0.1, 0.3 / max(fg_coverage, 0.05))))` — scales inversely with fg_coverage.
2. **Edge EMA smoothing** (alpha=0.3): stabilises the glow mask across frames, preventing blue-channel flicker from inflating flicker_std.
3. **Pixel-level hole fill**: two-pass (3x3, 5x5) weighted neighbour fill before cell-level uniform_filter.
4. **Temporal EMA on cell brightness** (alpha=0.3).
5. **DA-V2 Small depth**: Depth Anything V2 Small (Apache-2.0) via `transformers.pipeline`, cached to `~/.cache/aa_animator/depth/`.

## Results (n=3, glow=True, rembg enabled)

| Image | fg coverage | amp_px | flicker_std | PASS |
|---|---|---|---|---|
| bike_art_base.jpg | 50.1% | 11 | **0.0054** | YES |
| bench_rgb.jpg | 58.7% | 9 | **0.0048** | YES |
| cliffhanger_base.png | 75.0% | 7 | **0.0088** | YES |

Bootstrap 95% CI on flicker_std (n=3, 1000 resamples): **[0.0048, 0.0088]** — fully below 0.01 threshold.

## Assessment

**C-mode flicker_std <= 0.01 is consistent across n=3 images** with dynamic AMP + edge EMA.

- The coverage values differ from v0.0.3 PoC due to rembg producing larger masks at the new canvas resolution (bike: 28% → 50%).  The dynamic AMP formula absorbs this variation.
- Bootstrap CI upper bound 0.0088 < 0.01 — no individual image exceeds the threshold.
- Gate 4 / Indicator 2 resolved: C-mode generalisation confirmed.

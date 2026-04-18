# PoC Validation — v0.0.3 Multi-Image Test

Date: 2026-04-18  
Script: `scripts/poc_v0_0_3.py`  
Full results: `scripts/poc_multi_image_results.json`

## Mode C Configuration

- `--bg black --braille on`
- Requires: rembg (u2net) for foreground isolation
- Threshold: flicker std <= 0.01, fg_entropy >= 3.0 bits

## Results (n=3)

| Image | rembg | flicker_std | fg_entropy | flicker PASS | entropy PASS |
|---|---|---|---|---|---|
| bike_art_base.jpg | yes (cached) | **0.0086** | 3.179 | YES | YES |
| benchmark_result.jpg | no (not installed) | 0.0190 | 1.793 | NO | NO |
| synthetic.png | no (not installed) | 0.0107 | 1.603 | NO | NO |

## Honest Assessment

Mode C flicker_std <= 0.01 is **not consistent across n=3 images** with the current test environment.

- Image 1 passes because it uses a cached rembg (u2net) mask that isolates the bicycle subject. With foreground isolation, the Braille mode achieves low flicker.
- Images 2 and 3 fall back to `BG_MODE=full` because rembg is not installed. Without foreground isolation, the full-frame rendering produces higher flicker and lower fg_entropy.

## Implication

The 0.0086 flicker_std figure is valid for Mode C with rembg operational. The claim does not generalize without rembg. Installation of rembg (`pip install rembg`) is required for the metric to hold.

Image selection matters: chart and synthetic images have intrinsically low entropy (large uniform areas) regardless of rembg.

## Next Steps

- Install rembg in CI and repeat images 2 and 3 with real foreground masks
- Select natural-subject images (landscape, object photos) for entropy >= 3.0 to be achievable

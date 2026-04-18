# Copyright 2026 Hinano Hart <hinanohart@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for aa-animator.

Entry point: aa-animator (defined in pyproject.toml [project.scripts])

Subcommands:
    animate  INPUT [OUTPUT]  — render animated ASCII art MP4
    preview  INPUT           — single-frame terminal preview
    bake     INPUT OUTDIR    — render gallery of style × colour variants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from aa_animator_v2._version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aa-animator",
        description="Still image → animated ASCII art video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"aa-animator {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── animate ──────────────────────────────────────────────────────────────
    animate = sub.add_parser(
        "animate",
        help="render animated ASCII art to MP4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    animate.add_argument("input", metavar="INPUT", help="path to input image")
    animate.add_argument(
        "output",
        metavar="OUTPUT",
        nargs="?",
        default=None,
        help="output MP4 path (default: <input stem>_aa.mp4)",
    )
    animate.add_argument(
        "--style",
        choices=["density", "edge", "block", "braille"],
        default="density",
    )
    animate.add_argument(
        "--color",
        choices=["color", "mono", "matrix", "cyber", "amber", "gradient", "invert"],
        default="color",
    )
    animate.add_argument("--cols", type=int, default=100, help="character canvas width")
    animate.add_argument("--fps", type=int, default=30)
    animate.add_argument("--duration", type=float, default=4.0, help="clip length in seconds")
    animate.add_argument("--amp-deg", type=float, default=8.0, help="parallax orbit amplitude (degrees)")
    animate.add_argument("--ema", type=float, default=0.3, help="temporal EMA smoothing coefficient")
    animate.add_argument("--glow", action=argparse.BooleanOptionalAction, default=True)
    animate.add_argument(
        "--depth-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    animate.add_argument("--seed", type=int, default=42)
    animate.add_argument(
        "--subject-only",
        action="store_true",
        default=False,
        help="foreground-only AA (requires pip install 'aa-animator[matte]')",
    )
    # Shorthand aliases used by validation scripts
    animate.add_argument(
        "--mode",
        choices=["braille", "ascii"],
        default=None,
        help="rendering mode override (overrides --style)",
    )
    animate.add_argument(
        "--bg",
        choices=["black", "ghostty_fill"],
        default=None,
        help="background mode override (overrides --subject-only)",
    )
    animate.add_argument(
        "--dither",
        choices=["none", "bayer"],
        default="none",
        help="ordered dither mode for brightness quantisation",
    )

    # ── preview ───────────────────────────────────────────────────────────────
    preview = sub.add_parser(
        "preview",
        help="single-frame terminal preview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    preview.add_argument("input", metavar="INPUT", help="path to input image")
    preview.add_argument("--style", choices=["density", "edge", "block", "braille"], default="density")
    preview.add_argument(
        "--color",
        choices=["color", "mono", "matrix", "cyber", "amber", "gradient", "invert"],
        default="color",
    )
    preview.add_argument("--cols", type=int, default=100)
    preview.add_argument("--depth-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # ── bake ──────────────────────────────────────────────────────────────────
    bake = sub.add_parser(
        "bake",
        help="render gallery of style × colour variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bake.add_argument("input", metavar="INPUT", help="path to input image")
    bake.add_argument("outdir", metavar="OUTDIR", help="output directory for rendered variants")

    return parser


def _cmd_animate(args: argparse.Namespace) -> int:
    from aa_animator_v2.pipeline import AAAnimator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[aa-animator] error: input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_aa.mp4")

    # --mode overrides --style; --bg overrides --subject-only
    if args.mode is not None:
        mode = args.mode
    else:
        mode = "braille" if args.style == "braille" else "ascii"
    if args.bg is not None:
        bg: str = args.bg
    else:
        bg = "black" if args.subject_only else "ghostty_fill"

    animator = AAAnimator(
        mode=mode,  # type: ignore[arg-type]
        bg=bg,  # type: ignore[arg-type]
        fps=args.fps,
        cols=args.cols,
        n_frames=int(args.duration * args.fps),
        amp_px=args.amp_deg,
        glow=args.glow,
        dither=args.dither,  # type: ignore[arg-type]
    )

    print(
        f"[aa-animator] animate: input={input_path} output={output_path} "
        f"style={args.style} cols={args.cols} fps={args.fps} "
        f"duration={args.duration}s glow={args.glow}",
        file=sys.stderr,
    )

    try:
        animator.animate(input_path, output_path)
    except Exception as exc:
        print(f"[aa-animator] error: {exc}", file=sys.stderr)
        return 1

    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from aa_animator_v2.pipeline import AAAnimator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[aa-animator] error: input not found: {input_path}", file=sys.stderr)
        return 1

    mode = "braille" if args.style == "braille" else "ascii"
    output_path = input_path.with_name(input_path.stem + "_preview.png")

    animator = AAAnimator(mode=mode, bg="black", cols=args.cols, glow=True)  # type: ignore[arg-type]

    print(
        f"[aa-animator] preview: input={input_path} output={output_path} style={args.style}",
        file=sys.stderr,
    )

    try:
        animator.preview(input_path, output_path)
    except Exception as exc:
        print(f"[aa-animator] error: {exc}", file=sys.stderr)
        return 1

    return 0


def _cmd_bake(args: argparse.Namespace) -> int:
    from aa_animator_v2.pipeline import AAAnimator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[aa-animator] error: input not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.outdir)
    animator = AAAnimator(mode="braille", bg="black", glow=True)

    print(
        f"[aa-animator] bake: input={input_path} outdir={out_dir}",
        file=sys.stderr,
    )

    try:
        animator.bake(input_path, out_dir)
    except Exception as exc:
        print(f"[aa-animator] error: {exc}", file=sys.stderr)
        return 1

    return 0


def main() -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "animate": _cmd_animate,
        "preview": _cmd_preview,
        "bake": _cmd_bake,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()

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
    print(
        f"[aa-animator] animate: input={args.input!r} style={args.style} "
        f"color={args.color} cols={args.cols} fps={args.fps} "
        f"duration={args.duration}s amp={args.amp_deg}° "
        f"ema={args.ema} glow={args.glow} device={args.depth_device}",
        file=sys.stderr,
    )
    print("[aa-animator] NOT IMPLEMENTED — full pipeline coming in v0.1", file=sys.stderr)
    return 1


def _cmd_preview(args: argparse.Namespace) -> int:
    print(
        f"[aa-animator] preview: input={args.input!r} style={args.style}",
        file=sys.stderr,
    )
    print("[aa-animator] NOT IMPLEMENTED — full pipeline coming in v0.1", file=sys.stderr)
    return 1


def _cmd_bake(args: argparse.Namespace) -> int:
    print(
        f"[aa-animator] bake: input={args.input!r} outdir={args.outdir!r}",
        file=sys.stderr,
    )
    print("[aa-animator] NOT IMPLEMENTED — full pipeline coming in v0.1", file=sys.stderr)
    return 1


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

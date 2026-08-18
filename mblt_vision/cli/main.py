"""Standalone command-line entry point for Mobilint Vision."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .compile import add_compile_parser
from .predict import add_predict_parser
from .val import add_val_parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Vision command parser."""

    parser = argparse.ArgumentParser(
        prog="mblt-vision",
        description="Run, validate, and compile Mobilint Vision models.",
    )
    subparsers = parser.add_subparsers(help="mblt-vision commands")
    add_predict_parser(subparsers)
    add_val_parser(subparsers)
    add_compile_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone Vision command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "_handler"):
        return int(args._handler(args))
    parser.print_help()
    return 1

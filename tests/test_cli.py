"""Tests for the standalone Vision CLI parser."""

from __future__ import annotations

from mblt_vision.cli import build_parser
from mblt_vision.cli.compile import _run_compile
from mblt_vision.cli.predict import _cmd_predict
from mblt_vision.cli.val import _cmd_val


def test_standalone_cli_registers_vision_commands() -> None:
    """Expose the supported Vision commands from the standalone distribution."""

    parser = build_parser()

    predict_args = parser.parse_args(
        ["predict", "--source", "image.jpg", "--model", "resnet50"]
    )
    val_args = parser.parse_args(["val", "--model", "resnet50"])
    compile_args = parser.parse_args(
        ["compile", "--model-cls", "resnet50", "--target-device", "aries-rb"]
    )

    assert predict_args._handler is _cmd_predict
    assert val_args._handler is _cmd_val
    assert compile_args._handler is _run_compile
    assert compile_args.target_device == "aries-rb"
    assert predict_args.target_device == "aries-rb"
    assert val_args.target_device == "aries-rb"


def test_vision_cli_uses_single_core_mode_by_default_on_regulus() -> None:
    """Keep the CLI's implicit core mode compatible with a Regulus target."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--source",
            "image.jpg",
            "--model",
            "resnet50",
            "--target-device",
            "regulus-ra",
        ]
    )

    assert args.target_device == "regulus-ra"
    assert args.core_mode is None

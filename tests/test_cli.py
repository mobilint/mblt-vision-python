"""Tests for the standalone Vision CLI parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from mblt_vision.cli import build_parser
from mblt_vision.cli.compile import _run_compile
from mblt_vision.cli.predict import _cmd_predict
from mblt_vision.cli import val as val_module
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


@pytest.mark.parametrize("batch_size", ["0", "-1"])
def test_val_rejects_nonpositive_batch_sizes(batch_size: str) -> None:
    """Fail argument parsing before validation can construct a model."""

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["val", "--model", "resnet50", "--batch-size", batch_size]
        )


@pytest.mark.parametrize("removed_alias", ["classify", "detect", "pose", "segment"])
def test_predict_command_has_no_task_aliases(removed_alias: str) -> None:
    """Keep prediction discoverable through one task-agnostic command."""

    with pytest.raises(SystemExit):
        build_parser().parse_args([removed_alias])


def test_predict_help_explains_supported_workflows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Describe tasks, output behavior, framework choice, and examples in CLI help."""

    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args(["predict", "--help"])
    help_text = capsys.readouterr().out

    assert "Supported tasks:" in help_text
    assert "--framework onnx" in help_text
    assert "--target-device regulus-ra" in help_text


@pytest.mark.parametrize(
    ("cache_name", "candidates"),
    [
        ("nyu-depth", ["nyu-depth.zip", "nyu-depth"]),
    ],
)
def test_find_existing_source_never_returns_the_organized_cache(
    tmp_path: Path, cache_name: str, candidates: list[str]
) -> None:
    """An incomplete cache must not be handed back as its own raw source.

    Some datasets use a cache directory whose name is also a source
    candidate, so `data_path.parent / name` resolves back to `data_path`;
    organizing from it fails instead of downloading the default archive.
    """

    from mblt_vision.cli.val import _find_existing_source

    data_path = tmp_path / "datasets" / cache_name
    data_path.mkdir(parents=True)
    assert _find_existing_source(str(data_path), candidates) is None

    # A genuine sibling source is still discovered.
    real_source = data_path.parent / candidates[0]
    real_source.write_bytes(b"archive")
    assert _find_existing_source(str(data_path), candidates) == str(real_source)


def test_validation_default_dataset_path_uses_resolved_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep default organization alongside artifacts when the home cache falls back."""

    configured_path = Path.home() / ".mblt_model_zoo" / "datasets" / "coco"
    fallback_cache = tmp_path / "cache"
    monkeypatch.setattr(
        val_module,
        "get_dataset_config_for_task",
        lambda _task, _dataset: {"path": str(configured_path)},
    )
    monkeypatch.setattr(
        val_module, "get_mobilint_cache_dir", lambda: str(fallback_cache)
    )

    assert val_module._default_data_path_for_task("object_detection") == str(
        fallback_cache / "datasets" / "coco"
    )

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


def test_predict_parses_point_prompts_for_mask_generation() -> None:
    """Accept repeated `--point X,Y,LABEL` prompts and mask-generation path overrides."""

    args = build_parser().parse_args(
        [
            "predict",
            "--source",
            "image.jpg",
            "--model",
            "sam2-hiera-large",
            "--point",
            "320,240,1",
            "--point",
            "10.5,20.5,0",
            "--encoder-mxq-path",
            "encoder.mxq",
            "--decoder-mxq-path",
            "decoder.mxq",
            "--encoder-onnx-path",
            "encoder.onnx",
            "--decoder-onnx-path",
            "decoder.onnx",
        ]
    )
    assert args.points == [(320.0, 240.0, 1), (10.5, 20.5, 0)]
    assert args.encoder_mxq_path == "encoder.mxq"
    assert args.decoder_mxq_path == "decoder.mxq"
    assert args.encoder_onnx_path == "encoder.onnx"
    assert args.decoder_onnx_path == "decoder.onnx"


@pytest.mark.parametrize("bad_point", ["320,240", "320,240,2", "x,240,1"])
def test_predict_rejects_malformed_point_prompts(bad_point: str) -> None:
    """Fail argument parsing on malformed or out-of-range point prompts."""

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "predict",
                "--source",
                "image.jpg",
                "--model",
                "sam2-hiera-large",
                "--point",
                bad_point,
            ]
        )


def test_predict_rejects_points_for_non_mask_generation_models(
    synthetic_image_path: Path,
) -> None:
    """Reject `--point` before constructing an engine for a non-promptable model."""

    from mblt_vision.cli._vision import run_vision_inference

    args = build_parser().parse_args(
        [
            "predict",
            "--source",
            str(synthetic_image_path),
            "--model",
            "resnet50",
            "--point",
            "320,240,1",
        ]
    )
    with pytest.raises(SystemExit, match="only supported for mask generation"):
        run_vision_inference(args, command="predict")


@pytest.mark.parametrize(
    ("extra_args", "match"),
    [
        ([], "1 to 3 point prompts"),
        (
            [
                "--point",
                "1,1,1",
                "--point",
                "2,2,1",
                "--point",
                "3,3,1",
                "--point",
                "4,4,0",
            ],
            "1 to 3 point prompts",
        ),
        (["--point", "1,1,1", "--mxq-path", "model.mxq"], "encoder-mxq-path"),
        (["--point", "1,1,1", "--onnx-path", "model.onnx"], "encoder-onnx-path"),
    ],
)
def test_mask_generation_prompt_validation_fails_before_engine_construction(
    synthetic_image_path: Path, extra_args: list[str], match: str
) -> None:
    """Reject invalid mask-generation invocations without loading any backend."""

    from mblt_vision.cli._vision import run_vision_inference

    args = build_parser().parse_args(
        [
            "predict",
            "--source",
            str(synthetic_image_path),
            "--model",
            "sam2-hiera-large",
            *extra_args,
        ]
    )
    with pytest.raises(SystemExit, match=match):
        run_vision_inference(args, command="predict")


@pytest.mark.parametrize(
    ("path_arg", "match"),
    [
        (["--model-path", "model.mxq"], "encoder-mxq-path"),
        (["--mxq-path", "model.mxq"], "encoder-mxq-path"),
        (["--onnx-path", "model.onnx"], "encoder-onnx-path"),
    ],
)
def test_val_rejects_single_artifact_paths_for_mask_generation(
    path_arg: list[str], match: str
) -> None:
    """Refuse to silently evaluate the downloaded default instead of the request.

    Validation accepting and ignoring a single-artifact path would invalidate
    experiment results, so it must fail the same way prediction does.
    """

    from mblt_vision.cli._vision import create_mask_generation_engine

    args = build_parser().parse_args(["val", "--model", "sam2-hiera-large", *path_arg])
    with pytest.raises(SystemExit, match=match):
        create_mask_generation_engine(args)


def test_val_parses_mask_generation_options() -> None:
    """Expose SA-V evaluation protocol knobs and artifact-path overrides on val."""

    args = build_parser().parse_args(
        [
            "val",
            "--model",
            "sam2-hiera-large",
            "--num-samples",
            "20",
            "--num-points",
            "2",
            "--seed",
            "7",
            "--encoder-mxq-path",
            "encoder.mxq",
            "--decoder-mxq-path",
            "decoder.mxq",
            "--encoder-onnx-path",
            "encoder.onnx",
            "--decoder-onnx-path",
            "decoder.onnx",
        ]
    )
    assert args.num_samples == 20
    assert args.num_points == 2
    assert args.seed == 7
    assert args.encoder_mxq_path == "encoder.mxq"
    assert args.decoder_mxq_path == "decoder.mxq"
    assert args.encoder_onnx_path == "encoder.onnx"
    assert args.decoder_onnx_path == "decoder.onnx"


def test_val_defaults_match_the_reference_protocol() -> None:
    """Default to the reference-validated 200-sample single-point protocol."""

    args = build_parser().parse_args(["val", "--model", "sam2-hiera-large"])
    assert args.num_samples == 200
    assert args.num_points == 1
    assert args.seed == 0


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

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


@pytest.mark.parametrize(
    "bad_point",
    # nan/inf parse fine as floats, so they must be rejected explicitly:
    # they would otherwise contaminate Fourier prompt encoding downstream.
    ["320,240", "320,240,2", "x,240,1", "nan,240,1", "320,inf,1", "-inf,240,0"],
)
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


@pytest.mark.parametrize(
    ("cache_name", "candidates"),
    [
        ("sa-v", ["sav_val.tar", "sa-v", "sav_val"]),
        ("nyu-depth", ["nyu-depth.zip", "nyu-depth"]),
    ],
)
def test_find_existing_source_never_returns_the_organized_cache(
    tmp_path: Path, cache_name: str, candidates: list[str]
) -> None:
    """An incomplete cache must not be handed back as its own raw source.

    Several datasets use a cache directory whose name is also a source
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


def test_val_requires_a_manually_downloaded_sav_archive(tmp_path: Path) -> None:
    """SA-V is gated by Meta and not mirrored, so there is no default source.

    The error must name the portal, the layout reference, and the flags that
    accept the archive, rather than falling back to a URL.
    """

    from mblt_vision.cli.val import _resolve_sav_source

    args = build_parser().parse_args(["val", "--model", "sam2-hiera-large"])
    data_path = tmp_path / "datasets" / "sa-v"
    data_path.mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        _resolve_sav_source(args, str(data_path))
    message = str(excinfo.value)
    assert "sav_val.tar" in message
    assert "ai.meta.com" in message
    assert "sav_dataset" in message
    assert "--annotation-dir" in message

    # A manually downloaded archive beside the dataset path is accepted.
    archive = data_path.parent / "sav_val.tar"
    archive.write_bytes(b"archive")
    assert _resolve_sav_source(args, str(data_path)) == str(archive)


def test_val_finds_the_sav_archive_even_under_force_organize(tmp_path: Path) -> None:
    """--force-organize rebuilds the dataset, it does not ignore the source.

    SA-V has no fallback download URL, so skipping discovery would fail on a
    manual archive sitting in the very location the error message recommends.
    """

    from mblt_vision.cli.val import _resolve_sav_source

    data_path = tmp_path / "datasets" / "sa-v"
    data_path.mkdir(parents=True)
    archive = data_path.parent / "sav_val.tar"
    archive.write_bytes(b"archive")

    args = build_parser().parse_args(
        ["val", "--model", "sam2-hiera-large", "--force-organize"]
    )
    assert args.force_organize is True
    assert _resolve_sav_source(args, str(data_path)) == str(archive)


def test_benchmark_runner_excludes_unsupported_mask_generation() -> None:
    """The unified runner cannot build SAM2's engine or dispatch eval_sav.

    Offering the task would accept `--task mask_generation` and then produce an
    error row for every model.
    """

    from benchmark import benchmark_vision_models
    from mblt_vision._tasks import VISION_TASKS

    assert "mask_generation" in VISION_TASKS  # still a canonical Vision task
    assert "mask_generation" not in benchmark_vision_models.TASK_CHOICES
    assert set(benchmark_vision_models.TASK_CHOICES) == set(VISION_TASKS) - {
        "mask_generation"
    }

    with pytest.raises(SystemExit):
        benchmark_vision_models._parse_args(
            ["--models", "ResNet50", "--task", "mask_generation"]
        )


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

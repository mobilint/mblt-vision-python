"""Shared helpers for vision CLI commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from mblt_vision._tasks import normalize_vision_task

DEFAULT_OUTPUT_DIR = Path("runs") / "vision"


def parse_unit_interval(value: str) -> float:
    """Parse a floating-point value strictly between zero and one."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a number in the open interval (0, 1)"
        ) from exc
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError(
            f"expected a number in the open interval (0, 1), got {value}"
        )
    return parsed


def parse_target_cores(value: str | None) -> list[str] | None:
    """Parses a semicolon-separated target core list."""

    if value is None:
        return None
    cores = [item.strip() for item in value.split(";") if item.strip()]
    return cores or None


def parse_target_clusters(value: str | None) -> list[int] | None:
    """Parses a semicolon-separated target cluster list."""

    if value is None:
        return None
    try:
        clusters = [int(item.strip()) for item in value.split(";") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target clusters must be semicolon-separated integers"
        ) from exc
    return clusters or None


def parse_point(value: str) -> tuple[float, float, int]:
    """Parses an `X,Y,LABEL` point prompt for mask generation models."""

    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected a point as X,Y,LABEL")
    try:
        x, y = float(parts[0]), float(parts[1])
        label = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected numeric X,Y coordinates and an integer LABEL"
        ) from exc
    if label not in (0, 1):
        raise argparse.ArgumentTypeError(
            "point LABEL must be 1 (positive) or 0 (negative)"
        )
    return (x, y, label)


def resolve_cli_task(args: argparse.Namespace) -> str:
    """Resolves the selected model's task without constructing a runtime."""

    from mblt_vision.wrapper import resolve_model_config

    config = resolve_model_config(args.model, args.model_type)
    return normalize_vision_task(config["post_cfg"]["task"])


def add_common_vision_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments shared by all vision inference commands."""

    parser.add_argument("--source", required=True, help="Path to the source image.")
    parser.add_argument(
        "--model",
        required=True,
        help="Vision model name, for example `resnet50` or `yolo11m`.",
    )
    parser.add_argument(
        "--output",
        "--save-path",
        dest="output",
        help="Path to save the plotted result image.",
    )
    parser.add_argument(
        "--framework",
        default=None,
        choices=["mxq", "onnx"],
        help="Inference framework to use. When omitted, `--model-path` suffix is used first, then `mxq`.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default="",
        help="Optional generic local model path for MXQ or ONNX inference.",
    )
    parser.add_argument(
        "--mxq-path",
        dest="mxq_path",
        default="",
        help="Optional local MXQ model path. Preserved as a compatibility alias.",
    )
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        default="",
        help="Optional local ONNX model path.",
    )
    parser.add_argument(
        "--model-type",
        default="DEFAULT",
        help="Model variant from the YAML configuration.",
    )
    parser.add_argument(
        "--core-mode",
        default=None,
        choices=["single", "multi", "global4", "global8"],
        help="NPU core execution mode. Defaults to global8 on Aries and single on Regulus.",
    )
    parser.add_argument("--dev-no", type=int, default=0, help="NPU device number.")
    parser.add_argument(
        "--target-device",
        default="aries-rb",
        choices=["aries-rb", "regulus-ra", "regulus-rb"],
        help="NPU board target. Determines the backend implementation.",
    )
    parser.add_argument(
        "--target-cores",
        type=parse_target_cores,
        help="Optional semicolon-separated core list for single-core mode, for example `0:0;0:1`.",
    )
    parser.add_argument(
        "--target-clusters",
        type=parse_target_clusters,
        help="Optional semicolon-separated cluster list for multi/global modes, for example `0;1`.",
    )


def add_threshold_args(
    parser: argparse.ArgumentParser,
    *,
    conf_default: float | None = 0.25,
    iou_default: float | None = None,
) -> None:
    """Adds postprocess threshold arguments for dense vision tasks."""

    parser.add_argument(
        "--conf-thres",
        type=parse_unit_interval,
        default=conf_default,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou-thres",
        type=parse_unit_interval,
        default=iou_default,
        help="IoU threshold.",
    )


def parse_bool(value: str) -> bool:
    """Parses a case-insensitive boolean CLI value.

    Args:
        value: Boolean text to parse.

    Returns:
        Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a supported boolean spelling.
    """

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value: true or false")


def add_e2e_arg(parser: argparse.ArgumentParser) -> None:
    """Adds an optional YOLO end-to-end postprocessing mode override.

    Leaving the option unset preserves the model configuration's default.
    """

    parser.add_argument(
        "--e2e",
        nargs="?",
        const=True,
        type=parse_bool,
        default=None,
        help="Enable or disable YOLO end-to-end postprocessing (true/false). Bare `--e2e` means true.",
    )


def add_vision_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command: str,
    help_text: str,
    handler: Any,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Creates a vision command parser with common arguments."""

    parser = subparsers.add_parser(
        command,
        help=help_text,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(_handler=handler)
    add_common_vision_args(parser)
    return parser


def build_default_output_path(command: str, source: str, model: str) -> str:
    """Builds the default path used for plotted vision command output."""

    source_path = Path(source)
    suffix = source_path.suffix or ".jpg"
    return str(DEFAULT_OUTPUT_DIR / command / f"{source_path.stem}_{model}{suffix}")


def resolve_output_path(
    output: str | None, command: str, source: str, model: str
) -> str:
    """Returns an absolute result image path and ensures its parent exists."""

    save_path = Path(
        output or build_default_output_path(command, source, model)
    ).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_path.resolve())


def require_source_file(source: str) -> None:
    """Exits with a clear message when the source image is unavailable."""

    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise SystemExit(f"Source image not found: {source}")


def create_vision_engine(args: argparse.Namespace) -> Any:
    """Creates a vision engine from shared CLI model options.

    Args:
        args: Parsed command options containing common vision model arguments.

    Returns:
        Initialized vision inference engine.

    Raises:
        SystemExit: If the vision runtime dependencies are unavailable.
    """

    try:
        from mblt_vision import MBLT_Engine
        from mblt_vision.wrapper import normalize_core_mode
    except ImportError as exc:
        print(f"Missing dependencies for vision CLI: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    postprocess_kwargs: dict[str, Any] = {}
    if getattr(args, "e2e", None) is not None:
        postprocess_kwargs["e2e"] = args.e2e

    return MBLT_Engine(
        model_cls=args.model,
        model_type=args.model_type,
        framework=args.framework,
        model_path=args.model_path,
        mxq_path=args.mxq_path,
        onnx_path=args.onnx_path,
        dev_no=args.dev_no,
        target_device=args.target_device,
        core_mode=normalize_core_mode(
            args.core_mode
            or ("single" if args.target_device.startswith("regulus-") else "global8")
        ),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
        postprocess_kwargs=postprocess_kwargs,
    )


def reject_single_artifact_paths(args: argparse.Namespace) -> None:
    """Rejects single-artifact model paths for two-artifact mask generation models.

    Raises:
        SystemExit: If `--model-path`, `--mxq-path`, or `--onnx-path` is set.
    """

    if (
        getattr(args, "model_path", "")
        or getattr(args, "mxq_path", "")
        or getattr(args, "onnx_path", "")
    ):
        raise SystemExit(
            "Mask generation models load two artifacts; use "
            "`--encoder-mxq-path`/`--decoder-mxq-path` (or "
            "`--encoder-onnx-path`/`--decoder-onnx-path` with `--framework onnx`) "
            "instead of `--model-path`/`--mxq-path`/`--onnx-path`."
        )


def create_mask_generation_engine(args: argparse.Namespace) -> Any:
    """Creates a promptable mask generation engine from shared CLI model options.

    Mask generation models load two artifacts (encoder + decoder): MXQ by
    default, or ONNX with `--framework onnx`. The shared NPU options
    (`--dev-no`, `--core-mode`, `--target-cores`, `--target-clusters`) apply
    to both MXQ backends and are ignored for ONNX inference.

    The single-artifact path options are rejected here rather than in one
    command's handler, so every command that builds a mask generation engine
    fails loudly instead of silently evaluating the downloaded default in
    place of an explicitly requested local artifact.
    """

    reject_single_artifact_paths(args)

    try:
        import mblt_vision.mask_generation as mask_generation_module
        from mblt_vision.wrapper import _model_name_aliasing
    except ImportError as exc:
        print(f"Missing dependencies for vision CLI: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    class_name = Path(_model_name_aliasing(args.model)).stem
    model_class = getattr(mask_generation_module, class_name, None)
    if model_class is None:
        raise SystemExit(
            f"Mask generation model class '{class_name}' is not exported by "
            "mblt_vision.mask_generation."
        )
    return model_class(
        encoder_mxq_path=getattr(args, "encoder_mxq_path", None) or None,
        decoder_mxq_path=getattr(args, "decoder_mxq_path", None) or None,
        prompt_weights_path=getattr(args, "prompt_weights_path", None) or None,
        encoder_dev_no=args.dev_no,
        decoder_dev_no=args.dev_no,
        encoder_core_mode=args.core_mode,
        decoder_core_mode=args.core_mode,
        encoder_target_cores=args.target_cores,
        decoder_target_cores=args.target_cores,
        encoder_target_clusters=args.target_clusters,
        decoder_target_clusters=args.target_clusters,
        target_device=args.target_device,
        framework=args.framework,
        encoder_onnx_path=getattr(args, "encoder_onnx_path", None) or None,
        decoder_onnx_path=getattr(args, "decoder_onnx_path", None) or None,
    )


def run_mask_generation_inference(
    args: argparse.Namespace,
    *,
    command: str,
) -> Any:
    """Runs point-prompted mask generation for a CLI command."""

    point_prompts = getattr(args, "points", None) or []
    if not 1 <= len(point_prompts) <= 3:
        raise SystemExit(
            "Mask generation requires 1 to 3 point prompts; pass `--point X,Y,LABEL` "
            "(LABEL 1 positive, 0 negative) up to three times."
        )
    reject_single_artifact_paths(args)

    points = [[x, y] for x, y, _ in point_prompts]
    labels = [label for _, _, label in point_prompts]
    model = create_mask_generation_engine(args)
    try:
        result = model.predict(args.source, points, labels)
        save_path = resolve_output_path(args.output, command, args.source, args.model)
        result.plot(source_path=args.source, save_path=save_path)
        iou_text = ", ".join(f"{float(value):.4f}" for value in result.iou_predictions)
        print(
            f"Predicted IoU per mask candidate: [{iou_text}]; "
            f"selected mask index: {result.selected}"
        )
        print(f"Saved result to {os.path.relpath(save_path)}")
        return result
    finally:
        model.dispose()


def run_vision_inference(
    args: argparse.Namespace,
    *,
    command: str,
) -> Any:
    """Runs a complete vision inference pipeline for a CLI command."""

    require_source_file(args.source)
    if resolve_cli_task(args) == "mask_generation":
        return run_mask_generation_inference(args, command=command)
    if getattr(args, "points", None):
        raise SystemExit(
            "`--point` is only supported for mask generation models such as "
            "SAM2HieraLarge."
        )
    model = create_vision_engine(args)
    try:
        actual_task = normalize_vision_task(model.post_cfg.get("task", ""))
        plot_kwargs: dict[str, Any] = {}
        if actual_task == "image_classification":
            plot_kwargs["topk"] = args.topk
        elif actual_task in {
            "object_detection",
            "face_detection",
            "instance_segmentation",
            "pose_estimation",
            "obb",
        }:
            model.set_postprocess_thresholds(
                conf_thres=args.conf_thres, iou_thres=args.iou_thres
            )

        postprocess_kwargs: dict[str, Any] = {}
        if actual_task == "semantic_segmentation":
            input_img, metadata = model.preprocess_with_metadata(args.source)
            postprocess_kwargs["img0_shape"] = metadata["img0_shape"]
            postprocess_kwargs["ratio_pad"] = metadata.get("ratio_pad")
        else:
            input_img = model.preprocess(args.source)
        output = model(input_img)
        if not getattr(getattr(model, "postprocessor", None), "e2e", True):
            if args.output:
                raise SystemExit(
                    "`--output` is unavailable with `--e2e false`; use `--raw-output` instead."
                )

            raw_output = model.postprocessor(output)
            raw_output_path = getattr(args, "raw_output", None)
            if raw_output_path:
                output_path = Path(raw_output_path).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(raw_output, output_path)
                print(f"Saved raw postprocess output to {output_path}")
            else:
                print(
                    "Generated raw export-style postprocess output. Use `--raw-output` to save it."
                )
            return raw_output

        result = model.postprocess(output, **postprocess_kwargs)

        save_path = resolve_output_path(args.output, command, args.source, args.model)
        result.plot(source_path=args.source, save_path=save_path, **plot_kwargs)
        print(f"Saved result to {os.path.relpath(save_path)}")
        return result
    finally:
        model.dispose()

"""Run reproducible multi-model vision accuracy benchmarks.

This entry point follows the artifact contract used by the Transformers benchmark:
one results directory contains machine-readable JSON and CSV output, an optional
summary, and an accuracy chart.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mblt_vision.benchmark.argparse_utils import parse_positive_int
from mblt_vision.benchmark.io_utils import safe_filename, write_csv, write_json
from mblt_vision.benchmark.summary_utils import (
    collect_host_pc_info,
    markdown_table,
    write_summary_markdown,
)
from mblt_vision._model_paths import resolve_framework
from mblt_vision._tasks import VISION_TASKS, normalize_vision_task
from mblt_vision.wrapper import core_modes_for_target_device
from mblt_npu import normalize_target_device

CoreMode = Literal["single", "multi", "global4", "global8"]
CORE_MODES: tuple[CoreMode, ...] = cast(
    tuple[CoreMode, ...], core_modes_for_target_device("aries-rb")
)
# `mask_generation` is a canonical Vision task but the unified runner cannot
# execute it: it needs SAM2HieraLarge's two-artifact engine and point prompts
# rather than the generic MBLT_Engine `_run_target` builds, and `eval_sav`
# rather than the generic evaluators `_evaluate` dispatches. Offering it as a
# choice would only produce an error row for every model. Benchmark it with
# `mblt-vision val --model sam2-hiera-large` until the runner grows that path.
UNSUPPORTED_BENCHMARK_TASKS: tuple[str, ...] = ("mask_generation",)
TASK_CHOICES = tuple(
    task for task in VISION_TASKS if task not in UNSUPPORTED_BENCHMARK_TASKS
)
SUPPORTED_TARGET_DEVICES = frozenset({"aries-rb", "regulus-ra", "regulus-rb"})


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


def normalize_core_mode(core_mode: str) -> CoreMode:
    """Validate and narrow a benchmark NPU core mode."""

    if core_mode not in CORE_MODES:
        raise ValueError(
            f"Invalid core mode {core_mode!r}; expected one of {list(CORE_MODES)}."
        )
    return cast(CoreMode, core_mode)


def _parse_task(value: str) -> str:
    """Normalize a benchmark task for argparse."""

    try:
        # Validated against the runner's supported tasks, not every canonical
        # Vision task, so an unsupported one reports what can actually be run.
        return normalize_vision_task(value, supported=TASK_CHOICES)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_target_device(value: str) -> str:
    """Normalize and validate a benchmark target board."""

    try:
        target_device = normalize_target_device(value)
    except TypeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if target_device not in SUPPORTED_TARGET_DEVICES:
        raise argparse.ArgumentTypeError(
            f"unsupported target device {value!r}; expected one of "
            f"{sorted(SUPPORTED_TARGET_DEVICES)}."
        )
    return target_device


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standardized vision benchmark.

    Args:
        argv: Optional argument sequence. ``None`` reads process arguments.

    Returns:
        Parsed benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", required=True, help="Vision model classes to benchmark."
    )
    parser.add_argument(
        "--task",
        type=_parse_task,
        choices=TASK_CHOICES,
        required=True,
        help="Task shared by all requested models.",
    )
    parser.add_argument(
        "--model-type",
        default="DEFAULT",
        help="Model variant from the YAML configuration.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional local MXQ or ONNX model path for one target.",
    )
    parser.add_argument(
        "--mxq-path", default="", help="Compatibility alias for a local MXQ path."
    )
    parser.add_argument(
        "--onnx-path", default="", help="Optional local ONNX path for one target."
    )
    parser.add_argument(
        "--framework", choices=["mxq", "onnx"], help="Explicit inference framework."
    )
    parser.add_argument(
        "--target-device",
        type=_parse_target_device,
        default="aries-rb",
        help="NPU board: aries-rb, regulus-ra, or regulus-rb (legacy aries/regulus accepted).",
    )
    parser.add_argument(
        "--core-mode",
        default=None,
        choices=[*CORE_MODES, "all"],
        help=(
            "NPU core mode, or `all` to run every mode supported by the selected "
            "board. Defaults to global8 on Aries and single on Regulus."
        ),
    )
    parser.add_argument("--dev-no", type=int, default=0, help="NPU device number.")
    parser.add_argument(
        "--batch-size",
        type=parse_positive_int,
        default=1,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to an organized validation dataset."
    )
    parser.add_argument(
        "--conf-thres",
        type=parse_unit_interval,
        default=None,
        help="Optional confidence threshold override.",
    )
    parser.add_argument(
        "--iou-thres",
        type=parse_unit_interval,
        default=None,
        help="Optional IoU threshold override.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not write an accuracy chart."
    )
    parser.add_argument(
        "--collect-host-info",
        action="store_true",
        help="Collect host metadata with mblt-tracker.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed model run.",
    )
    args = parser.parse_args(argv)
    local_paths = [args.model_path, args.mxq_path, args.onnx_path]
    if len(args.models) != 1 and any(local_paths):
        parser.error(
            "--model-path, --mxq-path, and --onnx-path require exactly one --models target."
        )
    return args


def _core_modes(
    core_mode: str | None,
    framework: str | None = None,
    target_device: str = "aries-rb",
) -> tuple[str, ...]:
    """Expand the multi-run core-mode shorthand.

    Args:
        core_mode: Requested core mode, or ``None`` for the board default.
        framework: Resolved inference framework.
        target_device: Selected NPU board.

    Returns:
        One or more concrete core modes.
    """
    if framework == "onnx":
        return ("onnx",)
    supported_modes = core_modes_for_target_device(target_device)
    if core_mode is None:
        return (supported_modes[-1],)
    if core_mode == "all":
        return supported_modes
    normalized_core_mode = normalize_core_mode(core_mode)
    if normalized_core_mode not in supported_modes:
        raise ValueError(
            f"Core mode {normalized_core_mode!r} is not supported by "
            f"{normalize_target_device(target_device)}; expected one of "
            f"{list(supported_modes)}."
        )
    return (normalized_core_mode,)


def _evaluate(
    model: Any, args: argparse.Namespace, run_dir: Path
) -> tuple[float, str, dict[str, float]]:
    """Evaluate one model and normalize task metrics for benchmark artifacts.

    Args:
        model: Initialized vision engine.
        args: Parsed benchmark options.
        run_dir: Per-run output directory.

    Returns:
        Primary score, its name, and all normalized metrics.

    Raises:
        ValueError: If the model task differs from the requested benchmark task.
    """
    from mblt_vision.utils.evaluation import (
        eval_ade20k,
        eval_cityscapes,
        eval_coco_metrics,
        eval_dota,
        eval_imagenet_metrics,
        eval_nyu_depth,
        eval_widerface,
    )

    model_task = normalize_vision_task(model.post_cfg.get("task", ""))
    if model_task != args.task:
        raise ValueError(
            f"Model task '{model_task}' does not match requested task '{args.task}'."
        )
    if args.task == "image_classification":
        result = eval_imagenet_metrics(model, args.data_path, args.batch_size)
        return (
            float(result.primary_score),
            "top1_accuracy",
            {
                "top1_accuracy": float(result.top1),
                "top5_accuracy": float(result.top5),
            },
        )
    if args.task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        result = eval_coco_metrics(
            model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres
        )
        return (
            float(result.primary_score),
            "map50_95",
            {"map50_95": float(result.map5095), "map50": float(result.map50)},
        )
    if args.task == "depth_estimation":
        result = eval_nyu_depth(model, args.data_path, args.batch_size)
        return (
            float(result.primary_score),
            "delta1",
            {
                "delta1": float(result.delta1),
                "abs_rel": float(result.abs_rel),
                "rmse": float(result.rmse),
            },
        )
    if args.task == "semantic_segmentation":
        dataset = str(model.post_cfg.get("dataset", "")).lower()
        if dataset == "ade20k":
            result = eval_ade20k(model, args.data_path, args.batch_size)
        elif dataset == "cityscapes":
            result = eval_cityscapes(model, args.data_path, args.batch_size)
        else:
            raise ValueError(
                f"Unsupported semantic segmentation benchmark taxonomy {dataset!r}; expected 'ade20k' or 'cityscapes'."
            )
        return (
            float(result.primary_score),
            "miou",
            {
                "miou": float(result.miou),
                "pixel_accuracy": float(result.pixel_accuracy),
            },
        )
    if args.task == "obb":
        result = eval_dota(
            model,
            args.data_path,
            args.batch_size,
            args.conf_thres,
            args.iou_thres,
            str(run_dir / "dota_task1"),
        )
        return (
            float(result.primary_score),
            "map50_95",
            {
                "map50_95": float(result.map5095),
                "map50": float(result.map50),
            },
        )
    if args.task == "face_detection":
        result = eval_widerface(
            model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres
        )
        return (
            float(result.primary_score),
            "hard_ap",
            {
                "easy_ap": float(result.easy_ap),
                "medium_ap": float(result.medium_ap),
                "hard_ap": float(result.hard_ap),
            },
        )
    raise ValueError(f"Unsupported vision benchmark task: {args.task}")


def _run_target(
    model_name: str, core_mode: str, args: argparse.Namespace, results_dir: Path
) -> dict[str, Any]:
    """Run and record one model/core-mode benchmark target.

    Args:
        model_name: Vision model class name.
        core_mode: Concrete NPU core mode or the neutral ONNX runtime label.
        args: Parsed benchmark options.
        results_dir: Root directory for benchmark artifacts.

    Returns:
        A normalized benchmark result row.
    """
    from mblt_vision import MBLT_Engine

    label = f"{model_name}@{core_mode}"
    run_dir = results_dir / "runs" / safe_filename(label)
    run_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "model": model_name,
        "core_mode": core_mode,
        "task": args.task,
        "batch_size": args.batch_size,
        "target_device": args.target_device,
        "status": "error",
    }
    model = None
    started = time.perf_counter()
    try:
        engine_kwargs: dict[str, Any] = {
            "model_cls": model_name,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "mxq_path": args.mxq_path,
            "onnx_path": args.onnx_path,
            "framework": args.framework,
            "dev_no": args.dev_no,
            "target_device": args.target_device,
        }
        if core_mode != "onnx":
            engine_kwargs["core_mode"] = core_mode
        model = MBLT_Engine(
            **engine_kwargs,
        )
        score, score_name, metrics = _evaluate(model, args, run_dir)
        row.update(
            {"status": "ok", "score": score, "score_name": score_name, **metrics}
        )
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        NotImplementedError,
    ) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                if "error" in row:
                    row["cleanup_error"] = cleanup_error
                else:
                    row["error"] = cleanup_error
                row["status"] = "error"
        row["elapsed_s"] = round(time.perf_counter() - started, 6)
    return row


def _write_outputs(
    rows: list[dict[str, Any]], args: argparse.Namespace, results_dir: Path
) -> None:
    """Write the shared JSON, CSV, chart, and Markdown benchmark artifacts.

    Args:
        rows: Normalized benchmark rows.
        args: Parsed benchmark options.
        results_dir: Destination directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "benchmark": "vision",
        "task": args.task,
        "results": rows,
    }
    write_json(results_dir / "results.json", payload)
    write_csv(results_dir / "results.csv", rows)

    successful = [row for row in rows if row["status"] == "ok"]
    plot_paths: list[Path] = []
    if successful and not args.no_plot:
        from mblt_vision.benchmark.chart_utils import plot_simple_barh

        chart_path = results_dir / "accuracy.png"
        plot_simple_barh(
            labels=[f"{row['model']} ({row['core_mode']})" for row in successful],
            values=[float(row["score"]) for row in successful],
            x_label="Accuracy score",
            title=f"{args.task} benchmark accuracy",
            output_path=chart_path,
        )
        plot_paths.append(chart_path)

    host_info_path = (
        collect_host_pc_info(results_dir) if args.collect_host_info else None
    )
    table = markdown_table(
        ["Model", "Core mode", "Metric", "Score", "Elapsed (s)", "Status"],
        [
            [
                row["model"],
                row["core_mode"],
                row.get("score_name", "-"),
                f"{float(row['score']):.5f}" if row.get("score") is not None else "-",
                row["elapsed_s"],
                row["status"],
            ]
            for row in rows
        ],
    )
    table_path = results_dir / "results.md"
    table_path.write_text(table, encoding="utf-8")
    write_summary_markdown(
        results_dir / "summary.md",
        title=f"Vision benchmark: {args.task}",
        host_info_path=host_info_path,
        table_markdown_path=table_path,
        plot_paths=plot_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standardized vision benchmark.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero when every target succeeds, otherwise one.
    """
    args = _parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    # Keep compatibility-path routing aligned with MBLT_Engine: an explicit
    # ONNX path selects the ONNX backend unless an MXQ path takes precedence.
    framework_model_path = args.model_path
    if not framework_model_path and not args.mxq_path:
        framework_model_path = args.onnx_path
    framework = resolve_framework(args.framework, framework_model_path)
    rows: list[dict[str, Any]] = []
    try:
        core_modes = _core_modes(args.core_mode, framework, args.target_device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for model_name in args.models:
        for core_mode in core_modes:
            print(f"Benchmarking {model_name} with core mode {core_mode}...")
            row = _run_target(model_name, core_mode, args, results_dir)
            rows.append(row)
            if row["status"] == "ok":
                print(f"  {row['score_name']}: {row['score']:.5f}")
            else:
                print(f"  failed: {row['error']}")
                if args.fail_fast:
                    _write_outputs(rows, args, results_dir)
                    return 1
    _write_outputs(rows, args, results_dir)
    print(f"Saved benchmark artifacts to: {results_dir}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

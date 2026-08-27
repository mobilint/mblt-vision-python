"""Vision prediction CLI command."""

from __future__ import annotations

import argparse

from ._vision import (
    add_e2e_arg,
    add_threshold_args,
    add_vision_parser,
    parse_point,
    run_vision_inference,
)


def _cmd_predict(args: argparse.Namespace) -> int:
    """Runs vision inference on a source image."""

    run_vision_inference(args, command="predict")
    return 0


def add_predict_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Registers the unified vision prediction CLI command."""

    parser = add_vision_parser(
        subparsers,
        command="predict",
        help_text=(
            "Run vision inference for classification, depth estimation, detection, instance or semantic "
            "segmentation, OBB, pose, face detection, and point-prompted mask generation."
        ),
        description=(
            "Run a configured Vision model on one image. The selected model determines the task, "
            "preprocessing, postprocessing, and output visualization automatically."
        ),
        epilog="""Supported tasks:
  image classification, depth estimation, object and face detection, instance
  and semantic segmentation, oriented bounding boxes (OBB), pose estimation,
  and point-prompted mask generation (SAM2).

The command downloads the default MXQ artifact when no local model path is supplied,
then writes a plotted result under runs/vision/predict/ by default. Use --output to
choose the image destination. Use --framework onnx with --model-path or --onnx-path
for ONNX Runtime inference; MXQ is the default framework.

Mask generation models take 1-3 point prompts (--point X,Y,LABEL; LABEL 1 positive,
0 negative) and load two MXQ artifacts, overridable with --encoder-mxq-path and
--decoder-mxq-path instead of --model-path/--mxq-path.

Examples:
  mblt-vision predict --source image.jpg --model resnet50 --topk 3
  mblt-vision predict --source image.jpg --model yolo11m --conf-thres 0.4 --output result.jpg
  mblt-vision predict --source image.jpg --model yolo11m --framework onnx
  mblt-vision predict --source image.jpg --model yolo11m-pose --target-device regulus-ra --core-mode single
  mblt-vision predict --source image.jpg --model sam2-hiera-large --point 320,240,1

For export-style YOLO output, use --e2e false and optionally save it with --raw-output.""",
        handler=_cmd_predict,
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of classification labels to show."
    )
    parser.add_argument(
        "--raw-output",
        help="Path to save raw export-style output with `--e2e false`.",
    )
    parser.add_argument(
        "--point",
        action="append",
        dest="points",
        type=parse_point,
        metavar="X,Y,LABEL",
        help=(
            "Point prompt for mask generation models: pixel X,Y in the source image "
            "and LABEL (1 positive, 0 negative). Repeat up to 3 times."
        ),
    )
    parser.add_argument(
        "--encoder-mxq-path",
        dest="encoder_mxq_path",
        default="",
        help="Optional local encoder MXQ path for mask generation models.",
    )
    parser.add_argument(
        "--decoder-mxq-path",
        dest="decoder_mxq_path",
        default="",
        help="Optional local decoder MXQ path for mask generation models.",
    )
    parser.add_argument(
        "--prompt-weights-path",
        dest="prompt_weights_path",
        default="",
        help="Optional local prompt-encoder weights path for mask generation models.",
    )
    add_threshold_args(parser, conf_default=0.25, iou_default=None)
    add_e2e_arg(parser)

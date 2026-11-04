"""Native DOTA tiled submission CLI over the default library protocol."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mblt_vision.utils.datasets.dota_tiling import (
    ULTRALYTICS_COMMIT,
    ULTRALYTICS_VERSION,
    TiledConfig,
    get_eval_protocol,
)
from mblt_vision.utils.postprocess.tiled_obb import (
    UltralyticsTilePredictor,
    predict_tiled,
    write_task1,
)


def main(argv: list[str] | None = None) -> int:
    """Run an explicitly selected adapter on original images and write Task1 files.

    A callback factory is ``module:callable`` with no arguments returning a
    TilePredictor. Importing it executes trusted user code; no runtime is chosen
    implicitly. The reference adapter accepts only existing local checkpoints.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Original image or directory (not pre-split tiles)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New/empty output directory"
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--weights",
        type=Path,
        help="Trusted local OBB .pt, optional pinned Ultralytics CPU adapter",
    )
    backend.add_argument(
        "--predictor-factory",
        help="Trusted module:factory returning a native-tile predictor",
    )
    parser.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument(
        "--no-jpeg-roundtrip",
        action="store_true",
        help="Lossless mode, differs from split_test JPEG",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1024, help="Ultralytics adapter only"
    )
    parser.add_argument(
        "--conf", type=float, default=0.01, help="Ultralytics adapter only"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Ultralytics adapter per-tile NMS only; merge is 0.3",
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="Ultralytics adapter per-tile cap"
    )
    parser.add_argument(
        "--preprocessing",
        choices=("ultralytics-val", "direct-letterbox"),
        default=get_eval_protocol()["inference"]["preprocessing"],
        help="Ultralytics adapter input methodology; direct-letterbox is historical",
    )
    args = parser.parse_args(argv)
    config = TiledConfig(
        args.crop_size,
        args.gap,
        tuple(args.rates),
        jpeg_roundtrip=not args.no_jpeg_roundtrip,
    )
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error("--output must be a new or empty directory")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = (
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in extensions)
        if args.source.is_dir()
        else [args.source]
    )
    if not paths or any(not p.is_file() for p in paths):
        parser.error("--source must contain at least one existing image")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Original image stems must be unique")
    if args.weights:
        predictor = UltralyticsTilePredictor(
            args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            preprocessing=args.preprocessing,
        )
        from mblt_vision.datasets import get_dataset_class_names

        if predictor.names != tuple(get_dataset_class_names("dotav1")):
            parser.error("Reference checkpoint taxonomy must match DOTAv1")
    else:
        if (args.imgsz, args.conf, args.iou, args.max_det) != (
            1024,
            0.01,
            0.7,
            300,
        ) or args.preprocessing != get_eval_protocol()["inference"]["preprocessing"]:
            parser.error(
                "Adapter flags apply only to --weights; configure custom callbacks in the factory"
            )
        module, separator, name = args.predictor_factory.partition(":")
        if not separator or not module or not name:
            parser.error("--predictor-factory must be module:callable")
        predictor = getattr(importlib.import_module(module), name)()
        if not callable(predictor):
            parser.error("--predictor-factory must return a callable")
    predictions = {}
    for path in paths:
        # Match the raster read by split_test. Non-trivial EXIF orientation is
        # deliberately unsupported by this CLI (split_test mixes exif_size/read).
        from PIL import Image

        with Image.open(path) as original:
            if original.getexif().get(274, 1) != 1:
                raise ValueError(f"Normalize EXIF orientation before tiling: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode {path}")
        predictions[path.stem] = predict_tiled(image, path.stem, predictor, config)
    write_task1(predictions, args.output)
    metadata = {
        "ultralytics_version": ULTRALYTICS_VERSION,
        "ultralytics_commit": ULTRALYTICS_COMMIT,
        "crop_size": config.crop_size,
        "gap": config.gap,
        "rates": config.rates,
        "crop_pairs": config.crop_pairs,
        "jpeg_roundtrip": config.jpeg_roundtrip,
        "merge": "class-offset Fast-NMS probIoU >= 0.3, stable ties",
        "backend": str(args.weights.resolve())
        if args.weights
        else args.predictor_factory,
        "reference_adapter": {
            "preprocessing": args.preprocessing,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": False,
            "fuse": True,
            "rect": False,
            "multi_label": True,
        }
        if args.weights
        else None,
        "images": [str(path.resolve()) for path in paths],
        "scoring": "Task1 export only; no mAP computed or official-server claim",
    }
    (args.output / "pipeline.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Default DOTA multi-scale native-crop inference using any MBLT engine backend."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image

from ...obb.tiled import (
    Tile,
    TiledConfig,
    iter_tiles,
    merge_tile_predictions,
    write_task1,
)
from ..datasets import CustomDOTAv1, get_dotav1_class_num, get_dotav1_label
from ..postprocess.common import xywhr2xyxyxyxy
from ._pipeline import map_batched_inference
from .dota_polygon import score_original_polygons

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


def predict_engine_tiles(
    model: MBLT_Engine,
    images: Iterator[tuple[str, np.ndarray]],
    batch_size: int = 1,
    config: TiledConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Infer native BGR image tiles through the engine's unchanged raw backend.

    Returns original-pixel Nx7 [cx,cy,w,h,radians,score,class0] rows. Engine
    preprocessing accepts RGB arrays; cropping/JPEG match OpenCV split_test.
    Batching and inverse geometry use each tile's own metadata. Only inference
    enters the backend thread; decoding uses the bounded evaluation pipeline.
    """
    from .eval_dota import format_dota_results

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = config or TiledConfig()
    grouped: dict[str, list[tuple[Tile, torch.Tensor]]] = defaultdict(list)

    def batches() -> Iterator[
        tuple[list[Tile], np.ndarray, list[tuple[int, int]], list[Any]]
    ]:
        pending = []
        for image_id, image in images:
            grouped[image_id] = []
            for tile in iter_tiles(image, image_id, config):
                pending.append(tile)
                if len(pending) == batch_size:
                    yield prepare(pending)
                    pending = []
        if pending:
            yield prepare(pending)

    def prepare(
        tiles: list[Tile],
    ) -> tuple[list[Tile], np.ndarray, list[tuple[int, int]], list[Any]]:
        values, shapes, pads = [], [], []
        for tile in tiles:
            value, metadata = model.preprocess_with_metadata(
                np.ascontiguousarray(tile.image[..., ::-1])
            )
            values.append(value)
            shapes.append(tuple(tile.image.shape[:2]))
            pads.append(metadata.get("ratio_pad"))
        return tiles, np.stack(values), shapes, pads

    def infer(batch: Any) -> Any:
        return model(batch[1])

    def decode(batch: Any, raw: Any) -> list[tuple[Tile, torch.Tensor]]:
        tiles, values, shapes, pads = batch
        results = model.postprocess(raw)
        formatted = format_dota_results(
            results,
            tuple(values.shape[1:-1]),
            shapes,
            pads,
            tuple(tile.name for tile in tiles),
            model.postprocessor,
        )
        rows_by_name: dict[str, list[list[float]]] = defaultdict(list)
        for row in formatted:
            rows_by_name[row["image_id"]].append(
                [*row["rbox"], row["score"], row["category_id"]]
            )
        return [
            (
                tile,
                torch.tensor(rows_by_name[tile.name], dtype=torch.float32).reshape(
                    -1, 7
                ),
            )
            for tile in tiles
        ]

    for decoded in map_batched_inference(batches(), infer, decode):
        for tile, rows in decoded:
            grouped[tile.image_id].append(
                (replace(tile, image=np.empty((0, 0, 3), dtype=np.uint8)), rows)
            )
    return {
        image_id: merge_tile_predictions(predictions)
        for image_id, predictions in grouped.items()
    }


def eval_dota_tiled(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
    save_dir: str | None = None,
) -> Any:
    """Original-image polygon VOC07 primary AP, with legacy probabilistic diagnostic.

    map50 is original-image polygon AP50. map5095 is retained ONLY as a legacy
    probabilistic diagnostic for API compatibility, never official DOTA mAP.
    """
    from .eval_dota import DOTAResult, _load_ground_truths, evaluate_dota_predictions

    dataset = CustomDOTAv1(data_path)
    model.set_postprocess_thresholds(
        conf_thres=0.01 if conf_thres is None else conf_thres,
        iou_thres=0.7 if iou_thres is None else iou_thres,
    )

    def images() -> Iterator[tuple[str, np.ndarray]]:
        for image_id, path in zip(dataset.ids, dataset.image_paths, strict=True):
            with Image.open(path) as header:
                if header.getexif().get(274, 1) != 1:
                    raise ValueError(
                        "DOTA tiled inference requires EXIF-normalized original images"
                    )
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            yield image_id, image

    predictions = predict_engine_tiles(model, images(), batch_size)
    formatted = []
    for image_id, rows in predictions.items():
        polygons = xywhr2xyxyxyxy(rows[:, :5]).reshape(-1, 8).tolist()
        for row, polygon in zip(rows.tolist(), polygons, strict=True):
            formatted.append(
                {
                    "image_id": image_id,
                    "rbox": row[:5],
                    "category_id": int(row[6]),
                    "score": round(row[5], 3),
                    "poly": [round(value, 3) for value in polygon],
                }
            )
    names = [get_dotav1_label(index) for index in range(get_dotav1_class_num())]
    polygon_score = score_original_polygons(
        formatted, dataset.ids, Path(data_path) / "labels/val_original", names
    )
    diagnostic = evaluate_dota_predictions(
        _load_ground_truths(data_path, dataset), formatted
    )
    if save_dir is not None:
        import json

        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_task1(predictions, output / "predictions_merged_txt")
        (output / "original_polygon_score.json").write_text(
            json.dumps(polygon_score, indent=2), encoding="utf-8"
        )
    print(
        "DOTAv1 original-image tiled polygon VOC07 evaluation completed; no test-server submission"
    )
    return DOTAResult(polygon_score["map50"], diagnostic.map5095)

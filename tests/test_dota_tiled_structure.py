"""Dataset protocol/inheritance regressions and frozen pre-refactor Task1 bytes."""

from __future__ import annotations

import json
from pathlib import Path
from threading import current_thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mblt_vision.datasets import get_dataset_config
from mblt_vision.utils.datasets.dota_tiling import Tile, TiledConfig, get_eval_protocol
from mblt_vision.utils.evaluation.dota_polygon import (
    polygon_iou,
    score_original_polygons,
)
from mblt_vision.utils.postprocess.build_post import build_postprocess
from mblt_vision.utils.postprocess.tiled_obb import (
    DOTATiledOBBPostMixin,
    YOLOAnchorlessTiledOBBPost,
    YOLODFLFreeTiledOBBPost,
    merge_tile_predictions,
    write_task1,
)
from mblt_vision.utils.postprocess.yolo_anchorless_post import YOLOAnchorlessOBBPost
from mblt_vision.utils.postprocess.yolo_dflfree_post import YOLODFLFreeOBBPost
from mblt_vision.wrapper import resolve_model_config

FIXTURE = Path(__file__).parent / "fixtures/dota_tiled_da3ef806_task1.json"


def frozen_predictions():
    fixture = json.loads(FIXTURE.read_text())
    predictions = []
    for case in fixture["tiles"]:
        x, y, x2, y2 = case["window"]
        size = x2 - x
        tile = Tile(
            "P0001",
            f"P0001__{size}__{x}___{y}",
            case["rate"],
            size,
            500,
            (x, y, x2, y2),
            np.zeros((13, 19, 3), np.uint8),
        )
        predictions.append((tile, np.array(case["rows"], dtype=np.float32)))
    return fixture, predictions


def test_frozen_da3ef806_task1_bytes(tmp_path):
    fixture, predictions = frozen_predictions()
    assert fixture["vision_commit"] == "da3ef806c613eb101d9d53e2c53767df07cb16d6"
    assert fixture["reference_commit"] == "f8e6fddc89ec754b3a62eab87d5221e83058005a"
    output = write_task1(
        {"P0001": merge_tile_predictions(predictions)}, tmp_path, fixture["classes"]
    )
    assert {p.name: p.read_bytes() for p in output.glob("*.txt")} == {
        name: content.encode() for name, content in fixture["task1"].items()
    }


def test_dataset_protocol_defaults_and_selector_validation():
    protocol = get_dataset_config("dotav1")["eval_protocol"]
    assert get_eval_protocol(selector=protocol["id"]) == protocol
    assert TiledConfig() == TiledConfig(
        **{**protocol["tiling"], "rates": tuple(protocol["tiling"]["rates"])}
    )
    assert protocol["inference"] == {
        "conf_thres": 0.01,
        "iou_thres": 0.7,
        "max_det": 300,
    }
    with pytest.raises(ValueError, match="Unsupported eval_protocol"):
        get_eval_protocol(selector="not-a-supported-version.v2")


@pytest.mark.parametrize(
    "family,cls,base",
    [
        ("YOLOv8n-obb", YOLOAnchorlessTiledOBBPost, YOLOAnchorlessOBBPost),
        ("YOLO11n-obb", YOLOAnchorlessTiledOBBPost, YOLOAnchorlessOBBPost),
        ("YOLO26n-obb", YOLODFLFreeTiledOBBPost, YOLODFLFreeOBBPost),
    ],
)
@pytest.mark.parametrize("variant", ["DEFAULT", "TURBO"])
def test_existing_update_inherits_protocol_and_reuses_decode(
    family, cls, base, variant
):
    config = resolve_model_config(family, variant)
    post = config["post_cfg"]
    assert post["eval_protocol"] == get_eval_protocol()["id"]
    assert "conf_thres" not in post and "iou_thres" not in post
    processor = build_postprocess(config["pre_cfg"], post)
    assert isinstance(processor, cls) and isinstance(processor, DOTATiledOBBPostMixin)
    assert isinstance(processor, base)
    assert cls.decode is base.decode
    assert cls.nmsout2eval is base.nmsout2eval
    assert processor.conf_thres == 0.01 and processor.iou_thres == 0.7
    assert processor.tile_max_det == 300
    legacy_post = {k: v for k, v in post.items() if k != "eval_protocol"}
    legacy = build_postprocess(config["pre_cfg"], legacy_post)
    rows = torch.zeros((1, 4 + processor.nc + processor.n_extra))
    rows[0, :5] = torch.tensor([100, 100, 10, 5, 0.9])
    rows[0, -1] = 0.2
    # Dispatch wraps NMS but never replaces head-specific filtering or decode.
    torch.testing.assert_close(processor.nms([rows])[0], legacy.nms([rows])[0])


def test_all_obb_model_selectors_and_model_override():
    models = Path(__file__).parents[1] / "mblt_vision/models"
    for path in models.glob("*-obb.yaml"):
        config = resolve_model_config(path.stem, "TURBO")
        assert config["post_cfg"]["eval_protocol"] == get_eval_protocol()["id"]
    config = resolve_model_config("YOLOv8n-obb", "DEFAULT")
    config["post_cfg"].update(conf_thres=0.2, iou_thres=0.4, max_det=1)
    processor = build_postprocess(config["pre_cfg"], config["post_cfg"])
    assert (processor.conf_thres, processor.iou_thres, processor.tile_max_det) == (
        0.2,
        0.4,
        1,
    )
    rows = torch.zeros((2, 4 + processor.nc + processor.n_extra))
    rows[:, :5] = torch.tensor(
        [[100.0, 100.0, 10.0, 5.0, 0.9], [300.0, 300.0, 10.0, 5.0, 0.8]]
    )
    rows[:, -1] = 0.2
    assert len(processor.nms([rows])[0]) == 1


def test_merge_uses_explicit_protocol_settings():
    _, predictions = frozen_predictions()
    tile = predictions[0][0]
    rows = torch.tensor(
        [
            [100.0, 100.0, 10.0, 5.0, 0.2, 0.9, 0.0],
            [100.0, 100.0, 10.0, 5.0, 0.2, 0.8, 1.0],
        ]
    )
    assert len(merge_tile_predictions([(tile, rows)])) == 2
    protocol = get_eval_protocol()
    protocol["merge"]["classwise"] = False
    assert len(merge_tile_predictions([(tile, rows)], protocol)) == 1
    protocol["merge"]["iou_thres"] = 1.0
    assert len(merge_tile_predictions([(tile, rows)], protocol)) == 2


@pytest.mark.parametrize(
    "polygon", [[5.0, 0.0, 5.0, 0.0, 5.0, 10.0, 5.0, 10.0], [5.0] * 8]
)
def test_degenerate_prediction_is_false_positive_but_gt_is_strict(tmp_path, polygon):
    box = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    assert polygon_iou(polygon, box) == 0
    (tmp_path / "P.txt").write_text("0 0 10 0 10 10 0 10 plane 0\n")
    rows = [
        {"image_id": "P", "category_id": 0, "score": 0.9, "poly": polygon},
        {"image_id": "P", "category_id": 0, "score": 0.8, "poly": box},
    ]
    assert score_original_polygons(rows, ["P"], tmp_path, ["plane"])[
        "map50"
    ] == pytest.approx(0.5)
    (tmp_path / "P.txt").write_text(" ".join(map(str, polygon)) + " plane 0\n")
    with pytest.raises(ValueError, match="strictly convex"):
        score_original_polygons([], ["P"], tmp_path, ["plane"])


def test_engine_prepares_on_pool_and_merges_per_completed_image(monkeypatch):
    import importlib

    from mblt_vision.utils.evaluation.eval_dota_tiled import predict_engine_tiles

    tiled = importlib.import_module("mblt_vision.utils.evaluation.eval_dota_tiled")
    legacy = importlib.import_module("mblt_vision.utils.evaluation.eval_dota")
    prepared, inferred, merged, counts = [], [], [], []

    class Engine:
        postprocessor = object()

        def preprocess_with_metadata(self, rgb):
            prepared.append(current_thread().name)
            return rgb.astype(np.float32), {"ratio_pad": ((1.0, 1.0), (0.0, 0.0))}

        def __call__(self, values):
            inferred.append(current_thread().name)
            return values

        def postprocess(self, raw):
            return SimpleNamespace()

    def format_rows(results, shape, shapes, pads, ids, processor):
        return [
            {
                "image_id": key,
                "rbox": [2.0, 3.0, 1.0, 1.0, 0.0],
                "score": 0.9,
                "category_id": 0,
            }
            for key in ids
        ]

    original_merge = tiled.merge_tile_predictions

    def merge(rows):
        counts.append(len(rows))
        merged.append(rows[0][0].image_id)
        return original_merge(rows)

    monkeypatch.setattr(legacy, "format_dota_results", format_rows)
    monkeypatch.setattr(tiled, "merge_tile_predictions", merge)

    def images():
        for index in range(20):
            # Bounded prefetch may run ahead, but cannot defer every merge to EOF.
            if index == 19:
                assert merged
            yield f"P{index}", np.zeros((8, 8, 3), np.uint8)

    result = predict_engine_tiles(
        Engine(),
        images(),
        batch_size=2,
        config=TiledConfig(crop_size=8, gap=0, rates=(1.0,), jpeg_roundtrip=False),
    )
    assert list(result) == [f"P{i}" for i in range(20)]
    assert merged == list(result) and counts == [1] * 20
    assert all(name.startswith("mblt-vision-prepare") for name in prepared)
    assert set(inferred) == {current_thread().name}

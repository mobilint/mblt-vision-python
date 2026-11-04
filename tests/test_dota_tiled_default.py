"""Synthetic default routing and original-polygon DOTA protocol tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from mblt_vision.utils.evaluation.dota_polygon import (
    polygon_iou,
    score_original_polygons,
)
from mblt_vision.utils.evaluation.eval_dota import eval_dota

BOX = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]


def test_polygon_geometry_and_strict_threshold():
    assert polygon_iou(BOX, BOX) == pytest.approx(1.0)
    assert polygon_iou(BOX, [10.0, 0.0, 20.0, 0.0, 20.0, 10.0, 10.0, 10.0]) == 0.0
    assert polygon_iou(
        BOX, [0.0, 0.0, 5.0, 0.0, 5.0, 10.0, 0.0, 10.0]
    ) == pytest.approx(0.5)


def test_polygon_voc07_ignores_difficult_and_counts_duplicates(tmp_path):
    (tmp_path / "P.txt").write_text(
        "0 0 10 0 10 10 0 10 plane 0\n20 0 30 0 30 10 20 10 plane 1\n"
    )
    predictions = [
        {
            "image_id": "P",
            "category_id": 0,
            "score": 0.9,
            "poly": [20.0, 0.0, 30.0, 0.0, 30.0, 10.0, 20.0, 10.0],
        },
        {"image_id": "P", "category_id": 0, "score": 0.8, "poly": BOX},
        {"image_id": "P", "category_id": 0, "score": 0.7, "poly": BOX},
    ]
    result = score_original_polygons(predictions, ["P"], tmp_path, ["plane"])
    assert result["map50"] == pytest.approx(1.0)
    assert result["classes"][0]["positives"] == 1
    assert result["official_test_server"] is False


def test_iou_exact_half_is_false_positive(tmp_path):
    (tmp_path / "P.txt").write_text("0 0 10 0 10 10 0 10 plane 0\n")
    row = {
        "image_id": "P",
        "category_id": 0,
        "score": 0.9,
        "poly": [0.0, 0.0, 5.0, 0.0, 5.0, 10.0, 0.0, 10.0],
    }
    assert score_original_polygons([row], ["P"], tmp_path, ["plane"])["map50"] == 0.0


def test_default_routes_tiled_without_running_backend(monkeypatch):
    import mblt_vision.utils.evaluation.eval_dota_tiled as tiled

    expected = object()
    monkeypatch.setattr(tiled, "eval_dota_tiled", lambda *args: expected)
    model = SimpleNamespace(post_cfg={"task": "obb", "dataset": "dotav1"})
    assert eval_dota(model, "unused", 1) is expected


def test_engine_tile_color_and_geometry_seam(monkeypatch):
    import importlib

    legacy = importlib.import_module("mblt_vision.utils.evaluation.eval_dota")
    from mblt_vision.obb.tiled import TiledConfig
    from mblt_vision.utils.evaluation.eval_dota_tiled import predict_engine_tiles

    seen = []

    class Engine:
        postprocessor = object()

        def preprocess_with_metadata(self, rgb):
            seen.append(rgb.copy())
            return rgb.astype(np.float32), {"ratio_pad": ((1.0, 1.0), (0.0, 0.0))}

        def __call__(self, values):
            return values

        def postprocess(self, raw):
            return object()

    def formatted(results, shape, shapes, pads, ids, processor):
        return [
            {
                "image_id": key,
                "rbox": [2.0, 3.0, 1.0, 1.0, 0.0],
                "score": 0.9,
                "category_id": 0,
            }
            for key in ids
        ]

    monkeypatch.setattr(legacy, "format_dota_results", formatted)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    output = predict_engine_tiles(
        Engine(),
        iter([("P", image)]),
        config=TiledConfig(crop_size=8, gap=0, rates=(1.0,), jpeg_roundtrip=False),
    )
    assert seen[0][0, 0].tolist() == [0, 0, 200]
    assert output["P"].shape == (1, 7)
    assert output["P"][0, :2].tolist() == [2.0, 3.0]

"""Dataset settings/inheritance regressions and frozen pre-refactor Task1 bytes."""

from __future__ import annotations

import json
from pathlib import Path
from threading import current_thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mblt_vision.datasets import get_dataset_config
from mblt_vision.utils.datasets.dota_tiling import Tile, TiledConfig, get_dota_settings
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
    output = write_task1(
        {"P0001": merge_tile_predictions(predictions)}, tmp_path, fixture["classes"]
    )
    assert {p.name: p.read_bytes() for p in output.glob("*.txt")} == {
        name: content.encode() for name, content in fixture["task1"].items()
    }


def test_dataset_yaml_settings_defaults_and_owned_mapping():
    settings = get_dota_settings()
    dataset = get_dataset_config("dotav1")
    assert settings == {key: dataset[key] for key in ("tiling", "inference", "merge")}
    assert TiledConfig() == TiledConfig(
        **{**settings["tiling"], "rates": tuple(settings["tiling"]["rates"])}
    )
    assert settings["inference"] == {
        "preprocessing": "ultralytics-val",
        "conf_thres": 0.01,
        "iou_thres": 0.7,
        "max_det": 300,
    }
    settings["merge"]["iou_thres"] = 0.8
    assert get_dota_settings()["merge"]["iou_thres"] == 0.3


@pytest.mark.parametrize(
    "family,cls,base",
    [
        ("YOLOv8n-obb", YOLOAnchorlessTiledOBBPost, YOLOAnchorlessOBBPost),
        ("YOLO11n-obb", YOLOAnchorlessTiledOBBPost, YOLOAnchorlessOBBPost),
        ("YOLO26n-obb", YOLODFLFreeTiledOBBPost, YOLODFLFreeOBBPost),
    ],
)
@pytest.mark.parametrize("variant", ["DEFAULT", "TURBO"])
def test_existing_update_inherits_settings_and_reuses_decode(
    family, cls, base, variant
):
    config = resolve_model_config(family, variant)
    post = config["post_cfg"]
    assert post["tiled"] is True
    assert "conf_thres" not in post and "iou_thres" not in post
    processor = build_postprocess(config["pre_cfg"], post)
    assert isinstance(processor, cls) and isinstance(processor, DOTATiledOBBPostMixin)
    assert isinstance(processor, base)
    assert cls.decode is base.decode
    assert cls.nmsout2eval is base.nmsout2eval
    assert processor.conf_thres == 0.01 and processor.iou_thres == 0.7
    assert processor.tile_max_det == 300
    assert processor.tile_preprocessing == "ultralytics-val"
    legacy_post = {**post, "tiled": False}
    legacy = build_postprocess(config["pre_cfg"], legacy_post)
    rows = torch.zeros((1, 4 + processor.nc + processor.n_extra))
    rows[0, :5] = torch.tensor([100, 100, 10, 5, 0.9])
    rows[0, -1] = 0.2
    # Dispatch wraps NMS but never replaces head-specific filtering or decode.
    torch.testing.assert_close(processor.nms([rows])[0], legacy.nms([rows])[0])


def test_all_obb_models_enable_tiling_and_model_override():
    models = Path(__file__).parents[1] / "mblt_vision/models"
    for path in models.glob("*-obb.yaml"):
        config = resolve_model_config(path.stem, "TURBO")
        assert config["post_cfg"]["tiled"] is True
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


@pytest.mark.parametrize("family", ["YOLOv8n-obb", "YOLO26n-obb"])
def test_plain_yaml_nested_hyperparameters_override_dataset_defaults(tmp_path, family):
    import yaml

    base = resolve_model_config(family, "DEFAULT")
    path = tmp_path / "settings-obb.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "DEFAULT": {
                    **base,
                    "post_cfg": {
                        **base["post_cfg"],
                        "tiling": {
                            "crop_size": 64,
                            "gap": 16,
                            "rates": [1.0],
                            "max_tiles": 12,
                        },
                        "inference": {
                            "conf_thres": 0.2,
                            "iou_thres": 0.4,
                            "max_det": 7,
                        },
                        "merge": {"iou_thres": 0.5, "max_candidates": 40},
                    },
                },
                "TURBO": {
                    "update": "DEFAULT",
                    "post_cfg": {"inference": {"max_det": 3}},
                },
            }
        )
    )
    config = resolve_model_config(str(path), "DEFAULT")
    processor = build_postprocess(config["pre_cfg"], config["post_cfg"])
    assert processor.tile_config.crop_pairs == ((64, 16),)
    assert processor.tile_config.max_tiles == 12
    assert processor.tile_config.jpeg_roundtrip is True
    assert (processor.conf_thres, processor.iou_thres, processor.tile_max_det) == (
        0.2,
        0.4,
        7,
    )
    assert processor.tiled_settings["merge"]["iou_thres"] == 0.5
    assert processor.tiled_settings["merge"]["max_candidates"] == 40
    inherited = resolve_model_config(str(path), "TURBO")
    child = build_postprocess(inherited["pre_cfg"], inherited["post_cfg"])
    # Existing loader remains shallow: the child replaces inference's whole map,
    # then ordinary missing hyperparameters come from the dataset's YAML defaults.
    assert (child.conf_thres, child.iou_thres, child.tile_max_det) == (0.01, 0.7, 3)
    assert child.tile_config == processor.tile_config
    flat = get_dota_settings(
        overrides={"inference": {"conf_thres": 0.2}, "conf_thres": 0.4}
    )
    assert flat["inference"]["conf_thres"] == 0.4


def test_dotav1_obb_defaults_to_tiled_without_selector():
    config = resolve_model_config("YOLOv8n-obb", "DEFAULT")
    config["post_cfg"].pop("tiled")
    assert isinstance(
        build_postprocess(config["pre_cfg"], config["post_cfg"]), DOTATiledOBBPostMixin
    )
    config["post_cfg"]["tiled"] = False
    assert not isinstance(
        build_postprocess(config["pre_cfg"], config["post_cfg"]), DOTATiledOBBPostMixin
    )


def test_merge_uses_explicit_yaml_settings():
    _, predictions = frozen_predictions()
    tile = predictions[0][0]
    rows = torch.tensor(
        [
            [100.0, 100.0, 10.0, 5.0, 0.2, 0.9, 0.0],
            [100.0, 100.0, 10.0, 5.0, 0.2, 0.8, 1.0],
        ]
    )
    assert len(merge_tile_predictions([(tile, rows)])) == 2
    settings = get_dota_settings()
    settings["merge"]["classwise"] = False
    assert len(merge_tile_predictions([(tile, rows)], settings)) == 1
    settings["merge"]["iou_thres"] = 1.0
    assert len(merge_tile_predictions([(tile, rows)], settings)) == 2


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

        def postprocess(self, raw, *, multi_label):
            assert multi_label is True
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


@pytest.mark.parametrize("family", ["YOLOv8n-obb", "YOLO26n-obb"])
def test_engine_explicit_multilabel_keeps_secondary_class_through_inherited_post(
    family,
):
    from mblt_vision.utils.evaluation.eval_dota_tiled import predict_engine_tiles
    from mblt_vision.utils.preprocess.build_pre import build_preprocess
    from mblt_vision.wrapper import MBLT_Engine

    config = resolve_model_config(family, "DEFAULT")
    config["pre_cfg"]["LetterBox"]["img_size"] = [32, 32]
    processor = build_postprocess(config["pre_cfg"], config["post_cfg"])
    preprocessor = build_preprocess(config["pre_cfg"])
    forwarded = []

    class Engine:
        pre_cfg = config["pre_cfg"]
        post_cfg = config["post_cfg"]
        postprocessor = processor

        def preprocess_with_metadata(self, rgb):
            return preprocessor.with_metadata(rgb)

        def __call__(self, values):
            # One converted anchor, not already-final Nx7 rows: both families
            # run their inherited conversion, candidate filtering and rotated NMS.
            raw = torch.zeros((len(values), 4 + processor.nc + processor.n_extra, 1))
            raw[:, :4, 0] = torch.tensor([16.0, 16.0, 8.0, 4.0])
            raw[:, 4:6, 0] = torch.tensor([0.9, 0.8])
            raw[:, -1, 0] = 0.2
            return raw

        def postprocess(self, raw, *, multi_label):
            forwarded.append(multi_label)
            assert multi_label is True
            return MBLT_Engine.postprocess(self, raw, multi_label=multi_label)

    rows = predict_engine_tiles(
        Engine(),
        iter([("P", np.zeros((32, 32, 3), np.uint8))]),
        config=TiledConfig(crop_size=32, gap=0, rates=(1.0,), jpeg_roundtrip=False),
    )["P"]
    assert forwarded == [True]
    assert rows[:, 6].tolist() == [0.0, 1.0]
    torch.testing.assert_close(rows[:, 5], torch.tensor([0.9, 0.8]))
    torch.testing.assert_close(rows[:, :5], torch.tensor([[16, 16, 8, 4, 0.2]] * 2))


@pytest.mark.parametrize("jpeg_roundtrip", [False, True])
def test_engine_crop_copy_and_jpeg_run_on_bounded_prepare_pool(
    monkeypatch, jpeg_roundtrip
):
    import importlib

    import cv2
    from mblt_vision.utils.datasets.dota_tiling import iter_tiles
    from mblt_vision.utils.evaluation.eval_dota_tiled import predict_engine_tiles

    bridge = importlib.import_module("mblt_vision.utils.evaluation.eval_dota_tiled")
    legacy = importlib.import_module("mblt_vision.utils.evaluation.eval_dota")
    config = TiledConfig(
        crop_size=8, gap=2, rates=(1.0,), jpeg_roundtrip=jpeg_roundtrip
    )
    image = np.random.default_rng(0).integers(0, 256, (13, 19, 3), dtype=np.uint8)
    original = image.copy()
    expected = list(iter_tiles(image, "P", config))
    copies, materialized, codecs, inferred, decoded, enumerated = [], [], [], [], [], []

    class TrackedImage(np.ndarray):
        def copy(self, *args, **kwargs):
            copies.append(current_thread().name)
            return super().copy(*args, **kwargs)

    materialize = bridge._materialize_tile
    enumerate_windows = bridge._iter_tile_windows

    def tracked_windows(*args):
        for window in enumerate_windows(*args):
            enumerated.append(current_thread().name)
            yield window

    def tracked_materialize(source, window, cfg):
        materialized.append(current_thread().name)
        tile = materialize(source, window, cfg)
        reference = next(t for t in expected if t.name == tile.name)
        assert tile.window == reference.window
        np.testing.assert_array_equal(tile.image, reference.image)
        assert tile.image.flags.c_contiguous
        assert not np.shares_memory(tile.image, source)
        return tile

    for name in ("imencode", "imdecode"):
        function = getattr(cv2, name)

        def tracked_codec(*args, _function=function, _name=name, **kwargs):
            codecs.append((_name, current_thread().name))
            return _function(*args, **kwargs)

        monkeypatch.setattr(cv2, name, tracked_codec)
    monkeypatch.setattr(bridge, "_iter_tile_windows", tracked_windows)
    monkeypatch.setattr(bridge, "_materialize_tile", tracked_materialize)

    class Engine:
        postprocessor = object()

        def preprocess_with_metadata(self, rgb):
            return rgb.astype(np.float32), {"ratio_pad": ((1.0, 1.0), (0, 0))}

        def __call__(self, values):
            inferred.append(current_thread().name)
            return values

        def postprocess(self, raw, *, multi_label):
            assert multi_label is True
            decoded.append(current_thread().name)
            return SimpleNamespace()

    def formatted(results, shape, shapes, pads, ids, processor):
        return [
            {"image_id": key, "rbox": [2, 3, 1, 1, 0], "score": 0.9, "category_id": 0}
            for key in ids
        ]

    monkeypatch.setattr(legacy, "format_dota_results", formatted)
    output = predict_engine_tiles(
        Engine(), iter([("P", image.view(TrackedImage))]), batch_size=2, config=config
    )["P"]
    assert len(copies) == len(materialized) == len(expected)
    assert all(name.startswith("mblt-vision-prepare") for name in copies + materialized)
    assert len(codecs) == (2 * len(expected) if jpeg_roundtrip else 0)
    assert all(name.startswith("mblt-vision-prepare") for _, name in codecs)
    assert all(name.startswith("mblt-vision-decode") for name in decoded)
    assert set(inferred) == set(enumerated) == {current_thread().name}
    np.testing.assert_array_equal(image, original)
    reference = merge_tile_predictions(
        [(tile, torch.tensor([[2, 3, 1, 1, 0, 0.9, 0]])) for tile in expected]
    )
    torch.testing.assert_close(output, reference)

"""CPU synthetic fixtures for the explicit, opt-in DOTA tiled API.

Set MBLT_DOTA_REFERENCE to the pinned source's ultralytics directory to compare
AST-isolated official functions. No installed Ultralytics, weights or device is
needed. AST extraction executes only named definitions, never module imports.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from mblt_vision.obb.tiled import (
    TILED_API_VERSION,
    Tile,
    TiledConfig,
    UltralyticsTilePredictor,
    get_windows,
    iter_tiles,
    merge_tile_predictions,
    predict_tiled,
    prepare_tile,
    restore_tile_boxes,
    write_task1,
)
from mblt_vision.utils.postprocess.common import (
    batch_probiou,
    rotated_nms,
    xywhr2xyxyxyxy,
)
from PIL import Image


def _tile(x=0, y=0, size=1024, rate=1.0, shape=(13, 19)) -> Tile:
    return Tile(
        "P0001",
        f"P0001__{size}__{x}___{y}",
        rate,
        size,
        500,
        (x, y, x + size, y + size),
        np.zeros((*shape, 3), np.uint8),
    )


_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/dota_tiled_v8_3_228.json").read_text()
)


def _reference(relative: str, names: list[str], namespace: dict | None = None) -> dict:
    root = os.environ.get("MBLT_DOTA_REFERENCE")
    if not root:
        pytest.skip(
            "Set MBLT_DOTA_REFERENCE to the v8.3.228 ultralytics source directory"
        )
    path = Path(root) / relative
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == _FIXTURE["source_sha256"][relative]
    ), (
        "Reference source differs from pinned v8.3.228 commit; do not compare unpinned code"
    )
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    # Deferred annotations prevent references to unrelated upstream imports.
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    scope = {
        "np": np,
        "torch": torch,
        "itertools": itertools,
        "ceil": math.ceil,
        "Path": Path,
        "cv2": cv2,
        "Image": Image,
        **(namespace or {}),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)  # noqa: S102 -- isolated reference definitions
    return scope


def test_committed_reference_fixture_without_optional_source(tmp_path) -> None:
    for case in _FIXTURE["windows"]:
        np.testing.assert_array_equal(
            get_windows(tuple(case["shape"])), case["expected"]
        )
    restore = _FIXTURE["offset_restore"]
    result = merge_tile_predictions(
        [(_tile(1048, 349, 682, 1.5), np.array([restore["native_row"]]))]
    )
    torch.testing.assert_close(
        result, torch.tensor([restore["expected_original_row"]]), rtol=0, atol=0
    )
    task1 = _FIXTURE["task1"]
    output = write_task1(
        {"P0001": np.array([task1["merged_row"]])}, tmp_path, [task1["class"]]
    )
    assert (output / "Task1_plane.txt").read_text() == task1["line"]


def test_stable_tied_scores_preserve_earlier_tile_candidate() -> None:
    # A sufficiently long tie run exposes the reference CPU's unstable argsort.
    rows = torch.tensor(
        [
            [100.0 + index / 1000, 100.0, 20.0, 10.0, 0.2, 0.8, 0.0]
            for index in range(33)
        ]
    )
    result = merge_tile_predictions([(_tile(), rows)])
    assert len(result) == 1
    assert result[0, 0] == 100.0


def test_default_crop_pairs_and_api_marker() -> None:
    assert TILED_API_VERSION == 1
    assert TiledConfig().crop_pairs == ((2048, 1000), (1024, 500), (682, 333))


@pytest.mark.parametrize(
    "shape,expected_count",
    [((1, 1), 3), ((1024, 1024), 5), ((2048, 2048), 35), ((3000, 4000), 129)],
)
def test_window_counts_and_extent(shape, expected_count) -> None:
    windows = get_windows(shape)
    assert len(windows) == expected_count
    assert windows.dtype == np.int64
    assert (windows[:, :2] >= 0).all()
    # Last edge shifts back instead of padding when image is larger than crop.
    for x1, y1, x2, y2 in windows:
        size = x2 - x1
        assert x2 <= max(shape[1], size)
        assert y2 <= max(shape[0], size)


def test_boundary_shift_and_x_major_order() -> None:
    windows = get_windows((1100, 1300), TiledConfig(rates=(1.0,)))
    np.testing.assert_array_equal(
        windows,
        [
            [0, 0, 1024, 1024],
            [0, 76, 1024, 1100],
            [276, 0, 1300, 1024],
            [276, 76, 1300, 1100],
        ],
    )


def test_joint_area_filter_strict_threshold_and_fallback() -> None:
    config = TiledConfig(crop_size=10, gap=2, rates=(0.5, 1.0), im_rate_thr=0.6)
    # 6*10/100 == .6 is NOT greater than .6. Joint fallback retains only
    # near-max windows (the 10-sized crop), not both scales independently.
    np.testing.assert_array_equal(get_windows((6, 10), config), [[0, 0, 10, 10]])
    near = TiledConfig(
        crop_size=10, gap=2, rates=(1.0, 1.01), im_rate_thr=0.99, eps=0.01
    )
    assert len(get_windows((1, 1), near)) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rates": ()},
        {"rates": (0,)},
        {"rates": (float("nan"),)},
        {"gap": 1024},
        {"gap": -1},
        {"rates": (1, 0.999999)},  # both truncate to 1024
        {"rates": (1e9,)},
        {"im_rate_thr": 1},
        {"eps": 0},
    ],
)
def test_bad_config(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        TiledConfig(**kwargs)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (7, 19),
        (600, 1024),
        (1024, 1024),
        (1100, 1300),
        (2048, 2048),
        (3000, 4000),
        (17, 4096),
    ],
)
def test_windows_match_official(shape) -> None:
    official = _reference("data/split_dota.py", ["get_windows"])["get_windows"]
    for config in [
        TiledConfig(),
        TiledConfig(rates=(1.0,)),
        TiledConfig(im_rate_thr=0.95),
    ]:
        sizes, gaps = zip(*config.crop_pairs)
        expected = official(shape, sizes, gaps, config.im_rate_thr, config.eps)
        np.testing.assert_array_equal(get_windows(shape, config), expected)


def test_native_crop_no_resize_pad_and_ownership() -> None:
    source = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
    tiles = list(iter_tiles(source, "P0001", TiledConfig(jpeg_roundtrip=False)))
    assert len(tiles) == 3  # all tiny occupancy rates are within absolute eps=.01
    tile = tiles[-1]
    assert tile.name == "P0001__682__0___0"
    assert tile.rate == 1.5
    assert tile.window == (0, 0, 682, 682)
    np.testing.assert_array_equal(tile.image, source)
    assert tile.image.flags.c_contiguous
    tile.image[:] = 0
    assert source.any()


def test_jpeg_roundtrip_matches_official_split_test(tmp_path) -> None:
    reference = _reference(
        "data/split_dota.py",
        ["get_windows", "split_test"],
        {
            "glob": lambda pattern: [str(tmp_path / "source/images/test/P0001.png")],
            "TQDM": lambda values, **kwargs: values,
            "exif_size": lambda image: image.size,
        },
    )
    root = tmp_path / "source/images/test"
    root.mkdir(parents=True)
    image = np.random.default_rng(0).integers(0, 256, (27, 39, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "P0001.png"), image)
    config = TiledConfig(crop_size=32, gap=15)
    reference["split_test"](
        str(tmp_path / "source"), str(tmp_path / "official"), 32, 15, config.rates
    )
    tiles = list(iter_tiles(image, "P0001", config))
    output = tmp_path / "official/images/test"
    assert sorted(p.stem for p in output.glob("*.jpg")) == sorted(t.name for t in tiles)
    for tile in tiles:
        expected = cv2.imread(str(output / f"{tile.name}.jpg"))
        np.testing.assert_array_equal(tile.image, expected)


@pytest.mark.parametrize("shape", [(13, 19), (7, 31), (2048, 2048), (682, 682)])
def test_letterbox_pixels_colors_and_restoration(shape) -> None:
    tile = _tile(shape=shape)
    tile.image[:] = (10, 20, 30)
    tensor, geometry = prepare_tile(tile, 64)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32 and tensor.is_contiguous()
    left, top = geometry.pad
    torch.testing.assert_close(
        tensor[0, :, top, left], torch.tensor([30, 20, 10]) / 255
    )
    if top > 0:
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.full((3,), 114 / 255))
    original = torch.tensor([[shape[1] / 2, shape[0] / 2, 8.5, 3.25, 0.7, 0.9, 2.0]])
    network = original.clone()
    network[:, :4] *= geometry.ratio
    network[:, 0] += left
    network[:, 1] += top
    torch.testing.assert_close(restore_tile_boxes(network, geometry), original)
    assert torch.equal(network[:, 4:], original[:, 4:])


def test_letterbox_inverse_matches_official_without_clipping() -> None:
    scale = _reference("utils/ops.py", ["scale_boxes"])["scale_boxes"]
    tile = _tile(shape=(13, 19))
    _, geometry = prepare_tile(tile, 64)
    network = torch.tensor([[-8.0, 70.0, 11.0, 7.0, -0.3, 0.9, 0.0]])
    expected = scale(
        (64, 64),
        network[:, :5].clone(),
        (13, 19),
        ratio_pad=geometry.ratio_pad,
        xywh=True,
    )
    actual = restore_tile_boxes(network, geometry)
    torch.testing.assert_close(actual[:, :5], expected)
    assert actual[0, 0] < 0  # no axis-aligned clipping of rotated centers


def test_offset_only_each_scale_and_reference_round_before_offset() -> None:
    for size, rate in [(2048, 0.5), (1024, 1.0), (682, 1.5)]:
        rows = torch.tensor(
            [[12.34567, 23.45678, 8.76543, 4.12345, 0.12345, 0.987654, 0.0]]
        )
        result = merge_tile_predictions([(_tile(1048, 349, size, rate), rows)])
        expected = torch.tensor(
            [[1060.346, 372.457, 8.765, 4.123, 0.123, 0.98765, 0.0]]
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert rows[0, 0] == torch.tensor(12.34567)


def test_cross_tile_and_scale_dedup_class_offsets_and_fast_not_greedy() -> None:
    first = torch.tensor(
        [
            [600.0, 600.0, 100.0, 30.0, 0.2, 0.95, 0.0],
            [600.0, 600.0, 100.0, 30.0, 0.2, 0.90, 1.0],
        ]
    )
    duplicate = torch.tensor([[76.0, 76.0, 100.0, 30.0, 0.2, 0.85, 0.0]])
    result = merge_tile_predictions(
        [(_tile(), first), (_tile(524, 524, 682, 1.5), duplicate)]
    )
    assert result.shape == (2, 7)
    torch.testing.assert_close(result[:, 5], torch.tensor([0.95, 0.90]))
    # A overlaps B, B overlaps C, A does NOT overlap C: fast NMS still
    # suppresses C via B, unlike greedy NMS, even though B itself is suppressed.
    chain = torch.tensor(
        [
            [100.0, 100.0, 100.0, 20.0, 0.0, 0.9, 0.0],
            [145.0, 100.0, 100.0, 20.0, 0.0, 0.8, 0.0],
            [190.0, 100.0, 100.0, 20.0, 0.0, 0.7, 0.0],
        ]
    )
    assert batch_probiou(chain[:1, :5], chain[1:, :5])[0, 0] >= 0.3
    assert batch_probiou(chain[:1, :5], chain[1:, :5])[0, 1] < 0.3
    assert len(merge_tile_predictions([(_tile(), chain)])) == 1


def test_rotated_geometry_not_axis_aligned_and_threshold_equality() -> None:
    boxes = torch.tensor(
        [
            [100.0, 100.0, 100.0, 2.0, math.pi / 4],
            [100.0, 100.0, 100.0, 2.0, -math.pi / 4],
        ]
    )
    assert len(rotated_nms(boxes, torch.tensor([0.9, 0.8]), 0.3)) == 2
    overlap = lambda a, b: torch.full((len(a), len(b)), 0.3)
    assert rotated_nms(boxes, torch.tensor([0.9, 0.8]), 0.3, overlap).tolist() == [0]


def test_merge_matches_official_randomized_and_task1(tmp_path) -> None:
    metrics = _reference(
        "utils/metrics.py", ["_get_covariance_matrix", "batch_probiou"]
    )
    nms = _reference("utils/nms.py", ["TorchNMS"], {"box_iou": None})["TorchNMS"]
    ops = _reference("utils/ops.py", ["xywhr2xyxyxyxy"])
    # Use actual eval_json method with fake surrounding validator objects. This
    # exercises JSON quantization, dynamic offsets, Fast-NMS, polygon order and text.
    val = _reference(
        "models/yolo/obb/val.py",
        ["OBBValidator"],
        {
            "DetectionValidator": object,
            "LOGGER": SimpleNamespace(info=lambda *_: None),
            "ops": SimpleNamespace(**ops),
            "TorchNMS": nms,
            "batch_probiou": metrics["batch_probiou"],
        },
    )
    rng = np.random.default_rng(0)
    image_predictions, records = [], []
    for tile in [_tile(0, 0, 2048, 0.5), _tile(524, 524), _tile(349, 349, 682, 1.5)]:
        rows = np.column_stack(
            (
                rng.uniform(0, 500, (21, 2)),
                rng.uniform(10, 100, (21, 2)),
                rng.uniform(-math.pi, math.pi, 21),
                rng.uniform(0.1, 1.0, 21),
                rng.integers(0, 3, 21),
            )
        ).astype(np.float32)
        image_predictions.append((tile, rows))
        for row in rows.tolist():
            records.append(
                {
                    "image_id": tile.name,
                    "score": round(row[5], 5),
                    "category_id": int(row[6]) + 1,
                    "rbox": [round(v, 3) for v in row[:5]],
                    "poly": [
                        round(v, 3)
                        for v in xywhr2xyxyxyxy(np.array(row[:5])).reshape(-1).tolist()
                    ],
                }
            )
    import json

    official_dir = tmp_path / "official"
    official_dir.mkdir()
    (official_dir / "predictions.json").write_text(json.dumps(records))
    validator = SimpleNamespace(
        args=SimpleNamespace(save_json=True),
        is_dota=True,
        jdict=records,
        save_dir=official_dir,
        names={0: "plane", 1: "ship", 2: "large vehicle"},
    )
    assert val["OBBValidator"].eval_json(validator, {"sentinel": 1}) == {"sentinel": 1}
    actual = merge_tile_predictions(image_predictions)
    ours = write_task1(
        {"P0001": actual}, tmp_path / "ours", ["plane", "ship", "large vehicle"]
    )
    for file in (official_dir / "predictions_merged_txt").glob("*.txt"):
        assert (ours / file.name).read_text() == file.read_text()
    b1 = torch.tensor(rng.normal(size=(9, 5)), dtype=torch.float32)
    b1[:, 2:4].abs_()
    torch.testing.assert_close(
        batch_probiou(b1, b1), metrics["batch_probiou"](b1, b1), rtol=0, atol=0
    )


def test_empty_validation_and_no_input_mutation(tmp_path) -> None:
    assert merge_tile_predictions([]).shape == (0, 7)
    assert merge_tile_predictions([(_tile(), np.empty((0, 7)))]).shape == (0, 7)
    with pytest.raises(ValueError, match="only one original"):
        other = Tile("P0002", "unused", 1, 1024, 500, (0, 0, 1024, 1024), _tile().image)
        merge_tile_predictions(
            [(_tile(), torch.empty(0, 7)), (other, torch.empty(0, 7))]
        )
    for rows in [
        torch.zeros(1, 6),
        torch.full((1, 7), float("nan")),
        torch.tensor([[0.0, 0.0, -1.0, 1.0, 0.0, 0.5, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.1, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5]]),
    ]:
        with pytest.raises(ValueError):
            merge_tile_predictions([(_tile(), rows)])
    for name in ["", "P__bad", "../P", "P 1"]:
        with pytest.raises(ValueError):
            list(iter_tiles(np.zeros((2, 3, 3), np.uint8), name))
    with pytest.raises(TypeError):
        list(iter_tiles(np.zeros((2, 3, 3)), "P0001"))
    with pytest.raises(ValueError):
        write_task1(
            {"P0001": torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 15.0]])}, tmp_path
        )


def test_task1_rounding_corner_order_empty_classes_and_replacement(tmp_path) -> None:
    row = torch.tensor([[10.0, 20.0, 4.0, 2.0, 0.0, 0.98765, 0.0]])
    output = write_task1({"P0001": row}, tmp_path, ["plane", "large vehicle"])
    assert (
        output / "Task1_plane.txt"
    ).read_text() == "P0001 0.988 12.0 21.0 12.0 19.0 8.0 19.0 8.0 21.0\n"
    assert (output / "Task1_large-vehicle.txt").read_text() == ""
    write_task1({"P0001": row}, tmp_path, ["plane", "large vehicle"])
    assert len((output / "Task1_plane.txt").read_text().splitlines()) == 1


def test_predictor_callback_empty_and_nonempty() -> None:
    callback = Mock(return_value=torch.tensor([[3.0, 4.0, 2.0, 1.0, 0.0, 0.8, 0.0]]))
    result = predict_tiled(np.zeros((8, 12, 3), np.uint8), "P0001", callback)
    assert len(result) == 1
    assert callback.call_args.args[0].image.shape == (8, 12, 3)


def test_reference_adapter_mocked_raw_inference_contract(tmp_path, monkeypatch) -> None:
    import sys

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic, never deserialized")
    model = Mock()
    model.cpu.return_value = model
    model.float.return_value = model
    model.eval.return_value = model
    model.return_value = torch.zeros((1, 20, 10))
    yolo = Mock(
        return_value=SimpleNamespace(task="obb", model=model, names={0: "plane"})
    )
    nms = Mock(return_value=[torch.tensor([[32.0, 32.0, 8.0, 4.0, 0.9, 0.0, 0.3]])])
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(__version__="8.3.228", YOLO=yolo)
    )
    monkeypatch.setitem(
        sys.modules, "ultralytics.utils.nms", SimpleNamespace(non_max_suppression=nms)
    )
    adapter = UltralyticsTilePredictor(checkpoint, imgsz=64)
    output = adapter(_tile(shape=(64, 64)))
    torch.testing.assert_close(
        output, torch.tensor([[32.0, 32.0, 8.0, 4.0, 0.3, 0.9, 0.0]])
    )
    assert nms.call_args.kwargs == {
        "conf_thres": 0.01,
        "iou_thres": 0.7,
        "multi_label": True,
        "max_det": 300,
        "nc": 1,
        "rotated": True,
    }
    model.assert_called_once()
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(__version__="0.0", YOLO=yolo)
    )
    with pytest.raises(RuntimeError, match="8.3.228"):
        UltralyticsTilePredictor(checkpoint)
    with pytest.raises(ValueError, match="local"):
        UltralyticsTilePredictor("not-a-download.pt")


def test_cli_callback_synthetic_fixture(tmp_path, monkeypatch) -> None:
    import sys

    spec = importlib.util.spec_from_file_location(
        "dota_tiled_cli", Path(__file__).parents[1] / "benchmark/dota_tiled.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = tmp_path / "P0001.png"
    Image.new("RGB", (17, 11), (10, 20, 30)).save(source)
    monkeypatch.setitem(
        sys.modules,
        "synthetic_tile_callback",
        SimpleNamespace(
            factory=lambda: lambda tile: np.array([[3.0, 4.0, 2.0, 1.0, 0.0, 0.8, 0.0]])
        ),
    )
    args = [
        "--source",
        str(source),
        "--output",
        str(tmp_path / "submission"),
        "--predictor-factory",
        "synthetic_tile_callback:factory",
    ]
    assert cli.main(args) == 0
    assert (
        (tmp_path / "submission/Task1_plane.txt").read_text().startswith("P0001 0.8 ")
    )
    assert (tmp_path / "submission/pipeline.json").is_file()
    with pytest.raises(SystemExit):
        cli.main(args)  # no accidental append into an earlier submission

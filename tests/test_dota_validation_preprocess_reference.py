"""Pin upstream validation pixels and OBB restoration without model inference."""

from __future__ import annotations

import ast
import math
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from mblt_vision.utils.datasets.dota_tiling import Tile, prepare_tile
from mblt_vision.utils.postprocess.tiled_obb import (
    UltralyticsTilePredictor,
    restore_tile_boxes,
)


def _definition(relative, class_name, method=None, namespace=None):
    root = os.environ.get("MBLT_DOTA_REFERENCE")
    if not root:
        pytest.skip("Set MBLT_DOTA_REFERENCE to pinned v8.3.228 source")
    path = Path(root) / relative
    if not path.is_file():
        pytest.skip(f"Optional pinned preprocessing reference absent: {relative}")
    source = ast.parse(path.read_text(), filename=str(path))
    node = next(
        n
        for n in source.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == class_name
    )
    if method is not None:
        node = next(
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    scope = {"np": np, "cv2": cv2, "math": math, "Path": Path, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)  # noqa: S102 -- only pinned named definitions
    return scope[method or class_name]


@pytest.mark.parametrize(
    "shape,expected_equal",
    [((7, 31), False), ((31, 31), True), ((31, 7), False), ((803, 1025), False)],
)
def test_validation_loader_ceil_then_letterbox_not_direct_round(
    tmp_path, shape, expected_equal
):
    image = np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    load_image = _definition(
        "data/base.py",
        "BaseDataset",
        "load_image",
        {"imread": lambda *_args, **_kwargs: image.copy()},
    )
    letterbox = _definition("data/augment.py", "LetterBox")
    build_transforms = _definition(
        "data/dataset.py",
        "YOLODataset",
        "build_transforms",
        {
            "Compose": list,
            "LetterBox": letterbox,
            "Format": lambda **kwargs: kwargs,
        },
    )
    dataset = SimpleNamespace(
        ims=[None],
        im_files=["synthetic.png"],
        npy_files=[tmp_path / "not-cached.npy"],
        imgsz=64,
        cv2_flag=cv2.IMREAD_COLOR,
        augment=False,
        use_segments=False,
        use_keypoints=False,
        use_obb=True,
    )
    resized, original_shape, resized_shape = load_image(dataset, 0)
    ratio = 64 / max(shape)
    assert original_shape == shape
    assert resized_shape == (
        min(math.ceil(shape[0] * ratio), 64),
        min(math.ceil(shape[1] * ratio), 64),
    )
    transform = build_transforms(
        dataset, SimpleNamespace(mask_ratio=4, overlap_mask=True)
    )[0]
    assert transform.scaleup is False
    get_image_and_label = _definition(
        "data/base.py", "BaseDataset", "get_image_and_label", {"deepcopy": deepcopy}
    )
    dataset.labels = [{}]
    dataset.load_image = lambda _index: (resized, original_shape, resized_shape)
    dataset.rect = False
    dataset.update_labels_info = lambda labels: labels
    labels = get_image_and_label(dataset, 0)
    # No annotated instances are needed for this image/metadata-only reference.
    transform._update_labels = lambda labels, *_args: labels
    upstream = transform(labels)
    tile = Tile("P", "P__64__0___0", 1.0, 64, 0, (0, 0, 64, 64), image)
    direct, _ = prepare_tile(tile, 64, preprocessing="direct-letterbox")
    official, geometry = prepare_tile(tile, 64)
    expected_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(upstream["img"][..., ::-1].transpose(2, 0, 1))
        )
        .float()
        .unsqueeze(0)
        / 255
    )
    assert torch.equal(official, expected_tensor)
    assert torch.equal(direct, official) == expected_equal
    assert geometry.ratio_pad == upstream["ratio_pad"]
    assert geometry.ratio == upstream["ratio_pad"][0][0]
    if shape == (7, 31):
        assert resized_shape == (15, 64)  # historical direct round gives height14
        assert geometry.pad == (0, 24) and geometry.ratio == 15 / 7

    scale_boxes = _definition("utils/ops.py", "scale_boxes")
    scale_preds = _definition(
        "models/yolo/obb/val.py",
        "OBBValidator",
        "scale_preds",
        {"ops": SimpleNamespace(scale_boxes=scale_boxes)},
    )
    network = torch.tensor([[-8.0, 70.0, 11.0, 7.0, -0.3, 0.9, 0.0]])
    expected = scale_preds(
        None,
        {"bboxes": network[:, :5]},
        {"imgsz": (64, 64), "ori_shape": shape, "ratio_pad": upstream["ratio_pad"]},
    )["bboxes"]
    torch.testing.assert_close(restore_tile_boxes(network, geometry)[:, :5], expected)
    assert expected[0, 0] < 0  # xywh=True deliberately does not clip rotated boxes

    # Exercise the optional adapter's actual preparation/restoration boundary
    # with synthetic raw/NMS stubs, without constructing/loading/running a model.
    adapter = UltralyticsTilePredictor.__new__(UltralyticsTilePredictor)
    adapter.imgsz, adapter.preprocessing = 64, "ultralytics-val"
    adapter.conf, adapter.iou, adapter.max_det = 0.01, 0.7, 300
    adapter.names = ("plane",)

    def synthetic_raw(tensor):
        assert torch.equal(tensor, expected_tensor)
        return None

    adapter.model = synthetic_raw
    adapter._nms = lambda *_args, **_kwargs: [network[:, [0, 1, 2, 3, 5, 6, 4]]]
    torch.testing.assert_close(adapter(tile)[:, :5], expected)

    # Exercise the actual engine preparation -> inherited nmsout2eval -> merge
    # bridge, substituting only raw inference and the already-decoded NMS rows.
    from mblt_vision.utils.datasets.dota_tiling import TiledConfig
    from mblt_vision.utils.evaluation.eval_dota_tiled import predict_engine_tiles
    from mblt_vision.utils.postprocess.build_post import build_postprocess
    from mblt_vision.utils.preprocess.build_pre import build_preprocess
    from mblt_vision.wrapper import resolve_model_config

    config = resolve_model_config("YOLO11n-obb", "DEFAULT")
    config["pre_cfg"]["LetterBox"]["img_size"] = [64, 64]
    preprocessor = build_preprocess(config["pre_cfg"])

    class SyntheticEngine:
        pre_cfg = config["pre_cfg"]
        postprocessor = build_postprocess(config["pre_cfg"], config["post_cfg"])

        def preprocess_with_metadata(self, rgb):
            np.testing.assert_array_equal(rgb, upstream["img"][..., ::-1])
            return preprocessor.with_metadata(rgb)

        def __call__(self, values):
            np.testing.assert_array_equal(
                values[0], expected_tensor[0].permute(1, 2, 0).numpy()
            )
            return None

        def postprocess(self, _raw, *, multi_label):
            assert multi_label is True
            return SimpleNamespace(output=[network[:, [0, 1, 2, 3, 5, 6, 4]]])

    rows = predict_engine_tiles(
        SyntheticEngine(),
        iter([("P", image)]),
        config=TiledConfig(
            crop_size=max(shape), gap=0, rates=(1.0,), jpeg_roundtrip=False
        ),
    )["P"]
    rounded = torch.tensor([[round(v, 3) for v in expected[0].tolist()]])
    torch.testing.assert_close(rows[:, :5], rounded)


def test_invalid_preprocessing_fails_before_reference_model_load(tmp_path):
    tile = Tile(
        "P", "P__64__0___0", 1.0, 64, 0, (0, 0, 64, 64), np.zeros((7, 31, 3), np.uint8)
    )
    with pytest.raises(ValueError, match="preprocessing"):
        prepare_tile(tile, 64, preprocessing="unsupported")
    with pytest.raises(ValueError, match="preprocessing"):
        UltralyticsTilePredictor(tmp_path / "missing.pt", preprocessing="unsupported")

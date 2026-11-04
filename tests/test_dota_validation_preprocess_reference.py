"""Pin a known direct-letterbox versus upstream validation-loader difference.

This is an intermediate-only CPU check: no model, upstream imports or inference.
It deliberately does not change the frozen v1 callback/preprocessing contract.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from mblt_vision.utils.datasets.dota_tiling import Tile, prepare_tile

HASHES = {
    "data/base.py": "dac2668755540afc637ddbc401095d2bd3cb56cc3ea43563732a3c8022e56eea",
    "data/dataset.py": "2f941882273f0757b5cdf7e0440e65a8a0dde4f42e30383a3d955df914d403b1",
    "data/augment.py": "db2c9e2880014ea8087fbfeca54a8647ce3a91ac38d130e5965dfa098cfc6354",
}


def _definition(relative, class_name, method=None, namespace=None):
    root = os.environ.get("MBLT_DOTA_REFERENCE")
    if not root:
        pytest.skip("Set MBLT_DOTA_REFERENCE to pinned v8.3.228 source")
    path = Path(root) / relative
    if not path.is_file():
        pytest.skip(f"Optional pinned preprocessing reference absent: {relative}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[relative]
    source = ast.parse(path.read_text(), filename=str(path))
    node = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name
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


@pytest.mark.parametrize("shape,expected_equal", [((7, 31), False), ((31, 31), True)])
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
    upstream = transform(image=resized)
    tile = Tile("P", "P__64__0___0", 1.0, 64, 0, (0, 0, 64, 64), image)
    direct, _ = prepare_tile(tile, 64)
    ours = np.rint(direct[0].numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(
        np.uint8
    )
    assert np.array_equal(ours, upstream) == expected_equal
    if not expected_equal:
        assert resized_shape == (15, 64)  # upstream ceil; direct round is 14 pixels
        assert np.count_nonzero(ours != upstream) > 0

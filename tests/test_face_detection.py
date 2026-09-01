"""CPU regression tests for face-detection postprocessing and exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch
import yaml

import mblt_vision
from mblt_vision import YOLO11m_face, list_models
from mblt_vision.face_detection import YOLO11m_face as FaceDetectionYOLO11mFace
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.postprocess.base import YOLODetectionPostBase
from mblt_vision.utils.postprocess.common import (
    YOLOFaceDetectionMixin,
    nmsout2eval_face,
)
from mblt_vision.utils.postprocess.yolo_anchor_post import (
    YOLOAnchorDetectionPost,
    YOLOAnchorFaceDetectionPost,
)
from mblt_vision.utils.postprocess.yolo_anchorless_post import (
    YOLOAnchorlessDetectionPost,
    YOLOAnchorlessFaceDetectionPost,
)
from mblt_vision.utils.postprocess.yolo_dflfree_post import (
    YOLODFLFreeDetectionPost,
    YOLODFLFreeFaceDetectionPost,
)
from mblt_vision.utils.postprocess.yolo_nmsfree_post import (
    YOLONMSFreeDetectionPost,
    YOLONMSFreeFaceDetectionPost,
)
from mblt_vision.utils.results import Results

MODEL_CONFIG_DIR = Path(mblt_vision.__file__).parent / "models"


def _pre_cfg() -> dict[str, Any]:
    """Return a representative face preprocessing configuration."""

    return {"LetterBox": {"img_size": [640, 640]}}


def _post_cfg(**overrides: Any) -> dict[str, Any]:
    """Return a representative face postprocessing configuration."""

    return {
        "task": "face_detection",
        "nl": 3,
        "reg_max": 16,
        "conf_thres": 0.25,
        **overrides,
    }


@pytest.mark.parametrize(
    ("post_cfg", "expected_type", "expected_family"),
    [
        (
            {"nl": 3, "reg_max": 16},
            YOLOAnchorlessFaceDetectionPost,
            YOLOAnchorlessDetectionPost,
        ),
        (
            {"nl": 3, "dflfree": True},
            YOLODFLFreeFaceDetectionPost,
            YOLODFLFreeDetectionPost,
        ),
        (
            {"nl": 3, "nmsfree": True},
            YOLONMSFreeFaceDetectionPost,
            YOLONMSFreeDetectionPost,
        ),
        (
            {"anchors": [[10, 13, 16, 30, 33, 23]]},
            YOLOAnchorFaceDetectionPost,
            YOLOAnchorDetectionPost,
        ),
    ],
)
def test_face_detection_routes_postprocessors(
    post_cfg: dict[str, Any],
    expected_type: type[YOLODetectionPostBase],
    expected_family: type[YOLODetectionPostBase],
) -> None:
    """Route every supported face head family to its dedicated face postprocessor.

    Each dedicated class must inherit from both the matching detection-family
    base (so it decodes/NMS-suppresses identically) and ``YOLOFaceDetectionMixin``
    (so it evaluates with a single ``"face"`` label instead of COCO categories).
    """

    postprocessor = build_postprocess(_pre_cfg(), _post_cfg(**post_cfg))

    assert type(postprocessor) is expected_type
    assert isinstance(postprocessor, expected_family)
    assert isinstance(postprocessor, YOLOFaceDetectionMixin)
    assert cast(YOLODetectionPostBase, postprocessor).nc == 1


def test_face_detection_exports_and_plot_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Preserve legacy exports and render a face-specific detection label."""

    assert "YOLO11m_face" in list_models("face_detection")["face_detection"]
    assert YOLO11m_face is FaceDetectionYOLO11mFace
    source_path = tmp_path / "source.jpg"
    output_path = tmp_path / "face.jpg"
    cv2.imwrite(str(source_path), np.full((32, 32, 3), 255, dtype=np.uint8))
    labels: list[str] = []
    original_put_text = cv2.putText
    monkeypatch.setattr(
        cv2,
        "putText",
        lambda *args, **kwargs: (
            labels.append(args[1]),
            original_put_text(*args, **kwargs),
        )[1],
    )

    Results(
        _pre_cfg(),
        {"task": "face_detection"},
        [torch.tensor([[1, 1, 16, 16, 0.95, 0]])],
    ).plot(str(source_path), save_path=str(output_path))

    assert output_path.is_file()
    assert labels == ["face 95%"]


def test_face_detection_non_e2e_converted_and_raw_outputs() -> None:
    """Keep converted and raw face heads available through the legacy non-E2E contract."""

    postprocessor = build_postprocess(_pre_cfg(), _post_cfg(e2e=False))
    converted = torch.tensor(
        [
            [
                [10.0, 12.0, 0.9],
                [20.0, 18.0, 0.1],
                [8.0, 7.0, 0.2],
                [9.0, 6.0, 0.3],
                [0.8, 0.5, 0.1],
            ]
        ]
    )
    converted_result = postprocessor(converted)
    raw_heads = [torch.zeros((1, size, size, 64)) for size in (80, 40, 20)]
    raw_heads = [
        tensor
        for pair in zip(
            raw_heads, [torch.zeros((*tensor.shape[:3], 1)) for tensor in raw_heads]
        )
        for tensor in pair
    ]
    raw_result = postprocessor(raw_heads)

    assert isinstance(converted_result, torch.Tensor)
    assert converted_result.shape == (1, 5, 3)
    assert isinstance(raw_result, torch.Tensor)
    assert raw_result.shape == (1, 5, 8400)


def test_nmsout2eval_face_labels_every_row_face() -> None:
    """Convert single-class face rows without routing through COCO category IDs."""

    detections = torch.tensor([[10.0, 10.0, 20.0, 20.0, 0.9, 0.0]])

    labels, boxes, scores = nmsout2eval_face(detections, (100, 100), (100, 100))

    assert labels == [["face"]]
    assert scores == [[0.9]]
    assert boxes[0][0] == pytest.approx([10.0, 10.0, 10.0, 10.0])


def test_nmsout2eval_face_rejects_nonzero_class_ids() -> None:
    """Reject any class id other than 0 instead of silently mislabeling it."""

    detections = torch.tensor([[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]])

    with pytest.raises(ValueError, match="must all be 0"):
        nmsout2eval_face(detections, (100, 100), (100, 100))


ANCHOR_FACE_MODELS = (
    "YOLOv5n-face",
    "YOLOv5n-0.5-face",
    "YOLOv5s-face",
    "YOLOv5m-face",
    "YOLOv7-face",
    "YOLOv7s-face",
    "YOLOv7-tiny-face",
    "YOLOv7-lite-s-face",
    "YOLOv7-lite-t-face",
)


@pytest.mark.parametrize("model_name", ANCHOR_FACE_MODELS)
def test_anchor_face_yaml_builds_anchor_face_postprocessor(model_name: str) -> None:
    """Route every shipped anchor-based face YAML to the anchor face postprocessor.

    The ``YOLOv5*-face`` and ``YOLOv7*-face`` families are the only shipped face
    models that carry an ``anchors`` list, so they are what makes
    ``YOLOAnchorFaceDetectionPost`` reachable from the registry rather than from
    a synthetic ``post_cfg``.
    """

    config = yaml.safe_load(
        (MODEL_CONFIG_DIR / f"{model_name}.yaml").read_text(encoding="utf-8")
    )["DEFAULT"]
    post_cfg = config["post_cfg"]

    postprocessor = build_postprocess(config["pre_cfg"], post_cfg)

    assert type(postprocessor) is YOLOAnchorFaceDetectionPost
    assert config["pre_cfg"]["LetterBox"]["img_size"] == [640, 640]
    assert post_cfg["dataset"] == "widerface"
    assert post_cfg["iou_thres"] == 0.5
    assert len(post_cfg["anchors"]) == 3
    assert all(len(level) == 6 for level in post_cfg["anchors"])
    detection_post = cast(YOLODetectionPostBase, postprocessor)
    assert detection_post.nc == 1
    assert detection_post.na == 3
    assert detection_post.nl == 3

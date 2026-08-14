"""Focused tests for the vision postprocessor class hierarchy and builder."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
import torch
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.postprocess import common as common_module
from mblt_vision.utils.postprocess.base import PostBase, YOLODetectionPostBase
from mblt_vision.utils.postprocess.cls_post import ClsPost
from mblt_vision.utils.postprocess.depth_post import DepthPost
from mblt_vision.utils.postprocess.semantic_seg_post import SemanticSegPost
from mblt_vision.utils.postprocess.yolo_anchor_post import (
    YOLOAnchorDetectionPost,
    YOLOAnchorSegPost,
)
from mblt_vision.utils.postprocess.yolo_anchorless_post import (
    YOLOAnchorlessDetectionPost,
    YOLOAnchorlessOBBPost,
    YOLOAnchorlessPosePost,
    YOLOAnchorlessSegPost,
)
from mblt_vision.utils.postprocess.yolo_dflfree_post import (
    YOLODFLFreeDetectionPost,
    YOLODFLFreeOBBPost,
    YOLODFLFreePosePost,
    YOLODFLFreeSegPost,
)
from mblt_vision.utils.postprocess.yolo_nmsfree_post import YOLONMSFreeDetectionPost


def test_segmentation_rle_encoding_thresholds_resized_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep fractional bilinear boundaries from being truncated during RLE export."""

    resized_mask = common_module.scale_masks(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]), (3, 4)
    )
    captured_pixels: list[torch.Tensor] = []

    def _capture_encode(pixels: torch.Tensor) -> list[list[int]]:
        captured_pixels.append(pixels.clone())
        return [[pixels.numel()]]

    monkeypatch.setattr(common_module, "multi_encode", _capture_encode)

    common_module._encode_segmentation_masks(resized_mask)

    assert 0.5 < resized_mask[0, 0, 1] < 1.0
    assert captured_pixels[0].view(1, 4, 3).permute(0, 2, 1).tolist() == [
        [[1, 1, 0, 0]] * 3
    ]


def test_classification_postprocessor_rejects_wrong_taxonomy_width() -> None:
    """Reject local classification artifacts whose heads do not match ImageNet."""

    postprocessor = ClsPost({}, {"task": "image_classification", "dataset": "imagenet"})

    with pytest.raises(
        ValueError,
        match="Classification output has 999 classes, but dataset 'imagenet' requires 1000",
    ):
        postprocessor(torch.zeros((1, 999), dtype=torch.float32))


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")])
def test_classification_postprocessor_rejects_nonfinite_scores(
    invalid_score: float,
) -> None:
    """Do not convert malformed classification logits into plausible predictions."""

    scores = torch.zeros((1, 1000), dtype=torch.float32)
    scores[0, 0] = invalid_score

    with pytest.raises(ValueError, match="scores must all be finite"):
        ClsPost({}, {"task": "image_classification", "dataset": "imagenet"})(scores)


@pytest.mark.parametrize("invalid_class", [-1.0, 1.5, 2.0, float("nan")])
def test_already_decoded_detections_reject_invalid_task_class_ids(
    invalid_class: float,
) -> None:
    """Validate local decoded detection outputs against the configured taxonomy."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    detections = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0, 0.9, invalid_class]]], dtype=torch.float32
    )

    with pytest.raises(ValueError, match="Decoded detection class IDs"):
        postprocessor._final_detection_batches(detections)


@pytest.mark.parametrize(
    ("post_cfg", "expected_type"),
    [
        (
            {"task": "object_detection", "anchors": [[10, 13, 16, 30, 33, 23]]},
            YOLOAnchorDetectionPost,
        ),
        (
            {"task": "object_detection", "nl": 3, "reg_max": 16},
            YOLOAnchorlessDetectionPost,
        ),
        (
            {"task": "object_detection", "nl": 3, "dflfree": True},
            YOLODFLFreeDetectionPost,
        ),
        (
            {"task": "object_detection", "nl": 3, "reg_max": 16, "nmsfree": True},
            YOLONMSFreeDetectionPost,
        ),
        (
            {
                "task": "instance_segmentation",
                "anchors": [[10, 13, 16, 30, 33, 23]],
                "n_extra": 32,
            },
            YOLOAnchorSegPost,
        ),
        (
            {"task": "instance_segmentation", "nl": 3, "reg_max": 16, "n_extra": 32},
            YOLOAnchorlessSegPost,
        ),
        (
            {"task": "instance_segmentation", "nl": 3, "dflfree": True, "n_extra": 32},
            YOLODFLFreeSegPost,
        ),
        (
            {"task": "pose_estimation", "nl": 3, "reg_max": 16, "n_extra": 51},
            YOLOAnchorlessPosePost,
        ),
        (
            {"task": "pose_estimation", "nl": 3, "dflfree": True, "n_extra": 51},
            YOLODFLFreePosePost,
        ),
        ({"task": "obb", "nl": 3, "reg_max": 16, "n_extra": 1}, YOLOAnchorlessOBBPost),
        ({"task": "obb", "nl": 3, "dflfree": True, "n_extra": 1}, YOLODFLFreeOBBPost),
    ],
)
def test_builder_routes_detection_backends_without_warnings(
    post_cfg: dict[str, Any], expected_type: type[YOLODetectionPostBase]
) -> None:
    """Build every detection family through canonical warning-free imports."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        postprocessor = build_postprocess(
            {"LetterBox": {"img_size": [640, 640]}}, post_cfg
        )
    assert type(postprocessor) is expected_type


@pytest.mark.parametrize(
    ("pre_cfg", "post_cfg", "expected_type"),
    [
        ({}, {"task": "image_classification"}, ClsPost),
        ({"LetterBox": {"img_size": [8, 8]}}, {"task": "depth_estimation"}, DepthPost),
        (
            {"LetterBox": {"img_size": [8, 8]}},
            {"task": "semantic_segmentation", "dataset": "ade20k"},
            SemanticSegPost,
        ),
    ],
)
def test_builder_keeps_non_detection_routing_warning_free(
    pre_cfg: dict[str, Any], post_cfg: dict[str, Any], expected_type: type[PostBase]
) -> None:
    """Preserve classification and dense prediction builder routes."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        postprocessor = build_postprocess(pre_cfg, post_cfg)
    assert type(postprocessor) is expected_type


@pytest.mark.parametrize(
    ("postprocessor_type", "input_shape", "n_extra", "output_width"),
    [
        (YOLONMSFreeDetectionPost, (5, 1), 0, 6),
        (YOLODFLFreeDetectionPost, (5, 1), 0, 5),
        (YOLODFLFreePosePost, (56, 1), 51, 56),
        (YOLODFLFreeOBBPost, (6, 1), 1, 6),
    ],
)
def test_empty_anchorless_outputs_preserve_input_device_and_dtype(
    postprocessor_type: type[YOLODetectionPostBase],
    input_shape: tuple[int, int],
    n_extra: int,
    output_width: int,
) -> None:
    """Create empty decoded rows from the selected input tensor."""

    postprocessor = cast(Any, postprocessor_type.__new__(postprocessor_type))
    postprocessor.nc = 1
    postprocessor.n_extra = n_extra
    postprocessor.inv_conf_thres = 1.0
    input_tensor = torch.zeros(input_shape, dtype=torch.float64)

    output = postprocessor.process_box_cls(input_tensor)

    assert output.shape == (0, output_width)
    assert output.device == input_tensor.device
    assert output.dtype == input_tensor.dtype


def test_empty_anchor_outputs_preserve_input_device_and_dtype() -> None:
    """Keep converted and decoded empty anchor rows on the input device."""

    postprocessor = YOLOAnchorDetectionPost.__new__(YOLOAnchorDetectionPost)
    postprocessor.nc = 1
    postprocessor.n_extra = 0
    postprocessor.no = 6
    postprocessor.inv_conf_thres = 1.0
    postprocessor.conf_thres = 1.0
    decoded_input = torch.zeros((1, 6), dtype=torch.float64)
    converted_input = torch.zeros((1, 1, 6), dtype=torch.float64)

    decoded_output = postprocessor.process_box_cls(decoded_input)
    converted_output = postprocessor.filter_conversion(converted_input)[0]

    for output, input_tensor in (
        (decoded_output, decoded_input),
        (converted_output, converted_input),
    ):
        assert output.device == input_tensor.device
        assert output.dtype == input_tensor.dtype


def test_detection_postprocessor_rejects_missing_head_count() -> None:
    """Raise a stable runtime error when anchorless metadata omits nl."""

    with pytest.raises(ValueError, match="nl should be provided in post_cfg"):
        YOLOAnchorlessDetectionPost(
            {"LetterBox": {"img_size": [640, 640]}},
            {"task": "object_detection", "dataset": "coco"},
        )


@pytest.mark.parametrize(
    ("post_cfg", "channels"),
    [
        ({"task": "object_detection", "nl": 3, "reg_max": 16}, [64] * 3 + [80] * 2),
        (
            {"task": "instance_segmentation", "nl": 3, "reg_max": 16, "n_extra": 32},
            [32] * 4 + [64] * 3 + [80] * 2,
        ),
        (
            {"task": "pose_estimation", "nl": 3, "reg_max": 16, "n_extra": 51},
            [64] * 3 + [1] * 2 + [51] * 3,
        ),
        (
            {"task": "obb", "nl": 3, "reg_max": 16, "n_extra": 1},
            [64] * 3 + [15] * 2 + [1] * 3,
        ),
        ({"task": "object_detection", "nl": 3, "dflfree": True}, [4] * 3 + [80] * 2),
        (
            {"task": "instance_segmentation", "nl": 3, "dflfree": True, "n_extra": 32},
            [32] * 4 + [4] * 3 + [80] * 2,
        ),
        (
            {"task": "pose_estimation", "nl": 3, "dflfree": True, "n_extra": 51},
            [4] * 3 + [1] * 2 + [51] * 3,
        ),
        (
            {"task": "obb", "nl": 3, "dflfree": True, "n_extra": 1},
            [4] * 3 + [15] * 2 + [1] * 3,
        ),
    ],
)
def test_split_head_postprocessors_reject_incomplete_head_groups(
    post_cfg: dict[str, Any], channels: list[int]
) -> None:
    """Reject malformed raw output sets before zip can discard a detection scale."""

    postprocessor = build_postprocess({"LetterBox": {"img_size": [64, 64]}}, post_cfg)
    raw_outputs = [torch.zeros((1, 2, 2, channel)) for channel in channels]

    with pytest.raises(
        ValueError,
        match=r"Incomplete split-head outputs: expected 3 heads per group.*classification=2",
    ):
        postprocessor.rearrange(raw_outputs)  # type: ignore[attr-defined]


def test_segmentation_postprocessor_rejects_mismatched_prototype_batch() -> None:
    """Do not silently drop detections when prototypes have fewer images."""

    postprocessor = YOLODFLFreeSegPost(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "instance_segmentation", "nl": 3, "n_extra": 32},
    )
    detections = [torch.empty((0, 38)), torch.empty((0, 38))]
    prototypes = torch.empty((1, 32, 160, 160))

    with pytest.raises(
        ValueError,
        match="Detection and prototype batch sizes must match.*2 detections and 1 prototypes",
    ):
        postprocessor.masking(detections, prototypes)

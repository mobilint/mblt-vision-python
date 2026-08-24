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


def test_classification_postprocessor_keeps_batched_singleton_outputs() -> None:
    """Preserve batch size for local logits shaped [B, C, 1]."""

    postprocessor = ClsPost({}, {"task": "image_classification", "dataset": "imagenet"})
    output = postprocessor(torch.zeros((2, 1000, 1), dtype=torch.float32))

    assert output.shape == (2, 1000)


def test_nmsfree_postprocessor_accepts_decode_enabled_output_triplet() -> None:
    """Normalize QBCompiler's YOLOv10 score/confidence/box outputs."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "object_detection", "nl": 3, "reg_max": 16, "nmsfree": True},
    )
    classes = torch.zeros((1, 2, 80), dtype=torch.float32)
    classes[0, 0, 3] = 0.9
    classes[0, 1, 7] = 0.8
    confidence = torch.tensor([[[0.9], [0.1]]], dtype=torch.float32)
    boxes = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32
    )

    result = postprocessor([classes, confidence, boxes], conf_thres=0.25)

    assert len(result) == 1
    assert result[0].shape == (2, 6)
    assert result[0][0, 5].item() == 3


def test_nmsfree_single_class_output_uses_ordered_confidence_tensor() -> None:
    """Face models must not mistake their class-probability tensor for confidence."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "face_detection", "nl": 3, "reg_max": 16, "nmsfree": True},
    )
    classes = torch.tensor([[[0.9]]], dtype=torch.float32)
    confidence = torch.tensor([[[0.1]]], dtype=torch.float32)
    boxes = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], dtype=torch.float32)

    result = postprocessor([classes, confidence, boxes], conf_thres=0.25)

    assert len(result) == 1
    assert result[0].shape == (0, 6)


def test_nmsfree_decode_enabled_output_rejects_nonfinite_class_probabilities() -> None:
    """Do not turn malformed class values into plausible labels through argmax."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "object_detection", "nl": 3, "reg_max": 16, "nmsfree": True},
    )
    classes = torch.full((1, 1, 80), float("nan"))
    confidence = torch.tensor([[[0.9]]], dtype=torch.float32)
    boxes = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="class probabilities must be finite"):
        postprocessor([classes, confidence, boxes])


def test_nmsfree_decode_enabled_output_is_confidence_ranked_and_capped() -> None:
    """Keep decoded YOLOv10 output cardinality aligned with every other NMS-free path."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "object_detection", "nl": 3, "reg_max": 16, "nmsfree": True},
    )
    candidate_count = 301
    confidence = torch.linspace(0.3, 0.9, candidate_count).reshape(1, -1, 1)
    classes = torch.zeros((1, candidate_count, 80), dtype=torch.float32)
    classes[..., 0] = confidence[..., 0]
    boxes = torch.tensor([0.0, 0.0, 10.0, 10.0]).repeat(1, candidate_count, 1)

    result = postprocessor([classes, confidence, boxes], conf_thres=0.25)

    assert result[0].shape == (300, 6)
    assert torch.all(result[0][1:, 4] <= result[0][:-1, 4])
    assert result[0][0, 4].item() == pytest.approx(0.9)


def test_pose_postprocessor_normalizes_decode_enabled_keypoint_logits() -> None:
    """Keep QBCompiler's compact decoded pose output drawable."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "pose_estimation", "nl": 3, "reg_max": 16, "n_extra": 51},
    )
    converted = torch.zeros((1, 2, 56), dtype=torch.float32)
    converted[0, :, :5] = torch.tensor(
        [[20.0, 20.0, 10.0, 10.0, 0.9], [20.0, 20.0, 10.0, 10.0, 0.8]]
    )
    converted[0, :, 5::3] = 20.0
    converted[0, :, 6::3] = 20.0
    converted[0, :, 7::3] = 0.001
    converted[0, 0, 7] = 2.0

    result = postprocessor([converted], conf_thres=0.25)

    assert result[0].shape == (1, 57)
    assert result[0][0, 8].item() > 0.8


def test_non_e2e_anchorless_pose_preserves_batched_export_output() -> None:
    """Compact decoded pose tensors must bypass e2e-only final-output extraction."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "pose_estimation", "nl": 3, "reg_max": 16, "n_extra": 51},
        e2e=False,
    )
    converted = torch.zeros((1, 2, 56), dtype=torch.float32)

    result = postprocessor([converted])

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 56, 2)


def test_pose_evaluation_conversion_accepts_empty_nms_output() -> None:
    """An image without retained pose detections has no keypoints to scale."""
    labels, boxes, scores, keypoints = common_module.nmsout2eval_pose(
        [torch.empty((0, 57), dtype=torch.float32)],
        (640, 640),
        (480, 640),
    )

    assert labels == boxes == scores == keypoints == [[]]


def test_dflfree_pose_postprocessor_accepts_decode_enabled_output_parts() -> None:
    """Normalize YOLO26's split decoded pose tensors before rendering."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "pose_estimation", "nl": 3, "dflfree": True, "n_extra": 51},
    )
    scores = torch.tensor([[[0.9], [0.8]]], dtype=torch.float32)
    reduced_scores = scores.clone()
    boxes = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]]],
        dtype=torch.float32,
    )
    keypoints = torch.zeros((1, 2, 51), dtype=torch.float32)
    keypoints[..., 0::3] = 20.0
    keypoints[..., 1::3] = 30.0
    keypoints[..., 2::3] = 0.001
    keypoints[0, 0, 2] = 2.0

    result = postprocessor([keypoints, boxes, reduced_scores, scores], conf_thres=0.25)

    assert result[0].shape == (1, 57)
    assert result[0][0, 8].item() > 0.8


def test_non_e2e_dflfree_pose_preserves_batched_export_contract() -> None:
    """Decode-enabled pose parts must reach ``non_e2e`` when requested."""

    postprocessor = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "pose_estimation", "nl": 3, "dflfree": True, "n_extra": 51},
        e2e=False,
    )
    scores = torch.tensor([[[0.9], [0.8]]], dtype=torch.float32)
    boxes = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]]],
        dtype=torch.float32,
    )
    keypoints = torch.zeros((1, 2, 51), dtype=torch.float32)

    result = postprocessor([scores, scores.clone(), boxes, keypoints])

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 300, 57)


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[1.1] + [0.0] * 999]),
        torch.tensor([[0.5] + [0.0] * 999]),
    ],
)
def test_classification_probability_postprocessor_rejects_invalid_probabilities(
    scores: torch.Tensor,
) -> None:
    """Validate local artifacts that declare already-softmaxed outputs."""

    postprocessor = ClsPost(
        {}, {"task": "image_classification", "dataset": "imagenet", "softmax": True}
    )

    with pytest.raises(ValueError, match="probability outputs"):
        postprocessor(scores)


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")])
def test_classification_postprocessor_rejects_nonfinite_scores(
    invalid_score: float,
) -> None:
    """Do not convert malformed classification logits into plausible predictions."""

    scores = torch.zeros((1, 1000), dtype=torch.float32)
    scores[0, 0] = invalid_score

    with pytest.raises(ValueError, match="scores must all be finite"):
        ClsPost({}, {"task": "image_classification", "dataset": "imagenet"})(scores)


@pytest.mark.parametrize("invalid_class", [-1.0, 1.5, 2.0])
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
    ("column", "invalid_value"),
    [(0, float("nan")), (4, float("inf")), (5, float("-inf"))],
)
def test_already_decoded_detections_reject_nonfinite_rows(
    column: int, invalid_value: float
) -> None:
    """Reject malformed coordinates, scores, and labels before filtering them."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    detections = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.9, 1.0]]], dtype=torch.float32)
    detections[0, 0, column] = invalid_value

    with pytest.raises(ValueError, match="finite values"):
        postprocessor._final_detection_batches(detections)


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1])
def test_already_decoded_detections_reject_invalid_confidence(
    invalid_score: float,
) -> None:
    """Reject finite but impossible confidence values before thresholding."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    detections = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0, invalid_score, 1.0]]], dtype=torch.float32
    )

    with pytest.raises(ValueError, match=r"confidence values must be in \[0, 1\]"):
        postprocessor._final_detection_batches(detections)


@pytest.mark.parametrize("coordinates", [(1.0, 0.0, 1.0, 2.0), (0.0, 2.0, 1.0, 2.0)])
def test_already_decoded_detections_reject_nonpositive_box_area(
    coordinates: tuple[float, float, float, float],
) -> None:
    """Reject degenerate decoded XYXY boxes before they reach evaluation."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    detections = torch.tensor([[[*coordinates, 0.9, 1.0]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="positive xyxy area"):
        postprocessor._final_detection_batches(detections)


def test_already_decoded_detections_ignore_degenerate_padding_rows() -> None:
    """Validate geometry only after low-confidence fixed-size padding is removed."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    detections = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.1, 1.0]]], dtype=torch.float32)

    assert postprocessor._final_detection_batches(detections)[0].shape == (0, 6)


def test_already_decoded_obb_uses_width_height_geometry() -> None:
    """Do not interpret OBB center/size fields as XYXY corners."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    postprocessor.task = "obb"
    detections = torch.tensor(
        [[[500.0, 20.0, 100.0, 10.0, 0.9, 1.0]]], dtype=torch.float32
    )

    assert postprocessor._final_detection_batches(detections)[0].shape == (1, 6)


@pytest.mark.parametrize("prototype_first", [False, True])
def test_decoded_segmentation_propagates_invalid_mask_prototypes(
    prototype_first: bool,
) -> None:
    """Do not discard malformed required prototype tensors as unrelated outputs."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    postprocessor.n_extra = 2
    postprocessor.task = "instance_segmentation"
    detections = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0, 0.9, 1.0, 0.0, 0.0]]], dtype=torch.float32
    )
    invalid_proto = torch.full((1, 2, 2, 2), float("nan"))
    outputs = (
        [invalid_proto, detections] if prototype_first else [detections, invalid_proto]
    )

    with pytest.raises(
        ValueError, match="Mask prototype tensor must contain only finite"
    ):
        postprocessor.extract_final_outputs(outputs)


def test_decoded_segmentation_requires_mask_prototypes() -> None:
    """Reject decoded segmentation artifacts that cannot produce instance masks."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    postprocessor.conf_thres = 0.25
    postprocessor.nc = 2
    postprocessor.n_extra = 2
    postprocessor.task = "instance_segmentation"
    detections = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0, 0.9, 1.0, 0.0, 0.0]]], dtype=torch.float32
    )

    with pytest.raises(ValueError, match="require a mask prototype tensor"):
        postprocessor.extract_final_outputs([detections])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_detection_postprocessor_rejects_nonfinite_raw_heads(
    invalid_value: float,
) -> None:
    """Reject malformed raw detector heads before they reach decoding and NMS."""

    postprocessor = cast(
        Any, YOLOAnchorlessDetectionPost.__new__(YOLOAnchorlessDetectionPost)
    )
    postprocessor.device = torch.device("cpu")
    raw_head = torch.zeros((1, 6, 2, 2), dtype=torch.float32)
    raw_head[0, 0, 0, 0] = invalid_value

    with pytest.raises(
        ValueError, match="Detection output tensors must contain only finite"
    ):
        postprocessor.check_input(raw_head)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_detection_postprocessor_rejects_nonfinite_mask_prototypes(
    invalid_value: float,
) -> None:
    """Reject invalid mask prototypes from already-decoded segmentation artifacts."""

    postprocessor = cast(Any, YOLOAnchorlessSegPost.__new__(YOLOAnchorlessSegPost))
    postprocessor.device = torch.device("cpu")
    postprocessor.n_extra = 2
    prototype = torch.zeros((1, 2, 2, 2))
    prototype[0, 0, 0, 0] = invalid_value

    with pytest.raises(
        ValueError, match="Mask prototype tensor must contain only finite"
    ):
        postprocessor._normalize_proto_batch(prototype)


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


def test_anchor_postprocessor_rearranges_yolov7_onnx_heads() -> None:
    """Accept WongKinYiu YOLOv7's explicit-anchor ONNX head layout."""

    postprocessor = YOLOAnchorDetectionPost.__new__(YOLOAnchorDetectionPost)
    postprocessor.nl = 3
    postprocessor.na = 3
    postprocessor.no = 85
    raw_outputs = [
        torch.arange(3 * height * width * 85, dtype=torch.float32).reshape(
            1, 3, height, width, 85
        )
        for height, width in ((80, 80), (40, 40), (20, 20))
    ]

    output = postprocessor.rearrange(raw_outputs)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 25200, 85)
    assert torch.equal(output[0, 0], raw_outputs[0][0, 0, 0, 0])
    assert torch.equal(output[0, 19200], raw_outputs[1][0, 0, 0, 0])
    assert torch.equal(output[0, 24000], raw_outputs[2][0, 0, 0, 0])


def test_anchor_postprocessor_ignores_decode_enabled_auxiliary_output() -> None:
    """Use only raw heads when an MXQ also returns decoded YOLO predictions."""

    postprocessor = YOLOAnchorDetectionPost.__new__(YOLOAnchorDetectionPost)
    postprocessor.nl = 3
    postprocessor.na = 3
    postprocessor.no = 85
    raw_outputs = [
        torch.arange(3 * height * width * 85, dtype=torch.float32).reshape(
            1, 3, height, width, 85
        )
        for height, width in ((80, 80), (40, 40), (20, 20))
    ]
    decoded_output = torch.zeros((1, 25200, 85), dtype=torch.float32)

    output = postprocessor.rearrange([decoded_output, *raw_outputs])

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 25200, 85)
    assert torch.equal(output[0, 0], raw_outputs[0][0, 0, 0, 0])


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

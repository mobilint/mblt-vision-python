"""Tests for YOLO26 ADE20K semantic segmentation support."""

from __future__ import annotations


import pytest
import torch
from mblt_vision.utils.postprocess import SemanticSegPost


def test_semantic_postprocess_supports_logits_and_baked_maps() -> None:
    """Convert logits or baked maps to input-sized integer class maps."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
    )
    logits = torch.zeros((1, 150, 2, 2))
    logits[:, 7] = 1.0
    result = post(logits)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 4, 4)
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.full((1, 4, 4), 7))

    baked = post(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    assert isinstance(baked, torch.Tensor)
    assert baked.shape == (1, 4, 4)
    assert set(baked.unique().tolist()) == {1, 2, 3, 4}

    with pytest.raises(ValueError, match=r"expects \[B, 150, H, W\]"):
        post(torch.zeros((1, 19, 4, 4)))
    with pytest.raises(ValueError, match=r"must be in \[0, 149\]"):
        post(torch.full((1, 4, 4), 150))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_semantic_postprocess_rejects_non_finite_baked_class_ids(
    invalid_value: float,
) -> None:
    """Reject non-finite baked class IDs before converting their dtype."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "semantic_segmentation", "dataset": "cityscapes"},
    )
    class_map = torch.zeros((1, 2, 2), dtype=torch.float32)
    class_map[0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="must be finite"):
        post(class_map)


def test_semantic_postprocess_rejects_fractional_baked_class_ids() -> None:
    """Reject fractional baked class IDs instead of silently truncating them."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "semantic_segmentation", "dataset": "cityscapes"},
    )

    with pytest.raises(ValueError, match="must be integer-valued"):
        post(torch.tensor([[[0.0, 1.9], [2.0, 18.0]]]))


def test_semantic_postprocess_reports_semantic_letterbox_errors() -> None:
    """Use semantic task labels when validating preprocessing configuration."""

    with pytest.raises(
        ValueError,
        match=r"Semantic segmentation requires a LetterBox configuration in pre_cfg",
    ):
        SemanticSegPost({}, {"task": "semantic_segmentation", "dataset": "ade20k"})


def test_semantic_postprocess_supports_mxq_hwc_and_batched_nhwc_logits() -> None:
    """Convert Cityscapes MXQ channel-last logits before choosing class maps."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 8]}},
        {"task": "semantic_segmentation", "dataset": "cityscapes"},
    )
    hwc_logits = torch.zeros((4, 8, 19))
    hwc_logits[..., 6] = 2.0
    hwc_result = post(hwc_logits)
    assert isinstance(hwc_result, torch.Tensor)
    assert hwc_result.shape == (1, 4, 8)
    assert torch.equal(hwc_result, torch.full((1, 4, 8), 6))

    nhwc_logits = torch.zeros((2, 2, 4, 19))
    nhwc_logits[0, ..., 3] = 2.0
    nhwc_logits[1, ..., 9] = 2.0
    nhwc_result = post(nhwc_logits)
    assert isinstance(nhwc_result, torch.Tensor)
    assert nhwc_result.shape == (2, 4, 8)
    assert torch.equal(nhwc_result[0], torch.full((4, 8), 3))
    assert torch.equal(nhwc_result[1], torch.full((4, 8), 9))

    low_resolution_hwc_logits = torch.zeros((2, 4, 19))
    low_resolution_hwc_logits[..., 12] = 2.0
    low_resolution_result = post(low_resolution_hwc_logits)
    assert isinstance(low_resolution_result, torch.Tensor)
    assert low_resolution_result.shape == (1, 4, 8)
    assert torch.equal(low_resolution_result, torch.full((1, 4, 8), 12))

    with pytest.raises(ValueError, match=r"expects \[B, 19, H, W\]"):
        post(torch.zeros((1, 4, 8, 18)))
    with pytest.raises(ValueError, match=r"expects \[H, W, 19\] MXQ logits"):
        post(torch.zeros((4, 8, 18)))

    baked_width_matches_nc = torch.arange(19).reshape(1, 1, 19)
    baked_result = post(baked_width_matches_nc)
    assert isinstance(baked_result, torch.Tensor)
    assert baked_result.shape == (1, 4, 8)
    assert torch.equal(baked_result[0, 0], torch.tensor([0, 2, 4, 7, 9, 11, 14, 16]))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_semantic_postprocess_rejects_nonfinite_logits(invalid_value: float) -> None:
    """Reject invalid local logits before they can be converted by argmax."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "semantic_segmentation", "dataset": "cityscapes"},
    )
    logits = torch.zeros((1, 19, 2, 2))
    logits[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="logits must contain only finite values"):
        post(logits)


def test_semantic_postprocess_restores_letterbox_padding() -> None:
    """Crop padding before nearest-restoring a semantic map."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [8, 8]}},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
    )
    output = torch.zeros((1, 8, 8))
    output[:, 2:6, :] = 5
    restored = post(output, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    assert torch.equal(restored, torch.full((2, 4), 5))


def test_semantic_logits_restore_before_argmax_and_support_batches() -> None:
    """Bilinearly restore logits before choosing classes for each original shape."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 8]}},
        {"task": "semantic_segmentation", "dataset": "cityscapes"},
    )
    logits = torch.zeros((2, 19, 2, 4))
    logits[:, 0] = 1.0
    logits[:, 1, :, 2:] = 2.0
    restored = post(
        logits,
        img0_shape=[(2, 8), (1, 4)],
        ratio_pad=[((1.0, 1.0), (0.0, 1.0)), ((2.0, 2.0), (0.0, 1.0))],
    )

    assert isinstance(restored, list)
    assert [tuple(item.shape) for item in restored] == [(2, 8), (1, 4)]
    assert set(restored[0].unique().tolist()) == {0, 1}

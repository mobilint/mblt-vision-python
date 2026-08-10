"""Unit tests for YOLO26 depth-estimation support."""

from __future__ import annotations


import pytest
import torch
from mblt_vision.utils.postprocess import DepthPost

from mblt_vision.wrapper import resolve_model_config


@pytest.mark.parametrize(
    ("model_name", "task"),
    [
        ("yolo26s", "object_detection"),
        ("yolo26s-seg", "instance_segmentation"),
        ("yolo26s-pose", "pose_estimation"),
        ("yolo26s-obb", "obb"),
    ],
)
def test_other_yolo_validation_tasks_keep_letterbox(model_name: str, task: str) -> None:
    """Keep aspect-preserving letterbox preprocessing for non-depth YOLO tasks."""

    config = resolve_model_config(model_name)
    assert "LetterBox" in config["pre_cfg"]
    assert "Resize" not in config["pre_cfg"]
    assert config["post_cfg"]["task"] == task


def test_depth_post_restores_letterbox_padding() -> None:
    """Crop letterbox padding before bilinearly restoring an original image shape."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    output = torch.zeros((1, 1, 8, 8))
    output[:, :, 2:6, :] = 2.0
    restored = post(output, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    assert torch.allclose(restored, torch.full((2, 4), 2.0))
    with pytest.raises(ValueError, match=r"expects \[B, 1, H, W\] or \[B, H, W, 1\]"):
        post(torch.zeros((1, 2, 8, 8)))


def test_depth_post_normalizes_quarter_resolution_mxq_before_restoring() -> None:
    """Upsample the MXQ depth layout before undoing letterbox padding."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    mxq_depth = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    expected = torch.nn.functional.interpolate(
        mxq_depth[:, None], scale_factor=4.0, mode="bilinear", align_corners=False
    )[:, 0]

    normalized = post(mxq_depth)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, expected)

    restored = post(mxq_depth, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    expected_restored = torch.nn.functional.interpolate(
        expected[0, 2:6][None, None], size=(2, 4), mode="bilinear", align_corners=False
    )[0, 0]
    assert torch.equal(restored, expected_restored)


def test_depth_post_keeps_full_resolution_onnx_output_and_rejects_other_scales() -> (
    None
):
    """Keep ONNX depth unchanged while rejecting unsupported dense output scales."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    full_resolution = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    normalized = post(full_resolution)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, full_resolution[:, 0])

    with pytest.raises(ValueError, match="spatial shape must be"):
        post(torch.zeros((1, 3, 3)))


def test_depth_post_normalizes_full_resolution_channel_last_mxq_output() -> None:
    """Normalize batched and single-image baked MXQ resize outputs without resizing again."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    full_resolution = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8, 1)
    normalized = post(full_resolution)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, full_resolution[..., 0])

    single_image = full_resolution[0]
    normalized_single_image = post(single_image)
    assert isinstance(normalized_single_image, torch.Tensor)
    assert torch.equal(normalized_single_image, full_resolution[..., 0])

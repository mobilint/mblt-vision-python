"""Tests for ONNX inference across representative YOLO postprocess families."""

from __future__ import annotations

from pathlib import Path

import pytest

from mblt_vision import MBLT_Engine

pytestmark = pytest.mark.requires_network


@pytest.mark.parametrize(
    ("model_cls", "task"),
    [
        ("yolov5m", "object_detection"),
        ("yolov5m-seg", "instance_segmentation"),
        ("yolo11m", "object_detection"),
        ("yolo11m-seg", "instance_segmentation"),
        ("yolo11m-pose", "pose_estimation"),
        ("yolov10m", "object_detection"),
        ("yolo26m", "object_detection"),
        ("yolo26m-seg", "instance_segmentation"),
        ("yolo26m-pose", "pose_estimation"),
        ("yolov8m-obb", "obb"),
        ("yolo11m-obb", "obb"),
        ("yolo26m-obb", "obb"),
    ],
)
def test_onnx_yolo_inference(
    model_cls: str, task: str, tmp_path: Path, synthetic_image_path: Path
) -> None:
    """Run ONNX inference for representative YOLO postprocess families."""

    save_path = tmp_path / f"{model_cls}_visualization.jpg"
    model = MBLT_Engine(model_cls=model_cls, framework="onnx")

    try:
        input_img = model.preprocess(str(synthetic_image_path))
        output = model(input_img)
        result = model.postprocess(output)

        assert result is not None
        assert result.task == task
        assert result.output is not None
        result.plot(str(synthetic_image_path), save_path=str(save_path))
        assert save_path.is_file()
    finally:
        model.dispose()

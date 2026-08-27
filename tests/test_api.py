"""Tests for the standalone public Vision API."""

from __future__ import annotations

import pytest

import mblt_vision
from mblt_vision import list_models, list_tasks
from mblt_vision._tasks import normalize_vision_task


def test_public_discovery_exposes_all_supported_tasks() -> None:
    """Expose every standalone Vision task through the public API."""

    assert list_tasks() == [
        "image_classification",
        "depth_estimation",
        "object_detection",
        "instance_segmentation",
        "semantic_segmentation",
        "obb",
        "pose_estimation",
        "face_detection",
        "mask_generation",
    ]
    assert list_models("obb")["obb"]


def test_model_exports_are_discoverable_from_task_and_top_level_namespaces() -> None:
    """Keep task exports synchronized with lazy top-level compatibility exports."""

    from mblt_vision.image_classification import ResNet50
    from mblt_vision.object_detection import YOLO11m

    assert mblt_vision.ResNet50 is ResNet50
    assert mblt_vision.YOLO11m is YOLO11m
    assert "ResNet50" in list_models("image_classification")["image_classification"]
    assert "YOLO11m" in list_models("object_detection")["object_detection"]


@pytest.mark.parametrize("task", [None, 1, object()])
def test_task_normalization_rejects_non_strings(task: object) -> None:
    """Reject invalid task inputs with a stable, actionable error."""

    with pytest.raises(TypeError, match="must be a string"):
        normalize_vision_task(task)  # type: ignore[arg-type]

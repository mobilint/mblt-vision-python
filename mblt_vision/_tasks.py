"""Canonical task names used by the standalone Vision processing API."""

from __future__ import annotations

from collections.abc import Iterable

VISION_TASKS: tuple[str, ...] = (
    "image_classification",
    "depth_estimation",
    "object_detection",
    "instance_segmentation",
    "semantic_segmentation",
    "obb",
    "pose_estimation",
    "face_detection",
    "mask_generation",
)


def normalize_vision_task(task: str, *, supported: Iterable[str] | None = None) -> str:
    """Normalize a Vision task name and validate it against supported tasks."""

    if not isinstance(task, str):
        raise TypeError(f"Vision task must be a string, got {type(task).__name__}.")
    normalized = task.lower()
    supported_tasks = tuple(VISION_TASKS if supported is None else supported)
    if normalized not in supported_tasks:
        raise ValueError(
            f"Unsupported Vision task {task!r}; expected one of {sorted(supported_tasks)}."
        )
    return normalized

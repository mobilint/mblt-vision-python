"""Focused COCO evaluator regression tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

eval_coco_module = importlib.import_module("mblt_vision.utils.evaluation.eval_coco")


def test_pose_evaluation_uses_all_keypoints_annotation_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not omit COCO images that lack visible-person keypoint annotations."""

    constructed_kwargs: dict[str, Any] = {}
    evaluated_image_ids: list[int] | None = None

    class _Dataset:
        def __init__(self, root: str, annotation_path: str, **kwargs: Any) -> None:
            constructed_kwargs.update(
                root=root, annotation_path=annotation_path, **kwargs
            )
            self.ids = [1, 2]
            self.coco = object()

        def __len__(self) -> int:
            return len(self.ids)

    class _Model:
        post_cfg = {"task": "pose_estimation"}

        def set_postprocess_thresholds(
            self, *, conf_thres: float | None, iou_thres: float | None
        ) -> None:
            del conf_thres, iou_thres

        def preprocess_with_metadata(self, image: object) -> object:
            return image

    def _evaluate(
        coco_gt: object,
        coco_results: list[dict[str, Any]],
        task: str,
        img_ids: list[int] | None = None,
    ) -> SimpleNamespace:
        nonlocal evaluated_image_ids
        del coco_gt, coco_results
        assert task == "pose_estimation"
        evaluated_image_ids = img_ids
        return SimpleNamespace(stats=[SimpleNamespace(item=lambda: 0.1)] * 2)

    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", _Dataset)
    monkeypatch.setattr(eval_coco_module, "get_coco_loader", lambda *_: [])
    monkeypatch.setattr(eval_coco_module, "evaluate_predictions_on_coco", _evaluate)

    result = eval_coco_module.eval_coco_metrics(_Model(), "/dataset", batch_size=2)

    assert "min_keypoints" not in constructed_kwargs
    assert constructed_kwargs["annotation_path"].endswith(
        "person_keypoints_val2017.json"
    )
    assert evaluated_image_ids == [1, 2]
    assert result.map5095 == result.map50 == 0.1

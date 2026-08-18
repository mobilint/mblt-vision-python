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
            self.coco = SimpleNamespace(
                cats={1: {}},
                anns={},
                imgs={1: {"height": 1, "width": 1}, 2: {"height": 1, "width": 1}},
            )

        def __len__(self) -> int:
            return len(self.ids)

    class _Model:
        post_cfg = {"task": "pose_estimation", "dataset": "coco"}

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


def test_coco_evaluation_rejects_non_coco_model_taxonomy() -> None:
    """Do not evaluate another taxonomy using the hard-coded COCO ID mapping."""

    model = SimpleNamespace(post_cfg={"task": "object_detection", "dataset": "dotav1"})

    with pytest.raises(ValueError, match="post_cfg.dataset to be 'coco'"):
        eval_coco_module.eval_coco_metrics(model, "/dataset", batch_size=1)


def test_coco_evaluation_rejects_noncanonical_artifact_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not score direct COCO artifacts with an incompatible category taxonomy."""

    dataset = SimpleNamespace(coco=SimpleNamespace(cats={999: {}}, anns={}))
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match=r"unsupported category IDs: \[999\]"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(post_cfg={"task": "object_detection", "dataset": "coco"}),
            "/dataset",
            batch_size=1,
        )


def test_coco_evaluation_rejects_undeclared_annotation_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not let malformed annotation categories be omitted from direct AP."""

    dataset = SimpleNamespace(
        coco=SimpleNamespace(
            cats={1: {}},
            anns={7: {"category_id": 999}},
            imgs={1: {"height": 1, "width": 1}},
        )
    )
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match="invalid task-specific annotations"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(post_cfg={"task": "object_detection", "dataset": "coco"}),
            "/dataset",
            batch_size=1,
        )


def test_coco_evaluation_rejects_invalid_task_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use readiness-equivalent checks before direct COCO evaluation starts."""

    dataset = SimpleNamespace(
        coco=SimpleNamespace(
            cats={1: {}},
            imgs={1: {"height": 2, "width": 2}},
            anns={
                7: {
                    "id": 7,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": -1,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, 1, 0, 1, 1]],
                }
            },
        )
    )
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match="invalid task-specific annotations"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(
                post_cfg={"task": "instance_segmentation", "dataset": "coco"}
            ),
            "/dataset",
            batch_size=1,
        )


def test_coco_evaluation_rejects_polygons_outside_image_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not let a nonempty off-image polygon become an empty COCO mask."""

    dataset = SimpleNamespace(
        coco=SimpleNamespace(
            cats={1: {}},
            imgs={1: {"height": 10, "width": 10}},
            anns={
                7: {
                    "id": 7,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                    "segmentation": [[11, 0, 12, 0, 12, 1, 11, 1]],
                }
            },
        )
    )
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match="invalid task-specific annotations"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(
                post_cfg={"task": "instance_segmentation", "dataset": "coco"}
            ),
            "/dataset",
            batch_size=1,
        )


def test_coco_evaluation_rejects_polygons_without_rasterized_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuous subpixel overlap must not admit an empty raster mask."""

    dataset = SimpleNamespace(
        coco=SimpleNamespace(
            cats={1: {}},
            imgs={1: {"height": 10, "width": 10}},
            anns={
                7: {
                    "id": 7,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [9, 9, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                    "segmentation": [[9.9, 9.9, 9.99, 9.9, 9.99, 9.99]],
                }
            },
        )
    )
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match="invalid task-specific annotations"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(
                post_cfg={"task": "instance_segmentation", "dataset": "coco"}
            ),
            "/dataset",
            batch_size=1,
        )


def test_coco_evaluation_rejects_visible_keypoints_outside_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible pose coordinates must refer to pixels in the source image."""

    keypoints = [0.0, 0.0, 0.0] * 17
    keypoints[:3] = [11.0, 1.0, 2.0]
    dataset = SimpleNamespace(
        coco=SimpleNamespace(
            cats={1: {}},
            imgs={1: {"height": 10, "width": 10}},
            anns={
                7: {
                    "id": 7,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": 1,
                }
            },
        )
    )
    monkeypatch.setattr(eval_coco_module, "CustomCOCODataset", lambda *_: dataset)

    with pytest.raises(ValueError, match="invalid task-specific annotations"):
        eval_coco_module.eval_coco_metrics(
            SimpleNamespace(post_cfg={"task": "pose_estimation", "dataset": "coco"}),
            "/dataset",
            batch_size=1,
        )


def test_coco_result_formatter_rejects_truncated_postprocess_batch() -> None:
    """Require one decoded result for every submitted COCO image."""

    postprocess = SimpleNamespace(
        nmsout2eval=lambda *_args, **_kwargs: ([[1]], [[[0, 0, 1, 1]]], [[0.9]])
    )

    with pytest.raises(ValueError, match="batch cardinality mismatch"):
        eval_coco_module.format_coco_results(
            "object_detection",
            SimpleNamespace(output=[]),
            (640, 640),
            [(640, 640), (640, 640)],
            [None, None],
            [0, 1],
            [101, 102],
            postprocess,
        )

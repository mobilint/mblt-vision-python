"""Tests for DOTAv1 difficult-region evaluation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from mblt_vision.utils.datasets import CustomDOTAv1
from mblt_vision.utils.evaluation.eval_dota import (
    _load_ground_truths,
    evaluate_dota_predictions,
    format_dota_results,
)

eval_dota_module = importlib.import_module("mblt_vision.utils.evaluation.eval_dota")


def test_eval_dota_rejects_truncated_postprocess_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Do not silently omit ground truths when a batched model output is short."""

    class _FakeProgress:
        def __init__(self, iterable, **kwargs) -> None:
            del kwargs
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, value: str) -> None:
            del value

        def close(self) -> None:
            return None

    class _FakeModel:
        post_cfg = {"task": "obb", "dataset": "dotav1"}
        preprocess_with_metadata = object()

        def set_postprocess_thresholds(self, **kwargs) -> None:
            del kwargs

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def postprocess(self, outputs: torch.Tensor) -> SimpleNamespace:
            del outputs
            return SimpleNamespace(output=[torch.zeros((0, 7), dtype=torch.float32)])

    class _FakeDataset:
        def __len__(self) -> int:
            return 2

    batch = (
        torch.zeros((2, 8, 8, 3), dtype=torch.float32),
        [(8, 8), (8, 8)],
        [None, None],
        ("first", "second"),
    )
    monkeypatch.setattr(eval_dota_module, "CustomDOTAv1", lambda _: _FakeDataset())
    monkeypatch.setattr(eval_dota_module, "get_dota_loader", lambda *args: [batch])
    monkeypatch.setattr(eval_dota_module, "_load_ground_truths", lambda *args: {})
    monkeypatch.setattr(eval_dota_module, "tqdm", _FakeProgress)

    with pytest.raises(
        ValueError,
        match=r"DOTAv1 evaluation batch length mismatch: model outputs=1, input batch=2",
    ):
        eval_dota_module.eval_dota(_FakeModel(), str(tmp_path), batch_size=2)


def test_eval_dota_rejects_wrong_model_taxonomy(tmp_path) -> None:
    """Do not score another OBB taxonomy against DOTAv1 ground truth."""

    with pytest.raises(ValueError, match="post_cfg.dataset to be 'dotav1'"):
        eval_dota_module.eval_dota(
            SimpleNamespace(post_cfg={"task": "obb", "dataset": "coco"}),
            str(tmp_path),
            batch_size=1,
        )


def test_dota_ground_truth_requires_annotation_for_every_image(tmp_path) -> None:
    """Do not silently manufacture empty ground truth for unlabeled DOTAv1 samples."""

    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(
        FileNotFoundError, match="annotation not found for image 'image'"
    ):
        _load_ground_truths(str(tmp_path), dataset)


def test_normalized_difficult_flag_loads_as_an_ignored_region(tmp_path) -> None:
    """Read organizer-produced difficult metadata before selecting evaluation targets."""

    label_dir = tmp_path / "labels" / "val"
    label_dir.mkdir(parents=True)
    (label_dir / "image.txt").write_text(
        "0 0 0 0.2 0 0.2 0.2 0 0.2 0\n0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6 1\n",
        encoding="utf-8",
    )
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    ground_truth = _load_ground_truths(str(tmp_path), dataset)["image"]

    assert ground_truth["cls"].tolist() == [0]
    assert ground_truth["ignore_cls"].tolist() == [0]


def test_normalized_truncated_annotation_raises_with_file_and_line(tmp_path) -> None:
    """Reject nonblank normalized annotations without a complete OBB polygon."""

    label_dir = tmp_path / "labels" / "val"
    label_dir.mkdir(parents=True)
    label_path = label_dir / "image.txt"
    label_path.write_text("\n0 0 0 0.2\n", encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"image\.txt:2: expected at least 9 fields, got 4",
    ):
        _load_ground_truths(str(tmp_path), dataset)


def test_original_truncated_annotation_raises_with_file_and_line(tmp_path) -> None:
    """Require a difficulty flag when loading original DOTAv1 labels directly."""

    original_label_dir = tmp_path / "labels" / "val_original"
    original_label_dir.mkdir(parents=True)
    label_path = original_label_dir / "image.txt"
    label_path.write_text(
        "imagesource:GoogleEarth\n0 0 20 0 20 20 0 20 plane\n", encoding="utf-8"
    )
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"image\.txt:2: expected at least 10 fields, got 9",
    ):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "0 0 0 1 0 2 0 3 0"),
        ("labels/val_original/image.txt", "0 0 1 0 2 0 3 0 plane 0"),
    ],
)
def test_dota_annotations_reject_degenerate_polygons(
    tmp_path, label_path: str, annotation: str
) -> None:
    """Reject line-like quadrilaterals before deriving rotated boxes."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="polygon must have positive area"):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "0 0 0 0.2 0 0.2 0.2 0 0"),
        ("labels/val_original/image.txt", "0 0 20 0 20 20 0 0 plane 0"),
    ],
)
def test_dota_annotations_reject_repeated_vertices(
    tmp_path, label_path: str, annotation: str
) -> None:
    """DOTAv1 quadrilaterals cannot silently degrade into triangles."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="four distinct vertices"):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "0 1.1 0 1.2 0 1.2 0.1 1.1 0.1"),
        ("labels/val_original/image.txt", "110 0 120 0 120 10 110 10 plane 0"),
    ],
)
def test_dota_annotations_reject_polygons_outside_images(
    tmp_path, label_path: str, annotation: str
) -> None:
    """Ground-truth polygons must retain foreground within their source images."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="must overlap its source image"):
        _load_ground_truths(str(tmp_path), dataset)


def test_dota_normalized_coordinates_are_scaled_even_when_partially_clipped(
    tmp_path,
) -> None:
    """The normalized-label directory always uses normalized image coordinates."""

    label_path = tmp_path / "labels" / "val" / "image.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.9 0 1.6 0 1.6 0.1 0.9 0.1", encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    ground_truth = _load_ground_truths(str(tmp_path), dataset)["image"]

    assert ground_truth["polygons"][0, :, 0].tolist() == [90.0, 160.0, 160.0, 90.0]


def test_dota_ground_truth_rejects_orphan_label_files(tmp_path) -> None:
    """Direct evaluation must not ignore labels without a selected image."""

    label_path = tmp_path / "labels" / "val" / "orphan.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0 0 0.2 0 0.2 0.2 0 0.2", encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(ids=[], image_paths=[], _load_image=lambda _: None),
    )

    with pytest.raises(ValueError, match="no corresponding validation image"):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "-1 0 0 0.2 0 0.2 0.2 0 0.2"),
        ("labels/val_original/image.txt", "0 0 20 0 20 20 0 20 -1 0"),
    ],
)
def test_dota_annotations_reject_negative_class_indices(
    tmp_path, label_path: str, annotation: str
) -> None:
    """Reject invalid negative classes in normalized and original labels."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported DOTAv1 class index -1"):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "0 nan 0 0.2 0 0.2 0.2 0 0.2"),
        ("labels/val_original/image.txt", "inf 0 20 0 20 20 0 20 plane 0"),
    ],
)
def test_dota_annotations_reject_nonfinite_coordinates(
    tmp_path, label_path: str, annotation: str
) -> None:
    """Reject poisoned normalized and original DOTA polygon coordinates."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match=r"coordinates must be finite.*image\.txt:1"):
        _load_ground_truths(str(tmp_path), dataset)


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "0 0 0 0.2 0 0.2 0.2 0 0.2 3"),
        ("labels/val_original/image.txt", "0 0 20 0 20 20 0 20 plane invalid"),
    ],
)
def test_dota_annotations_reject_unknown_difficulty_flags(
    tmp_path, label_path: str, annotation: str
) -> None:
    """Keep invalid difficult-region metadata from changing positive counts."""

    path = tmp_path / label_path
    path.parent.mkdir(parents=True)
    path.write_text(annotation, encoding="utf-8")
    dataset = cast(
        CustomDOTAv1,
        SimpleNamespace(
            ids=["image"],
            image_paths=["unused"],
            _load_image=lambda _: np.zeros((100, 100, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported DOTAv1 difficulty flag"):
        _load_ground_truths(str(tmp_path), dataset)


def test_difficult_regions_do_not_count_as_positive_or_false_positive() -> None:
    """Ignore a detection on a difficult region while retaining positive matching."""

    ground_truths = {
        "image": {
            "cls": torch.tensor([0]),
            "bboxes": torch.tensor([[10.0, 10.0, 4.0, 4.0, 0.0]]),
            "ignore_cls": torch.tensor([0]),
            "ignore_bboxes": torch.tensor([[30.0, 30.0, 4.0, 4.0, 0.0]]),
        }
    }
    predictions = [
        {
            "image_id": "image",
            "category_id": 0,
            "score": 0.99,
            "rbox": [30.0, 30.0, 4.0, 4.0, 0.0],
        },
        {
            "image_id": "image",
            "category_id": 0,
            "score": 0.90,
            "rbox": [10.0, 10.0, 4.0, 4.0, 0.0],
        },
    ]

    result = evaluate_dota_predictions(ground_truths, predictions)
    baseline = evaluate_dota_predictions(ground_truths, predictions[1:])

    assert result == baseline


def test_dota_ap_interpolation_uses_terminal_recall_sentinel() -> None:
    """Preserve the reference AP curve after the final observed recall point."""

    ap, _, recall_curve = eval_dota_module._compute_ap(np.array([0.5]), np.array([1.0]))

    assert recall_curve.tolist() == [0.0, 0.5, 1.0]
    assert ap == pytest.approx(0.75)


def test_dota_metrics_require_non_ignored_ground_truth() -> None:
    """Empty or difficult-only labels must not yield a normal AP result."""

    with pytest.raises(ValueError, match="at least one non-ignored target"):
        eval_dota_module._evaluate_stats(eval_dota_module._empty_stats())


def test_dota_export_rejects_truncated_converted_batches() -> None:
    """Do not silently omit Task1 files when converted output batches are short."""

    postprocess = SimpleNamespace(
        nmsout2eval=lambda *_args, **_kwargs: ([[]], [[]], [[]], [[]])
    )

    with pytest.raises(ValueError, match="export batch length mismatch"):
        format_dota_results(
            SimpleNamespace(output=[]),
            (640, 640),
            [(640, 640), (640, 640)],
            [None, None],
            ("first", "second"),
            postprocess,
        )


def test_dota_export_rejects_mismatched_detection_fields() -> None:
    """Require a polygon, score, and rotated box for every exported label."""

    postprocess = SimpleNamespace(
        nmsout2eval=lambda *_args, **_kwargs: ([["plane"]], [[]], [[0.9]], [[]])
    )

    with pytest.raises(ValueError, match="export detection length mismatch"):
        format_dota_results(
            SimpleNamespace(output=[]),
            (640, 640),
            [(640, 640)],
            [None],
            ("first",),
            postprocess,
        )

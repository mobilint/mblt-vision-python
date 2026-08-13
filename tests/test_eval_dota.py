"""Tests for DOTAv1 difficult-region evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from mblt_vision.utils.datasets import CustomDOTAv1
from mblt_vision.utils.evaluation.eval_dota import (
    _load_ground_truths,
    evaluate_dota_predictions,
)


def test_normalized_difficult_flag_loads_as_an_ignored_region(tmp_path) -> None:
    """Read organizer-produced difficult metadata before selecting evaluation targets."""

    label_dir = tmp_path / "labels" / "val"
    label_dir.mkdir(parents=True)
    (label_dir / "image.txt").write_text(
        "0 0 0 0.2 0 0.2 0.2 0 0.2 0\n" "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6 1\n",
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


@pytest.mark.parametrize(
    ("label_path", "annotation"),
    [
        ("labels/val/image.txt", "-1 0 0 0.2 0 0.2 0.2 0 0.2"),
        ("labels/val_original/image.txt", "0 0 20 0 20 20 0 20 -1"),
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

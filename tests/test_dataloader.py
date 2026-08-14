"""Tests for vision dataset sample discovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mblt_vision.utils.datasets.dataloader import (
    CustomCOCODataset,
    CustomImageFolder,
    CustomWiderFaceDataset,
)


@pytest.mark.parametrize(
    ("dataset_class", "class_name", "expected_count", "expected_suffixes"),
    [
        (CustomImageFolder, "class-a", 1, {".jpg"}),
        (CustomWiderFaceDataset, "0--Parade", 2, {".jpg", ".png"}),
    ],
)
def test_class_based_datasets_ignore_non_image_files(
    tmp_path: Path,
    dataset_class: type[CustomImageFolder] | type[CustomWiderFaceDataset],
    class_name: str,
    expected_count: int,
    expected_suffixes: set[str],
) -> None:
    """Keep incidental files out of ImageNet and WiderFace sample lists."""

    class_dir = tmp_path / class_name
    nested_dir = class_dir / "nested"
    nested_dir.mkdir(parents=True)
    (class_dir / "image.JPG").write_bytes(b"image")
    (nested_dir / "image.png").write_bytes(b"image")
    (class_dir / ".DS_Store").write_bytes(b"metadata")
    (nested_dir / "labels.txt").write_text("metadata", encoding="utf-8")

    dataset = dataset_class(str(tmp_path))

    assert len(dataset) == expected_count
    assert {
        Path(sample[0]).suffix.lower() for sample in dataset.samples
    } == expected_suffixes


def test_coco_dataset_rejects_annotation_geometry_mismatching_image() -> None:
    """Do not evaluate COCO labels with geometry different from their decoded image."""

    dataset = SimpleNamespace(
        ids=[1],
        coco=SimpleNamespace(imgs={1: {"height": 4, "width": 5}}),
        _load_image=lambda _: np.zeros((3, 5, 3), dtype=np.uint8),
    )

    with pytest.raises(
        ValueError, match=r"image ID 1: annotation \(4, 5\), image \(3, 5\)"
    ):
        CustomCOCODataset.__getitem__(dataset, 0)

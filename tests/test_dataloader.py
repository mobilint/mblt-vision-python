"""Tests for vision dataset sample discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from mblt_vision.utils.datasets.dataloader import (
    CustomImageFolder,
    CustomWiderFaceDataset,
)


@pytest.mark.parametrize(
    ("dataset_class", "class_name"),
    [
        (CustomImageFolder, "class-a"),
        (CustomWiderFaceDataset, "0--Parade"),
    ],
)
def test_class_based_datasets_ignore_non_image_files(
    tmp_path: Path,
    dataset_class: type[CustomImageFolder] | type[CustomWiderFaceDataset],
    class_name: str,
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

    assert len(dataset) == 2
    assert {Path(sample[0]).suffix.lower() for sample in dataset.samples} == {
        ".jpg",
        ".png",
    }

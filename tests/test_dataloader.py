"""Tests for vision dataset sample discovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest
from PIL import Image

from mblt_vision.utils.datasets.dataloader import (
    CustomCOCODataset,
    CustomADE20K,
    CustomCityscapes,
    CustomDOTAv1,
    CustomImageFolder,
    CustomNYUDepth,
    CustomWiderFaceDataset,
)


@pytest.mark.parametrize(
    ("dataset_class", "class_name", "expected_count", "expected_suffixes"),
    [
        (CustomImageFolder, "class-a", 1, {".jpg"}),
        (CustomWiderFaceDataset, "0--Parade", 1, {".jpg"}),
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

    dataset = cast(
        CustomCOCODataset,
        SimpleNamespace(
            ids=[1],
            coco=SimpleNamespace(imgs={1: {"height": 4, "width": 5}}),
            _load_image=lambda _: np.zeros((3, 5, 3), dtype=np.uint8),
        ),
    )

    with pytest.raises(
        ValueError, match=r"image ID 1: annotation \(4, 5\), image \(3, 5\)"
    ):
        CustomCOCODataset.__getitem__(dataset, 0)


@pytest.mark.parametrize("file_name", ["/tmp/outside.jpg", "../outside.jpg"])
def test_coco_dataset_rejects_unsafe_annotation_file_names(file_name: str) -> None:
    """Do not let COCO metadata access images outside the configured root."""

    dataset = cast(
        CustomCOCODataset,
        SimpleNamespace(
            root="/dataset",
            coco=SimpleNamespace(
                loadImgs=lambda _: [{"file_name": file_name}],
            ),
        ),
    )

    with pytest.raises(ValueError, match="unsafe file_name"):
        CustomCOCODataset._load_image(dataset, 1)


def test_dota_dataset_rejects_duplicate_image_stems(tmp_path: Path) -> None:
    """Direct DOTAv1 loading must not silently collapse same-stem images."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"jpg")
    (image_dir / "sample.png").write_bytes(b"png")

    with pytest.raises(ValueError, match="duplicate filename stems"):
        CustomDOTAv1(str(tmp_path))


@pytest.mark.parametrize(
    ("dataset_class", "source_id"),
    [(CustomADE20K, 0), (CustomCityscapes, 255)],
)
def test_dense_semantic_datasets_reject_all_ignored_targets(
    tmp_path: Path,
    dataset_class: type[CustomADE20K] | type[CustomCityscapes],
    source_id: int,
) -> None:
    """Reject direct semantic evaluation samples that contain no valid labels."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "sample.png")
    Image.fromarray(np.full((2, 2), source_id, dtype=np.uint8)).save(
        annotation_dir / "sample.png"
    )
    dataset = dataset_class(str(tmp_path))

    with pytest.raises(ValueError, match="contains no evaluable class IDs"):
        dataset[0]


@pytest.mark.parametrize(
    ("dataset_class", "target_dir", "target_names"),
    [
        (CustomNYUDepth, "depth", ("sample.npy", "sample.NPY")),
        (CustomADE20K, "annotations", ("sample.png", "sample.PNG")),
        (CustomCityscapes, "annotations", ("sample.png", "sample.PNG")),
    ],
)
def test_dense_datasets_reject_duplicate_target_stems(
    tmp_path: Path,
    dataset_class: type[CustomNYUDepth] | type[CustomADE20K] | type[CustomCityscapes],
    target_dir: str,
    target_names: tuple[str, str],
) -> None:
    """Do not let case variants silently overwrite a dense target mapping."""

    (tmp_path / "images").mkdir()
    (tmp_path / target_dir).mkdir()
    (tmp_path / "images" / "sample.jpg").write_bytes(b"image")
    for target_name in target_names:
        (tmp_path / target_dir / target_name).write_bytes(b"target")

    with pytest.raises(ValueError, match="duplicate filename stems"):
        dataset_class(str(tmp_path))


def test_nyu_dataset_rejects_negative_depth_targets(tmp_path: Path) -> None:
    """Do not silently exclude corrupted negative depths during evaluation."""

    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    image_dir.mkdir()
    depth_dir.mkdir()
    assert cv2.imwrite(str(image_dir / "sample.jpg"), np.zeros((1, 1, 3), np.uint8))
    np.save(depth_dir / "sample.npy", np.array([[-1]], dtype=np.float32))

    with pytest.raises(ValueError, match="must not contain negative values"):
        CustomNYUDepth(str(tmp_path))[0]


def test_cityscapes_dataset_rejects_unknown_source_ids(tmp_path: Path) -> None:
    """Reject corrupted Cityscapes labels instead of remapping them to ignore."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    assert cv2.imwrite(
        str(image_dir / "sample.png"), np.zeros((2, 2, 3), dtype=np.uint8)
    )
    Image.fromarray(np.full((2, 2), 200, dtype=np.uint8)).save(
        annotation_dir / "sample.png"
    )

    with pytest.raises(ValueError, match=r"unsupported source IDs \[200\]"):
        CustomCityscapes(str(tmp_path))[0]


@pytest.mark.parametrize(
    "mask", [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint16)]
)
def test_ade20k_dataset_rejects_noncanonical_mask_encodings(
    tmp_path: Path, mask: np.ndarray
) -> None:
    """Reject color and high-bit-depth ADE20K masks before source-ID conversion."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    assert cv2.imwrite(
        str(image_dir / "sample.png"), np.zeros((2, 2, 3), dtype=np.uint8)
    )
    Image.fromarray(mask).save(annotation_dir / "sample.png")

    with pytest.raises(ValueError, match="single-channel 8-bit PNG masks"):
        CustomADE20K(str(tmp_path))[0]

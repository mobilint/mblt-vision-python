"""Tests for organized vision dataset identity and completeness checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from mblt_vision.utils.datasets import readiness


def _write_file(path: Path) -> None:
    """Create a placeholder dataset file and its parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def _write_widerface_metadata(path: Path, event_images: dict[str, list[str]]) -> None:
    """Write the WiderFace event and image-name cell arrays used by readiness."""

    event_list = np.empty((len(event_images), 1), dtype=object)
    file_list = np.empty((len(event_images), 1), dtype=object)
    for index, (event_name, image_stems) in enumerate(event_images.items()):
        event_list[index, 0] = event_name
        file_list[index, 0] = np.array([[stem] for stem in image_stems], dtype=object)
    savemat(path, {"event_list": event_list, "file_list": file_list})


def test_imagenet_readiness_requires_complete_official_class_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject incomplete or non-ImageNet classification directory trees."""

    monkeypatch.setattr(readiness, "IMAGENET_CLASS_COUNT", 2)
    monkeypatch.setattr(readiness, "IMAGENET_IMAGES_PER_CLASS", 2)
    monkeypatch.setattr(readiness, "IMAGENET_SYNSETS", {"n00000000", "n00000001"})
    for class_index in range(2):
        for image_index in range(2):
            if (class_index, image_index) == (1, 1):
                continue
            _write_file(
                tmp_path
                / f"n{class_index:08d}"
                / f"ILSVRC2012_val_{class_index * 2 + image_index + 1:08d}.JPEG"
            )

    assert not readiness.dataset_ready(tmp_path, "image_classification", "imagenet")

    _write_file(tmp_path / "n00000001" / "ILSVRC2012_val_00000004.JPEG")

    assert readiness.dataset_ready(tmp_path, "image_classification", "imagenet")
    assert not readiness.dataset_ready(tmp_path, "image_classification", "coco")


@pytest.mark.parametrize(
    ("task", "annotation_name"),
    [
        ("object_detection", "instances_val2017.json"),
        ("instance_segmentation", "instances_val2017.json"),
        ("pose_estimation", "person_keypoints_val2017.json"),
    ],
)
def test_coco_readiness_matches_images_to_task_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    annotation_name: str,
) -> None:
    """Require every official COCO image in the task-specific annotation file."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, annotation_name, 2)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, annotation_name, 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    image_names = ["000000000001.jpg", "000000000002.jpg"]
    for image_name in image_names:
        _write_file(tmp_path / "val2017" / image_name)
    (tmp_path / annotation_name).write_text(
        json.dumps({"images": [{"id": 1, "file_name": image_names[0]}]}),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, task, "coco")

    (tmp_path / annotation_name).write_text(
        json.dumps(
            {
                "images": [
                    {"id": index, "file_name": image_name}
                    for index, image_name in enumerate(image_names, start=1)
                ],
                "categories": [{"id": 1}],
                "annotations": [
                    {
                        "id": index,
                        "image_id": index,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "area": 1,
                        "iscrowd": 0,
                        **(
                            {"segmentation": [[0, 0, 1, 0, 1, 1]]}
                            if task == "instance_segmentation"
                            else {}
                        ),
                        **(
                            {"keypoints": [0, 0, 0] * 17, "num_keypoints": 0}
                            if task == "pose_estimation"
                            else {}
                        ),
                    }
                    for index in range(1, 3)
                ],
            }
        ),
        encoding="utf-8",
    )

    assert readiness.dataset_ready(tmp_path, task, "coco")
    assert not readiness.dataset_ready(tmp_path, task, "imagenet")


@pytest.mark.parametrize(
    ("task", "invalid_field"),
    [
        ("instance_segmentation", "segmentation"),
        ("pose_estimation", "keypoints"),
        ("pose_estimation", "num_keypoints"),
    ],
)
def test_coco_readiness_rejects_missing_task_specific_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    invalid_field: str,
) -> None:
    """Reject COCO metadata missing the payload required by its evaluation task."""

    annotation_name = (
        "person_keypoints_val2017.json"
        if task == "pose_estimation"
        else "instances_val2017.json"
    )
    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, annotation_name, 1)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, annotation_name, 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    _write_file(tmp_path / "val2017" / "000000000001.jpg")
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0, 0, 1, 1],
        "area": 1,
        "iscrowd": 0,
        "segmentation": [[0, 0, 1, 0, 1, 1]],
        "keypoints": [0, 0, 0] * 17,
        "num_keypoints": 0,
    }
    del annotation[invalid_field]
    (tmp_path / annotation_name).write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg"}],
                "categories": [{"id": 1}],
                "annotations": [annotation],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, task, "coco")


def test_coco_readiness_rejects_noncanonical_category_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Require the canonical COCO category-ID set, not merely its count."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    image_name = "000000000001.jpg"
    _write_file(tmp_path / "val2017" / image_name)
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image_name}],
                "categories": [{"id": 2}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2}],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, "object_detection", "coco")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("segmentation", [[0, 0, 1, 1, 2, 2]]),
        ("area", 0),
        ("iscrowd", 2),
    ],
)
def test_coco_readiness_rejects_invalid_evaluation_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Reject metadata that can alter COCO matching or area-range evaluation."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    _write_file(tmp_path / "val2017" / "000000000001.jpg")
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0, 0, 1, 1],
        "area": 1,
        "iscrowd": 0,
        "segmentation": [[0, 0, 1, 0, 1, 1]],
    }
    annotation[field] = value
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg"}],
                "categories": [{"id": 1}],
                "annotations": [annotation],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, "instance_segmentation", "coco")


@pytest.mark.parametrize("bbox", [[0, 0, 0, 1], [0, 0, 1, float("nan")]])
def test_coco_readiness_rejects_invalid_instance_box_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bbox: list[float]
) -> None:
    """Reject invalid instance boxes before a malformed cache reaches COCOeval."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    _write_file(tmp_path / "val2017" / "000000000001.jpg")
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg"}],
                "categories": [{"id": 1}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, "object_detection", "coco")


@pytest.mark.parametrize(
    ("counts", "expected_ready"),
    [([3, 1], True), ("31", True), ([4], False), ([3], False), ("4", False)],
)
def test_coco_readiness_decodes_and_validates_rle_segmentations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    counts: list[int] | str,
    expected_ready: bool,
) -> None:
    """Require nonempty, complete RLE masks with the referenced image geometry."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    _write_file(tmp_path / "val2017" / "000000000001.jpg")
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "000000000001.jpg",
                        "height": 2,
                        "width": 2,
                    }
                ],
                "categories": [{"id": 1}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "area": 1,
                        "iscrowd": 0,
                        "segmentation": {"size": [2, 2], "counts": counts},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        readiness.dataset_ready(tmp_path, "instance_segmentation", "coco")
        is expected_ready
    )


def test_coco_annotation_identity_digest_rejects_wrong_validation_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bind a complete COCO annotation table to its canonical image identities."""

    images = [
        {"id": index, "file_name": f"{index:012d}.jpg"} for index in range(1, 5001)
    ]
    payload = "".join(f"{item['id']}:{item['file_name']}\n" for item in images).encode()
    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 5000)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 0)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    monkeypatch.setattr(
        readiness,
        "COCO_VALIDATION_IMAGE_IDENTITIES_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    annotation_path = tmp_path / "instances_val2017.json"
    annotation_path.write_text(
        json.dumps({"images": images, "categories": [{"id": 1}], "annotations": []}),
        encoding="utf-8",
    )

    assert readiness._load_coco_image_names(annotation_path) is not None
    images[0]["file_name"] = "999999999999.jpg"
    annotation_path.write_text(
        json.dumps({"images": images, "categories": [{"id": 1}], "annotations": []}),
        encoding="utf-8",
    )
    assert readiness._load_coco_image_names(annotation_path) is None


def test_coco_readiness_rejects_duplicate_image_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject COCO metadata whose duplicate IDs would overwrite image records."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 2)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    image_names = ["000000000001.jpg", "000000000002.jpg"]
    for image_name in image_names:
        _write_file(tmp_path / "val2017" / image_name)
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": image_names[0]},
                    {"id": 1, "file_name": image_names[1]},
                ],
                "categories": [{"id": 1}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1},
                    {"id": 2, "image_id": 1, "category_id": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, "object_detection", "coco")


def test_coco_readiness_rejects_truncated_annotation_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not reuse complete-looking images with incomplete COCO ground truth."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setitem(readiness.COCO_ANNOTATION_COUNTS, "instances_val2017.json", 2)
    monkeypatch.setitem(readiness.COCO_CATEGORY_COUNTS, "instances_val2017.json", 1)
    monkeypatch.setattr(readiness, "COCO_CATEGORY_IDS", frozenset({1}))
    image_names = ["000000000001.jpg", "000000000002.jpg"]
    for image_name in image_names:
        _write_file(tmp_path / "val2017" / image_name)
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": index, "file_name": image_name}
                    for index, image_name in enumerate(image_names, start=1)
                ],
                "categories": [{"id": 1}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            }
        ),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, "object_detection", "coco")


def test_widerface_readiness_rejects_invalid_difficulty_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject present-but-malformed difficulty files before evaluation indexes them."""

    monkeypatch.setattr(readiness, "WIDERFACE_EVENT_COUNT", 1)
    monkeypatch.setattr(readiness, "WIDERFACE_VALIDATION_SAMPLE_COUNT", 1)
    _write_widerface_metadata(
        tmp_path / "wider_face_val.mat", {"0--Parade": ["sample"]}
    )
    for file_name in (
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    ):
        _write_file(tmp_path / file_name)
    _write_file(tmp_path / "images" / "0--Parade" / "sample.jpg")

    assert not readiness.dataset_ready(tmp_path, "face_detection", "widerface")


def test_widerface_readiness_rejects_empty_face_ground_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not accept empty face arrays that evaluation would silently skip."""

    face_boxes = np.empty((1, 1), dtype=object)
    event_faces = np.empty((1, 1), dtype=object)
    event_faces[0, 0] = np.empty((0, 4), dtype=np.float64)
    face_boxes[0, 0] = event_faces
    difficulty = np.empty((1, 1), dtype=object)
    event_indices = np.empty((1, 1), dtype=object)
    event_indices[0, 0] = np.empty((0,), dtype=np.int64)
    difficulty[0, 0] = event_indices

    def _loadmat(path: Path) -> dict[str, np.ndarray]:
        if path.name == "wider_face_val.mat":
            return {"face_bbx_list": face_boxes}
        return {"gt_list": difficulty}

    monkeypatch.setattr(readiness, "loadmat", _loadmat)

    assert not readiness._widerface_difficulty_metadata_ready(
        tmp_path, {"0--Parade": {"sample.jpg"}}
    )


def test_widerface_readiness_rejects_duplicate_difficulty_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Difficulty masks must identify each eligible face at most once."""

    face_boxes = np.empty((1, 1), dtype=object)
    event_faces = np.empty((1, 1), dtype=object)
    event_faces[0, 0] = np.array([[0, 0, 1, 1]], dtype=np.float64)
    face_boxes[0, 0] = event_faces
    difficulty = np.empty((1, 1), dtype=object)
    event_indices = np.empty((1, 1), dtype=object)
    event_indices[0, 0] = np.array([1, 1], dtype=np.int64)
    difficulty[0, 0] = event_indices

    def _loadmat(path: Path) -> dict[str, np.ndarray]:
        if path.name == "wider_face_val.mat":
            return {"face_bbx_list": face_boxes}
        return {"gt_list": difficulty}

    monkeypatch.setattr(readiness, "loadmat", _loadmat)

    assert not readiness._widerface_difficulty_metadata_ready(
        tmp_path, {"0--Parade": {"sample.jpg"}}
    )


def test_widerface_readiness_rejects_row_vector_difficulty_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject row vectors that the evaluator would count as one eligible face."""

    face_boxes = np.empty((1, 1), dtype=object)
    event_faces = np.empty((1, 1), dtype=object)
    event_faces[0, 0] = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float64)
    face_boxes[0, 0] = event_faces
    difficulty = np.empty((1, 1), dtype=object)
    event_indices = np.empty((1, 1), dtype=object)
    event_indices[0, 0] = np.array([[1, 2]], dtype=np.int64)
    difficulty[0, 0] = event_indices

    def _loadmat(path: Path) -> dict[str, np.ndarray]:
        if path.name == "wider_face_val.mat":
            return {"face_bbx_list": face_boxes}
        return {"gt_list": difficulty}

    monkeypatch.setattr(readiness, "loadmat", _loadmat)

    assert not readiness._widerface_difficulty_metadata_ready(
        tmp_path, {"0--Parade": {"sample.jpg"}}
    )


def test_widerface_readiness_accepts_empty_difficulty_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep valid images with no eligible faces in a difficulty split."""

    face_boxes = np.empty((1, 1), dtype=object)
    event_faces = np.empty((1, 1), dtype=object)
    event_faces[0, 0] = np.array([[0, 0, 1, 1]], dtype=np.float64)
    face_boxes[0, 0] = event_faces
    difficulty = np.empty((1, 1), dtype=object)
    event_indices = np.empty((1, 1), dtype=object)
    event_indices[0, 0] = np.empty((0, 0), dtype=np.int64)
    difficulty[0, 0] = event_indices

    def _loadmat(path: Path) -> dict[str, np.ndarray]:
        if path.name == "wider_face_val.mat":
            return {"face_bbx_list": face_boxes}
        return {"gt_list": difficulty}

    monkeypatch.setattr(readiness, "loadmat", _loadmat)

    assert readiness._widerface_difficulty_metadata_ready(
        tmp_path, {"0--Parade": {"sample.jpg"}}
    )


def test_widerface_readiness_rejects_face_boxes_outside_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Face ground truth must retain foreground in its decoded source image."""

    face_boxes = np.empty((1, 1), dtype=object)
    event_faces = np.empty((1, 1), dtype=object)
    event_faces[0, 0] = np.array([[11, 0, 1, 1]], dtype=np.float64)
    face_boxes[0, 0] = event_faces
    difficulty = np.empty((1, 1), dtype=object)
    event_indices = np.empty((1, 1), dtype=object)
    event_indices[0, 0] = np.array([1], dtype=np.int64)
    difficulty[0, 0] = event_indices

    def _loadmat(path: Path) -> dict[str, np.ndarray]:
        if path.name == "wider_face_val.mat":
            return {"face_bbx_list": face_boxes}
        return {"gt_list": difficulty}

    monkeypatch.setattr(readiness, "loadmat", _loadmat)

    assert not readiness._widerface_difficulty_metadata_ready(
        tmp_path,
        {"0--Parade": {"sample.jpg"}},
        image_shapes=[[(10, 10)]],
    )


@pytest.mark.parametrize("relative_image_dir", ["images", "images/val"])
def test_dotav1_readiness_requires_complete_image_label_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_image_dir: str,
) -> None:
    """Accept flat and legacy DOTA images only when every image has a label."""

    monkeypatch.setattr(readiness, "DOTAV1_VALIDATION_SAMPLE_COUNT", 2)
    for stem in ("P0001", "P0002"):
        _write_file(tmp_path / relative_image_dir / f"{stem}.png")
    _write_file(tmp_path / "labels" / "val_original" / "P0001.txt")

    assert not readiness.dataset_ready(tmp_path, "obb", "dotav1")

    _write_file(tmp_path / "labels" / "val" / "P0002.txt")

    assert readiness.dataset_ready(tmp_path, "obb", "dotav1")

    external_image = tmp_path / "external.png"
    external_label = tmp_path / "external.txt"
    _write_file(external_image)
    _write_file(external_label)
    image_path = tmp_path / relative_image_dir / "P0001.png"
    label_path = tmp_path / "labels" / "val" / "P0002.txt"
    image_path.unlink()
    label_path.unlink()
    image_path.symlink_to(external_image)
    label_path.symlink_to(external_label)

    assert readiness.dataset_ready(tmp_path, "obb", "dotav1")


def test_widerface_readiness_requires_complete_event_tree_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require all WiderFace validation events, images, and evaluation metadata."""

    monkeypatch.setattr(readiness, "WIDERFACE_EVENT_COUNT", 2)
    monkeypatch.setattr(readiness, "WIDERFACE_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setattr(
        readiness, "_widerface_difficulty_metadata_ready", lambda *_, **__: True
    )
    monkeypatch.setattr(
        readiness, "_widerface_image_shapes", lambda *_: [[(1, 1)], [(1, 1)]]
    )
    _write_widerface_metadata(
        tmp_path / "wider_face_val.mat",
        {"0--Parade": ["sample-0"], "1--Handshaking": ["sample-1"]},
    )
    for file_name in (
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    ):
        _write_file(tmp_path / file_name)
    _write_file(tmp_path / "images" / "0--Parade" / "sample-0.jpg")
    (tmp_path / "images" / "1--Handshaking").mkdir()

    assert not readiness.dataset_ready(tmp_path, "face_detection", "widerface")

    _write_file(tmp_path / "images" / "1--Handshaking" / "sample-1.jpg")

    assert readiness.dataset_ready(tmp_path, "face_detection", "widerface")
    assert not readiness.dataset_ready(tmp_path, "face_detection", "coco")


@pytest.mark.parametrize(
    ("event_name", "image_name"),
    [
        ("0--Parade", "stale"),
        ("1--Handshaking", "expected"),
    ],
)
def test_widerface_readiness_rejects_tree_not_named_by_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    event_name: str,
    image_name: str,
) -> None:
    """Reject complete-looking trees whose event or image identity differs from metadata."""

    monkeypatch.setattr(readiness, "WIDERFACE_EVENT_COUNT", 1)
    monkeypatch.setattr(readiness, "WIDERFACE_VALIDATION_SAMPLE_COUNT", 1)
    _write_widerface_metadata(
        tmp_path / "wider_face_val.mat", {"0--Parade": ["expected"]}
    )
    for file_name in (
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    ):
        _write_file(tmp_path / file_name)
    _write_file(tmp_path / "images" / event_name / f"{image_name}.jpg")

    assert not readiness.dataset_ready(tmp_path, "face_detection", "widerface")


def test_ade20k_readiness_requires_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require both ADE20K metadata files before reusing an organized cache."""

    monkeypatch.setattr(readiness, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    _write_file(tmp_path / "images" / "ADE_val_00000001.jpg")
    _write_file(tmp_path / "annotations" / "ADE_val_00000001.png")

    assert not readiness.dataset_ready(tmp_path, "semantic_segmentation", "ade20k")

    _write_file(tmp_path / "objectInfo150.txt")
    assert not readiness.dataset_ready(tmp_path, "semantic_segmentation", "ade20k")

    _write_file(tmp_path / "sceneCategories.txt")
    assert readiness.dataset_ready(tmp_path, "semantic_segmentation", "ade20k")


@pytest.mark.parametrize(
    ("dataset", "task", "relative_path"),
    [
        ("nyu-depth", "depth_estimation", "images/sample.jpg"),
        ("nyu-depth", "depth_estimation", "depth/sample.npy"),
        ("nyu-depth", "depth_estimation", "images/extra.jpg"),
        ("ade20k", "semantic_segmentation", "images/ADE_val_00000001.jpg"),
        ("ade20k", "semantic_segmentation", "annotations/ADE_val_00000001.png"),
        ("ade20k", "semantic_segmentation", "objectInfo150.txt"),
    ],
)
def test_dense_readiness_rejects_symlinked_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: str,
    task: str,
    relative_path: str,
) -> None:
    """Do not reuse a complete-looking dense cache containing symlinked files."""

    monkeypatch.setattr(readiness, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(readiness, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    if dataset == "nyu-depth":
        _write_file(tmp_path / "images" / "sample.jpg")
        _write_file(tmp_path / "depth" / "sample.npy")
    else:
        _write_file(tmp_path / "images" / "ADE_val_00000001.jpg")
        _write_file(tmp_path / "annotations" / "ADE_val_00000001.png")
        for file_name in readiness.ADE20K_METADATA_FILES:
            _write_file(tmp_path / file_name)
    external_file = tmp_path.parent / f"{tmp_path.name}-outside"
    external_file.write_bytes(b"outside dataset")
    source_path = tmp_path / relative_path
    if source_path.exists():
        source_path.unlink()
    source_path.symlink_to(external_file)

    assert not readiness.dataset_ready(tmp_path, task, dataset)


def test_dense_readiness_rejects_symlinked_root_ancestors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not reuse complete dense roots reached through a symlinked parent."""

    monkeypatch.setattr(readiness, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(readiness, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    target_parent = tmp_path / "target"
    nyu_root = target_parent / "nyu-depth"
    _write_file(nyu_root / "images" / "sample.jpg")
    _write_file(nyu_root / "depth" / "sample.npy")
    ade20k_root = target_parent / "ade20k"
    _write_file(ade20k_root / "images" / "ADE_val_00000001.jpg")
    _write_file(ade20k_root / "annotations" / "ADE_val_00000001.png")
    for file_name in readiness.ADE20K_METADATA_FILES:
        _write_file(ade20k_root / file_name)
    symlinked_parent = tmp_path / "datasets"
    symlinked_parent.symlink_to(target_parent, target_is_directory=True)

    assert not readiness.dataset_ready(
        symlinked_parent / "nyu-depth", "depth_estimation", "nyu-depth"
    )
    assert not readiness.dataset_ready(
        symlinked_parent / "ade20k", "semantic_segmentation", "ade20k"
    )

    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    traversed_parent = existing_dir / ".." / symlinked_parent.name
    assert not readiness.dataset_ready(
        traversed_parent / "nyu-depth", "depth_estimation", "nyu-depth"
    )
    assert not readiness.dataset_ready(
        traversed_parent / "ade20k", "semantic_segmentation", "ade20k"
    )

    target_child = target_parent / "child"
    target_child.mkdir()
    traversal_link = tmp_path / "traversal-link"
    traversal_link.symlink_to(target_child, target_is_directory=True)
    symlink_traversed_parent = traversal_link / ".."
    assert not readiness.dataset_ready(
        symlink_traversed_parent / "nyu-depth", "depth_estimation", "nyu-depth"
    )
    assert not readiness.dataset_ready(
        symlink_traversed_parent / "ade20k", "semantic_segmentation", "ade20k"
    )

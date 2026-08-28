"""Identity and completeness checks for organized vision validation datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from faster_coco_eval import mask as coco_mask
from PIL import Image
from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from ..._tasks import normalize_vision_task
from ...datasets import get_dataset_category_ids
from .cityscapes import CITYSCAPES_SOURCE_TO_TRAIN_ID

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_CLASS_COUNT = 1000
IMAGENET_IMAGES_PER_CLASS = 50
COCO_VALIDATION_SAMPLE_COUNT = 5000
DOTAV1_VALIDATION_SAMPLE_COUNT = 458
WIDERFACE_EVENT_COUNT = 61
WIDERFACE_VALIDATION_SAMPLE_COUNT = 3226
NYU_DEPTH_VALIDATION_SAMPLE_COUNT = 654
ADE20K_VALIDATION_SAMPLE_COUNT = 2000
CITYSCAPES_VALIDATION_SAMPLE_COUNT = 500
SAV_VALIDATION_VIDEO_COUNT = 155
SAV_VALIDATION_MASKLET_COUNT = 293
# Total per-object annotated masks across every masklet, measured from the
# organized sha256-pinned official `sav_val.tar` (see `datasets/sa-v.yaml` and
# `PINNED_ARCHIVE_SHA256`). Masklet and video counts alone accept a truncated
# source that keeps every masklet but only a few of its annotated frames,
# which would silently change the evaluation corpus; this pins the inventory.
SAV_VALIDATION_MASK_COUNT = 31967
ADE20K_METADATA_FILES = ("objectInfo150.txt", "sceneCategories.txt")
IMAGENET_CLASS_PATTERN = re.compile(r"n\d{8}")
IMAGENET_IMAGE_PATTERN = re.compile(r"ILSVRC2012_val_\d{8}")
COCO_IMAGE_PATTERN = re.compile(r"\d{12}")
WIDERFACE_EVENT_PATTERN = re.compile(r"\d+--\S.*")
CITYSCAPES_SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<city>[A-Za-z][A-Za-z0-9-]*)_\d{6}_\d{6}$"
)
SAV_VIDEO_ID_PATTERN = re.compile(r"^sav_\d{6}$")
SAV_OBJECT_ID_PATTERN = re.compile(r"^\d{3}$")
SAV_FRAME_STEM_PATTERN = re.compile(r"^\d{5}$")
CITYSCAPES_VALIDATION_CITY_COUNTS = {"frankfurt": 267, "lindau": 59, "munster": 174}
IMAGENET_SYNSET_ORDER = tuple(
    files("mblt_vision.datasets")
    .joinpath("imagenet_synsets.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
IMAGENET_SYNSETS = frozenset(IMAGENET_SYNSET_ORDER)
COCO_ANNOTATION_COUNTS = {
    "instances_val2017.json": 36781,
    "person_keypoints_val2017.json": 11004,
}
COCO_CATEGORY_COUNTS = {
    "instances_val2017.json": 80,
    "person_keypoints_val2017.json": 1,
}
COCO_CATEGORY_IDS = frozenset(get_dataset_category_ids("coco"))
COCO_PERSON_KEYPOINT_CATEGORY_IDS = frozenset({1})
COCO_VALIDATION_IMAGE_IDENTITIES_SHA256 = (
    "f57f71ba25171a0fd99be8c425a91d4a6fdd43d25aadc7e5be51dd37a73281a7"
)


def _path_has_symlink_component(path: Path) -> bool:
    """Return whether a path traversal or its normalized ancestors contain a symlink."""

    expanded_path = path.expanduser()
    traversal_path = (
        expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path
    )
    normalized_path = Path(os.path.abspath(expanded_path))
    candidates = (
        traversal_path,
        *traversal_path.parents,
        normalized_path,
        *normalized_path.parents,
    )
    return any(component.is_symlink() for component in candidates)


def _files_by_stem(
    directory: Path,
    suffixes: set[str],
    *,
    reject_symlinks: bool = False,
) -> dict[str, Path] | None:
    """Collect direct child files with supported suffixes by stem.

    Args:
        directory: Directory containing candidate files.
        suffixes: Accepted lowercase file suffixes.
        reject_symlinks: Whether any symlinked directory or entry invalidates
            the file collection.

    Returns:
        Files keyed by stem, an empty mapping for a missing directory, or
        ``None`` for duplicate stems or rejected symlinks.
    """

    if (reject_symlinks and directory.is_symlink()) or not directory.is_dir():
        return {}
    entries = list(directory.iterdir())
    if reject_symlinks and any(path.is_symlink() for path in entries):
        return None
    paths = [
        path for path in entries if path.is_file() and path.suffix.lower() in suffixes
    ]
    files = {path.stem: path for path in paths}
    return files if len(files) == len(paths) else None


def _has_positive_polygon_area(polygon: list[int | float]) -> bool:
    """Return whether a finite flat polygon encloses non-zero signed area."""

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    signed_double_area = np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(
        points[:, 1], np.roll(points[:, 0], -1)
    )
    return bool(abs(signed_double_area) > 0)


def _polygon_union_has_rasterized_foreground(
    polygons: list[list[int | float]], image_shape: tuple[int, int] | None
) -> bool:
    """Return whether the combined COCO polygons rasterize to foreground.

    COCO polygons describe one instance as a union.  Individual components can
    be thinner than a pixel (as occurs in the official validation annotations),
    so only the combined raster needs to contain foreground.
    """

    if image_shape is None:
        return False
    height, width = image_shape
    try:
        encoded = coco_mask.frPyObjects(polygons, height, width)
        decoded = np.asarray(coco_mask.decode(encoded))
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(np.any(decoded))


def _canonicalize_quadrilateral(
    coordinates: list[int | float] | tuple[int | float, ...],
) -> tuple[float, ...]:
    """Return a quadrilateral key independent of its start vertex and winding."""

    if len(coordinates) != 8:
        raise ValueError(
            "A quadrilateral must contain exactly four two-dimensional vertices."
        )
    points = tuple(
        (float(coordinates[index]), float(coordinates[index + 1]))
        for index in range(0, len(coordinates), 2)
    )
    candidates = []
    for winding in (points, tuple(reversed(points))):
        candidates.extend(winding[index:] + winding[:index] for index in range(4))
    return tuple(coordinate for point in min(candidates) for coordinate in point)


def _polygon_has_positive_image_overlap(
    polygon: list[int | float], image_shape: tuple[int, int] | None
) -> bool:
    """Return whether a polygon covers non-zero area within an image rectangle."""

    if image_shape is None:
        return True
    height, width = image_shape
    if height <= 0 or width <= 0:
        return False
    points = [
        tuple(point) for point in np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    ]

    def _clip(
        vertices: list[tuple[float, float]],
        inside: Any,
        intersect: Any,
    ) -> list[tuple[float, float]]:
        clipped: list[tuple[float, float]] = []
        if not vertices:
            return clipped
        previous = vertices[-1]
        previous_inside = bool(inside(previous))
        for current in vertices:
            current_inside = bool(inside(current))
            if current_inside != previous_inside:
                clipped.append(intersect(previous, current))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        return clipped

    def _vertical_intersection(
        x: float, start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        delta_x = end[0] - start[0]
        if delta_x == 0:
            return x, start[1]
        ratio = (x - start[0]) / delta_x
        return x, start[1] + ratio * (end[1] - start[1])

    def _horizontal_intersection(
        y: float, start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        delta_y = end[1] - start[1]
        if delta_y == 0:
            return start[0], y
        ratio = (y - start[1]) / delta_y
        return start[0] + ratio * (end[0] - start[0]), y

    points = _clip(
        points,
        lambda point: point[0] >= 0,
        lambda a, b: _vertical_intersection(0, a, b),
    )
    points = _clip(
        points,
        lambda point: point[0] <= width,
        lambda a, b: _vertical_intersection(width, a, b),
    )
    points = _clip(
        points,
        lambda point: point[1] >= 0,
        lambda a, b: _horizontal_intersection(0, a, b),
    )
    points = _clip(
        points,
        lambda point: point[1] <= height,
        lambda a, b: _horizontal_intersection(height, a, b),
    )
    if len(points) < 3:
        return False
    clipped = [coordinate for point in points for coordinate in point]
    return _has_positive_polygon_area(clipped)


def _imagenet_ready(root: Path) -> bool:
    """Check the organizer's complete ImageNet-1k validation class tree."""

    if not root.is_dir():
        return False
    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    if {path.name for path in class_dirs} != IMAGENET_SYNSETS:
        return False
    image_names: set[str] = set()
    for class_dir in class_dirs:
        if IMAGENET_CLASS_PATTERN.fullmatch(class_dir.name) is None:
            return False
        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(images) != IMAGENET_IMAGES_PER_CLASS or any(
            path.suffix != ".JPEG"
            or IMAGENET_IMAGE_PATTERN.fullmatch(path.stem) is None
            for path in images
        ):
            return False
        image_names.update(path.name for path in images)
    return len(image_names) == IMAGENET_CLASS_COUNT * IMAGENET_IMAGES_PER_CLASS


def _load_coco_image_names(
    annotation_path: Path, task: str = "object_detection"
) -> set[str] | None:
    """Load unique validation image filenames from a COCO annotation file."""

    try:
        annotation: Any = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    if not isinstance(annotation, dict):
        return None
    image_records = annotation.get("images")
    if (
        not isinstance(image_records, list)
        or len(image_records) != COCO_VALIDATION_SAMPLE_COUNT
    ):
        return None
    names: list[str] = []
    image_ids: list[int] = []
    image_shapes: dict[int, tuple[int, int] | None] = {}
    for record in image_records:
        if not isinstance(record, dict):
            continue
        file_name = record.get("file_name")
        image_id = record.get("id")
        if (
            isinstance(file_name, str)
            and isinstance(image_id, int)
            and not isinstance(image_id, bool)
        ):
            names.append(file_name)
            image_ids.append(image_id)
            height, width = record.get("height"), record.get("width")
            if (
                isinstance(height, int)
                and not isinstance(height, bool)
                and height > 0
                and isinstance(width, int)
                and not isinstance(width, bool)
                and width > 0
            ):
                image_shapes[image_id] = (height, width)
            else:
                return None
    if (
        len(names) != len(image_records)
        or len(names) != len(set(names))
        or len(image_ids) != len(set(image_ids))
    ):
        return None
    if len(image_records) == 5000:
        identity_payload = "".join(
            f"{image_id}:{file_name}\n"
            for image_id, file_name in sorted(zip(image_ids, names, strict=True))
        ).encode()
        if (
            hashlib.sha256(identity_payload).hexdigest()
            != COCO_VALIDATION_IMAGE_IDENTITIES_SHA256
        ):
            return None
    annotation_records = annotation.get("annotations")
    categories = annotation.get("categories")
    expected_annotations = COCO_ANNOTATION_COUNTS.get(annotation_path.name)
    expected_categories = COCO_CATEGORY_COUNTS.get(annotation_path.name)
    if (
        not isinstance(annotation_records, list)
        or not isinstance(categories, list)
        or len(annotation_records) != expected_annotations
        or len(categories) != expected_categories
    ):
        return None
    category_ids: list[int] = []
    for category in categories:
        if not isinstance(category, dict):
            return None
        category_id = category.get("id")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            return None
        category_ids.append(category_id)
    if len(category_ids) != len(set(category_ids)):
        return None
    expected_category_ids = (
        COCO_PERSON_KEYPOINT_CATEGORY_IDS
        if annotation_path.name == "person_keypoints_val2017.json"
        else COCO_CATEGORY_IDS
    )
    if set(category_ids) != expected_category_ids:
        return None
    if not _coco_task_annotations_valid(
        annotation_records,
        image_ids=set(image_ids),
        category_ids=set(category_ids),
        image_shapes=image_shapes,
        task=task,
    ):
        return None
    return set(names)


def _coco_task_annotations_valid(
    annotation_records: list[Any],
    *,
    image_ids: set[int],
    category_ids: set[int],
    image_shapes: dict[int, tuple[int, int] | None],
    task: str,
) -> bool:
    """Validate task-specific COCO annotation payloads for readiness and APIs."""

    annotation_ids: list[int] = []
    for record in annotation_records:
        if not isinstance(record, dict):
            return False
        annotation_id = record.get("id")
        image_id = record.get("image_id")
        category_id = record.get("category_id")
        if (
            not isinstance(annotation_id, int)
            or isinstance(annotation_id, bool)
            or not isinstance(image_id, int)
            or isinstance(image_id, bool)
            or image_id not in image_ids
            or not isinstance(category_id, int)
            or isinstance(category_id, bool)
            or category_id not in category_ids
        ):
            return False
        image_shape = image_shapes.get(image_id)
        if image_shape is None:
            return False
        image_height, image_width = image_shape
        bbox = record.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                for value in bbox
            )
            or bbox[2] <= 0
            or bbox[3] <= 0
            or bbox[0] >= image_width
            or bbox[0] + bbox[2] <= 0
            or bbox[1] >= image_height
            or bbox[1] + bbox[3] <= 0
        ):
            return False
        area = record.get("area")
        iscrowd = record.get("iscrowd")
        if (
            not isinstance(area, (int, float))
            or isinstance(area, bool)
            or not np.isfinite(area)
            or area <= 0
            or area > image_height * image_width
            or not isinstance(iscrowd, int)
            or isinstance(iscrowd, bool)
            or iscrowd not in {0, 1}
        ):
            return False
        if task == "pose_estimation" and not iscrowd and area > bbox[2] * bbox[3]:
            return False
        if task == "instance_segmentation":
            segmentation = record.get("segmentation")
            if isinstance(segmentation, list):
                if (
                    not segmentation
                    or any(
                        not isinstance(polygon, list)
                        or len(polygon) < 6
                        or len(polygon) % 2
                        or any(
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not np.isfinite(value)
                            for value in polygon
                        )
                        or not _has_positive_polygon_area(polygon)
                        or not _polygon_has_positive_image_overlap(
                            polygon, image_shapes.get(image_id)
                        )
                        for polygon in segmentation
                    )
                    or not _polygon_union_has_rasterized_foreground(
                        segmentation, image_shapes.get(image_id)
                    )
                ):
                    return False
            elif isinstance(segmentation, dict):
                if not _valid_coco_rle(segmentation, image_shapes.get(image_id)):
                    return False
            else:
                return False
        if task == "pose_estimation":
            keypoints = record.get("keypoints")
            num_keypoints = record.get("num_keypoints")
            if (
                not isinstance(keypoints, list)
                or len(keypoints) != 51
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                    for value in keypoints
                )
                or not isinstance(num_keypoints, int)
                or isinstance(num_keypoints, bool)
                or not 0 <= num_keypoints <= 17
                or any(
                    keypoints[index] not in {0, 1, 2}
                    for index in range(2, len(keypoints), 3)
                )
                or num_keypoints
                != sum(keypoints[index] > 0 for index in range(2, len(keypoints), 3))
            ):
                return False
            height, width = image_shape
            if any(
                keypoints[index + 2] > 0
                and not (
                    0 <= keypoints[index] < width and 0 <= keypoints[index + 1] < height
                )
                for index in range(0, len(keypoints), 3)
            ):
                return False
        annotation_ids.append(annotation_id)
    return len(annotation_ids) == len(set(annotation_ids))


def _decode_coco_rle_counts(counts: str) -> list[int] | None:
    """Decode COCO's compact RLE run-length string without trusting its payload."""

    run_counts: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            if position >= len(counts):
                return None
            code = ord(counts[position]) - 48
            position += 1
            if not 0 <= code <= 0x3F:
                return None
            value |= (code & 0x1F) << shift
            more = bool(code & 0x20)
            shift += 5
            if shift > 60:
                return None
            if not more and code & 0x10:
                value |= -1 << shift
        if len(run_counts) > 2:
            value += run_counts[-2]
        if value < 0:
            return None
        run_counts.append(value)
    return run_counts


def _valid_coco_rle(
    segmentation: dict[str, Any], image_shape: tuple[int, int] | None
) -> bool:
    """Validate and decode an RLE mask against its referenced COCO image shape."""

    counts = segmentation.get("counts")
    size = segmentation.get("size")
    if (
        image_shape is None
        or not isinstance(size, list)
        or len(size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in size
        )
        or tuple(size) != image_shape
        or not isinstance(counts, (str, list))
    ):
        return False
    if isinstance(counts, list):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            return False
        run_counts = counts
    else:
        run_counts = _decode_coco_rle_counts(counts)
        if run_counts is None:
            return False
    if sum(run_counts) != math.prod(size):
        return False
    try:
        encoded = (
            coco_mask.frPyObjects(segmentation, size[0], size[1])
            if isinstance(counts, list)
            else segmentation
        )
        decoded = np.asarray(coco_mask.decode(encoded))
    except (RuntimeError, TypeError, ValueError):
        return False
    return decoded.shape == tuple(size) and bool(np.any(decoded))


def _coco_ready(root: Path, task: str) -> bool:
    """Check the complete COCO 2017 image split and task annotation metadata."""

    annotation_name = (
        "person_keypoints_val2017.json"
        if task == "pose_estimation"
        else "instances_val2017.json"
    )
    annotation_names = _load_coco_image_names(root / annotation_name, task)
    if annotation_names is None:
        return False
    image_dir = root / "val2017"
    if not image_dir.is_dir():
        return False
    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(image_paths) != COCO_VALIDATION_SAMPLE_COUNT or any(
        path.suffix.lower() != ".jpg" or COCO_IMAGE_PATTERN.fullmatch(path.stem) is None
        for path in image_paths
    ):
        return False
    if {path.name for path in image_paths} != annotation_names:
        return False
    try:
        payload = json.loads((root / annotation_name).read_text(encoding="utf-8"))
        image_records = payload["images"]
        image_shapes = {
            record["file_name"]: (record["height"], record["width"])
            for record in image_records
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    for image_path in image_paths:
        expected_shape = image_shapes.get(image_path.name)
        if (
            not isinstance(expected_shape, tuple)
            or len(expected_shape) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in expected_shape
            )
        ):
            return False
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
        except OSError:
            return False
        if (height, width) != expected_shape:
            return False
    return True


def _dotav1_ready(root: Path) -> bool:
    """Check complete paired DOTAv1 validation images and labels."""

    flat_image_dir = root / "images"
    flat_images = _files_by_stem(flat_image_dir, IMAGE_SUFFIXES)
    if flat_images is None:
        return False
    image_dir = flat_image_dir if flat_images else flat_image_dir / "val"
    images = flat_images if flat_images else _files_by_stem(image_dir, IMAGE_SUFFIXES)
    if images is None or len(images) != DOTAV1_VALIDATION_SAMPLE_COUNT:
        return False

    normalized_labels = _files_by_stem(root / "labels" / "val", {".txt"})
    original_labels = _files_by_stem(root / "labels" / "val_original", {".txt"})
    if normalized_labels is None or original_labels is None:
        return False
    label_stems = normalized_labels.keys() | original_labels.keys()
    return images.keys() == label_stems


def _widerface_ready(root: Path) -> bool:
    """Check the complete WiderFace validation image tree and metadata files."""

    required_files = (
        "wider_face_val.mat",
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    )
    if not all((root / file_name).is_file() for file_name in required_files):
        return False
    expected_images = _load_widerface_image_names(root / "wider_face_val.mat")
    if expected_images is None:
        return False
    image_root = root / "images"
    if not image_root.is_dir():
        return False
    event_dirs = [path for path in image_root.iterdir() if path.is_dir()]
    if len(event_dirs) != WIDERFACE_EVENT_COUNT or any(
        WIDERFACE_EVENT_PATTERN.fullmatch(path.name) is None for path in event_dirs
    ):
        return False
    if {path.name for path in event_dirs} != expected_images.keys():
        return False
    actual_images = {
        event_dir.name: {
            path.name
            for path in event_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        for event_dir in event_dirs
    }
    image_shapes = _widerface_image_shapes(root, expected_images)
    return (
        actual_images == expected_images
        and sum(len(image_names) for image_names in actual_images.values())
        == WIDERFACE_VALIDATION_SAMPLE_COUNT
        and image_shapes is not None
        and _widerface_difficulty_metadata_ready(
            root, expected_images, image_shapes=image_shapes
        )
    )


def _widerface_difficulty_metadata_ready(
    root: Path,
    expected_images: dict[str, set[str]],
    *,
    image_shapes: list[list[tuple[int, int]]] | None = None,
) -> bool:
    """Validate WiderFace difficulty metadata and its decoded-image geometry."""

    try:
        main = loadmat(root / "wider_face_val.mat")
        face_boxes = main["face_bbx_list"]
        difficulties = [
            loadmat(root / file_name)["gt_list"]
            for file_name in (
                "wider_easy_val.mat",
                "wider_medium_val.mat",
                "wider_hard_val.mat",
            )
        ]
    except (IndexError, KeyError, MatReadError, OSError, TypeError, ValueError):
        return False
    if len(face_boxes) != len(expected_images) or any(
        len(table) != len(expected_images) for table in difficulties
    ):
        return False
    if image_shapes is not None and len(image_shapes) != len(expected_images):
        return False
    difficulty_has_eligible_face = [False] * len(difficulties)
    for event_index, image_names in enumerate(expected_images.values()):
        try:
            event_faces = face_boxes[event_index][0]
        except (IndexError, TypeError):
            return False
        if len(event_faces) != len(image_names):
            return False
        if image_shapes is not None and len(image_shapes[event_index]) != len(
            image_names
        ):
            return False
        face_counts: list[int] = []
        for image_index, face_entry in enumerate(event_faces):
            try:
                face_array = np.asarray(face_entry[0])
            except (IndexError, TypeError):
                return False
            if (
                face_array.ndim != 2
                or face_array.shape[1] != 4
                or len(face_array) == 0
                or not np.issubdtype(face_array.dtype, np.number)
                or np.issubdtype(face_array.dtype, np.complexfloating)
                or not np.isfinite(face_array).all()
            ):
                return False
            no_face_sentinel = face_array.shape == (1, 4) and not bool(
                np.any(face_array)
            )
            face_counts.append(0 if no_face_sentinel else len(face_array))
        for difficulty_index, table in enumerate(difficulties):
            try:
                event_indices = table[event_index][0]
            except (IndexError, TypeError):
                return False
            if len(event_indices) != len(image_names):
                return False
            for image_index in range(len(event_faces)):
                try:
                    keep_indices = np.asarray(event_indices[image_index][0])
                except (IndexError, TypeError):
                    return False
                if keep_indices.ndim == 0:
                    keep_indices = keep_indices.reshape(1)
                elif keep_indices.ndim == 2:
                    if keep_indices.size == 0:
                        if keep_indices.shape[0] != 0:
                            return False
                        keep_indices = keep_indices.reshape(0)
                    elif keep_indices.shape[1] == 1:
                        keep_indices = keep_indices[:, 0]
                    else:
                        return False
                elif keep_indices.ndim != 1:
                    return False
                try:
                    valid_indices = (
                        np.isfinite(keep_indices).all()
                        and np.equal(keep_indices, np.trunc(keep_indices)).all()
                    )
                except (TypeError, ValueError):
                    return False
                if not valid_indices:
                    return False
                if keep_indices.size and (
                    int(keep_indices.min()) < 1
                    or int(keep_indices.max()) > face_counts[image_index]
                ):
                    return False
                if len(np.unique(keep_indices)) != keep_indices.size:
                    return False
                difficulty_has_eligible_face[difficulty_index] |= bool(
                    keep_indices.size
                )
    if not all(difficulty_has_eligible_face):
        return False
    return True


def _flatten_matlab_strings(value: Any) -> list[str]:
    """Flatten strings stored inside nested MATLAB cell arrays."""

    if isinstance(value, (str, np.str_)):
        return [str(value)]
    if isinstance(value, np.ndarray):
        strings: list[str] = []
        for item in value.flat:
            strings.extend(_flatten_matlab_strings(item))
        return strings
    return []


def _load_widerface_image_names(annotation_path: Path) -> dict[str, set[str]] | None:
    """Load exact event and image identities from WiderFace validation metadata."""

    try:
        annotation = loadmat(annotation_path)
        event_list = annotation["event_list"]
        file_list = annotation["file_list"]
    except (IndexError, KeyError, MatReadError, OSError, TypeError, ValueError):
        return None
    if len(event_list) != len(file_list):
        return None

    expected: dict[str, set[str]] = {}
    for event_cell, file_cell in zip(event_list, file_list, strict=True):
        event_names = _flatten_matlab_strings(event_cell)
        image_stems = _flatten_matlab_strings(file_cell)
        if (
            len(event_names) != 1
            or not image_stems
            or len(image_stems) != len(set(image_stems))
            or WIDERFACE_EVENT_PATTERN.fullmatch(event_names[0]) is None
            or any(not stem or Path(stem).name != stem for stem in image_stems)
            or event_names[0] in expected
        ):
            return None
        expected[event_names[0]] = {f"{stem}.jpg" for stem in image_stems}
    if len(expected) != WIDERFACE_EVENT_COUNT:
        return None
    return expected


def _widerface_image_shapes(
    root: Path, expected_images: dict[str, set[str]]
) -> list[list[tuple[int, int]]] | None:
    """Load WiderFace image shapes in the annotation's event and file order."""

    try:
        annotation = loadmat(root / "wider_face_val.mat")
        event_list = annotation["event_list"]
        file_list = annotation["file_list"]
    except (IndexError, KeyError, MatReadError, OSError, TypeError, ValueError):
        return None
    if len(event_list) != len(file_list) or len(event_list) != len(expected_images):
        return None

    all_shapes: list[list[tuple[int, int]]] = []
    for event_cell, file_cell in zip(event_list, file_list, strict=True):
        event_names = _flatten_matlab_strings(event_cell)
        image_stems = _flatten_matlab_strings(file_cell)
        if (
            len(event_names) != 1
            or not image_stems
            or expected_images.get(event_names[0])
            != {f"{stem}.jpg" for stem in image_stems}
        ):
            return None
        event_shapes: list[tuple[int, int]] = []
        for stem in image_stems:
            image_path = root / "images" / event_names[0] / f"{stem}.jpg"
            if image_path.is_symlink():
                return None
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
            except OSError:
                return None
            event_shapes.append((height, width))
        all_shapes.append(event_shapes)
    return all_shapes


def dense_dataset_ready(data_path: str | Path, dataset: str) -> bool:
    """Return whether a dense dataset matches its taxonomy and full validation split.

    Args:
        data_path: Organized dataset root.
        dataset: Dense validation taxonomy.

    Returns:
        Whether the dataset has the expected filename identity, matched targets,
        and complete validation sample count.
    """

    root = Path(data_path).expanduser()
    if _path_has_symlink_component(root) or not root.is_dir():
        return False
    normalized = dataset.lower()
    if normalized == "nyu-depth":
        images = _files_by_stem(
            root / "images", {".jpg", ".jpeg", ".png"}, reject_symlinks=True
        )
        depths = _files_by_stem(root / "depth", {".npy"}, reject_symlinks=True)
        if images is None or depths is None:
            return False
        if not (
            len(images) == NYU_DEPTH_VALIDATION_SAMPLE_COUNT
            and len(depths) == NYU_DEPTH_VALIDATION_SAMPLE_COUNT
            and images.keys() == depths.keys()
        ):
            return False
        for stem, image_path in images.items():
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_shape = (image.height, image.width)
                raw_depth = np.load(depths[stem], allow_pickle=False)
            except (OSError, ValueError):
                return False
            if not np.issubdtype(raw_depth.dtype, np.number) or np.issubdtype(
                raw_depth.dtype, np.complexfloating
            ):
                return False
            with np.errstate(over="ignore", invalid="ignore"):
                depth = np.asarray(raw_depth, dtype=np.float32)
            if (
                depth.ndim != 2
                or depth.shape != image_shape
                or not np.isfinite(depth).all()
                or bool((depth < 0).any())
                or not bool(((depth > 0.001) & (depth < 100.0)).any())
            ):
                return False
        return True

    images = _files_by_stem(
        root / "images", {".jpg", ".jpeg", ".png"}, reject_symlinks=True
    )
    annotations = _files_by_stem(root / "annotations", {".png"}, reject_symlinks=True)
    if images is None or annotations is None or images.keys() != annotations.keys():
        return False

    for stem, image_path in images.items():
        try:
            with Image.open(image_path) as image:
                image.load()
                image_shape = (image.height, image.width)
            with Image.open(annotations[stem]) as annotation_image:
                annotation = np.asarray(annotation_image)
        except OSError:
            return False
        if normalized == "cityscapes" and annotation.ndim == 3:
            if (
                annotation.shape[2] not in {3, 4}
                or not np.array_equal(annotation[..., 0], annotation[..., 1])
                or not np.array_equal(annotation[..., 0], annotation[..., 2])
            ):
                return False
            annotation = annotation[..., 0]
        if annotation.ndim != 2 or annotation.shape != image_shape:
            return False
        if normalized == "ade20k":
            if annotation.dtype != np.uint8 or (
                annotation.size and int(annotation.max()) > 150
            ):
                return False
            if not bool((annotation > 0).any()):
                return False
        elif normalized == "cityscapes":
            if annotation.size and (
                int(annotation.min()) < 0 or int(annotation.max()) > 255
            ):
                return False
            if not bool(np.all((annotation <= 33) | (annotation == 255))):
                return False
            train_ids = CITYSCAPES_SOURCE_TO_TRAIN_ID[annotation.astype(np.uint8)]
            if not bool((train_ids != 255).any()):
                return False

    if normalized == "ade20k":
        return (
            len(images) == ADE20K_VALIDATION_SAMPLE_COUNT
            and all(stem.startswith("ADE_val_") for stem in images)
            and all(
                path.suffix.lower() in {".jpg", ".jpeg"} for path in images.values()
            )
        )
    if normalized == "cityscapes":
        city_counts: dict[str, int] = {}
        for stem in images:
            match = CITYSCAPES_SAMPLE_ID_PATTERN.fullmatch(stem)
            if match is not None:
                city = match.group("city")
                city_counts[city] = city_counts.get(city, 0) + 1
        return (
            len(images) == CITYSCAPES_VALIDATION_SAMPLE_COUNT
            and all(
                CITYSCAPES_SAMPLE_ID_PATTERN.fullmatch(stem) is not None
                for stem in images
            )
            and all(path.suffix.lower() == ".png" for path in images.values())
            and (
                CITYSCAPES_VALIDATION_SAMPLE_COUNT != 500
                or city_counts == CITYSCAPES_VALIDATION_CITY_COUNTS
            )
        )
    return False


def _sav_ready(root: Path) -> bool:
    """Return whether an organized SA-V validation split is complete.

    Validates the id list, per-video image/annotation directory identity, the
    total masklet count, and mask/frame stem consistency. Decodes one
    frame/mask pair per video (155 decodes) rather than the full multi-
    thousand-file corpus, which would make this readiness probe too slow for
    its per-`val`-invocation call site; the organizer's staged payload
    validation decodes every mask once at install time instead.
    """

    if _path_has_symlink_component(root) or not root.is_dir():
        return False
    video_ids_path = root / "video_ids.txt"
    if video_ids_path.is_symlink() or not video_ids_path.is_file():
        return False
    try:
        video_ids = [
            line.strip()
            for line in video_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError):
        return False
    if len(video_ids) != SAV_VALIDATION_VIDEO_COUNT or len(set(video_ids)) != len(
        video_ids
    ):
        return False
    if any(SAV_VIDEO_ID_PATTERN.fullmatch(video_id) is None for video_id in video_ids):
        return False

    image_root = root / "images"
    annotation_root = root / "annotations"
    for directory in (image_root, annotation_root):
        if directory.is_symlink() or not directory.is_dir():
            return False
        entries = list(directory.iterdir())
        if any(entry.is_symlink() for entry in entries):
            return False
        if {entry.name for entry in entries if entry.is_dir()} != set(video_ids):
            return False

    total_masklets = 0
    total_masks = 0
    for video_id in video_ids:
        images = _files_by_stem(image_root / video_id, {".jpg"}, reject_symlinks=True)
        if not images:
            return False
        if any(SAV_FRAME_STEM_PATTERN.fullmatch(stem) is None for stem in images):
            return False

        annotation_dir = annotation_root / video_id
        object_dirs = sorted(
            entry for entry in annotation_dir.iterdir() if entry.is_dir()
        )
        if not object_dirs or any(entry.is_symlink() for entry in object_dirs):
            return False
        if any(
            SAV_OBJECT_ID_PATTERN.fullmatch(entry.name) is None for entry in object_dirs
        ):
            return False
        total_masklets += len(object_dirs)

        # Paired into one variable (rather than two, set-together but separately
        # declared) so the invariant that they are always assigned on the same
        # branch is visible to a reader and provable by static analysis, not
        # just true by construction (object_dirs and masks are both checked
        # non-empty above, so the loop body always reaches this assignment on
        # its first iteration if it does not already return False).
        first_pair: tuple[Path, Path] | None = None
        for object_dir in object_dirs:
            masks = _files_by_stem(object_dir, {".png"}, reject_symlinks=True)
            if not masks or not set(masks) <= set(images):
                return False
            total_masks += len(masks)
            if first_pair is None:
                first_stem = sorted(masks)[0]
                first_pair = (masks[first_stem], images[first_stem])
        assert first_pair is not None
        first_mask_path, first_image_path = first_pair
        try:
            with Image.open(first_image_path) as image:
                image.load()
                image_shape = (image.height, image.width)
            with Image.open(first_mask_path) as mask_image:
                mask = np.asarray(mask_image)
        except OSError:
            return False
        # Counting unique values alone would accept a two-valued mask such as
        # {1, 2}, which CustomSAV's `> 0` binarization turns entirely into
        # foreground; require every non-zero value to be a single object ID.
        if (
            mask.ndim != 2
            or mask.shape != image_shape
            or len(set(np.unique(mask).tolist()) - {0}) > 1
        ):
            return False

    return (
        total_masklets == SAV_VALIDATION_MASKLET_COUNT
        and total_masks == SAV_VALIDATION_MASK_COUNT
    )


def dataset_ready(data_path: str | Path, task: str, dataset: str | None = None) -> bool:
    """Return whether an organized dataset matches its task, taxonomy, and full validation split.

    Args:
        data_path: Organized dataset root.
        task: Canonical vision task.
        dataset: Optional validation taxonomy.

    Returns:
        Whether the dataset has the expected identity, metadata, and sample count.
    """

    root = Path(data_path).expanduser()
    normalized_task = normalize_vision_task(task)
    expected_dataset = {
        "image_classification": "imagenet",
        "object_detection": "coco",
        "instance_segmentation": "coco",
        "pose_estimation": "coco",
        "face_detection": "widerface",
        "obb": "dotav1",
        "depth_estimation": "nyu-depth",
        "mask_generation": "sa-v",
    }.get(normalized_task)
    normalized_dataset = (dataset or expected_dataset or "").lower()

    if normalized_task == "semantic_segmentation":
        return dense_dataset_ready(root, normalized_dataset or "ade20k")
    if expected_dataset is None or normalized_dataset != expected_dataset:
        return False
    if normalized_task == "image_classification":
        return _imagenet_ready(root)
    if normalized_task in {
        "object_detection",
        "instance_segmentation",
        "pose_estimation",
    }:
        return _coco_ready(root, normalized_task)
    if normalized_task == "face_detection":
        return _widerface_ready(root)
    if normalized_task == "obb":
        return _dotav1_ready(root)
    if normalized_task == "depth_estimation":
        return dense_dataset_ready(root, normalized_dataset)
    if normalized_task == "mask_generation":
        return _sav_ready(root)
    return False

"""
Utilities for organizing datasets.
"""

from __future__ import annotations

import concurrent.futures
import cv2
import hashlib
import math
import os
import re
import shutil
import stat
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from time import sleep
from typing import Protocol, TypeGuard
from urllib.parse import urlparse

import requests
import numpy as np
from gdown.download import download
from gdown.download_folder import download_folder
from PIL import Image
from tqdm import tqdm

from ...datasets import get_dataset_config
from .readiness import (
    ADE20K_METADATA_FILES,
    ADE20K_VALIDATION_SAMPLE_COUNT,
    CITYSCAPES_SAMPLE_ID_PATTERN,
    CITYSCAPES_VALIDATION_SAMPLE_COUNT,
    DOTAV1_VALIDATION_SAMPLE_COUNT,
    NYU_DEPTH_VALIDATION_SAMPLE_COUNT,
    _path_has_symlink_component,
    dataset_ready,
)

DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024
DOWNLOAD_RETRY_LIMIT = 4
DOWNLOAD_RETRY_BACKOFF_SECONDS = 2.0
DOWNLOAD_TIMEOUT = (10, 30)
DOTAV1_DOWNLOAD_CONFIG = get_dataset_config("dotav1")["download"]
DOTAV1_GOOGLE_DRIVE_ARCHIVES = {
    DOTAV1_DOWNLOAD_CONFIG["images_archive"],
    DOTAV1_DOWNLOAD_CONFIG["labels_archive"],
}
DOTAV1_CLASS_TO_IDX = {
    name: int(index) for index, name in get_dataset_config("dotav1")["names"].items()
}
COCO_DOWNLOAD_CONFIG = get_dataset_config("coco")["download"]
ADE20K_DOWNLOAD_CONFIG = get_dataset_config("ade20k")["download"]
NYU_DEPTH_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip"
)
ADE20K_URL = ADE20K_DOWNLOAD_CONFIG["url"]
CITYSCAPES_IMAGE_SUFFIX = "_leftImg8bit.png"
CITYSCAPES_ANNOTATION_SUFFIX = "_gtFine_labelIds.png"
IMAGENET_SYNSET_PATTERN = re.compile(r"n\d{8}")
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429})
CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
UNSATISFIABLE_CONTENT_RANGE_PATTERN = re.compile(r"^bytes \*/(\d+)$")
PINNED_ARCHIVE_SHA256 = {
    COCO_DOWNLOAD_CONFIG["images"]: COCO_DOWNLOAD_CONFIG["images_sha256"],
    COCO_DOWNLOAD_CONFIG["annotations"]: COCO_DOWNLOAD_CONFIG["annotations_sha256"],
    ADE20K_DOWNLOAD_CONFIG["url"]: ADE20K_DOWNLOAD_CONFIG["sha256"],
}


def _resolve_organizer_output_dir(output_dir: str | None, dataset_name: str) -> str:
    """Return an explicit output directory or the lazily resolved artifact cache."""

    if output_dir is not None:
        return os.path.expanduser(output_dir)
    from mblt_vision.wrapper import get_mobilint_cache_dir

    return os.path.join(get_mobilint_cache_dir(), "datasets", dataset_name)


def _replace_staged_directories(
    replacements: Iterable[tuple[str, str]],
    output_parent_dir: str,
    backup_prefix: str,
) -> None:
    """Atomically install staged directories while preserving failed rollback backups.

    Args:
        replacements: Pairs of staged and destination directories.
        output_parent_dir: Parent directory where the backup directory is created.
        backup_prefix: Prefix identifying the temporary backup directory.

    Raises:
        OSError: If installation or rollback fails. A failed rollback leaves its
            backup directory in place and includes its path in the error.
    """

    replacement_list = list(replacements)
    backup_dir = mkdtemp(dir=output_parent_dir, prefix=backup_prefix)
    backups: dict[str, str] = {}
    installed_dirs: list[str] = []
    try:
        for _, destination_dir in replacement_list:
            if os.path.lexists(destination_dir):
                backup_path = os.path.join(
                    backup_dir, os.path.basename(destination_dir)
                )
                os.replace(destination_dir, backup_path)
                backups[destination_dir] = backup_path
        for staged_dir, destination_dir in replacement_list:
            os.makedirs(os.path.dirname(destination_dir), exist_ok=True)
            os.replace(staged_dir, destination_dir)
            installed_dirs.append(destination_dir)
    except OSError:
        try:
            for directory in installed_dirs:
                if os.path.isdir(directory) and not os.path.islink(directory):
                    shutil.rmtree(directory)
                elif os.path.lexists(directory):
                    os.remove(directory)
            for destination_dir, backup_path in backups.items():
                os.makedirs(os.path.dirname(destination_dir), exist_ok=True)
                os.replace(backup_path, destination_dir)
        except OSError as rollback_error:
            raise OSError(
                f"Dataset installation rollback failed; backups are preserved at {backup_dir}."
            ) from rollback_error
        shutil.rmtree(backup_dir)
        raise
    shutil.rmtree(backup_dir)


def _validate_staged_dataset(
    staged_output_dir: str,
    dataset: str,
    tasks: Iterable[str],
) -> None:
    """Validate a complete staged dataset before replacing its managed cache.

    Args:
        staged_output_dir: Root of the staged organized dataset.
        dataset: Validation dataset taxonomy.
        tasks: Tasks whose required metadata and files must all be ready.

    Raises:
        ValueError: If the staged dataset is incomplete or has mismatched identity.
    """

    if not all(dataset_ready(staged_output_dir, task, dataset) for task in tasks):
        raise ValueError(
            f"Staged {dataset} validation dataset is incomplete or has mismatched metadata; "
            "the existing dataset cache was not replaced."
        )
    _validate_staged_payloads(Path(staged_output_dir), dataset)


def _validate_staged_payloads(staged_root: Path, dataset: str) -> None:
    """Decode staged data files before a structurally valid cache is replaced."""

    image_roots = {
        "imagenet": (staged_root,),
        "coco": (staged_root / "val2017",),
        "widerface": (staged_root / "images",),
        "dotav1": (staged_root / "images",),
        "ade20k": (staged_root / "images",),
        "cityscapes": (staged_root / "images",),
    }
    for image_root in image_roots.get(dataset, ()):
        for image_path in image_root.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in {
                ".bmp",
                ".jpeg",
                ".jpg",
                ".png",
                ".tif",
                ".tiff",
                ".webp",
            }:
                if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                    raise ValueError(
                        f"Staged {dataset} image is unreadable: {image_path}."
                    )

    if dataset in {"ade20k", "cityscapes"}:
        _validate_staged_semantic_masks(staged_root, dataset)
    elif dataset == "dotav1":
        _validate_staged_dotav1_labels(staged_root)


def _validate_staged_semantic_masks(staged_root: Path, dataset: str) -> None:
    """Validate decoded semantic targets against their paired staged images."""

    image_dir = staged_root / "images"
    annotation_dir = staged_root / "annotations"
    for annotation_path in sorted(annotation_dir.glob("*.png")):
        image_path = next(
            (
                candidate
                for candidate in image_dir.glob(f"{annotation_path.stem}.*")
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            None,
        )
        if image_path is None:
            raise ValueError(
                f"Staged {dataset} target has no paired image: {annotation_path}."
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        try:
            with Image.open(annotation_path) as annotation_image:
                annotation = np.asarray(annotation_image)
        except OSError as exc:
            raise ValueError(
                f"Staged {dataset} annotation is unreadable: {annotation_path}."
            ) from exc
        if dataset == "cityscapes" and annotation.ndim == 3:
            if (
                annotation.shape[2] not in {3, 4}
                or not np.array_equal(annotation[..., 0], annotation[..., 1])
                or not np.array_equal(annotation[..., 0], annotation[..., 2])
            ):
                raise ValueError(
                    f"Staged Cityscapes annotation must be grayscale or RGB-grayscale: {annotation_path}."
                )
            annotation = annotation[..., 0]
        if image is None or annotation.ndim != 2 or annotation.shape != image.shape[:2]:
            raise ValueError(
                f"Staged {dataset} image and annotation geometry is invalid: {annotation_path}."
            )
        if dataset == "ade20k" and (
            annotation.dtype != np.uint8
            or (annotation.size and int(annotation.max()) > 150)
        ):
            raise ValueError(
                f"Staged ADE20K annotation must be an 8-bit mask with values in [0, 150]: {annotation_path}."
            )


def _validate_staged_dotav1_labels(staged_root: Path) -> None:
    """Validate both DOTAv1 label representations before cache replacement."""

    label_dirs = {
        "normalized": staged_root / "labels" / "val",
        "original": staged_root / "labels" / "val_original",
    }
    valid_indices = set(DOTAV1_CLASS_TO_IDX.values())
    for kind, label_dir in label_dirs.items():
        for label_path in sorted(label_dir.glob("*.txt")):
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if not fields or (
                    kind == "original"
                    and (
                        fields[0].startswith("imagesource:")
                        or fields[0].startswith("gsd:")
                    )
                ):
                    continue
                min_fields = 9 if kind == "normalized" else 10
                if len(fields) < min_fields:
                    raise ValueError(
                        f"Malformed staged DOTAv1 {kind} annotation at "
                        f"{label_path}:{line_number}: expected at least {min_fields} fields."
                    )
                try:
                    coordinates = [
                        float(value)
                        for value in (
                            fields[1:9] if kind == "normalized" else fields[:8]
                        )
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed staged DOTAv1 coordinates at {label_path}:{line_number}."
                    ) from exc
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(
                        f"Staged DOTAv1 coordinates must be finite at {label_path}:{line_number}."
                    )
                points = np.asarray(coordinates, dtype=np.float64).reshape(4, 2)
                signed_double_area = np.dot(
                    points[:, 0], np.roll(points[:, 1], -1)
                ) - np.dot(points[:, 1], np.roll(points[:, 0], -1))
                if abs(signed_double_area) <= 0:
                    raise ValueError(
                        f"Staged DOTAv1 polygon must have positive area at {label_path}:{line_number}."
                    )
                if kind == "normalized":
                    try:
                        class_index = int(fields[0])
                    except ValueError as exc:
                        raise ValueError(
                            f"Malformed staged DOTAv1 class index at {label_path}:{line_number}."
                        ) from exc
                    difficulty = fields[9] if len(fields) >= 10 else "0"
                    if class_index not in valid_indices:
                        raise ValueError(
                            f"Unsupported staged DOTAv1 class index at {label_path}:{line_number}."
                        )
                else:
                    difficulty = fields[9]
                    if fields[8] not in DOTAV1_CLASS_TO_IDX:
                        raise ValueError(
                            f"Unsupported staged DOTAv1 class at {label_path}:{line_number}."
                        )
                if difficulty not in {"0", "1", "2"}:
                    raise ValueError(
                        f"Unsupported staged DOTAv1 difficulty flag at {label_path}:{line_number}."
                    )


class _GoogleDriveDownloadEntry(Protocol):
    """The public attributes needed from a gdown folder-listing entry."""

    id: str
    path: str


def _is_google_drive_download_entry(
    value: object,
) -> TypeGuard[_GoogleDriveDownloadEntry]:
    """Returns whether a folder-listing value has the Google Drive file attributes needed here."""

    return isinstance(getattr(value, "id", None), str) and isinstance(
        getattr(value, "path", None), str
    )


def _is_url(path_or_url: str) -> bool:
    """Returns whether the given string looks like an HTTP(S) URL."""
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _verify_archive_sha256(
    archive_path: str, expected_sha256: str, source_url: str
) -> None:
    """Verify a downloaded archive before it can be extracted."""

    digest = hashlib.sha256()
    with open(archive_path, "rb") as archive:
        for chunk in iter(lambda: archive.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        Path(archive_path).unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded archive SHA-256 mismatch for {source_url}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def _has_expected_resume_offset(content_range: str | None, existing_size: int) -> bool:
    """Return whether a partial response begins exactly after local archive bytes."""

    if content_range is None:
        return False
    match = CONTENT_RANGE_PATTERN.fullmatch(content_range)
    if match is None:
        return False
    start, end, total = match.groups()
    return (
        int(start) == existing_size
        and int(end) >= int(start)
        and (total == "*" or int(end) < int(total))
    )


def _is_completed_range_response(content_range: str | None, existing_size: int) -> bool:
    """Return whether a 416 confirms that the local archive is complete."""

    if content_range is None:
        return False
    match = UNSATISFIABLE_CONTENT_RANGE_PATTERN.fullmatch(content_range)
    return match is not None and int(match.group(1)) == existing_size


def _restart_partial_download(local_path: str, url: str) -> None:
    """Discard an invalid partial archive before retrying from byte zero."""

    Path(local_path).unlink(missing_ok=True)
    print(
        f"Server returned an invalid resume response for {os.path.basename(local_path)}; "
        "restarting from byte zero."
    )


def _download_url(url: str, local_path: str, expected_sha256: str | None = None) -> str:
    """Downloads a URL to a local file with progress and resume support.

    Args:
        url: HTTP(S) URL to download.
        local_path: Destination file path.
        expected_sha256: Optional pinned SHA-256 digest to verify before return.

    Returns:
        The local destination path.

    Raises:
        RuntimeError: If all download attempts fail.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, DOWNLOAD_RETRY_LIMIT + 1):
        existing_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        headers: dict[str, str] = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            with requests.get(
                url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers
            ) as response:
                if response.status_code == 416 and existing_size > 0:
                    if _is_completed_range_response(
                        response.headers.get("Content-Range"), existing_size
                    ):
                        if expected_sha256 is not None:
                            _verify_archive_sha256(local_path, expected_sha256, url)
                        return local_path
                    _restart_partial_download(local_path, url)
                    continue
                response.raise_for_status()

                if response.status_code == 200 and existing_size > 0:
                    existing_size = 0
                    mode = "wb"
                elif existing_size > 0 and (
                    response.status_code != 206
                    or not _has_expected_resume_offset(
                        response.headers.get("Content-Range"), existing_size
                    )
                ):
                    _restart_partial_download(local_path, url)
                    continue

                total_size = response.headers.get("Content-Length")
                total_bytes = (
                    existing_size + int(total_size) if total_size is not None else None
                )

                desc = f"Downloading {os.path.basename(local_path)}"
                with tqdm(
                    total=total_bytes,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as pbar:
                    with open(local_path, mode) as file_obj:
                        for chunk in response.iter_content(
                            chunk_size=DOWNLOAD_CHUNK_SIZE
                        ):
                            if not chunk:
                                continue
                            file_obj.write(chunk)
                            pbar.update(len(chunk))
            if expected_sha256 is not None:
                _verify_archive_sha256(local_path, expected_sha256, url)
            return local_path
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.HTTPError,
        ) as exc:
            if isinstance(exc, requests.HTTPError):
                status_code = getattr(exc.response, "status_code", None)
                if not isinstance(status_code, int) or (
                    status_code not in RETRYABLE_HTTP_STATUS_CODES
                    and not 500 <= status_code < 600
                ):
                    raise
            if attempt == DOWNLOAD_RETRY_LIMIT:
                raise RuntimeError(
                    f"Failed to download {url} after {DOWNLOAD_RETRY_LIMIT} attempts."
                ) from exc
            resumed_size = (
                os.path.getsize(local_path) if os.path.exists(local_path) else 0
            )
            print(
                f"Download attempt failed for {os.path.basename(local_path)}; "
                f"retrying from {resumed_size} bytes (attempt {attempt + 1}/{DOWNLOAD_RETRY_LIMIT})..."
            )
            sleep(DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Failed to download {url} after {DOWNLOAD_RETRY_LIMIT} attempts."
    )


def _should_download_serially(path_or_urls: list[str]) -> bool:
    """Returns whether URL inputs should be downloaded one by one.

    Dataset hosts such as ImageNet often throttle concurrent archive downloads
    from the same origin. Serializing same-host downloads is slower in the best
    case, but much more stable for the large validation archives used here.
    """

    hosts = [
        urlparse(path_or_url).netloc
        for path_or_url in path_or_urls
        if _is_url(path_or_url)
    ]
    return len(hosts) > 1 and len(set(hosts)) == 1


def _download_if_url(path_or_url: str, download_dir: str) -> str:
    """Downloads a remote dataset archive when needed.

    Args:
        path_or_url: Local path or HTTP(S) URL pointing to a dataset archive.
        download_dir: Directory to store downloaded archives.

    Returns:
        A local filesystem path to the archive or directory.

    Raises:
        ValueError: If the URL path does not contain a filename.
    """
    if not _is_url(path_or_url):
        return path_or_url

    parsed = urlparse(path_or_url)
    if parsed.scheme != "https":
        raise ValueError(
            "Dataset archive URLs must use HTTPS. Download the archive locally "
            f"and provide its path instead: {path_or_url}"
        )
    filename = os.path.basename(parsed.path)
    if not filename:
        raise ValueError(f"Unable to determine a filename from URL: {path_or_url}")

    local_path = os.path.join(download_dir, filename)
    print(f"Downloading dataset archive from {path_or_url} to {local_path}...")
    _download_url(
        path_or_url,
        local_path,
        expected_sha256=PINNED_ARCHIVE_SHA256.get(path_or_url),
    )
    print("Download completed")
    return local_path


def _resolve_source(path_or_url: str, download_dir: str) -> str:
    """Resolves a local path for a dataset source."""

    return _download_if_url(path_or_url, download_dir)


def _resolve_sources(path_or_urls: list[str], download_dir: str) -> list[str]:
    """Resolves multiple dataset sources, downloading URL inputs in parallel."""

    if _should_download_serially(path_or_urls):
        return [
            _resolve_source(path_or_url, download_dir) for path_or_url in path_or_urls
        ]

    local_paths: list[str | None] = [None] * len(path_or_urls)
    futures: dict[concurrent.futures.Future[str], int] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, len(path_or_urls))
    ) as executor:
        for idx, path_or_url in enumerate(path_or_urls):
            if _is_url(path_or_url):
                futures[executor.submit(_resolve_source, path_or_url, download_dir)] = (
                    idx
                )
            else:
                local_paths[idx] = path_or_url

        for future in concurrent.futures.as_completed(futures):
            local_paths[futures[future]] = future.result()

    return [path for path in local_paths if path is not None]


def _get_object_name(obj: ET.Element, xml_file: str) -> str:
    """Extracts a non-empty object name from an ImageNet annotation node.

    Args:
        obj: XML ``object`` element from an annotation file.
        xml_file: Source XML filename used for error context.

    Returns:
        The validated object name.

    Raises:
        ValueError: If the object name node is missing or empty.
    """
    name_element = obj.find("name")
    if name_element is None or name_element.text is None:
        raise ValueError(f"XML file {xml_file} has an object without a valid name")

    object_name = name_element.text.strip()
    if not object_name:
        raise ValueError(f"XML file {xml_file} has an object with an empty name")
    if IMAGENET_SYNSET_PATTERN.fullmatch(object_name) is None:
        raise ValueError(
            f"XML file {xml_file} has invalid ImageNet synset name {object_name!r}; "
            "expected n########."
        )

    return object_name


def _imagenet_class_output_dir(staged_output_dir: str, object_name: str) -> Path:
    """Return a containment-checked class directory below the staging root."""

    staged_root = Path(staged_output_dir).resolve()
    class_dir = (staged_root / object_name).resolve()
    if class_dir.parent != staged_root:
        raise ValueError(
            f"ImageNet class directory escapes staging root: {object_name!r}."
        )
    return class_dir


def construct_imagenet(image_dir: str, xml_dir: str, output_dir: str) -> None:
    """Constructs the ImageNet dataset by organizing images into category folders.

    Args:
        image_dir (str): Directory containing the ImageNet validation images.
        xml_dir (str): Directory containing the ImageNet bounding box XML files.
        output_dir (str): Directory where the organized dataset will be stored.

    Raises:
        ValueError: If an XML file has no objects or contains multiple object names.
        ValueError: If the number of XML files and images do not match.
    """

    xml_count = len(os.listdir(xml_dir + "/val"))
    image_count = len(os.listdir(image_dir))
    if xml_count != image_count:
        raise ValueError(
            f"Number of XML and image files do not match: {xml_count} != {image_count}."
        )

    # validate the XML files
    pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Validating XML files")
    for xml_file in pbar:
        xml_path = os.path.join(xml_dir + "/val", xml_file)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()

        if len(root.findall("object")) < 1:
            raise ValueError(
                f"XML file {xml_file} has no object, but expected at least 1"
            )

        # check whether the object names in the XML files are the same
        object_names = [
            _get_object_name(obj, xml_file) for obj in root.findall("object")
        ]
        if len(set(object_names)) != 1:
            raise ValueError(
                f"Object names in XML file {xml_file} are not the same. "
                f"It has {len(set(object_names))} different object names."
            )

    pbar.close()

    output_dir = os.path.abspath(output_dir)
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".imagenet-staging-"
    ) as staging_dir:
        staged_output_dir = os.path.join(staging_dir, "imagenet")

        # construct the ImageNet dataset
        pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Constructing ImageNet dataset")
        for xml_file in pbar:
            xml_path = os.path.join(xml_dir + "/val", xml_file)
            xml_tree = ET.parse(xml_path)
            root = xml_tree.getroot()
            object_name = _get_object_name(root.findall("object")[0], xml_file)
            image_path = os.path.join(image_dir, xml_file.replace(".xml", ".JPEG"))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            class_dir = _imagenet_class_output_dir(staged_output_dir, object_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(
                image_path,
                class_dir / os.path.basename(image_path),
            )
        pbar.close()

        # validate the staged ImageNet dataset before replacing the managed output root
        pbar = tqdm(os.listdir(staged_output_dir), desc="Validating ImageNet dataset")
        print(f"Number of categories: {len(os.listdir(staged_output_dir))}")
        for object_name in pbar:
            num_images = len(os.listdir(os.path.join(staged_output_dir, object_name)))
            if num_images != 50:
                raise ValueError(
                    f"Object {object_name} has {num_images} images, but expected 50"
                )
        pbar.close()
        _validate_staged_dataset(
            staged_output_dir, "imagenet", ("image_classification",)
        )
        _replace_staged_directories(
            ((staged_output_dir, output_dir),),
            output_parent_dir,
            ".imagenet-backup-",
        )
    print("Each category has 50 images")
    print("ImageNet dataset constructed successfully")


def organize_imagenet(
    image_dir: str,
    xml_dir: str,
    output_dir: str | None = None,
) -> None:
    """Organizes the ImageNet dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image directory or archive (.tar).
        xml_dir (str): Path or URL to the XML directory or archive (.tgz).
        output_dir: Directory to store the organized dataset. Defaults to the
            resolved Mobilint cache directory.
    """
    output_dir = _resolve_organizer_output_dir(output_dir, "imagenet")
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_xml_dir = _resolve_sources(
            [image_dir, xml_dir], temp_dir
        )

        if local_image_dir.endswith(".tar") and local_xml_dir.endswith(".tgz"):
            print("Unpacking image and XML files to temporary directory...")
            _safe_unpack_archive(
                local_image_dir, os.path.join(temp_dir, "ILSVRC2012_img_val")
            )
            _safe_unpack_archive(
                local_xml_dir, os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3")
            )
            print("Unpacking completed")
            construct_imagenet(
                os.path.join(temp_dir, "ILSVRC2012_img_val"),
                os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3"),
                output_dir,
            )
            return

        construct_imagenet(local_image_dir, local_xml_dir, output_dir)


def construct_coco(image_dir: str, annotation_dir: str, output_dir: str) -> None:
    """Constructs the COCO dataset by copying images and annotations to a target directory.

    Args:
        image_dir (str): Directory containing COCO images.
        annotation_dir (str): Directory containing COCO annotations.
        output_dir (str): Directory where the organized dataset will be stored.
    """
    print(
        f"Constructing COCO dataset from {image_dir} and {annotation_dir} to {output_dir}"
    )
    output_dir = os.path.abspath(output_dir)
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".coco-staging-"
    ) as staging_dir:
        staged_output_dir = os.path.join(staging_dir, "coco")
        shutil.copytree(image_dir, os.path.join(staged_output_dir, "val2017"))
        for file in os.listdir(os.path.join(annotation_dir, "annotations")):
            if file.endswith("_val2017.json"):
                shutil.copy(
                    os.path.join(annotation_dir, "annotations", file),
                    os.path.join(staged_output_dir, file),
                )
        _validate_staged_dataset(
            staged_output_dir,
            "coco",
            ("object_detection", "pose_estimation"),
        )
        _replace_staged_directories(
            ((staged_output_dir, output_dir),),
            output_parent_dir,
            ".coco-backup-",
        )
    print("Constructing COCO dataset completed")


def organize_coco(
    image_dir: str,
    annotation_dir: str,
    output_dir: str | None = None,
) -> None:
    """Organizes the COCO dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image zip file or directory.
        annotation_dir (str): Path or URL to the annotation zip file or directory.
        output_dir: Directory to store the organized dataset. Defaults to the
            resolved Mobilint cache directory.
    """
    output_dir = _resolve_organizer_output_dir(output_dir, "coco")
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_annotation_dir = _resolve_sources(
            [image_dir, annotation_dir], temp_dir
        )

        if local_image_dir.endswith(".zip") and local_annotation_dir.endswith(".zip"):
            print("Unpacking image and annotation files to temporary directory...")
            _safe_unpack_archive(local_image_dir, temp_dir)
            _safe_unpack_archive(
                local_annotation_dir, os.path.join(temp_dir, "annotations_trainval2017")
            )
            print("Unpacking completed")
            construct_coco(
                os.path.join(temp_dir, "val2017"),
                os.path.join(temp_dir, "annotations_trainval2017"),
                output_dir,
            )
            return

        construct_coco(local_image_dir, local_annotation_dir, output_dir)


def construct_widerface(image_dir: str, annotation_dir: str, output_dir: str) -> None:
    """Constructs the WiderFace dataset by copying images and annotations to a target directory.

    Args:
        image_dir (str): Directory containing WiderFace images.
        annotation_dir (str): Directory containing WiderFace annotations.
        output_dir (str): Directory where the organized dataset will be stored.
    """
    print(
        f"Constructing WiderFace dataset from {image_dir} and {annotation_dir} to {output_dir}"
    )
    output_dir = os.path.abspath(output_dir)
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".widerface-staging-"
    ) as staging_dir:
        staged_output_dir = os.path.join(staging_dir, "widerface")
        shutil.copytree(
            os.path.join(image_dir, "images"),
            os.path.join(staged_output_dir, "images"),
        )
        for file in os.listdir(annotation_dir):
            if "_val" in file:
                shutil.copy(os.path.join(annotation_dir, file), staged_output_dir)
        _validate_staged_dataset(staged_output_dir, "widerface", ("face_detection",))
        _replace_staged_directories(
            ((staged_output_dir, output_dir),),
            output_parent_dir,
            ".widerface-backup-",
        )
    print("Constructing WiderFace dataset completed")


def organize_widerface(
    image_dir: str,
    annotation_dir: str,
    output_dir: str | None = None,
) -> None:
    """Organizes the WiderFace dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image zip file or directory.
        annotation_dir (str): Path or URL to the annotation zip file or directory.
        output_dir: Directory to store the organized dataset. Defaults to the
            resolved Mobilint cache directory.
    """
    output_dir = _resolve_organizer_output_dir(output_dir, "widerface")
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_annotation_dir = _resolve_sources(
            [image_dir, annotation_dir], temp_dir
        )

        if local_image_dir.endswith(".zip") and local_annotation_dir.endswith(".zip"):
            print("Unpacking image and annotation files to temporary directory...")
            _safe_unpack_archive(local_image_dir, temp_dir)
            _safe_unpack_archive(local_annotation_dir, temp_dir)
            print("Unpacking completed")
            construct_widerface(
                os.path.join(temp_dir, "WIDER_val"),
                os.path.join(temp_dir, "wider_face_split"),
                output_dir,
            )
            return

        construct_widerface(local_image_dir, local_annotation_dir, output_dir)


def _resolve_nyu_depth_validation_dirs(dataset_dir: str) -> tuple[str, str, str]:
    """Resolves NYU Depth validation image and depth directories.

    Args:
        dataset_dir: Directory containing the NYU Depth root or its parent.

    Returns:
        Paths to the selected dataset root, validation image directory, and
        validation depth directory.

    Raises:
        ValueError: If the expected NYU Depth layout is not present.
    """

    roots = (os.path.join(dataset_dir, "nyu-depth"), dataset_dir)
    for root in roots:
        candidates = (
            (os.path.join(root, "images", "val"), os.path.join(root, "depth", "val")),
            (os.path.join(root, "val", "images"), os.path.join(root, "val", "depth")),
            (os.path.join(root, "images"), os.path.join(root, "depth")),
        )
        for image_dir, depth_dir in candidates:
            if os.path.isdir(image_dir) and os.path.isdir(depth_dir):
                return root, image_dir, depth_dir
    raise ValueError(
        f"NYU Depth dataset must contain matching images/ and depth/ directories: {dataset_dir}"
    )


def _validate_dense_source_file(source_path: str, dataset_root: Path) -> str:
    """Resolve a non-symlink regular file contained by a dense dataset root.

    Args:
        source_path: Candidate data or metadata file.
        dataset_root: Resolved root of the extracted dataset.

    Returns:
        Resolved source path safe to copy.

    Raises:
        ValueError: If the source is a symlink, is not a regular file, cannot be
            resolved, or escapes the dataset root.
    """

    source = Path(source_path)
    if source.is_symlink():
        raise ValueError(f"Dense dataset source file must not be a symlink: {source}.")
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Unable to resolve dense dataset source file {source}: {exc}."
        ) from exc
    if not resolved_source.is_file():
        raise ValueError(f"Dense dataset source must be a regular file: {source}.")
    if not resolved_source.is_relative_to(dataset_root):
        raise ValueError(
            f"Dense dataset source must remain within dataset root: {source}."
        )
    return str(resolved_source)


def _validate_dense_output_root(
    output_dir: str,
    dataset_name: str,
    layout_names: Iterable[str],
) -> str:
    """Reject symlinks in a dense managed root before organization.

    Args:
        output_dir: Requested managed dataset root.
        dataset_name: Human-readable dataset name for error reporting.
        layout_names: Dataset-specific directories managed below the root.

    Returns:
        Expanded absolute output path.

    Raises:
        ValueError: If the managed root, an ancestor, or a managed layout
            directory is a symlink.
    """

    requested_path = Path(output_dir).expanduser()
    output_path = Path(os.path.abspath(requested_path))
    if _path_has_symlink_component(requested_path):
        raise ValueError(
            f"{dataset_name} output directory and its existing parents must not be symlinks: {output_path}. "
            "Remove the symlink or choose a path beneath regular directories."
        )
    for layout_name in layout_names:
        layout_path = output_path / layout_name
        if layout_path.is_symlink():
            raise ValueError(
                f"{dataset_name} output layout directories must not be symlinks: {layout_path}. "
                "Remove the symlink or choose a different output directory."
            )
    return str(output_path)


def _collect_nyu_depth_validation_files(
    image_dir: str,
    depth_dir: str,
    dataset_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validates and returns matching NYU Depth validation image/depth pairs."""

    images = {
        os.path.splitext(os.path.basename(path))[0]: _validate_dense_source_file(
            path, dataset_root
        )
        for path in _iter_files(image_dir, [".jpg", ".jpeg", ".png"])
    }
    depths = {
        os.path.splitext(os.path.basename(path))[0]: _validate_dense_source_file(
            path, dataset_root
        )
        for path in _iter_files(depth_dir, [".npy"])
    }
    missing_depths = sorted(set(images) - set(depths))
    missing_images = sorted(set(depths) - set(images))
    if missing_depths or missing_images:
        details = []
        if missing_depths:
            details.append(
                f"images without depth maps: {', '.join(missing_depths[:5])}"
            )
        if missing_images:
            details.append(
                f"depth maps without images: {', '.join(missing_images[:5])}"
            )
        raise ValueError(
            f"NYU Depth validation image/depth mismatch ({'; '.join(details)})."
        )
    if len(images) != NYU_DEPTH_VALIDATION_SAMPLE_COUNT:
        raise ValueError(
            "NYU Depth validation dataset must contain "
            f"{NYU_DEPTH_VALIDATION_SAMPLE_COUNT} matching image/depth pairs, found {len(images)}."
        )
    return images, depths


def construct_nyu_depth(dataset_dir: str, output_dir: str) -> None:
    """Constructs the NYU Depth layout from an extracted dataset directory.

    Args:
        dataset_dir: Directory containing the NYU Depth root or its parent.
        output_dir: Directory where the organized dataset will be stored.
    """

    output_dir = _validate_dense_output_root(
        output_dir, "NYU Depth", ("images", "depth")
    )
    selected_root, image_dir, depth_dir = _resolve_nyu_depth_validation_dirs(
        dataset_dir
    )
    try:
        dataset_root = Path(selected_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Unable to resolve NYU Depth dataset root {selected_root}: {exc}."
        ) from exc
    images, depths = _collect_nyu_depth_validation_files(
        image_dir, depth_dir, dataset_root
    )
    print(
        f"Constructing NYU Depth validation dataset from {dataset_dir} to {output_dir}"
    )

    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".nyu-depth-staging-"
    ) as staging_dir:
        staged_image_dir = os.path.join(staging_dir, "images")
        staged_depth_dir = os.path.join(staging_dir, "depth")
        os.makedirs(staged_image_dir)
        os.makedirs(staged_depth_dir)
        for sample_id in sorted(images):
            shutil.copy2(
                images[sample_id],
                os.path.join(staged_image_dir, os.path.basename(images[sample_id])),
            )
            shutil.copy2(
                depths[sample_id],
                os.path.join(staged_depth_dir, os.path.basename(depths[sample_id])),
            )

        _validate_staged_nyu_depth(staging_dir)

        replacements = (
            (staged_image_dir, os.path.join(output_dir, "images")),
            (staged_depth_dir, os.path.join(output_dir, "depth")),
        )
        _replace_staged_directories(
            replacements, output_parent_dir, ".nyu-depth-backup-"
        )
    print(
        f"Constructed NYU Depth validation dataset with {len(images)} image/depth pairs"
    )


def _validate_staged_nyu_depth(staging_dir: str) -> None:
    """Decode staged NYU pairs before they can replace an existing cache."""

    image_dir = Path(staging_dir) / "images"
    depth_dir = Path(staging_dir) / "depth"
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        depth_path = depth_dir / f"{image_path.stem}.npy"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Staged NYU Depth image is unreadable: {image_path}.")
        try:
            raw_depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Unable to load staged NYU Depth target {depth_path}: {exc}."
            ) from exc
        if not np.issubdtype(raw_depth.dtype, np.number) or np.issubdtype(
            raw_depth.dtype, np.complexfloating
        ):
            raise ValueError(
                "Staged NYU Depth target must use a real numeric dtype, "
                f"got {raw_depth.dtype}: {depth_path}."
            )
        depth = np.asarray(raw_depth, dtype=np.float32)
        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise ValueError(
                "Staged NYU Depth image and target shapes must match: "
                f"image {image.shape[:2]}, depth {depth.shape}: {image_path}."
            )
        if not bool(np.isfinite(depth).all()):
            raise ValueError(
                f"Staged NYU Depth target must contain only finite values: {depth_path}."
            )
        if bool((depth < 0).any()):
            raise ValueError(
                f"Staged NYU Depth target must not contain negative values: {depth_path}."
            )


def organize_nyu_depth(
    dataset_path: str = NYU_DEPTH_URL,
    output_dir: str | None = None,
) -> None:
    """Organizes NYU Depth, downloading and unpacking an archive when necessary.

    Args:
        dataset_path: Path or URL to the NYU Depth zip file or extracted dataset directory.
        output_dir: Directory to store the organized dataset. Defaults to the
            resolved Mobilint cache directory.
    """

    output_dir = _resolve_organizer_output_dir(output_dir, "nyu-depth")
    output_dir = _validate_dense_output_root(
        output_dir, "NYU Depth", ("images", "depth")
    )
    with TemporaryDirectory() as temp_dir:
        local_dataset_path = _resolve_source(dataset_path, temp_dir)
        if local_dataset_path.endswith(".zip"):
            print("Unpacking NYU Depth files to temporary directory...")
            _safe_unpack_archive(local_dataset_path, temp_dir)
            print("Unpacking completed")
            construct_nyu_depth(temp_dir, output_dir)
            return

        construct_nyu_depth(local_dataset_path, output_dir)


def _resolve_ade20k_validation_dirs(dataset_dir: str) -> tuple[str, str, str]:
    """Resolves the ADE20K root and validation image/mask directories."""

    for root in (os.path.join(dataset_dir, "ADEChallengeData2016"), dataset_dir):
        for image_dir, annotation_dir in (
            (
                os.path.join(root, "images", "validation"),
                os.path.join(root, "annotations", "validation"),
            ),
            (os.path.join(root, "images"), os.path.join(root, "annotations")),
        ):
            if os.path.isdir(image_dir) and os.path.isdir(annotation_dir):
                return root, image_dir, annotation_dir
    raise ValueError(
        f"ADE20K dataset must contain matching images/ and annotations/ directories: {dataset_dir}"
    )


def construct_ade20k(dataset_dir: str, output_dir: str) -> None:
    """Constructs the flat ADE20K validation layout from an extracted dataset.

    Args:
        dataset_dir: Directory containing the ADE20K root or its parent.
        output_dir: Directory where the organized validation dataset will be stored.

    Raises:
        ValueError: If the source does not contain 2,000 matched validation image/mask pairs.
    """

    output_dir = _validate_dense_output_root(
        output_dir, "ADE20K", ("images", "annotations")
    )
    dataset_root, image_dir, annotation_dir = _resolve_ade20k_validation_dirs(
        dataset_dir
    )
    try:
        resolved_dataset_root = Path(dataset_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"Unable to resolve ADE20K dataset root {dataset_root}: {exc}."
        ) from exc
    images = {
        os.path.splitext(file_name)[0]: _validate_dense_source_file(
            os.path.join(image_dir, file_name),
            resolved_dataset_root,
        )
        for file_name in os.listdir(image_dir)
        if file_name.startswith("ADE_val_") and file_name.lower().endswith(".jpg")
    }
    annotations = {
        os.path.splitext(file_name)[0]: _validate_dense_source_file(
            os.path.join(annotation_dir, file_name),
            resolved_dataset_root,
        )
        for file_name in os.listdir(annotation_dir)
        if file_name.startswith("ADE_val_") and file_name.lower().endswith(".png")
    }
    if set(images) != set(annotations):
        raise ValueError(
            "ADE20K validation images and annotations must have matching file stems."
        )
    if len(images) != ADE20K_VALIDATION_SAMPLE_COUNT:
        raise ValueError(
            f"ADE20K validation dataset must contain {ADE20K_VALIDATION_SAMPLE_COUNT} pairs, found {len(images)}."
        )
    metadata: dict[str, str] = {}
    for file_name in ADE20K_METADATA_FILES:
        metadata_path = os.path.join(dataset_root, file_name)
        if not os.path.lexists(metadata_path):
            raise ValueError(
                f"ADE20K dataset is missing required metadata files: {file_name}."
            )
        metadata[file_name] = _validate_dense_source_file(
            metadata_path, resolved_dataset_root
        )

    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".ade20k-staging-"
    ) as staging_dir:
        staged_output_dir = os.path.join(staging_dir, "ade20k")
        staged_image_dir = os.path.join(staged_output_dir, "images")
        staged_annotation_dir = os.path.join(staged_output_dir, "annotations")
        os.makedirs(staged_image_dir)
        os.makedirs(staged_annotation_dir)
        for sample_id in sorted(images):
            shutil.copy2(
                images[sample_id],
                os.path.join(staged_image_dir, os.path.basename(images[sample_id])),
            )
            shutil.copy2(
                annotations[sample_id],
                os.path.join(
                    staged_annotation_dir, os.path.basename(annotations[sample_id])
                ),
            )
        for file_name in ADE20K_METADATA_FILES:
            shutil.copy2(
                metadata[file_name],
                os.path.join(staged_output_dir, file_name),
            )

        _validate_staged_dataset(
            staged_output_dir, "ade20k", ("semantic_segmentation",)
        )
        _replace_staged_directories(
            ((staged_output_dir, output_dir),),
            output_parent_dir,
            ".ade20k-backup-",
        )
    print(f"Constructed ADE20K validation dataset with {len(images)} image/mask pairs")


def organize_ade20k(
    dataset_path: str = ADE20K_URL,
    output_dir: str | None = None,
) -> None:
    """Organizes ADE20K validation data, downloading and unpacking when necessary."""

    output_dir = _resolve_organizer_output_dir(output_dir, "ADEChallengeData2016")
    output_dir = _validate_dense_output_root(
        output_dir, "ADE20K", ("images", "annotations")
    )
    with TemporaryDirectory() as temp_dir:
        local_dataset_path = _resolve_source(dataset_path, temp_dir)
        if local_dataset_path.endswith(".zip"):
            _safe_unpack_archive(local_dataset_path, temp_dir)
            construct_ade20k(temp_dir, output_dir)
            return
        construct_ade20k(local_dataset_path, output_dir)


def _validate_cityscapes_zip(archive_path: str, source_name: str) -> str:
    """Validate one official Cityscapes ZIP source.

    Args:
        archive_path: Path to the raw Cityscapes archive.
        source_name: Human-readable source description for errors.

    Returns:
        Expanded absolute archive path.

    Raises:
        ValueError: If the path is missing, is not a file, is not a ZIP, or contains duplicate members.
    """

    resolved_path = os.path.abspath(os.path.expanduser(archive_path))
    if not os.path.isfile(resolved_path):
        raise ValueError(
            f"Cityscapes {source_name} archive does not exist or is not a file: {resolved_path}."
        )
    if not zipfile.is_zipfile(resolved_path):
        raise ValueError(
            f"Cityscapes {source_name} source must be a valid ZIP archive: {resolved_path}."
        )

    with zipfile.ZipFile(resolved_path) as archive:
        seen_members: set[str] = set()
        duplicate_members: set[str] = set()
        for member in archive.infolist():
            if member.filename in seen_members:
                duplicate_members.add(member.filename)
            seen_members.add(member.filename)
    if duplicate_members:
        raise ValueError(
            f"Cityscapes {source_name} archive contains duplicate members: {', '.join(sorted(duplicate_members)[:3])}."
        )
    return resolved_path


def _collect_cityscapes_validation_files(
    split_dir: str,
    suffix: str,
    source_name: str,
) -> dict[str, str]:
    """Collect official Cityscapes validation files keyed by shared sample ID.

    Args:
        split_dir: Extracted ``leftImg8bit/val`` or ``gtFine/val`` directory.
        suffix: Required official file suffix.
        source_name: Human-readable source description for errors.

    Returns:
        Mapping from ``<city>_<sequence>_<frame>`` to source path.

    Raises:
        ValueError: If a candidate filename is malformed, misplaced, or duplicates an ID.
    """

    files: dict[str, str] = {}
    if not os.path.isdir(split_dir):
        return files

    for current_root, _, file_names in os.walk(split_dir):
        relative_root = os.path.relpath(current_root, split_dir)
        for file_name in file_names:
            if not file_name.endswith(suffix):
                continue
            if relative_root == "." or os.sep in relative_root:
                raise ValueError(
                    f"Malformed Cityscapes {source_name} path: "
                    f"{os.path.relpath(os.path.join(current_root, file_name), split_dir)}."
                )
            sample_id = file_name.removesuffix(suffix)
            match = CITYSCAPES_SAMPLE_ID_PATTERN.fullmatch(sample_id)
            if match is None or match.group("city") != relative_root:
                raise ValueError(
                    f"Malformed Cityscapes {source_name} filename: {file_name}."
                )
            if sample_id in files:
                raise ValueError(
                    f"Duplicate Cityscapes {source_name} sample ID: {sample_id}."
                )
            files[sample_id] = os.path.join(current_root, file_name)
    return files


def organize_cityscapes(
    image_dir: str,
    annotation_dir: str,
    output_dir: str | None = None,
) -> None:
    """Install official Cityscapes validation archives as lossless flat PNG pairs.

    Only validation RGB images and ``gtFine_labelIds`` masks are selected.
    Training, test, and auxiliary annotation files remain excluded.

    Args:
        image_dir: Path to ``leftImg8bit_trainvaltest.zip``.
        annotation_dir: Path to ``gtFine_trainvaltest.zip``.
        output_dir: Directory where the organized validation dataset is stored.
            Defaults to the resolved Mobilint cache directory.

    Raises:
        ValueError: If either source is invalid or does not contain exactly 500 matching pairs.
        OSError: If extraction, copying, or atomic installation fails.
    """

    output_dir = _resolve_organizer_output_dir(output_dir, "cityscapes")
    output_dir = _validate_dense_output_root(
        output_dir, "Cityscapes", ("images", "annotations")
    )
    image_archive = _validate_cityscapes_zip(image_dir, "image")
    annotation_archive = _validate_cityscapes_zip(annotation_dir, "annotation")
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".cityscapes-staging-"
    ) as staging_dir:
        extracted_image_dir = os.path.join(staging_dir, "raw-images")
        extracted_annotation_dir = os.path.join(staging_dir, "raw-annotations")
        _safe_unpack_archive(image_archive, extracted_image_dir)
        _safe_unpack_archive(annotation_archive, extracted_annotation_dir)
        images = _collect_cityscapes_validation_files(
            os.path.join(extracted_image_dir, "leftImg8bit", "val"),
            CITYSCAPES_IMAGE_SUFFIX,
            "image",
        )
        annotations = _collect_cityscapes_validation_files(
            os.path.join(extracted_annotation_dir, "gtFine", "val"),
            CITYSCAPES_ANNOTATION_SUFFIX,
            "annotation",
        )
        missing_annotations = sorted(images.keys() - annotations.keys())
        missing_images = sorted(annotations.keys() - images.keys())
        if missing_annotations or missing_images:
            details = []
            if missing_annotations:
                details.append(
                    f"missing annotations for {', '.join(missing_annotations[:3])}"
                )
            if missing_images:
                details.append(f"missing images for {', '.join(missing_images[:3])}")
            raise ValueError(
                f"Cityscapes validation image/annotation mismatch ({'; '.join(details)})."
            )
        if len(images) != CITYSCAPES_VALIDATION_SAMPLE_COUNT:
            raise ValueError(
                "Cityscapes validation archives must contain "
                f"{CITYSCAPES_VALIDATION_SAMPLE_COUNT} pairs, found {len(images)}."
            )

        staged_image_dir = os.path.join(staging_dir, "images")
        staged_annotation_dir = os.path.join(staging_dir, "annotations")
        os.makedirs(staged_image_dir)
        os.makedirs(staged_annotation_dir)
        for sample_id in sorted(images):
            shutil.copy2(
                images[sample_id], os.path.join(staged_image_dir, f"{sample_id}.png")
            )
            shutil.copy2(
                annotations[sample_id],
                os.path.join(staged_annotation_dir, f"{sample_id}.png"),
            )

        if not dataset_ready(staging_dir, "semantic_segmentation", "cityscapes"):
            raise ValueError(
                "Staged Cityscapes validation data failed identity and completeness checks."
            )
        _validate_staged_payloads(Path(staging_dir), "cityscapes")

        replacements = (
            (staged_image_dir, os.path.join(output_dir, "images")),
            (staged_annotation_dir, os.path.join(output_dir, "annotations")),
        )
        os.makedirs(output_dir, exist_ok=True)
        _replace_staged_directories(
            replacements, output_parent_dir, ".cityscapes-backup-"
        )
    print(
        f"Constructed Cityscapes validation dataset with {len(images)} image/mask pairs"
    )


def _resolve_dotav1_root(dataset_dir: str) -> str:
    """Resolves a DOTAv1 dataset root from a directory path.

    Args:
        dataset_dir: Directory containing the DOTAv1 dataset or its parent.

    Returns:
        Path to the DOTAv1 dataset root.
    """
    dotav1_dir = os.path.join(dataset_dir, "DOTAv1")
    if os.path.isdir(dotav1_dir):
        return dotav1_dir
    return dataset_dir


def _is_google_drive_folder_url(path_or_url: str) -> bool:
    """Returns whether a URL points to a Google Drive folder."""

    parsed = urlparse(path_or_url)
    return parsed.hostname == "drive.google.com" and bool(
        re.fullmatch(r"/drive(?:/u/[^/]+)?/folders/[^/]+/?", parsed.path)
    )


def _download_dotav1_google_drive_archives(
    folder_url: str, download_dir: str
) -> tuple[str, str]:
    """Downloads the DOTAv1 image and v1.0-label archives from a Google Drive folder.

    Args:
        folder_url: Public Google Drive folder URL containing the DOTAv1 archives.
        download_dir: Directory where the selected archives will be stored.

    Returns:
        Paths to the image archive and original v1.0-label archive.

    Raises:
        ValueError: If the required archives are absent from the Drive folder.
        RuntimeError: If gdown fails to download a required archive.
    """

    print(f"Retrieving DOTAv1 archive list from {folder_url}...")
    folder_entries = download_folder(
        url=folder_url, output=download_dir, quiet=True, skip_download=True
    )
    if folder_entries is None:
        raise RuntimeError(
            f"Failed to retrieve the DOTAv1 Google Drive folder listing: {folder_url}"
        )
    files = [
        entry for entry in folder_entries if _is_google_drive_download_entry(entry)
    ]
    archives: dict[str, _GoogleDriveDownloadEntry] = {}
    for archive_path in DOTAV1_GOOGLE_DRIVE_ARCHIVES:
        matches = [
            drive_file
            for drive_file in files
            if drive_file.path == archive_path
            or drive_file.path.endswith(f"/{archive_path}")
        ]
        if len(matches) == 1:
            archives[archive_path] = matches[0]
            continue

        available = ", ".join(sorted(drive_file.path for drive_file in files)) or "none"
        if not matches:
            raise ValueError(
                f"DOTAv1 Drive folder is missing {archive_path}. Available files: {available}."
            )
        ambiguous = ", ".join(sorted(drive_file.path for drive_file in matches))
        raise ValueError(
            f"DOTAv1 Drive folder has ambiguous matches for {archive_path}: {ambiguous}."
        )

    local_archives: dict[str, str] = {}
    for archive_path in sorted(DOTAV1_GOOGLE_DRIVE_ARCHIVES):
        drive_file = archives[archive_path]
        local_path = os.path.join(download_dir, os.path.basename(archive_path))
        print(f"Downloading DOTAv1 {archive_path}...")
        downloaded_path = download(
            id=drive_file.id, output=local_path, quiet=False, resume=True
        )
        if not isinstance(downloaded_path, str):
            raise RuntimeError(
                f"Failed to download DOTAv1 archive {archive_path} from {folder_url}."
            )
        local_archives[archive_path] = downloaded_path

    return (
        local_archives[DOTAV1_DOWNLOAD_CONFIG["images_archive"]],
        local_archives[DOTAV1_DOWNLOAD_CONFIG["labels_archive"]],
    )


def _iter_files(root: str, extensions: Iterable[str]) -> Iterable[str]:
    """Yields files below a directory with one of the requested suffixes."""

    suffixes = tuple(extension.lower() for extension in extensions)
    for current_root, _, file_names in os.walk(root):
        for file_name in file_names:
            if file_name.lower().endswith(suffixes):
                yield os.path.join(current_root, file_name)


def _safe_archive_member_path(member_name: str, destination: str) -> str:
    """Return an archive member destination after enforcing staging-directory containment.

    Args:
        member_name: Path stored in an archive member.
        destination: Archive extraction directory.

    Returns:
        Absolute destination path for the member.

    Raises:
        ValueError: If the member path is absolute or escapes the extraction directory.
    """

    root = os.path.abspath(destination)
    target = os.path.abspath(os.path.join(root, member_name))
    if os.path.commonpath((root, target)) != root:
        raise ValueError(f"Unsafe archive member path: {member_name!r}.")
    return target


def _safe_unpack_archive(archive_path: str, destination: str) -> None:
    """Extract an archive while rejecting links, special files, and escaping paths.

    Args:
        archive_path: ZIP or tar-family dataset archive.
        destination: Empty staging directory where archive members are written.

    Raises:
        ValueError: If the archive format or any member is unsafe or unsupported.
        OSError: If a validated archive cannot be read or written.
    """

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            targets: set[str] = set()
            for member in members:
                target = _safe_archive_member_path(member.filename, destination)
                if target in targets:
                    raise ValueError(
                        f"Duplicate archive member path: {member.filename!r}."
                    )
                targets.add(target)
                file_type = stat.S_IFMT(member.external_attr >> 16)
                if file_type and not (
                    stat.S_ISREG(file_type) or stat.S_ISDIR(file_type)
                ):
                    raise ValueError(
                        f"Unsafe archive member type: {member.filename!r}."
                    )
            for member in members:
                target = _safe_archive_member_path(member.filename, destination)
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(member) as source, open(target, "wb") as output_file:
                    shutil.copyfileobj(source, output_file)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            targets: set[str] = set()
            for member in members:
                target = _safe_archive_member_path(member.name, destination)
                if target in targets:
                    raise ValueError(f"Duplicate archive member path: {member.name!r}.")
                targets.add(target)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member type: {member.name!r}.")
            for member in members:
                target = _safe_archive_member_path(member.name, destination)
                if member.isdir():
                    os.makedirs(target, exist_ok=True)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read archive member: {member.name!r}.")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with source, open(target, "wb") as output_file:
                    shutil.copyfileobj(source, output_file)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}.")


def _write_dotav1_yolo_labels(
    image_path: str, original_label_path: str, output_path: str
) -> None:
    """Converts one official DOTAv1 label file into normalized OBB label format."""

    with Image.open(image_path) as image:
        width, height = image.size
    converted_lines: list[str] = []
    with open(original_label_path, encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            fields = line.split()
            if fields and (
                fields[0].startswith("imagesource:") or fields[0].startswith("gsd:")
            ):
                continue
            if len(fields) < 10:
                raise ValueError(
                    "Malformed DOTAv1 annotation in "
                    f"{original_label_path} at line {line_number}: expected at least "
                    f"10 fields, got {len(fields)}."
                )
            class_name = fields[8]
            if class_name not in DOTAV1_CLASS_TO_IDX:
                raise ValueError(
                    f"Unsupported DOTAv1 class in {original_label_path}: {class_name}"
                )
            coordinates = [float(value) for value in fields[:8]]
            if not all(math.isfinite(coordinate) for coordinate in coordinates):
                raise ValueError(
                    f"DOTAv1 coordinates must be finite in {original_label_path} "
                    f"at line {line_number}."
                )
            if fields[9] not in {"0", "1", "2"}:
                raise ValueError(
                    f"Unsupported DOTAv1 difficulty flag {fields[9]!r} in "
                    f"{original_label_path} at line {line_number}."
                )
            normalized = [
                coordinate / (width if index % 2 == 0 else height)
                for index, coordinate in enumerate(coordinates)
            ]
            converted_lines.append(
                f"{DOTAV1_CLASS_TO_IDX[class_name]} "
                + " ".join(f"{coordinate:.8g}" for coordinate in normalized)
                # The trailing flag is normalized-label metadata, not a YOLO OBB
                # coordinate. It preserves official difficult regions for evaluation.
                + f" {int(fields[9] in {'1', '2'})}"
            )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(converted_lines))
        if converted_lines:
            output_file.write("\n")


def construct_dotav1_from_archives(
    image_archive: str, label_archive: str, output_dir: str
) -> None:
    """Constructs the DOTAv1 validation layout from the Google Drive archives.

    Args:
        image_archive: Path to the DOTAv1 validation-image archive.
        label_archive: Path to the original DOTAv1 v1.0 label archive.
        output_dir: Directory where the organized validation dataset will be stored.

    Raises:
        ValueError: If the archives have no validation files or their image and label stems differ.
        OSError: If staging or replacing the organized dataset files fails.
    """

    with TemporaryDirectory() as extract_dir:
        image_dir = os.path.join(extract_dir, "images")
        label_dir = os.path.join(extract_dir, "labels")
        _safe_unpack_archive(image_archive, image_dir)
        _safe_unpack_archive(label_archive, label_dir)

        label_paths = list(_iter_files(label_dir, [".txt"]))
        labels = {
            os.path.splitext(os.path.basename(path))[0]: path for path in label_paths
        }
        if len(labels) != len(label_paths):
            raise ValueError("DOTAv1 archive contains duplicate label stems.")
        if not labels:
            raise ValueError(f"No DOTAv1 label files found in {label_archive}.")

        image_paths = list(
            _iter_files(image_dir, [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"])
        )
        images = {
            os.path.splitext(os.path.basename(path))[0]: path for path in image_paths
        }
        if len(images) != len(image_paths):
            raise ValueError("DOTAv1 archive contains duplicate image stems.")
        image_ids = set(images)
        label_ids = set(labels)
        missing_labels = sorted(image_ids - label_ids)
        missing_images = sorted(label_ids - image_ids)
        if missing_labels or missing_images:
            details = []
            if missing_labels:
                details.append(
                    f"images without labels: {', '.join(missing_labels[:5])}"
                )
            if missing_images:
                details.append(
                    f"labels without images: {', '.join(missing_images[:5])}"
                )
            raise ValueError(f"DOTAv1 archive stem mismatch ({'; '.join(details)}).")
        matching_ids = sorted(image_ids)
        if len(matching_ids) != DOTAV1_VALIDATION_SAMPLE_COUNT:
            raise ValueError(
                "DOTAv1 validation dataset must contain "
                f"{DOTAV1_VALIDATION_SAMPLE_COUNT} matching image/label pairs, found {len(matching_ids)}."
            )

        output_dir = os.path.abspath(output_dir)
        output_parent_dir = os.path.dirname(output_dir)
        os.makedirs(output_parent_dir, exist_ok=True)
        with TemporaryDirectory(
            dir=output_parent_dir, prefix=".dotav1-staging-"
        ) as staging_dir:
            staged_output_dir = os.path.join(staging_dir, "dotav1")
            staged_image_dir = os.path.join(staged_output_dir, "images")
            staged_label_dir = os.path.join(staged_output_dir, "labels", "val")
            staged_original_label_dir = os.path.join(
                staged_output_dir, "labels", "val_original"
            )
            os.makedirs(staged_image_dir)
            os.makedirs(staged_label_dir)
            os.makedirs(staged_original_label_dir)

            for image_id in matching_ids:
                image_path = images[image_id]
                shutil.copy2(
                    image_path,
                    os.path.join(staged_image_dir, os.path.basename(image_path)),
                )

            for image_id in matching_ids:
                shutil.copy2(
                    labels[image_id],
                    os.path.join(staged_original_label_dir, f"{image_id}.txt"),
                )
                _write_dotav1_yolo_labels(
                    images[image_id],
                    labels[image_id],
                    os.path.join(staged_label_dir, f"{image_id}.txt"),
                )

            _validate_staged_dataset(staged_output_dir, "dotav1", ("obb",))
            _replace_staged_directories(
                ((staged_output_dir, output_dir),),
                output_parent_dir,
                ".dotav1-backup-",
            )

    print(f"Constructed DOTAv1 validation dataset with {len(matching_ids)} images")


def _copy_dotav1_layout_to_staging(dataset_root: str, staged_output_dir: str) -> None:
    """Copy a flat or legacy DOTAv1 validation layout into canonical staging."""

    image_root = os.path.join(dataset_root, "images")
    supported_image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    flat_image_files = (
        [
            file_name
            for file_name in os.listdir(image_root)
            if os.path.isfile(os.path.join(image_root, file_name))
            and file_name.lower().endswith(supported_image_suffixes)
        ]
        if os.path.isdir(image_root)
        else []
    )
    source_image_dir = (
        image_root if flat_image_files else os.path.join(image_root, "val")
    )
    if not os.path.isdir(source_image_dir):
        raise ValueError(f"No DOTAv1 validation images found in {dataset_root}")

    staged_image_dir = os.path.join(staged_output_dir, "images")
    os.makedirs(staged_image_dir)
    for file_name in os.listdir(source_image_dir):
        source_path = os.path.join(source_image_dir, file_name)
        if os.path.isfile(source_path) and file_name.lower().endswith(
            supported_image_suffixes
        ):
            shutil.copy2(source_path, os.path.join(staged_image_dir, file_name))

    for label_directory in ("val", "val_original"):
        source_label_dir = os.path.join(dataset_root, "labels", label_directory)
        if not os.path.isdir(source_label_dir):
            continue
        staged_label_dir = os.path.join(staged_output_dir, "labels", label_directory)
        os.makedirs(staged_label_dir, exist_ok=True)
        for file_name in os.listdir(source_label_dir):
            source_path = os.path.join(source_label_dir, file_name)
            if os.path.isfile(source_path) and file_name.lower().endswith(".txt"):
                shutil.copy2(source_path, os.path.join(staged_label_dir, file_name))


def construct_dotav1(dataset_dir: str, output_dir: str) -> None:
    """Constructs a validation-only DOTAv1 dataset.

    Args:
        dataset_dir: Directory containing a DOTAv1 dataset or its parent.
        output_dir: Directory where the organized validation dataset will be stored.

    Raises:
        ValueError: If the staged validation dataset is incomplete or mismatched.
        OSError: If staging or replacing the organized dataset files fails.
    """
    dataset_root = _resolve_dotav1_root(dataset_dir)
    print(f"Constructing DOTAv1 validation dataset from {dataset_root} to {output_dir}")
    output_dir = os.path.abspath(output_dir)
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(
        dir=output_parent_dir, prefix=".dotav1-staging-"
    ) as staging_dir:
        staged_output_dir = os.path.join(staging_dir, "dotav1")
        os.makedirs(staged_output_dir)
        _copy_dotav1_layout_to_staging(dataset_root, staged_output_dir)
        _validate_staged_dataset(staged_output_dir, "dotav1", ("obb",))
        _replace_staged_directories(
            ((staged_output_dir, output_dir),),
            output_parent_dir,
            ".dotav1-backup-",
        )

    print("Constructing DOTAv1 validation dataset completed")


def organize_dotav1(
    dataset_path: str,
    output_dir: str | None = None,
) -> None:
    """Organizes a validation-only DOTAv1 dataset.

    Args:
        dataset_path: Path or URL to the DOTAv1 zip file or extracted dataset directory.
        output_dir: Directory to store the organized dataset. Defaults to the
            resolved Mobilint cache directory.
    """
    output_dir = _resolve_organizer_output_dir(output_dir, "dotav1")
    with TemporaryDirectory() as temp_dir:
        if _is_google_drive_folder_url(dataset_path):
            image_archive, label_archive = _download_dotav1_google_drive_archives(
                dataset_path, temp_dir
            )
            construct_dotav1_from_archives(image_archive, label_archive, output_dir)
            return

        local_dataset_path = _resolve_source(dataset_path, temp_dir)

        if local_dataset_path.endswith(".zip"):
            print("Unpacking DOTAv1 files to temporary directory...")
            _safe_unpack_archive(local_dataset_path, temp_dir)
            print("Unpacking completed")
            construct_dotav1(temp_dir, output_dir)
            return

        construct_dotav1(local_dataset_path, output_dir)

"""Tests for dataset download organization helpers."""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import os
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zipfile import ZipFile

import pytest
import requests
import numpy as np
from PIL import Image

import mblt_vision.utils.datasets.organizer as organizer
from mblt_vision.utils.datasets import readiness as readiness_module


class _DummyTqdm:
    """Minimal tqdm stub for download tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.updated = 0

    def __enter__(self) -> _DummyTqdm:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def update(self, value: int) -> None:
        self.updated += value


def test_dotav1_normalized_labels_preserve_difficult_metadata(tmp_path: Path) -> None:
    """Keep official difficult regions in the normalized label export."""

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 50)).save(image_path)
    source_path = tmp_path / "source.txt"
    source_path.write_text(
        "imagesource:GoogleEarth\n"
        "0 0 20 0 20 20 0 20 plane 0\n"
        "30 0 50 0 50 20 30 20 plane 1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "normalized.txt"

    organizer._write_dotav1_yolo_labels(
        str(image_path), str(source_path), str(output_path)
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "0 0 0 0.2 0 0.2 0.4 0 0.4 0",
        "0 0.3 0 0.5 0 0.5 0.4 0.3 0.4 1",
    ]


def test_dotav1_organizer_rejects_truncated_raw_annotation_rows(
    tmp_path: Path,
) -> None:
    """Do not silently remove malformed source annotations during organization."""

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 50)).save(image_path)
    source_path = tmp_path / "source.txt"
    source_path.write_text(
        "imagesource:GoogleEarth\ngsd:0.5\n0 0 20 0 20 20 0 plane 0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"source\.txt at line 3: expected at least 10 fields, got 9",
    ):
        organizer._write_dotav1_yolo_labels(
            str(image_path), str(source_path), str(tmp_path / "normalized.txt")
        )


def test_dotav1_organizer_rejects_repeated_polygon_vertices(tmp_path: Path) -> None:
    """Keep malformed triangle-like rows out of the staged normalized labels."""

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 50)).save(image_path)
    source_path = tmp_path / "source.txt"
    source_path.write_text("0 0 20 0 20 20 0 0 plane 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="four distinct vertices"):
        organizer._write_dotav1_yolo_labels(
            str(image_path), str(source_path), str(tmp_path / "normalized.txt")
        )


def test_dotav1_organizer_rejects_duplicate_targets(tmp_path: Path) -> None:
    """Reject duplicate raw targets before creating normalized labels."""

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 50)).save(image_path)
    source_path = tmp_path / "source.txt"
    target = "0 0 20 0 20 20 0 20 plane 0\n"
    source_path.write_text(target * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate DOTAv1 annotation target"):
        organizer._write_dotav1_yolo_labels(
            str(image_path), str(source_path), str(tmp_path / "normalized.txt")
        )


def test_dotav1_organizer_rejects_reordered_duplicate_targets(
    tmp_path: Path,
) -> None:
    """Raw DOTAv1 labels cannot duplicate a target with another start vertex."""

    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 50)).save(image_path)
    source_path = tmp_path / "source.txt"
    source_path.write_text(
        "0 0 20 0 20 20 0 20 plane 0\n20 0 20 20 0 20 0 0 plane 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate DOTAv1 annotation target"):
        organizer._write_dotav1_yolo_labels(
            str(image_path), str(source_path), str(tmp_path / "normalized.txt")
        )


def test_dotav1_stage_requires_a_non_difficult_target(tmp_path: Path) -> None:
    """Do not replace a usable DOTAv1 cache with an unscorable staged split."""

    image_path = tmp_path / "images" / "P0001.png"
    image_path.parent.mkdir()
    Image.new("RGB", (100, 50)).save(image_path)
    label_path = tmp_path / "labels" / "val_original" / "P0001.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0 20 0 20 20 0 20 plane 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one non-difficult target"):
        organizer._validate_staged_dotav1_labels(tmp_path)


@pytest.mark.parametrize(
    ("organize_dataset", "dataset_name"),
    [
        (organizer.organize_imagenet, "imagenet"),
        (organizer.organize_coco, "coco"),
        (organizer.organize_widerface, "widerface"),
        (organizer.organize_nyu_depth, "nyu-depth"),
        (organizer.organize_ade20k, "ade20k"),
        (organizer.organize_cityscapes, "cityscapes"),
        (organizer.organize_dotav1, "dotav1"),
        (organizer.organize_sav, "sa-v"),
    ],
)
def test_organizer_defaults_use_the_lazy_cache_resolver(
    monkeypatch: pytest.MonkeyPatch,
    organize_dataset: Callable[..., Any],
    dataset_name: str,
    tmp_path: Path,
) -> None:
    """Keep direct organizer APIs consistent with artifact cache fallback behavior."""

    output_dir = inspect.signature(organize_dataset).parameters["output_dir"].default
    assert output_dir is None

    import mblt_vision.wrapper as wrapper

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(wrapper, "get_mobilint_cache_dir", lambda: str(cache_dir))
    assert organizer._resolve_organizer_output_dir(None, dataset_name) == str(
        cache_dir / "datasets" / dataset_name
    )


class _FakeResponse:
    """Simple streaming response test double."""

    def __init__(
        self, status_code: int, headers: dict[str, str], chunks: list[bytes | Exception]
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def iter_content(self, chunk_size: int) -> Any:
        del chunk_size
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def test_download_url_retries_and_resumes_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume a partial archive download after a transient connection failure."""

    first_chunk = b"abc"
    second_chunk = b"def"
    calls: list[dict[str, str]] = []
    responses = [
        _FakeResponse(
            status_code=200,
            headers={"Content-Length": str(len(first_chunk) + len(second_chunk))},
            chunks=[first_chunk, requests.ConnectionError("interrupted")],
        ),
        _FakeResponse(
            status_code=206,
            headers={
                "Content-Length": str(len(second_chunk)),
                "Content-Range": "bytes 3-5/6",
            },
            chunks=[second_chunk],
        ),
    ]

    def _fake_get(
        url: str, stream: bool, timeout: tuple[int, int], headers: dict[str, str]
    ) -> _FakeResponse:
        del url, stream, timeout
        calls.append(dict(headers))
        return responses.pop(0)

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    monkeypatch.setattr(organizer, "sleep", lambda _: None)

    local_path = tmp_path / "archive.tar"
    result = organizer._download_url("https://example.com/archive.tar", str(local_path))

    assert result == str(local_path)
    assert local_path.read_bytes() == first_chunk + second_chunk
    assert calls == [{}, {"Range": "bytes=3-"}]


def test_download_url_restarts_when_resume_offset_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Discard a partial file when a range response starts at another offset."""

    calls: list[dict[str, str]] = []
    responses = [
        _FakeResponse(
            status_code=206,
            headers={"Content-Length": "3", "Content-Range": "bytes 0-2/6"},
            chunks=[b"bad"],
        ),
        _FakeResponse(
            status_code=200,
            headers={"Content-Length": "6"},
            chunks=[b"fresh!"],
        ),
    ]

    def _fake_get(
        url: str, stream: bool, timeout: tuple[int, int], headers: dict[str, str]
    ) -> _FakeResponse:
        del url, stream, timeout
        calls.append(dict(headers))
        return responses.pop(0)

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    local_path = tmp_path / "archive.tar"
    local_path.write_bytes(b"old")

    assert organizer._download_url(
        "https://example.com/archive.tar", str(local_path)
    ) == str(local_path)
    assert local_path.read_bytes() == b"fresh!"
    assert calls == [{"Range": "bytes=3-"}, {}]


def test_download_url_accepts_completed_archive_on_range_not_satisfiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Treat a matching 416 total as confirmation that the local file is complete."""

    payload = b"complete"
    calls: list[dict[str, str]] = []

    def _fake_get(
        url: str, stream: bool, timeout: tuple[int, int], headers: dict[str, str]
    ) -> _FakeResponse:
        del url, stream, timeout
        calls.append(dict(headers))
        return _FakeResponse(
            status_code=416,
            headers={"Content-Range": f"bytes */{len(payload)}"},
            chunks=[],
        )

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    local_path = tmp_path / "archive.tar"
    local_path.write_bytes(payload)

    assert organizer._download_url(
        "https://example.com/archive.tar",
        str(local_path),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    ) == str(local_path)
    assert local_path.read_bytes() == payload
    assert calls == [{"Range": f"bytes={len(payload)}-"}]


def test_download_url_restarts_after_incomplete_range_not_satisfiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not accept a 416 when the local partial file has the wrong size."""

    responses = [
        _FakeResponse(
            status_code=416,
            headers={"Content-Range": "bytes */6"},
            chunks=[],
        ),
        _FakeResponse(
            status_code=200,
            headers={"Content-Length": "6"},
            chunks=[b"fresh!"],
        ),
    ]

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        del url, kwargs
        return responses.pop(0)

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    local_path = tmp_path / "archive.tar"
    local_path.write_bytes(b"old")

    assert organizer._download_url(
        "https://example.com/archive.tar", str(local_path)
    ) == str(local_path)
    assert local_path.read_bytes() == b"fresh!"


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_download_url_retries_transient_http_responses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_code: int,
) -> None:
    """Retry transient HTTP responses before accepting a successful download."""

    calls = 0
    responses = [
        _FakeResponse(status_code=status_code, headers={}, chunks=[]),
        _FakeResponse(status_code=200, headers={"Content-Length": "2"}, chunks=[b"ok"]),
    ]

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return responses.pop(0)

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    monkeypatch.setattr(organizer, "sleep", lambda _: None)

    local_path = tmp_path / "archive.tar"
    assert organizer._download_url(
        "https://example.com/archive.tar", str(local_path)
    ) == str(local_path)
    assert local_path.read_bytes() == b"ok"
    assert calls == 2


def test_download_url_does_not_retry_permanent_http_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail immediately for permanent client errors such as HTTP 404."""

    calls = 0

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return _FakeResponse(status_code=404, headers={}, chunks=[])

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)

    with pytest.raises(requests.HTTPError):
        organizer._download_url(
            "https://example.com/archive.tar", str(tmp_path / "archive.tar")
        )

    assert calls == 1


def test_download_url_verifies_pinned_archive_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept only downloaded archive bytes matching the configured digest."""

    payload = b"verified archive"

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        del url, kwargs
        return _FakeResponse(
            status_code=200,
            headers={"Content-Length": str(len(payload))},
            chunks=[payload],
        )

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    local_path = tmp_path / "archive.zip"

    assert organizer._download_url(
        "https://example.com/archive.zip",
        str(local_path),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    ) == str(local_path)
    assert local_path.read_bytes() == payload


def test_download_url_rejects_and_removes_digest_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Never leave an unverified archive available for later extraction."""

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        del url, kwargs
        return _FakeResponse(
            status_code=200,
            headers={"Content-Length": "9"},
            chunks=[b"malicious"],
        )

    monkeypatch.setattr("mblt_vision.utils.datasets.organizer.requests.get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    local_path = tmp_path / "archive.zip"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        organizer._download_url(
            "https://example.com/archive.zip",
            str(local_path),
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )

    assert not local_path.exists()


def test_download_if_url_passes_registered_archive_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use the package-pinned digest whenever a default archive is downloaded."""

    source_url, expected_sha256 = next(iter(organizer.PINNED_ARCHIVE_SHA256.items()))
    recorded: dict[str, str | None] = {}

    def _fake_download(
        url: str, local_path: str, expected_sha256: str | None = None
    ) -> str:
        recorded.update(url=url, local_path=local_path, expected_sha256=expected_sha256)
        return local_path

    monkeypatch.setattr(organizer, "_download_url", _fake_download)

    assert organizer._download_if_url(source_url, str(tmp_path)) == str(
        tmp_path / Path(urlparse(source_url).path).name
    )
    assert recorded["url"] == source_url
    assert recorded["expected_sha256"] == expected_sha256


def test_download_if_url_rejects_plaintext_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not fetch benchmark inputs over an unauthenticated transport."""

    monkeypatch.setattr(
        "mblt_vision.utils.datasets.organizer.requests.get",
        lambda *args, **kwargs: pytest.fail("plaintext URL must not be requested"),
    )

    with pytest.raises(ValueError, match="must use HTTPS"):
        organizer._download_if_url("http://example.com/archive.zip", str(tmp_path))


def test_coco_and_ade20k_downloads_use_pinned_https_archives() -> None:
    """Keep default benchmark datasets on authenticated, integrity-checked URLs."""

    assert all(url.startswith("https://") for url in organizer.PINNED_ARCHIVE_SHA256)
    assert all(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in organizer.PINNED_ARCHIVE_SHA256.values()
    )


def test_should_download_serially_for_same_host_urls() -> None:
    """Serialize same-host dataset archive downloads to avoid throttling."""

    assert organizer._should_download_serially(
        [
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz",
        ]
    )

    assert not organizer._should_download_serially(
        [
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "https://example.com/data/annotations.tgz",
        ]
    )


def _create_imagenet_source(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small structurally valid ImageNet organizer source."""

    image_dir = tmp_path / "source-images"
    xml_dir = tmp_path / "source-xml" / "val"
    image_dir.mkdir()
    xml_dir.mkdir(parents=True)
    for index in range(50):
        stem = f"ILSVRC2012_val_{index + 1:08d}"
        (image_dir / f"{stem}.JPEG").write_bytes(b"image")
        (xml_dir / f"{stem}.xml").write_text(
            "<annotation><object><name>n00000001</name></object></annotation>",
            encoding="utf-8",
        )
    return image_dir, xml_dir.parent


def _create_coco_source(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small COCO organizer source."""

    image_dir = tmp_path / "source-images"
    annotation_dir = tmp_path / "source-annotations" / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir(parents=True)
    (image_dir / "000000000001.jpg").write_bytes(b"image")
    (annotation_dir / "instances_val2017.json").write_text("{}", encoding="utf-8")
    return image_dir, annotation_dir.parent


def _create_widerface_source(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small WiderFace organizer source."""

    image_dir = tmp_path / "source-images" / "images" / "0--Parade"
    annotation_dir = tmp_path / "source-annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"image")
    (annotation_dir / "wider_face_val.mat").write_bytes(b"metadata")
    return image_dir.parent.parent, annotation_dir


def test_construct_imagenet_replaces_stale_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace a managed ImageNet root only after staged readiness succeeds."""

    image_dir, xml_dir = _create_imagenet_source(tmp_path)
    readiness_calls: list[tuple[str, str]] = []

    def _dataset_ready(_: str, task: str, dataset: str) -> bool:
        readiness_calls.append((task, dataset))
        return True

    monkeypatch.setattr(organizer, "dataset_ready", _dataset_ready)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)

    output_dir = tmp_path / "imagenet"
    (output_dir / "stale-class").mkdir(parents=True)
    (output_dir / "stale-class" / "stale.JPEG").write_bytes(b"stale")

    organizer.construct_imagenet(str(image_dir), str(xml_dir), str(output_dir))

    assert {path.name for path in output_dir.iterdir()} == {"n00000001"}
    assert len(list((output_dir / "n00000001").iterdir())) == 50
    assert readiness_calls == [("image_classification", "imagenet")]


@pytest.mark.parametrize("class_name", ["../../escaped", "/tmp/escaped"])
def test_construct_imagenet_rejects_unsafe_xml_class_name(
    tmp_path: Path, class_name: str
) -> None:
    """Reject XML class names that could escape the ImageNet staging root."""

    image_dir, xml_dir = _create_imagenet_source(tmp_path)
    xml_path = next((xml_dir / "val").glob("*.xml"))
    xml_path.write_text(
        f"<annotation><object><name>{class_name}</name></object></annotation>",
        encoding="utf-8",
    )
    output_dir = tmp_path / "imagenet"

    with pytest.raises(ValueError, match="invalid ImageNet synset name"):
        organizer.construct_imagenet(str(image_dir), str(xml_dir), str(output_dir))

    assert not (tmp_path / "escaped").exists()
    assert not output_dir.exists()


def test_construct_coco_replaces_stale_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace a managed COCO root only after staged readiness succeeds."""

    image_dir, annotation_dir = _create_coco_source(tmp_path)
    readiness_calls: list[tuple[str, str]] = []

    def _dataset_ready(_: str, task: str, dataset: str) -> bool:
        readiness_calls.append((task, dataset))
        return True

    monkeypatch.setattr(organizer, "dataset_ready", _dataset_ready)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)

    output_dir = tmp_path / "coco"
    (output_dir / "val2017").mkdir(parents=True)
    (output_dir / "val2017" / "stale.jpg").write_bytes(b"stale")
    (output_dir / "stale_val2017.json").write_text("{}", encoding="utf-8")

    organizer.construct_coco(str(image_dir), str(annotation_dir), str(output_dir))

    assert {path.name for path in (output_dir / "val2017").iterdir()} == {
        "000000000001.jpg"
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "val2017",
        "instances_val2017.json",
    }
    assert readiness_calls == [
        ("object_detection", "coco"),
        ("instance_segmentation", "coco"),
        ("pose_estimation", "coco"),
    ]


def test_construct_widerface_replaces_stale_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace a managed WiderFace root only after staged readiness succeeds."""

    image_dir, annotation_dir = _create_widerface_source(tmp_path)
    readiness_calls: list[tuple[str, str]] = []

    def _dataset_ready(_: str, task: str, dataset: str) -> bool:
        readiness_calls.append((task, dataset))
        return True

    monkeypatch.setattr(organizer, "dataset_ready", _dataset_ready)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)

    output_dir = tmp_path / "widerface"
    (output_dir / "images" / "stale-event").mkdir(parents=True)
    (output_dir / "images" / "stale-event" / "stale.jpg").write_bytes(b"stale")
    (output_dir / "stale_val.mat").write_bytes(b"stale")

    organizer.construct_widerface(str(image_dir), str(annotation_dir), str(output_dir))

    assert {path.name for path in (output_dir / "images").iterdir()} == {"0--Parade"}
    assert {path.name for path in output_dir.iterdir()} == {
        "images",
        "wider_face_val.mat",
    }
    assert readiness_calls == [("face_detection", "widerface")]


@pytest.mark.parametrize("dataset", ["imagenet", "coco", "widerface"])
def test_incomplete_staged_dataset_preserves_existing_cache(
    tmp_path: Path,
    dataset: str,
) -> None:
    """Reject incomplete staged roots before replacing any existing cache."""

    source_builders = {
        "imagenet": _create_imagenet_source,
        "coco": _create_coco_source,
        "widerface": _create_widerface_source,
    }
    constructors = {
        "imagenet": organizer.construct_imagenet,
        "coco": organizer.construct_coco,
        "widerface": organizer.construct_widerface,
    }
    first_source, second_source = source_builders[dataset](tmp_path)

    output_dir = tmp_path / dataset
    output_dir.mkdir()
    (output_dir / "valid-cache-marker").write_bytes(b"existing")

    with pytest.raises(ValueError, match="existing dataset cache was not replaced"):
        constructors[dataset](str(first_source), str(second_source), str(output_dir))

    assert (output_dir / "valid-cache-marker").read_bytes() == b"existing"


def test_safe_unpack_archive_preserves_regular_tar_layout(tmp_path: Path) -> None:
    """Extract regular files and directories from a supported tar archive."""

    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("dataset/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        payload = b"dataset contents"
        member = tarfile.TarInfo("dataset/sample.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    extract_dir = tmp_path / "extracted"
    organizer._safe_unpack_archive(str(archive_path), str(extract_dir))

    assert (extract_dir / "dataset" / "sample.txt").read_bytes() == b"dataset contents"


@pytest.mark.parametrize("suffix", [".zip", ".tar"])
def test_safe_unpack_archive_rejects_duplicate_member_paths(
    tmp_path: Path, suffix: str
) -> None:
    """Reject duplicate archive members before any staged file can be replaced."""

    archive_path = tmp_path / f"dataset{suffix}"
    if suffix == ".zip":
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("dataset/sample.txt", b"first")
            archive.writestr("dataset/sample.txt", b"second")
    else:
        with tarfile.open(archive_path, "w") as archive:
            for payload in (b"first", b"second"):
                member = tarfile.TarInfo("dataset/sample.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Duplicate archive member path"):
        organizer._safe_unpack_archive(str(archive_path), str(tmp_path / "extracted"))


def test_safe_unpack_archive_rejects_tar_traversal_before_writing(
    tmp_path: Path,
) -> None:
    """Reject a traversal member without extracting earlier valid members."""

    archive_path = tmp_path / "dataset.tar"
    outside_path = tmp_path / "outside.txt"
    with tarfile.open(archive_path, "w") as archive:
        safe_payload = b"safe"
        safe_member = tarfile.TarInfo("dataset/safe.txt")
        safe_member.size = len(safe_payload)
        archive.addfile(safe_member, io.BytesIO(safe_payload))
        outside_payload = b"outside"
        outside_member = tarfile.TarInfo("../outside.txt")
        outside_member.size = len(outside_payload)
        archive.addfile(outside_member, io.BytesIO(outside_payload))

    extract_dir = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        organizer._safe_unpack_archive(str(archive_path), str(extract_dir))

    assert not outside_path.exists()
    assert not (extract_dir / "dataset" / "safe.txt").exists()


def test_organize_imagenet_rejects_tar_symlink_escape(tmp_path: Path) -> None:
    """Reject a tar symlink before ImageNet extraction can write through it."""

    image_archive = tmp_path / "images.tar"
    xml_archive = tmp_path / "annotations.tgz"
    outside_path = tmp_path / "outside.txt"
    with tarfile.open(image_archive, "w") as archive:
        link = tarfile.TarInfo("redirect")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path)
        archive.addfile(link)
        payload = b"outside"
        member = tarfile.TarInfo("redirect/outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(xml_archive, "w:gz"):
        pass

    with pytest.raises(ValueError, match="Unsafe archive member type"):
        organizer.organize_imagenet(
            str(image_archive), str(xml_archive), str(tmp_path / "imagenet")
        )

    assert not outside_path.exists()


def test_organize_nyu_depth_extracts_only_validation_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Install only NYU Depth validation image/depth pairs from an archive."""

    monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(organizer, "_validate_staged_nyu_depth", lambda _: None)
    archive_path = tmp_path / "nyu-depth.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nyu-depth/images/train/nyu_train.jpg", b"training image")
        archive.writestr("nyu-depth/depth/train/nyu_train.npy", b"training depth")
        archive.writestr("nyu-depth/images/val/nyu_0000.jpg", b"validation image")
        archive.writestr("nyu-depth/depth/val/nyu_0000.npy", b"validation depth")

    output_dir = tmp_path / "organized"
    organizer.organize_nyu_depth(str(archive_path), str(output_dir))

    assert archive_path.is_file()
    assert (output_dir / "images" / "nyu_0000.jpg").read_bytes() == b"validation image"
    assert (output_dir / "depth" / "nyu_0000.npy").read_bytes() == b"validation depth"
    assert not (output_dir / "images" / "train").exists()
    assert not (output_dir / "depth" / "train").exists()


@pytest.mark.parametrize("relative_path", ["images/sample.jpg", "depth/sample.npy"])
def test_construct_nyu_depth_rejects_symlinked_data_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
) -> None:
    """Reject NYU data symlinks without replacing an existing managed cache."""

    monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = tmp_path / "source"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "depth").mkdir()
    (dataset_dir / "images" / "sample.jpg").write_bytes(b"image")
    (dataset_dir / "depth" / "sample.npy").write_bytes(b"depth")
    external_file = tmp_path / "secret"
    external_file.write_bytes(b"outside dataset")
    source_path = dataset_dir / relative_path
    source_path.unlink()
    source_path.symlink_to(external_file)
    output_dir = tmp_path / "organized"
    output_dir.mkdir()
    marker = output_dir / "valid-cache-marker"
    marker.write_bytes(b"existing")

    with pytest.raises(ValueError, match="must not be a symlink"):
        organizer.construct_nyu_depth(str(dataset_dir), str(output_dir))

    assert marker.read_bytes() == b"existing"
    assert not (output_dir / Path(relative_path).name).exists()


def test_construct_nyu_depth_rejects_source_outside_resolved_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject regular files reached through a directory symlink escaping the dataset root."""

    monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = tmp_path / "source"
    selected_root = dataset_dir / "nyu-depth"
    selected_root.mkdir(parents=True)
    external_images = dataset_dir / "external-images"
    external_images.mkdir()
    (external_images / "sample.jpg").write_bytes(b"outside dataset")
    (selected_root / "images").symlink_to(external_images, target_is_directory=True)
    (selected_root / "depth").mkdir()
    (selected_root / "depth" / "sample.npy").write_bytes(b"depth")

    with pytest.raises(ValueError, match="must remain within dataset root"):
        organizer.construct_nyu_depth(str(dataset_dir), str(tmp_path / "organized"))


def test_construct_nyu_depth_rejects_duplicate_nested_source_stems(
    tmp_path: Path,
) -> None:
    """Do not let recursive NYU source traversal overwrite a flattened sample."""

    dataset_dir = tmp_path / "source"
    for relative_path in (
        "images/first/sample.jpg",
        "images/second/sample.png",
        "depth/sample.npy",
    ):
        path = dataset_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")

    with pytest.raises(
        ValueError, match="NYU Depth images contain duplicate filename stem"
    ):
        organizer._collect_nyu_depth_validation_files(
            str(dataset_dir / "images"),
            str(dataset_dir / "depth"),
            dataset_dir.resolve(),
        )


def test_nyu_depth_install_preserves_backups_when_rollback_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Leave recoverable NYU backups outside staging when rollback fails."""

    monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(organizer, "_validate_staged_nyu_depth", lambda _: None)
    dataset_dir = tmp_path / "source"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "depth").mkdir()
    (dataset_dir / "images" / "sample.jpg").write_bytes(b"new image")
    (dataset_dir / "depth" / "sample.npy").write_bytes(b"new depth")

    output_dir = tmp_path / "organized"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "depth").mkdir()
    (output_dir / "images" / "keep.jpg").write_bytes(b"old image")
    (output_dir / "depth" / "keep.npy").write_bytes(b"old depth")

    real_replace = os.replace

    def _fail_install_and_rollback(source: str, destination: str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(".nyu-depth-staging-")
            and source_path.name == "depth"
        ):
            raise OSError("simulated install failure")
        if (
            source_path.parent.name.startswith(".nyu-depth-backup-")
            and destination_path == output_dir / "images"
        ):
            raise OSError("simulated rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "mblt_vision.utils.datasets.organizer.os.replace", _fail_install_and_rollback
    )

    with pytest.raises(OSError, match="backups are preserved"):
        organizer.construct_nyu_depth(str(dataset_dir), str(output_dir))

    backup_dirs = list(tmp_path.glob(".nyu-depth-backup-*"))
    assert len(backup_dirs) == 1
    assert (backup_dirs[0] / "images" / "keep.jpg").read_bytes() == b"old image"
    assert (backup_dirs[0] / "depth" / "keep.npy").read_bytes() == b"old depth"


@pytest.mark.parametrize(
    "depth",
    [
        np.array([[np.nan]], dtype=np.float32),
        np.array([[1]], dtype=np.complex64),
        np.array([[-1]], dtype=np.float32),
        np.array([[0]], dtype=np.float32),
        np.array([[100]], dtype=np.float32),
    ],
)
def test_staged_nyu_depth_validation_rejects_malformed_payloads(
    tmp_path: Path, depth: np.ndarray
) -> None:
    """Reject corrupt staged NYU targets before cache replacement can begin."""

    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    image_dir.mkdir()
    depth_dir.mkdir()
    Image.new("RGB", (1, 1)).save(image_dir / "sample.png")
    np.save(depth_dir / "sample.npy", depth)

    with pytest.raises(
        ValueError,
        match="real numeric dtype|finite values|negative values|valid metric depth",
    ):
        organizer._validate_staged_nyu_depth(str(tmp_path))


@pytest.mark.parametrize("dataset", ["ade20k", "dotav1"])
def test_staged_payload_validation_rejects_corrupt_files(
    tmp_path: Path, dataset: str
) -> None:
    """Do not replace a valid cache with structurally complete corrupt data."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    if dataset == "ade20k":
        annotation_dir = tmp_path / "annotations"
        annotation_dir.mkdir()
        (image_dir / "ADE_val_00000001.jpg").write_bytes(b"not an image")
        (annotation_dir / "ADE_val_00000001.png").write_bytes(b"not a PNG")
    else:
        Image.new("RGB", (1, 1)).save(image_dir / "P0001.png")
        label_dir = tmp_path / "labels" / "val"
        original_label_dir = tmp_path / "labels" / "val_original"
        label_dir.mkdir(parents=True)
        original_label_dir.mkdir()
        (label_dir / "P0001.txt").write_text("bad label", encoding="utf-8")
        (original_label_dir / "P0001.txt").write_text("bad label", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable|Malformed staged DOTAv1"):
        organizer._validate_staged_payloads(tmp_path, dataset)


def test_staged_cityscapes_payload_validation_rejects_unknown_source_ids(
    tmp_path: Path,
) -> None:
    """Reject source IDs the Cityscapes loader cannot classify consistently."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "sample.png")
    Image.fromarray(np.array([[0, 34], [255, 7]], dtype=np.uint8)).save(
        annotation_dir / "sample.png"
    )

    with pytest.raises(ValueError, match="unsupported source IDs.*34"):
        organizer._validate_staged_payloads(tmp_path, "cityscapes")


@pytest.mark.parametrize(
    ("dataset", "source_id"),
    [("ade20k", 0), ("cityscapes", 255)],
)
def test_staged_semantic_payload_validation_rejects_all_ignored_targets(
    tmp_path: Path, dataset: str, source_id: int
) -> None:
    """Do not install semantic masks that cannot contribute to validation metrics."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "sample.png")
    Image.fromarray(np.full((2, 2), source_id, dtype=np.uint8)).save(
        annotation_dir / "sample.png"
    )

    with pytest.raises(ValueError, match="contains no evaluable class IDs"):
        organizer._validate_staged_payloads(tmp_path, dataset)


def test_staged_coco_payload_validation_rejects_annotation_geometry_mismatch(
    tmp_path: Path,
) -> None:
    """Preserve a usable cache when JSON dimensions differ from staged image bytes."""

    image_dir = tmp_path / "val2017"
    image_dir.mkdir()
    Image.new("RGB", (2, 1)).save(image_dir / "000000000001.jpg")
    (tmp_path / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "000000000001.jpg",
                        "height": 2,
                        "width": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="geometry does not match"):
        organizer._validate_staged_payloads(tmp_path, "coco")


def _create_ade20k_source(tmp_path: Path) -> Path:
    """Create a compact extracted ADE20K validation source."""

    dataset_dir = tmp_path / "ADEChallengeData2016"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "annotations").mkdir()
    (dataset_dir / "images" / "ADE_val_00000001.jpg").write_bytes(b"validation")
    (dataset_dir / "annotations" / "ADE_val_00000001.png").write_bytes(b"annotation")
    (dataset_dir / "objectInfo150.txt").write_bytes(b"labels")
    (dataset_dir / "sceneCategories.txt").write_bytes(b"scenes")
    return dataset_dir


def test_organize_ade20k_extracts_flat_validation_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Install only ADE20K validation image/mask pairs in the reference layout."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(readiness_module, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)
    archive_path = tmp_path / "ADEChallengeData2016.zip"
    image_buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image_buffer, format="JPEG")
    annotation_buffer = io.BytesIO()
    Image.new("L", (1, 1), color=1).save(annotation_buffer, format="PNG")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "ADEChallengeData2016/images/training/ADE_train_00000001.jpg", b"training"
        )
        archive.writestr(
            "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
            b"training",
        )
        archive.writestr(
            "ADEChallengeData2016/images/validation/ADE_val_00000001.jpg",
            image_buffer.getvalue(),
        )
        archive.writestr(
            "ADEChallengeData2016/annotations/validation/ADE_val_00000001.png",
            annotation_buffer.getvalue(),
        )
        archive.writestr("ADEChallengeData2016/objectInfo150.txt", b"labels")
        archive.writestr("ADEChallengeData2016/sceneCategories.txt", b"scenes")

    output_dir = tmp_path / "organized"
    organizer.organize_ade20k(str(archive_path), str(output_dir))

    assert (
        output_dir / "images" / "ADE_val_00000001.jpg"
    ).read_bytes() == image_buffer.getvalue()
    assert (
        output_dir / "annotations" / "ADE_val_00000001.png"
    ).read_bytes() == annotation_buffer.getvalue()
    assert (output_dir / "objectInfo150.txt").read_bytes() == b"labels"
    assert (output_dir / "sceneCategories.txt").read_bytes() == b"scenes"
    assert not (output_dir / "images" / "training").exists()


def test_construct_ade20k_requires_metadata_before_replacing_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve the cache when an ADE20K source omits required metadata."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = _create_ade20k_source(tmp_path)
    (dataset_dir / "sceneCategories.txt").unlink()
    output_dir = tmp_path / "organized"
    output_dir.mkdir()
    (output_dir / "valid-cache-marker").write_bytes(b"existing")

    with pytest.raises(ValueError, match="sceneCategories.txt"):
        organizer.construct_ade20k(str(dataset_dir), str(output_dir))

    assert (output_dir / "valid-cache-marker").read_bytes() == b"existing"


def test_construct_ade20k_rejects_duplicate_source_stems(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject flat ADE20K source names that would overwrite during staging."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = _create_ade20k_source(tmp_path)
    image_path = dataset_dir / "images" / "ADE_val_00000001.jpg"
    shutil.copy2(image_path, image_path.with_suffix(".JPG"))

    with pytest.raises(
        ValueError, match="ADE20K images contain duplicate filename stem"
    ):
        organizer.construct_ade20k(str(dataset_dir), str(tmp_path / "organized"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "images/ADE_val_00000001.jpg",
        "annotations/ADE_val_00000001.png",
        "objectInfo150.txt",
        "sceneCategories.txt",
    ],
)
def test_construct_ade20k_rejects_symlinked_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
) -> None:
    """Reject ADE20K data and metadata symlinks before replacing a managed cache."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = _create_ade20k_source(tmp_path)
    external_file = tmp_path / "secret"
    external_file.write_bytes(b"outside dataset")
    source_path = dataset_dir / relative_path
    source_path.unlink()
    source_path.symlink_to(external_file)
    output_dir = tmp_path / "organized"
    output_dir.mkdir()
    marker = output_dir / "valid-cache-marker"
    marker.write_bytes(b"existing")

    with pytest.raises(ValueError, match="must not be a symlink"):
        organizer.construct_ade20k(str(dataset_dir), str(output_dir))

    assert marker.read_bytes() == b"existing"


def test_construct_ade20k_preserves_cache_when_metadata_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve the cache when copying required ADE20K metadata into staging fails."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    dataset_dir = _create_ade20k_source(tmp_path)
    output_dir = tmp_path / "organized"
    output_dir.mkdir()
    (output_dir / "valid-cache-marker").write_bytes(b"existing")
    real_copy2 = shutil.copy2

    def _fail_scene_metadata_copy(source: str, destination: str) -> str:
        if Path(source).name == "sceneCategories.txt":
            raise OSError("simulated metadata copy failure")
        return real_copy2(source, destination)

    monkeypatch.setattr(
        "mblt_vision.utils.datasets.organizer.shutil.copy2",
        _fail_scene_metadata_copy,
    )

    with pytest.raises(OSError, match="simulated metadata copy failure"):
        organizer.construct_ade20k(str(dataset_dir), str(output_dir))

    assert (output_dir / "valid-cache-marker").read_bytes() == b"existing"


def _write_cityscapes_archives(
    tmp_path: Path, sample_ids: list[str]
) -> tuple[Path, Path]:
    """Create compact official-layout Cityscapes ZIP fixtures."""

    image_archive = tmp_path / "leftImg8bit_trainvaltest.zip"
    annotation_archive = tmp_path / "gtFine_trainvaltest.zip"
    with ZipFile(image_archive, "w") as archive:
        archive.writestr(
            "leftImg8bit/train/aachen/aachen_000000_000001_leftImg8bit.png",
            b"train image",
        )
        archive.writestr(
            "leftImg8bit/test/berlin/berlin_000000_000001_leftImg8bit.png",
            b"test image",
        )
        for sample_id in sample_ids:
            city = sample_id.split("_", maxsplit=1)[0]
            archive.writestr(
                f"leftImg8bit/val/{city}/{sample_id}_leftImg8bit.png",
                f"image:{sample_id}".encode(),
            )
    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr(
            "gtFine/train/aachen/aachen_000000_000001_gtFine_labelIds.png",
            b"train mask",
        )
        for sample_id in sample_ids:
            city = sample_id.split("_", maxsplit=1)[0]
            archive.writestr(
                f"gtFine/val/{city}/{sample_id}_gtFine_labelIds.png",
                f"mask:{sample_id}".encode(),
            )
            archive.writestr(
                f"gtFine/val/{city}/{sample_id}_gtFine_color.png", b"color"
            )
            archive.writestr(
                f"gtFine/val/{city}/{sample_id}_gtFine_instanceIds.png", b"instance"
            )
            archive.writestr(
                f"gtFine/val/{city}/{sample_id}_gtFine_polygons.json", b"{}"
            )
            archive.writestr(
                f"gtFine/val/{city}/{sample_id}_gtFine_trainIds.png", b"train IDs"
            )
    return image_archive, annotation_archive


@pytest.mark.parametrize("dataset", ["nyu-depth", "ade20k", "cityscapes"])
def test_dense_organizers_reject_symlinked_output_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: str,
) -> None:
    """Fail before organizing through a symlinked managed root or ancestor."""

    if dataset == "nyu-depth":
        monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
        source_dir = tmp_path / "nyu-source"
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "depth").mkdir()
        (source_dir / "images" / "sample.jpg").write_bytes(b"image")
        (source_dir / "depth" / "sample.npy").write_bytes(b"depth")

        def organizer_fn(output_dir: str) -> None:
            organizer.organize_nyu_depth(str(source_dir), output_dir)
    elif dataset == "ade20k":
        monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
        source_dir = _create_ade20k_source(tmp_path)

        def organizer_fn(output_dir: str) -> None:
            organizer.organize_ade20k(str(source_dir), output_dir)
    else:
        monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
        image_archive, annotation_archive = _write_cityscapes_archives(
            tmp_path,
            ["lindau_000000_000019"],
        )

        def organizer_fn(output_dir: str) -> None:
            organizer.organize_cityscapes(
                str(image_archive),
                str(annotation_archive),
                output_dir,
            )

    for topology in (
        "root",
        "ancestor",
        "normalized-ancestor",
        "symlink-parent-traversal",
    ):
        if topology == "root":
            protected_dir = tmp_path / "root-target"
            protected_dir.mkdir()
            output_dir = tmp_path / "managed"
            output_dir.symlink_to(protected_dir, target_is_directory=True)
        elif topology == "ancestor":
            target_parent = tmp_path / "ancestor-target"
            protected_dir = target_parent / "managed"
            protected_dir.mkdir(parents=True)
            symlinked_parent = tmp_path / "datasets-link"
            symlinked_parent.symlink_to(target_parent, target_is_directory=True)
            output_dir = symlinked_parent / "managed"
        elif topology == "normalized-ancestor":
            target_parent = tmp_path / "normalized-ancestor-target"
            protected_dir = target_parent / "managed"
            protected_dir.mkdir(parents=True)
            symlinked_parent = tmp_path / "normalized-datasets-link"
            symlinked_parent.symlink_to(target_parent, target_is_directory=True)
            output_dir = tmp_path / "missing" / ".." / symlinked_parent.name / "managed"
        else:
            target_parent = tmp_path / "traversal-target"
            target_child = target_parent / "child"
            target_child.mkdir(parents=True)
            protected_dir = target_parent / "traversed-managed"
            protected_dir.mkdir()
            symlinked_parent = tmp_path / "traversal-link"
            symlinked_parent.symlink_to(target_child, target_is_directory=True)
            output_dir = symlinked_parent / ".." / protected_dir.name
        marker = protected_dir / "keep"
        marker.write_bytes(b"existing")

        with pytest.raises(ValueError, match="existing parents must not be symlinks"):
            organizer_fn(str(output_dir))

        assert marker.read_bytes() == b"existing"
        assert list(protected_dir.iterdir()) == [marker]


@pytest.mark.parametrize(
    ("dataset", "layout_name"),
    [
        ("nyu-depth", "images"),
        ("nyu-depth", "depth"),
        ("ade20k", "images"),
        ("ade20k", "annotations"),
        ("cityscapes", "images"),
        ("cityscapes", "annotations"),
    ],
)
def test_dense_organizers_reject_symlinked_output_layout_directories(
    tmp_path: Path,
    dataset: str,
    layout_name: str,
) -> None:
    """Reject symlinked managed layout children without modifying their targets."""

    output_dir = tmp_path / "managed"
    output_dir.mkdir()
    protected_dir = tmp_path / "protected-layout"
    protected_dir.mkdir()
    marker = protected_dir / "keep"
    marker.write_bytes(b"existing")
    (output_dir / layout_name).symlink_to(protected_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="layout directories must not be symlinks"):
        if dataset == "nyu-depth":
            organizer.organize_nyu_depth("unused", str(output_dir))
        elif dataset == "ade20k":
            organizer.organize_ade20k("unused", str(output_dir))
        else:
            organizer.organize_cityscapes(
                "unused-images", "unused-annotations", str(output_dir)
            )

    assert marker.read_bytes() == b"existing"
    assert list(protected_dir.iterdir()) == [marker]
    assert (output_dir / layout_name).is_symlink()


def test_organize_cityscapes_materializes_lossless_validation_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Copy only exactly paired validation images and label-ID masks without transcoding."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setattr(organizer, "dataset_ready", lambda *_: True)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)
    sample_ids = ["frankfurt_000000_000294", "munster_000001_000019"]
    image_archive, annotation_archive = _write_cityscapes_archives(tmp_path, sample_ids)
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "stale.png").write_bytes(b"stale")
    (output_dir / "annotations" / "stale.png").write_bytes(b"stale")

    organizer.organize_cityscapes(
        str(image_archive), str(annotation_archive), str(output_dir)
    )

    image_paths = sorted((output_dir / "images").glob("*.png"))
    annotation_paths = sorted((output_dir / "annotations").glob("*.png"))
    assert [path.name for path in image_paths] == [
        f"{sample_id}.png" for sample_id in sample_ids
    ]
    assert [path.name for path in annotation_paths] == [
        f"{sample_id}.png" for sample_id in sample_ids
    ]
    assert image_paths[0].read_bytes() == f"image:{sample_ids[0]}".encode()
    assert annotation_paths[0].read_bytes() == f"mask:{sample_ids[0]}".encode()
    assert not list(output_dir.rglob("*train*"))
    assert not list(output_dir.rglob("*color*"))
    assert not list(output_dir.rglob("*instance*"))
    assert not list(output_dir.rglob("*.json"))


def test_organize_cityscapes_enforces_validation_pair_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject incomplete validation sources before replacing existing data."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    image_archive, annotation_archive = _write_cityscapes_archives(
        tmp_path, ["lindau_000000_000019"]
    )
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    marker = output_dir / "images" / "keep.png"
    marker.write_bytes(b"keep")

    with pytest.raises(ValueError, match="must contain 2 pairs"):
        organizer.organize_cityscapes(
            str(image_archive), str(annotation_archive), str(output_dir)
        )

    assert marker.read_bytes() == b"keep"


def test_organize_cityscapes_rejects_mismatched_and_malformed_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require exact official stems and reject malformed validation candidates."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    image_archive, annotation_archive = _write_cityscapes_archives(
        tmp_path, ["lindau_000000_000019"]
    )
    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr(
            "gtFine/val/lindau/lindau_000000_000020_gtFine_labelIds.png", b"mask"
        )
    with pytest.raises(ValueError, match="mismatch"):
        organizer.organize_cityscapes(
            str(image_archive), str(annotation_archive), str(tmp_path / "mismatched")
        )

    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr(
            "gtFine/val/lindau/frankfurt_000000_000019_gtFine_labelIds.png", b"mask"
        )
    with pytest.raises(ValueError, match="Malformed Cityscapes annotation filename"):
        organizer.organize_cityscapes(
            str(image_archive), str(annotation_archive), str(tmp_path / "malformed")
        )


def test_organize_cityscapes_rejects_non_zip_duplicate_and_unsafe_inputs(
    tmp_path: Path,
) -> None:
    """Reject invalid ZIPs, duplicate members, and traversal paths before installation."""

    invalid_archive = tmp_path / "invalid.zip"
    invalid_archive.write_bytes(b"not a zip")
    valid_image_archive, valid_annotation_archive = _write_cityscapes_archives(
        tmp_path,
        ["lindau_000000_000019"],
    )
    with pytest.raises(ValueError, match="does not exist"):
        organizer.organize_cityscapes(
            str(tmp_path / "missing.zip"),
            str(valid_annotation_archive),
            str(tmp_path / "missing"),
        )
    with pytest.raises(ValueError, match="must be a valid ZIP"):
        organizer.organize_cityscapes(
            str(invalid_archive),
            str(valid_annotation_archive),
            str(tmp_path / "invalid"),
        )

    duplicate_archive = tmp_path / "duplicate.zip"
    duplicate_member = "leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(duplicate_archive, "w") as archive:
            archive.writestr(duplicate_member, b"first")
            archive.writestr(duplicate_member, b"second")
    with pytest.raises(ValueError, match="duplicate members"):
        organizer.organize_cityscapes(
            str(duplicate_archive),
            str(valid_annotation_archive),
            str(tmp_path / "duplicate"),
        )

    unsafe_archive = tmp_path / "unsafe.zip"
    outside_marker = tmp_path / "outside.png"
    with ZipFile(unsafe_archive, "w") as archive:
        archive.writestr("../outside.png", b"outside")
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        organizer.organize_cityscapes(
            str(unsafe_archive), str(valid_annotation_archive), str(tmp_path / "unsafe")
        )
    assert not outside_marker.exists()


def test_organize_cityscapes_rolls_back_failed_atomic_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restore both previous directories if the staged installation fails."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(organizer, "dataset_ready", lambda *_: True)
    monkeypatch.setattr(organizer, "_validate_staged_payloads", lambda *_: None)
    image_archive, annotation_archive = _write_cityscapes_archives(
        tmp_path, ["lindau_000000_000019"]
    )
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "keep.png").write_bytes(b"old image")
    (output_dir / "annotations" / "keep.png").write_bytes(b"old annotation")
    real_replace = os.replace
    failed = False

    def _fail_annotation_install(source: str, destination: str) -> None:
        nonlocal failed
        if (
            not failed
            and Path(source).name == "annotations"
            and Path(destination) == output_dir / "annotations"
        ):
            failed = True
            raise OSError("simulated install failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "mblt_vision.utils.datasets.organizer.os.replace", _fail_annotation_install
    )

    with pytest.raises(OSError, match="simulated"):
        organizer.organize_cityscapes(
            str(image_archive), str(annotation_archive), str(output_dir)
        )

    assert (output_dir / "images" / "keep.png").read_bytes() == b"old image"
    assert (output_dir / "annotations" / "keep.png").read_bytes() == b"old annotation"


def test_organize_cityscapes_preserves_cache_when_staging_fails_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validate the staged tree before it can replace a usable Cityscapes cache."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    monkeypatch.setattr(organizer, "dataset_ready", lambda *_: False)
    image_archive, annotation_archive = _write_cityscapes_archives(
        tmp_path, ["lindau_000000_000019"]
    )
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "keep.png").write_bytes(b"old image")
    (output_dir / "annotations" / "keep.png").write_bytes(b"old annotation")

    with pytest.raises(ValueError, match="failed identity and completeness checks"):
        organizer.organize_cityscapes(
            str(image_archive), str(annotation_archive), str(output_dir)
        )

    assert (output_dir / "images" / "keep.png").read_bytes() == b"old image"
    assert (output_dir / "annotations" / "keep.png").read_bytes() == b"old annotation"


def test_dense_install_preserves_backups_when_rollback_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep recoverable backups outside staging when rollback cannot finish."""

    staging_dir = tmp_path / ".staging"
    staged_images = staging_dir / "images"
    staged_annotations = staging_dir / "annotations"
    staged_images.mkdir(parents=True)
    staged_annotations.mkdir()
    (staged_images / "new.png").write_bytes(b"new image")
    (staged_annotations / "new.png").write_bytes(b"new annotation")

    output_dir = tmp_path / "organized"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "keep.png").write_bytes(b"old image")
    (output_dir / "annotations" / "keep.png").write_bytes(b"old annotation")

    real_replace = os.replace

    def _fail_install_and_rollback(source: str, destination: str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == staged_annotations:
            raise OSError("simulated install failure")
        if (
            source_path.parent.name.startswith(".dense-backup-")
            and destination_path == output_dir / "images"
        ):
            raise OSError("simulated rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "mblt_vision.utils.datasets.organizer.os.replace", _fail_install_and_rollback
    )

    replacements = (
        (str(staged_images), str(output_dir / "images")),
        (str(staged_annotations), str(output_dir / "annotations")),
    )
    with pytest.raises(OSError, match="backups are preserved"):
        organizer._replace_staged_directories(
            replacements, str(tmp_path), ".dense-backup-"
        )

    backup_dirs = list(tmp_path.glob(".dense-backup-*"))
    assert len(backup_dirs) == 1
    assert (backup_dirs[0] / "images" / "keep.png").read_bytes() == b"old image"
    assert (
        backup_dirs[0] / "annotations" / "keep.png"
    ).read_bytes() == b"old annotation"


def _write_sav_fixture_tree(root: Path, *, video_id: str = "sav_000001") -> None:
    """Write a tiny official-layout SA-V validation tree with real payloads."""

    frame = Image.new("RGB", (16, 12), color=(30, 60, 90))
    mask = Image.new("L", (16, 12), color=0)
    mask.paste(255, (4, 3, 12, 9))
    (root / "sav_val.txt").write_text(f"{video_id}\n", encoding="utf-8")
    frame_dir = root / "JPEGImages_24fps" / video_id
    object_dir = root / "Annotations_6fps" / video_id / "000"
    frame_dir.mkdir(parents=True)
    object_dir.mkdir(parents=True)
    for stem in ("00000", "00001", "00002", "00003", "00004"):
        frame.save(frame_dir / f"{stem}.jpg")
    for stem in ("00000", "00004"):
        mask.save(object_dir / f"{stem}.png")


def test_organize_sav_keeps_only_annotated_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Install only annotated SA-V frames and masks through staged validation."""

    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_VIDEO_COUNT", 1)
    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_MASKLET_COUNT", 1)
    # _write_sav_fixture_tree annotates 2 of its 5 frames for the one masklet.
    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_MASK_COUNT", 2)
    source_root = tmp_path / "source" / "sav_val"
    source_root.mkdir(parents=True)
    _write_sav_fixture_tree(source_root)
    archive_path = tmp_path / "sav_val.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source_root, arcname="sav_val")

    output_dir = tmp_path / "organized"
    organizer.organize_sav(str(archive_path), str(output_dir))

    assert archive_path.is_file()
    video_images = output_dir / "images" / "sav_000001"
    assert sorted(path.name for path in video_images.iterdir()) == [
        "00000.jpg",
        "00004.jpg",
    ]
    masks = output_dir / "annotations" / "sav_000001" / "000"
    assert sorted(path.name for path in masks.iterdir()) == ["00000.png", "00004.png"]
    assert (output_dir / "video_ids.txt").read_text(encoding="utf-8") == "sav_000001\n"
    assert readiness_module.dataset_ready(output_dir, "mask_generation", "sa-v")


@pytest.mark.parametrize(
    "malicious_id", ["../../escape", "sav_000001/../../escape", "/abs/escape"]
)
def test_construct_sav_rejects_traversal_video_ids(
    tmp_path: Path, malicious_id: str
) -> None:
    """Reject ids that would resolve outside staging, before any file is written.

    `sav_val.txt` is file content rather than a directory listing, so an entry
    like `../../escape` could otherwise copy attacker-controlled PNGs into the
    output parent before organization failed.
    """

    source_root = tmp_path / "source" / "sav_val"
    source_root.mkdir(parents=True)
    _write_sav_fixture_tree(source_root)
    (source_root / "sav_val.txt").write_text(f"{malicious_id}\n", encoding="utf-8")
    escaped = source_root / "JPEGImages_24fps" / malicious_id
    output_dir = tmp_path / "organized"
    before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(ValueError, match="unsupported video ids|escapes the staging"):
        organizer.construct_sav(str(source_root), str(output_dir))

    del escaped
    # Nothing was written outside the intended output directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_construct_sav_rejects_missing_annotated_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail organization when an annotated frame has no matching JPEG."""

    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_VIDEO_COUNT", 1)
    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_MASKLET_COUNT", 1)
    # _write_sav_fixture_tree annotates 2 of its 5 frames for the one masklet.
    monkeypatch.setattr(readiness_module, "SAV_VALIDATION_MASK_COUNT", 2)
    source_root = tmp_path / "sav_val"
    source_root.mkdir(parents=True)
    _write_sav_fixture_tree(source_root)
    (source_root / "JPEGImages_24fps" / "sav_000001" / "00004.jpg").unlink()

    with pytest.raises(ValueError, match="missing annotated frame 00004.jpg"):
        organizer.construct_sav(str(source_root.parent), str(tmp_path / "organized"))


def test_validate_staged_sav_masks_rejects_non_binary_masks(tmp_path: Path) -> None:
    """Reject a staged SA-V mask carrying more than one object id."""

    staged_root = tmp_path / "staged"
    image_dir = staged_root / "images" / "sav_000001"
    object_dir = staged_root / "annotations" / "sav_000001" / "000"
    image_dir.mkdir(parents=True)
    object_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "00000.jpg")
    multi_object = Image.new("L", (8, 8), color=0)
    multi_object.paste(1, (0, 0, 2, 2))
    multi_object.paste(2, (4, 4, 6, 6))
    multi_object.save(object_dir / "00000.png")

    with pytest.raises(ValueError, match="single-object binary map"):
        organizer._validate_staged_sav_masks(staged_root)

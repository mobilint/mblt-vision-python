"""Prepared query/gallery person crops; no download or identity relabeling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...datasets import get_dataset_config_for_task

_FILENAME = re.compile(r"^(-?\d+)_c(\d+)(?:s\d+)?(?:_|\.)")


@dataclass(frozen=True)
class ReIDSample:
    """One crop with source identity, zero-based camera and query/gallery split."""

    image: Path
    pid: int
    camid: int
    split: Literal["query", "gallery"]


class ReIDDataset:
    """Read a prepared benchmark root in source-defined deterministic order.

    ``root`` is the directory containing ``query`` and ``bounding_box_test``
    (the ops prepared ``eval`` directory). Counts are checked by default;
    ``validate_counts=False`` explicitly supports caller-owned small subsets.
    """

    name: str

    def __init__(
        self, root: str | Path | None = None, *, validate_counts: bool = True
    ) -> None:
        self.config = get_dataset_config_for_task("person_reid", self.name)
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path(self.config["path"]) / self.config["val"]
        )
        samples = []
        for split in ("query", "gallery"):
            options = self.config["retrieval_splits"][split]
            directory = self.root / options["directory"]
            if not directory.is_dir():
                raise FileNotFoundError(f"ReID split directory is missing: {directory}")
            rows = []
            for path in sorted(directory.glob("*.jpg")):
                match = _FILENAME.match(path.name)
                if match is None:
                    raise ValueError(
                        f"ReID file name carries no <pid>_c<cam> token: {path}"
                    )
                pid, camera = map(int, match.groups())
                if pid < -1:
                    raise ValueError(
                        f"Unsupported negative person identity {pid}: {path}"
                    )
                if pid == -1:
                    if self.config["skip_junk"]:
                        continue
                    raise ValueError(f"Unexpected junk identity: {path}")
                if not 1 <= camera <= self.config["cameras"]:
                    raise ValueError(f"Camera {camera} outside benchmark range: {path}")
                rows.append(ReIDSample(path, pid, camera - 1, split))
            if not rows:
                raise ValueError(f"ReID split directory holds no images: {directory}")
            if validate_counts and len(rows) != options["count"]:
                raise ValueError(
                    f"{split} has {len(rows)} crops, expected {options['count']}"
                )
            samples.extend(rows)
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReIDSample:
        return self.samples[index]


class Market1501Dataset(ReIDDataset):
    """Market-1501: discard pid -1; retain pid 0 background distractors."""

    name = "market1501"


class DukeMTMCReIDDataset(ReIDDataset):
    """DukeMTMC-reID prepared held-out query/gallery crops."""

    name = "dukemtmc-reid"


class OccludedDukeDataset(ReIDDataset):
    """Occluded-Duke prepared query split and Duke gallery."""

    name = "occluded-duke"

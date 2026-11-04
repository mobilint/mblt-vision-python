"""Prepared aligned face pairs, preserving source file order and fold labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ...datasets import get_dataset_config_for_task


@dataclass(frozen=True)
class FacePair:
    """Source basenames, same/different label and official one-based fold."""

    image_a: str
    image_b: str
    is_same: bool
    fold: int


@dataclass(frozen=True)
class PairImage:
    """One prepared 112x112 aligned crop, keyed by its source basename."""

    image: Path
    name: str


class FacePairDataset:
    """Read prepared pairs.csv and already aligned crops without realignment.

    Root contains pairs.csv and the configured aligned/images directory.
    Source pair counts are enforced unless validate_counts=False explicitly
    selects a caller-owned subset. Basenames, labels and folds are always checked.
    """

    name: str

    def __init__(
        self, root: str | Path | None = None, *, validate_counts: bool = True
    ) -> None:
        self.config = get_dataset_config_for_task("face_recognition", self.name)
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path(self.config["path"]) / self.config["val"]
        )
        path = self.root / "pairs.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Pair file is missing: {path}")
        pairs = []
        ids = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = {"pair_id", "fold_id", "is_same", "image_a", "image_b"}
            if reader.fieldnames is None or not columns <= set(reader.fieldnames):
                raise ValueError(f"{path} lacks the pair columns {sorted(columns)}")
            for row in reader:
                for key in ("image_a", "image_b"):
                    name = row[key]
                    if (
                        not name
                        or name in {".", ".."}
                        or any(c in name for c in "/\\:")
                    ):
                        raise ValueError(f"Invalid pair image basename: {name!r}")
                if row["is_same"] not in {"0", "1"}:
                    raise ValueError("Pair is_same must be 0 or 1")
                try:
                    fold, pair_id = int(row["fold_id"]), int(row["pair_id"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        "Pair fold_id and pair_id must be integers"
                    ) from exc
                if not 1 <= fold <= 10 or pair_id in ids:
                    raise ValueError(
                        "Pair fold must be 1..10 and pair_id must be unique"
                    )
                ids.add(pair_id)
                pairs.append(
                    FacePair(
                        row["image_a"], row["image_b"], row["is_same"] == "1", fold
                    )
                )
        if not pairs or (validate_counts and len(pairs) != self.config["pairs"]):
            raise ValueError(
                f"Pair file holds {len(pairs)} pairs, expected {self.config['pairs']}"
            )
        self.pairs = tuple(pairs)
        image_dir = self.root / self.config["image_directory"]
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory is missing: {image_dir}")
        names = sorted({p.image_a for p in pairs} | {p.image_b for p in pairs})
        images = []
        for name in names:
            path = image_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Pair image is missing: {path}")
            # Alignment is a dataset preparation contract, never re-estimated here.
            with Image.open(path) as image:
                if image.size != (112, 112):
                    raise ValueError(f"Prepared aligned crop must be 112x112: {path}")
            images.append(PairImage(path, name))
        self.samples = tuple(images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PairImage:
        return self.samples[index]


class LFWDataset(FacePairDataset):
    """LFW official 6000 prepared aligned pairs."""

    name = "lfw"


class AgeDB30Dataset(FacePairDataset):
    """Prepared AgeDB-30 mirror pair protocol, without changing its folds."""

    name = "agedb-30"


class CFPFPDataset(FacePairDataset):
    """Prepared CFP-FP mirror protocol; source caveats remain in the YAML."""

    name = "cfp-fp"

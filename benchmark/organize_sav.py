"""Organize the SA-V validation dataset for local use.

SA-V is distributed by Meta through a form-gated download portal and is not
mirrored by this package, so `--dataset-path` is required: download
`sav_val.tar` yourself and pass it here, exactly as Cityscapes requires its
official archives. Both the archive and an already-extracted `sav_val`
directory are accepted.

The official source layout is documented in the SAM 2 repository:
https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md

    sav_val
    |-- sav_val.txt              # video ids in the split
    |-- JPEGImages_24fps/{video_id}/{frame:05d}.jpg
    `-- Annotations_6fps/{video_id}/{object_id:03d}/{frame:05d}.png

Only the annotated frames are installed: annotations exist at 6fps while
frames are extracted at 24fps, and evaluation can only use annotated frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mblt_vision.datasets import get_dataset_config
from mblt_vision.utils.datasets import organize_sav

SAV_DOWNLOAD_CONFIG = get_dataset_config("sa-v")["download"]


def main() -> None:
    """Parse organizer options and materialize SA-V validation data."""

    parser = argparse.ArgumentParser(
        description=(
            "Organize SA-V validation data from the official sav_val.tar archive. "
            f"Download it at {SAV_DOWNLOAD_CONFIG['source']} "
            f"(layout: {SAV_DOWNLOAD_CONFIG['documentation']})."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            f"Path to the manually downloaded {SAV_DOWNLOAD_CONFIG['archive']} "
            "or its extracted sav_val directory"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Destination for the organized images/, annotations/, and "
            "video_ids.txt (defaults to cache)"
        ),
    )
    args = parser.parse_args()
    organize_sav(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

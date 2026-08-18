"""Organize the NYU Depth dataset for local use.

The script downloads the published NYU Depth archive by default, or accepts a
local zip file or extracted dataset directory.
"""

import argparse
import sys
from pathlib import Path

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mblt_vision.utils.datasets import organize_nyu_depth
from mblt_vision.utils.datasets.organizer import NYU_DEPTH_URL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize NYU Depth dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=NYU_DEPTH_URL,
        help="Local path or download URL for the NYU Depth zip file or extracted dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the directory to save the organized dataset (defaults to cache)",
    )
    args = parser.parse_args()

    organize_nyu_depth(dataset_path=args.dataset_path, output_dir=args.output_dir)

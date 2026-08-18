"""
Script to organize the COCO dataset.

This script takes local archives or downloadable sources for the COCO dataset
and organizes them into a structure suitable for the model zoo.
"""

import argparse
import sys
from pathlib import Path

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mblt_vision.datasets import get_dataset_config
from mblt_vision.utils.datasets import organize_coco

COCO_DOWNLOAD_CONFIG = get_dataset_config("coco")["download"]
DEFAULT_COCO_IMAGE_SOURCE = COCO_DOWNLOAD_CONFIG["images"]
DEFAULT_COCO_ANNOTATION_SOURCE = COCO_DOWNLOAD_CONFIG["annotations"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize COCO dataset")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_COCO_IMAGE_SOURCE,
        help="Local path or download URL for the image zip file",
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default=DEFAULT_COCO_ANNOTATION_SOURCE,
        help="Local path or download URL for the annotation zip file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the directory to save the organized dataset (defaults to cache)",
    )
    args = parser.parse_args()

    organize_coco(
        image_dir=args.image_dir,
        annotation_dir=args.ann_dir,
        output_dir=args.output_dir,
    )

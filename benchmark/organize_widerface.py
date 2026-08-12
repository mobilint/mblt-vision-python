"""
Script to organize the WiderFace dataset.

This script takes local archives or downloadable sources for the WiderFace
dataset and organizes them into a structure suitable for the model zoo.
"""

import argparse
import os
import sys
from pathlib import Path

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mblt_vision.utils.datasets import organize_widerface

DEFAULT_WIDERFACE_IMAGE_SOURCE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"
DEFAULT_WIDERFACE_ANNOTATION_SOURCE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize WiderFace dataset")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_WIDERFACE_IMAGE_SOURCE,
        help="Local path or download URL for the image zip file",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=DEFAULT_WIDERFACE_ANNOTATION_SOURCE,
        help="Local path or download URL for the annotation zip file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/widerface"),
        help="Path to the directory to save the organized dataset",
    )
    args = parser.parse_args()

    organize_widerface(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
    )

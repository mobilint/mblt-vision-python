"""Organize the ADE20K validation dataset for local use."""

import argparse
import os

from mblt_vision.utils.datasets import organize_ade20k
from mblt_vision.utils.datasets.organizer import ADE20K_URL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize ADE20K validation dataset")
    parser.add_argument(
        "--dataset-path",
        default=ADE20K_URL,
        help="Local path or download URL for ADE20K",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/.mblt_vision/datasets/ADEChallengeData2016"),
        help="Path to the organized validation dataset",
    )
    args = parser.parse_args()
    organize_ade20k(dataset_path=args.dataset_path, output_dir=args.output_dir)

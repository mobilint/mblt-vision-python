"""
Image reader preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from .base import PreOps


class Reader(PreOps):
    """
    Reader for loading images from file paths or converting existing objects.
    Supports "pil" and "numpy" reading styles.

    For ``style="pil"``, arrays must be ``uint8`` RGB values or finite floating-point
    RGB values. Floating-point arrays in ``[0, 1]`` are treated as normalized RGB and
    scaled to ``[0, 255]``; other floating-point arrays must already be in ``[0, 255]``.
    """

    def __init__(self, style: str) -> None:
        """Initializes the Reader operation.

        Args:
            style (str): Reading style, either "pil" or "numpy".
        """
        super().__init__()
        if not isinstance(style, str):
            raise TypeError(
                f"Reader style must be a string, got {type(style).__name__}."
            )
        if style.lower() not in {"pil", "numpy"}:
            raise ValueError(
                f"Unsupported Reader style {style!r}; expected 'pil' or 'numpy'."
            )
        self.style = style.lower()

    def __call__(
        self, x: str | Path | TensorLike | Image.Image
    ) -> np.ndarray | Image.Image:
        """Reads/converts the input into an image object.

        Args:
            x (str | Path | TensorLike | Image.Image): Input image path or image object.

        Returns:
            np.ndarray | Image.Image: Read image in the specified style.
        """
        if self.style == "numpy":
            if isinstance(x, np.ndarray):
                return x
            elif isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, (str, Path)):
                image = cv2.imread(str(x))
                if image is None:
                    raise FileNotFoundError(f"Image not found: {x}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(x, Image.Image):
                return np.array(x)
            else:
                raise TypeError(
                    f"Reader(style='numpy') does not support input type {type(x).__name__}."
                )
        elif self.style == "pil":
            if isinstance(x, np.ndarray):
                return Image.fromarray(self._to_uint8_image_array(x))
            elif isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
                return Image.fromarray(self._to_uint8_image_array(x))
            elif isinstance(x, (str, Path)):
                return Image.open(x).convert("RGB")
            elif isinstance(x, Image.Image):
                return x
            else:
                raise TypeError(
                    f"Reader(style='pil') does not support input type {type(x).__name__}."
                )
        else:
            raise RuntimeError(
                f"Reader has an invalid validated style: {self.style!r}."
            )

    @staticmethod
    def _to_uint8_image_array(image: np.ndarray) -> np.ndarray:
        """Convert a validated array to byte RGB values for Pillow."""

        if image.dtype == np.uint8:
            return image
        if not np.issubdtype(image.dtype, np.floating):
            raise TypeError(
                "Reader(style='pil') accepts uint8 arrays or floating-point arrays "
                "with RGB values in [0, 1] or [0, 255]; "
                f"got {image.dtype}."
            )
        if not np.isfinite(image).all():
            raise ValueError(
                "Reader(style='pil') requires floating-point image arrays to contain "
                "only finite RGB values."
            )

        min_value = float(image.min())
        max_value = float(image.max())
        if min_value < 0.0 or max_value > 255.0:
            raise ValueError(
                "Reader(style='pil') accepts floating-point RGB values only in "
                "[0, 1] or [0, 255]; "
                f"got range [{min_value}, {max_value}]."
            )
        if max_value <= 1.0:
            image = image * 255.0
        return np.rint(image).astype(np.uint8)

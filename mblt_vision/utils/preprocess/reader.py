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
from ._validation import normalize_uint8_rgb_array
from .base import PreOps


class Reader(PreOps):
    """
    Reader for loading images from file paths or converting existing objects.
    Supports "pil" and "numpy" reading styles.

    For ``style="pil"``, arrays must be ``uint8`` RGB values or finite floating-point
    RGB values. Floating-point arrays in ``[0, 1]`` are treated as normalized RGB and
    scaled to ``[0, 255]``; other floating-point arrays must already be in ``[0, 255]``.

    Array, tensor, and PIL inputs are RGB, and image files are decoded to RGB.
    ``color_mode="BGR"`` (``numpy`` style only) hands the model BGR instead, for
    networks trained on cv2's native channel order such as YOLOX: arrays are
    flipped into a new array, never in place, and files keep cv2's decoding.
    """

    def __init__(self, style: str, color_mode: str = "RGB") -> None:
        """Initializes the Reader operation.

        Args:
            style (str): Reading style, either "pil" or "numpy".
            color_mode: Channel order handed to the model, "RGB" or "BGR".

        Raises:
            ValueError: If the style or color mode is unsupported, or BGR is
                requested from the RGB-only ``pil`` style.
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
        if not isinstance(color_mode, str) or color_mode.upper() not in {"RGB", "BGR"}:
            raise ValueError(
                f"Unsupported Reader color_mode {color_mode!r}; expected 'RGB' or 'BGR'."
            )
        self.color_mode = color_mode.upper()
        if self.style == "pil" and self.color_mode == "BGR":
            raise ValueError("Reader(style='pil') produces RGB images only.")

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
            if isinstance(x, (str, Path)):
                image = cv2.imread(str(x))
                if image is None:
                    raise FileNotFoundError(f"Image not found: {x}")
                if self.color_mode == "BGR":
                    return image
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if isinstance(x, np.ndarray):
                rgb = x
            elif isinstance(x, torch.Tensor):
                rgb = x.detach().cpu().numpy()
            elif isinstance(x, Image.Image):
                rgb = np.array(x)
            else:
                raise TypeError(
                    f"Reader(style='numpy') does not support input type {type(x).__name__}."
                )
            if self.color_mode == "RGB":
                return rgb
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(
                    f"Reader(color_mode='BGR') expects an HWC RGB image, got shape {rgb.shape}."
                )
            return np.ascontiguousarray(rgb[..., ::-1])
        elif self.style == "pil":
            if isinstance(x, np.ndarray):
                return Image.fromarray(
                    normalize_uint8_rgb_array(x, operation="Reader(style='pil')")
                )
            elif isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
                return Image.fromarray(
                    normalize_uint8_rgb_array(x, operation="Reader(style='pil')")
                )
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

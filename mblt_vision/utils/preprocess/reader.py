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
    """

    def __init__(self, style: str, color_mode: str = "RGB") -> None:
        """Initializes the Reader operation.

        Args:
            style (str): Reading style, either "pil" or "numpy".
            color_mode (str): Channel order this model expects, "RGB" or "BGR".
                Every family shipped before YOLOX wants RGB, so that stays the
                default; YOLOX is trained on cv2's native BGR and its scores drop
                without it. The flag describes the emitted array, so an already
                loaded array is taken as RGB — the library's own convention — and
                converted when BGR is asked for.

        Raises:
            TypeError: If style or color_mode is not a string.
            ValueError: If style or color_mode is unsupported, or BGR is asked of
                the PIL reader, which has no such mode.
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
        if not isinstance(color_mode, str):
            raise TypeError(
                f"Reader color_mode must be a string, got {type(color_mode).__name__}."
            )
        if color_mode.upper() not in {"RGB", "BGR"}:
            raise ValueError(
                f"Unsupported Reader color_mode {color_mode!r}; expected 'RGB' or 'BGR'."
            )
        self.style = style.lower()
        self.color_mode = color_mode.upper()
        if self.style == "pil" and self.color_mode == "BGR":
            raise ValueError(
                "Reader(style='pil') emits PIL RGB images; use style='numpy' for BGR."
            )

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
                return self._as_color_mode(x, source="RGB")
            elif isinstance(x, torch.Tensor):
                return self._as_color_mode(x.detach().cpu().numpy(), source="RGB")
            elif isinstance(x, (str, Path)):
                image = cv2.imread(str(x))
                if image is None:
                    raise FileNotFoundError(f"Image not found: {x}")
                # cv2 decodes to BGR, so RGB is the conversion and BGR is the file.
                return self._as_color_mode(image, source="BGR")
            elif isinstance(x, Image.Image):
                return self._as_color_mode(np.array(x), source="RGB")
            else:
                raise TypeError(
                    f"Reader(style='numpy') does not support input type {type(x).__name__}."
                )
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

    def _as_color_mode(self, image: np.ndarray, source: str) -> np.ndarray:
        """Return a three-channel array in the configured channel order."""

        if source == self.color_mode or image.ndim != 3 or image.shape[2] != 3:
            return image
        return image[..., ::-1].copy()

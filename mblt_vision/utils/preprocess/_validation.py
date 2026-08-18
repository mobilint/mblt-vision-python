"""Validation helpers shared by image preprocessing operations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_uint8_rgb_array(image: np.ndarray, *, operation: str) -> np.ndarray:
    """Return byte RGB data after validating or scaling a floating-point image.

    ``[0, 1]`` floating-point input is treated as normalized RGB and scaled to
    ``[0, 255]``. Other floating-point input must already lie in ``[0, 255]``.
    """

    if image.dtype == np.uint8:
        return image
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(
            f"{operation} accepts uint8 arrays or floating-point arrays with RGB "
            f"values in [0, 1] or [0, 255]; got {image.dtype}."
        )
    if not np.isfinite(image).all():
        raise ValueError(
            f"{operation} requires floating-point image arrays to contain only "
            "finite RGB values."
        )

    min_value = float(image.min())
    max_value = float(image.max())
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError(
            f"{operation} accepts floating-point RGB values only in [0, 1] or "
            f"[0, 255]; got range [{min_value}, {max_value}]."
        )
    if max_value <= 1.0:
        image = image * 255.0
    return np.rint(image).astype(np.uint8)


def normalize_image_size(size: int | Sequence[int], *, name: str = "size") -> list[int]:
    """Normalize a positive scalar or two-dimensional image size to ``[height, width]``."""

    if isinstance(size, bool):
        raise TypeError(
            f"{name} must be a positive integer or a two-item integer sequence, got bool."
        )
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{name} must be positive, got {size}.")
        return [size, size]
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) != 2:
            raise ValueError(f"{name} must contain exactly two items, got {size!r}.")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in size
        ):
            raise TypeError(f"{name} items must be integers, got {size!r}.")
        normalized = [int(size[0]), int(size[1])]
        if any(value <= 0 for value in normalized):
            raise ValueError(f"{name} items must be positive, got {size!r}.")
        return normalized
    raise TypeError(
        f"{name} must be a positive integer or a two-item integer sequence, got {type(size).__name__}."
    )

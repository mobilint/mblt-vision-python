"""Shared forward and inverse geometry for aspect-preserving letterboxing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

RatioPad: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class LetterBoxGeometry:
    """Geometry shared by letterbox preprocessing and output restoration."""

    input_shape: tuple[int, int]
    original_shape: tuple[int, int]
    ratio: float
    resized_shape: tuple[int, int]
    pad: tuple[int, int]

    @classmethod
    def from_shapes(
        cls,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
        center: bool = True,
    ) -> LetterBoxGeometry:
        """Calculate letterbox geometry, centered by default as Ultralytics does.

        Args:
            input_shape: Target shape as ``(height, width)``.
            original_shape: Source shape as ``(height, width)``.
            center: Split the padding evenly around the resized image. ``False``
                anchors the image at the top-left corner and pads only the
                bottom and right, as YOLOX and DAMO-YOLO do.

        Returns:
            Calculated resize ratio, resized shape, and top-left padding.
        """

        input_height, input_width = input_shape
        original_height, original_width = original_shape
        ratio = min(input_height / original_height, input_width / original_width)
        resized_height = int(round(original_height * ratio))
        resized_width = int(round(original_width * ratio))
        if center:
            left = int(round((input_width - resized_width) / 2 - 0.1))
            top = int(round((input_height - resized_height) / 2 - 0.1))
        else:
            left = top = 0
        return cls(
            input_shape=input_shape,
            original_shape=original_shape,
            ratio=ratio,
            resized_shape=(resized_height, resized_width),
            pad=(left, top),
        )

    @property
    def ratio_pad(self) -> RatioPad:
        """Return metadata consumed by inverse letterbox operations."""

        return ((self.ratio, self.ratio), self.pad)

    @property
    def borders(self) -> tuple[int, int, int, int]:
        """Return OpenCV border widths as ``(top, bottom, left, right)``."""

        input_height, input_width = self.input_shape
        resized_height, resized_width = self.resized_shape
        left, top = self.pad
        return (
            top,
            input_height - resized_height - top,
            left,
            input_width - resized_width - left,
        )

    def crop_bounds(
        self,
        output_shape: tuple[int, int],
        pad: tuple[float, float] | None = None,
    ) -> tuple[int, int, int, int]:
        """Scale inverse-letterbox crop bounds to a dense output shape.

        Args:
            output_shape: Dense output shape as ``(height, width)``.
            pad: Optional exact top-left padding metadata as ``(x, y)``.

        Returns:
            Crop bounds as ``(top, bottom, left, right)``.
        """

        output_height, output_width = output_shape
        input_height, input_width = self.input_shape
        scale_x = output_width / input_width
        scale_y = output_height / input_height
        pad_x, pad_y = self.pad if pad is None else pad
        left = int(round(pad_x * scale_x))
        top = int(round(pad_y * scale_y))
        resized_height, resized_width = self.resized_shape
        right = left + int(round(resized_width * scale_x))
        bottom = top + int(round(resized_height * scale_y))
        return top, bottom, left, right


def resolve_ratio_pad(
    input_shape: tuple[int, int],
    original_shape: tuple[int, int],
    ratio_pad: RatioPad | None = None,
    center: bool = True,
) -> RatioPad:
    """Return supplied letterbox metadata or derive it from image shapes.

    Args:
        input_shape: Letterboxed shape as ``(height, width)``.
        original_shape: Source shape as ``(height, width)``.
        ratio_pad: Optional metadata recorded during preprocessing.
        center: Anchoring used to derive missing metadata; see
            ``LetterBoxGeometry.from_shapes``.

    Returns:
        Resize ratios and top-left padding as ``((ratio_x, ratio_y), (pad_x, pad_y))``.
    """

    if ratio_pad is not None:
        return ratio_pad
    return LetterBoxGeometry.from_shapes(input_shape, original_shape, center).ratio_pad


def letterbox_center(pre_cfg: Mapping[str, Any]) -> bool:
    """Return whether a model's configured letterbox centers the image.

    Geometry derived from image shapes alone must use the model's own anchoring:
    assuming the Ultralytics default would shift every box of a top-left model by
    the whole padding.

    Args:
        pre_cfg: Model preprocessing configuration.

    Returns:
        ``pre_cfg.LetterBox.center``, ``True`` when absent.

    Raises:
        TypeError: If ``center`` is not a boolean.
    """

    letterbox_cfg = pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, Mapping):
        return True
    center = letterbox_cfg.get("center", True)
    if not isinstance(center, bool):
        raise TypeError(
            f"pre_cfg.LetterBox.center must be a boolean, got {type(center).__name__}."
        )
    return center

"""Shared forward and inverse geometry for aspect-preserving letterboxing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

RatioPad: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class LetterBoxGeometry:
    """Geometry shared by letterbox preprocessing and output restoration."""

    input_shape: tuple[int, int]
    original_shape: tuple[int, int]
    ratio: float
    resized_shape: tuple[int, int]
    pad: tuple[int, int]
    # Set only when the resize does not preserve aspect ratio, where one scale no
    # longer describes both axes. Left None for every centered YOLO letterbox, so
    # `ratio_pad` keeps emitting the `(ratio, ratio)` pair its callers already read.
    ratio_xy: tuple[float, float] | None = None

    @classmethod
    def from_shapes(
        cls,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
        *,
        center: bool = True,
        keep_ratio: bool = True,
    ) -> LetterBoxGeometry:
        """Calculate letterbox geometry for one of the three published conventions.

        Args:
            input_shape: Target shape as ``(height, width)``.
            original_shape: Source shape as ``(height, width)``.
            center: Center the resized image inside the canvas, as Ultralytics does.
                ``False`` anchors it at the top-left corner, which is what YOLOX's
                ``preproc()`` fills and therefore what its boxes are relative to.
            keep_ratio: Preserve aspect ratio and pad the remainder. ``False`` stretches
                the image onto the canvas with no padding at all, which is DAMO-YOLO's
                ``Resize(keep_ratio=False)``; the two axes then scale differently and the
                geometry reports both.

        Returns:
            Calculated resize ratio, resized shape, and top-left padding.
        """

        input_height, input_width = input_shape
        original_height, original_width = original_shape
        if not keep_ratio:
            ratio_x = input_width / original_width
            ratio_y = input_height / original_height
            return cls(
                input_shape=input_shape,
                original_shape=original_shape,
                ratio=min(ratio_x, ratio_y),
                resized_shape=(input_height, input_width),
                pad=(0, 0),
                ratio_xy=(ratio_x, ratio_y),
            )
        ratio = min(input_height / original_height, input_width / original_width)
        resized_height = int(round(original_height * ratio))
        resized_width = int(round(original_width * ratio))
        if center:
            left = int(round((input_width - resized_width) / 2 - 0.1))
            top = int(round((input_height - resized_height) / 2 - 0.1))
        else:
            left, top = 0, 0
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

        return (self.ratio_xy or (self.ratio, self.ratio), self.pad)

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
    *,
    center: bool = True,
    keep_ratio: bool = True,
) -> RatioPad:
    """Return supplied letterbox metadata or derive it from image shapes.

    Args:
        input_shape: Letterboxed shape as ``(height, width)``.
        original_shape: Source shape as ``(height, width)``.
        ratio_pad: Optional metadata recorded during preprocessing.

    Returns:
        Resize ratios and top-left padding as ``((ratio_x, ratio_y), (pad_x, pad_y))``.
    """

    if ratio_pad is not None:
        return ratio_pad
    return LetterBoxGeometry.from_shapes(
        input_shape, original_shape, center=center, keep_ratio=keep_ratio
    ).ratio_pad

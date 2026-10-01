"""Shared forward and inverse geometry for aspect-preserving letterboxing."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

RatioPad: TypeAlias = tuple[tuple[float, float], tuple[float, float]]
SizeRounding: TypeAlias = Literal["round", "floor"]
_SIZE_ROUNDINGS = ("round", "floor")


@dataclass(frozen=True)
class LetterBoxLayout:
    """How a model's letterbox sizes, places, and later inverts its resize.

    The defaults are Ultralytics': Python ``round`` for the resized size, the
    padding split around the image, and one uniform ratio for restoration.
    Upstream YOLOX (``data_augment.preproc``) and DAMO-YOLO (``Resize`` of the
    December 2022 release) truncate the size with ``int(w * r)`` and anchor the
    image top-left. YOLOX restores boxes by dividing by ``r``; DAMO-YOLO's
    ``BoxList.resize`` scales each axis by the original over the truncated size.

    Attributes:
        center: Split the padding around the image, or anchor it top-left.
        size_rounding: ``"round"`` (half to even) or ``"floor"`` (upstream ``int()``).
        per_axis_ratio: Restore x and y by the resized-to-original ratio of each
            axis instead of the one aspect-preserving ratio.
    """

    center: bool = True
    size_rounding: SizeRounding = "round"
    per_axis_ratio: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.center, bool):
            raise TypeError(
                f"LetterBox center must be a boolean, got {type(self.center).__name__}."
            )
        if self.size_rounding not in _SIZE_ROUNDINGS:
            raise ValueError(
                f"LetterBox size_rounding must be 'round' or 'floor', got {self.size_rounding!r}."
            )
        if not isinstance(self.per_axis_ratio, bool):
            raise TypeError(
                "LetterBox per_axis_ratio must be a boolean, "
                f"got {type(self.per_axis_ratio).__name__}."
            )

    @property
    def is_default(self) -> bool:
        """Whether this is the Ultralytics layout the shape-only helpers assume."""

        return self == LetterBoxLayout()


def deprecated_center_argument(
    layout: LetterBoxLayout | bool, center: bool | None, owner: str
) -> LetterBoxLayout | bool:
    """Resolve the ``center=`` keyword that ``layout=`` replaced.

    Args:
        layout: The ``layout`` argument as passed, ``True`` when omitted.
        center: The deprecated ``center`` keyword, ``None`` when omitted.
        owner: Function name used in the warning and error.

    Returns:
        ``center`` when it was given, otherwise ``layout``.

    Raises:
        TypeError: If both ``layout`` and ``center`` were given.
    """

    if center is None:
        return layout
    if layout is not True:
        raise TypeError(f"{owner}() takes layout or the deprecated center, not both.")
    warnings.warn(
        f"{owner}(center=...) is deprecated; pass layout=LetterBoxLayout(center=...).",
        DeprecationWarning,
        stacklevel=3,
    )
    return center


def _as_layout(layout: LetterBoxLayout | bool) -> LetterBoxLayout:
    """Accept the older ``center`` boolean wherever a layout is expected."""

    if isinstance(layout, LetterBoxLayout):
        return layout
    return LetterBoxLayout(center=layout)


@dataclass(frozen=True)
class LetterBoxGeometry:
    """Geometry shared by letterbox preprocessing and output restoration."""

    input_shape: tuple[int, int]
    original_shape: tuple[int, int]
    ratio: float
    resized_shape: tuple[int, int]
    pad: tuple[int, int]
    per_axis_ratio: bool = False

    @classmethod
    def from_shapes(
        cls,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
        layout: LetterBoxLayout | bool = True,
        *,
        center: bool | None = None,
    ) -> LetterBoxGeometry:
        """Calculate letterbox geometry, Ultralytics' by default.

        Args:
            input_shape: Target shape as ``(height, width)``.
            original_shape: Source shape as ``(height, width)``.
            layout: The model's ``LetterBoxLayout``. A boolean is read as
                ``LetterBoxLayout(center=layout)``: ``False`` anchors the image at
                the top-left corner and pads only the bottom and right.
            center: Deprecated spelling of a boolean ``layout``.

        Returns:
            Calculated resize ratio, resized shape, and top-left padding.
        """

        layout = _as_layout(
            deprecated_center_argument(layout, center, "LetterBoxGeometry.from_shapes")
        )
        input_height, input_width = input_shape
        original_height, original_width = original_shape
        ratio = min(input_height / original_height, input_width / original_width)
        if layout.size_rounding == "floor":
            resized_height = int(original_height * ratio)
            resized_width = int(original_width * ratio)
        else:
            resized_height = int(round(original_height * ratio))
            resized_width = int(round(original_width * ratio))
        if layout.center:
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
            per_axis_ratio=layout.per_axis_ratio,
        )

    @property
    def ratio_pad(self) -> RatioPad:
        """Return metadata consumed by inverse letterbox operations.

        The ratios are ``(x, y)``: both the aspect-preserving ratio, or with
        ``per_axis_ratio`` each axis' resized-over-original size.
        """

        if self.per_axis_ratio:
            resized_height, resized_width = self.resized_shape
            original_height, original_width = self.original_shape
            return (
                (resized_width / original_width, resized_height / original_height),
                self.pad,
            )
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
    layout: LetterBoxLayout | bool = True,
    *,
    center: bool | None = None,
) -> RatioPad:
    """Return supplied letterbox metadata or derive it from image shapes.

    Args:
        input_shape: Letterboxed shape as ``(height, width)``.
        original_shape: Source shape as ``(height, width)``.
        ratio_pad: Optional metadata recorded during preprocessing.
        layout: Layout used to derive missing metadata; see
            ``LetterBoxGeometry.from_shapes``.
        center: Deprecated spelling of a boolean ``layout``.

    Returns:
        Resize ratios and top-left padding as ``((ratio_x, ratio_y), (pad_x, pad_y))``.
    """

    layout = deprecated_center_argument(layout, center, "resolve_ratio_pad")
    if ratio_pad is not None:
        return ratio_pad
    return LetterBoxGeometry.from_shapes(input_shape, original_shape, layout).ratio_pad


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

    return letterbox_layout(pre_cfg).center


def letterbox_layout(pre_cfg: Mapping[str, Any]) -> LetterBoxLayout:
    """Return a model's configured ``LetterBoxLayout``.

    Args:
        pre_cfg: Model preprocessing configuration.

    Returns:
        ``pre_cfg.LetterBox``'s ``center``, ``size_rounding`` and
        ``per_axis_ratio``, each defaulting to Ultralytics' behaviour.

    Raises:
        TypeError: If ``center`` or ``per_axis_ratio`` is not a boolean.
        ValueError: If ``size_rounding`` is not ``"round"`` or ``"floor"``.
    """

    letterbox_cfg = pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, Mapping):
        return LetterBoxLayout()
    try:
        return LetterBoxLayout(
            center=letterbox_cfg.get("center", True),
            size_rounding=letterbox_cfg.get("size_rounding", "round"),
            per_axis_ratio=letterbox_cfg.get("per_axis_ratio", False),
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"pre_cfg.{error}") from error

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from ..letterbox import LetterBoxGeometry, LetterBoxLayout, RatioPad, SizeRounding
from ..types import TensorLike
from ._validation import normalize_image_size, normalize_uint8_rgb_array
from .base import PreOps


def _apply_letterbox(
    image: np.ndarray,
    img_size: list[int],
    interpolation: int,
    padding_value: int | tuple[int, int, int],
    layout: LetterBoxLayout | bool = True,
) -> tuple[np.ndarray, RatioPad]:
    """Resize and pad an array while preserving its aspect ratio.

    Args:
        image: Image or two-dimensional semantic mask.
        img_size: Target size as ``[height, width]``.
        interpolation: OpenCV interpolation mode.
        padding_value: Constant border value.
        layout: The model's ``LetterBoxLayout``, or the older ``center`` boolean.

    Returns:
        The letterboxed array and its resize/padding metadata.
    """

    input_shape = (int(img_size[0]), int(img_size[1]))
    original_shape = (int(image.shape[0]), int(image.shape[1]))
    geometry = LetterBoxGeometry.from_shapes(input_shape, original_shape, layout)
    resized_height, resized_width = geometry.resized_shape
    if image.shape[:2] != geometry.resized_shape:
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )
    top, bottom, left, right = geometry.borders
    # cv2's border value fills only the first channel and zeros the rest when
    # given a bare scalar on a multi-channel image (cv::Scalar's single-value
    # constructor), so broadcast a scalar to one value per channel rather than
    # pass it through as-is -- a no-op for the already-per-channel tuple caller
    # and for the single-channel semantic-mask caller.
    channels = image.shape[2] if image.ndim == 3 else 1
    border_value: cv2.typing.Scalar = (
        (float(padding_value),) * channels
        if isinstance(padding_value, int)
        else tuple(float(component) for component in padding_value)
    )
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
    return image, geometry.ratio_pad


def letterbox_semantic_mask(
    mask: np.ndarray,
    img_size: list[int],
    ignore_label: int = 255,
    layout: LetterBoxLayout | bool = True,
) -> tuple[np.ndarray, RatioPad]:
    """Letterbox a semantic mask without interpolating class IDs.

    Args:
        mask: Two-dimensional semantic class map.
        img_size: Target size as ``[height, width]``.
        ignore_label: Class value used for padded pixels.
        layout: The model's ``letterbox_layout(pre_cfg)`` (or the older
            ``center`` boolean), so the target matches its image's geometry.

    Returns:
        The letterboxed mask and its resize/padding metadata.

    Raises:
        ValueError: If the mask is not two-dimensional.
    """

    if mask.ndim != 2:
        raise ValueError(
            f"Semantic masks must be two-dimensional, got shape {mask.shape}."
        )
    return _apply_letterbox(mask, img_size, cv2.INTER_NEAREST, ignore_label, layout)


class LetterBox(PreOps):
    """Preprocessing for YOLO models, implementing letterbox resizing.

    Resizes the image while maintaining aspect ratio, adding padding to meet
    target dimensions. Floating-point RGB inputs in ``[0, 1]`` are scaled to
    byte RGB; other floating-point values must be finite and in ``[0, 255]``.
    The defaults are Ultralytics' letterbox: centered, padded with 114, the
    resized size rounded. Upstream YOLOX truncates the size with ``int()`` and
    anchors it top-left with 114; DAMO-YOLO's December 2022 checkpoints do the
    same with zeros and restore each axis by its own ratio. PIL images, as
    ``Reader(style="pil")`` returns them, are converted with ``np.asarray`` the
    way DAMO-YOLO's dataset does.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    """

    def __init__(
        self,
        img_size: list[int],
        center: bool = True,
        padding_value: int = 114,
        size_rounding: SizeRounding = "round",
        per_axis_ratio: bool = False,
    ) -> None:
        """Initializes LetterBox with target image size.

        Args:
            img_size (list[int]): Target image size [h, w].
            center: Split the padding around the image, or anchor it top-left.
            padding_value: Byte value filling every channel of the padding.
            size_rounding: ``"round"`` the resized size, or ``"floor"`` it as
                upstream YOLOX and DAMO-YOLO do with ``int(w * r)``.
            per_axis_ratio: Record each axis' resized-over-original ratio for
                restoration instead of the aspect-preserving one.

        Raises:
            TypeError: If ``center`` or ``per_axis_ratio`` is not a boolean, or
                ``padding_value`` not an integer.
            ValueError: If ``padding_value`` is outside ``[0, 255]`` or
                ``size_rounding`` is unsupported.
        """
        super().__init__()
        self.img_size = normalize_image_size(img_size, name="img_size")
        self.layout = LetterBoxLayout(center, size_rounding, per_axis_ratio)
        if isinstance(padding_value, bool) or not isinstance(padding_value, int):
            raise TypeError(
                "LetterBox padding_value must be an integer, "
                f"got {type(padding_value).__name__}."
            )
        if not 0 <= padding_value <= 255:
            raise ValueError(
                f"LetterBox padding_value must be in [0, 255], got {padding_value}."
            )
        self.center = center
        self.size_rounding = size_rounding
        self.per_axis_ratio = per_axis_ratio
        self.padding_value = padding_value
        self.ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __call__(self, x: TensorLike | Image.Image) -> torch.Tensor:
        """Executes YOLO preprocessing (letterbox resizing).

        The call's geometry is also left in ``self.ratio_pad`` for compatibility.
        That attribute is shared by every caller of this instance, so concurrent
        callers must use ``with_ratio_pad`` instead.

        Args:
            x (TensorLike | Image.Image): Input image.

        Returns:
            torch.Tensor: Preprocessed image in HWC format on the selected device.
        """
        img, self.ratio_pad = self.with_ratio_pad(x)
        return img

    def with_ratio_pad(
        self, x: TensorLike | Image.Image
    ) -> tuple[torch.Tensor, RatioPad]:
        """Letterbox an image and return its geometry without touching instance state.

        Args:
            x (TensorLike | Image.Image): Input image.

        Returns:
            The preprocessed HWC image on the selected device and its ``ratio_pad``.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, Image.Image):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError(
                "LetterBox expects a NumPy array, tensor or PIL image, "
                f"got {type(x).__name__}."
            )
        if x.ndim != 3:
            raise ValueError(f"LetterBox expects an HWC image, got shape {x.shape}.")
        x = normalize_uint8_rgb_array(x, operation="LetterBox")
        img, ratio_pad = _apply_letterbox(
            x,
            self.img_size,
            cv2.INTER_LINEAR,
            (self.padding_value,) * 3,
            self.layout,
        )
        return torch.from_numpy(img).to(self.device).byte(), ratio_pad

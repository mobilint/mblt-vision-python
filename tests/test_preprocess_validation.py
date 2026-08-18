"""Runtime validation tests for shared Vision preprocessing."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from mblt_vision.utils.preprocess.center_crop import CenterCrop
from mblt_vision.utils.preprocess.letterbox import LetterBox
from mblt_vision.utils.preprocess.normalize import Normalize
from mblt_vision.utils.preprocess.order import SetOrder
from mblt_vision.utils.preprocess.reader import Reader
from mblt_vision.utils.preprocess.resize import Resize


@pytest.mark.parametrize(
    "operation", [CenterCrop(2), Resize(2, "bilinear"), LetterBox([2, 2])]
)
def test_image_size_operations_reject_nonpositive_sizes(operation: object) -> None:
    """Keep size validation active when Python assertions are optimized away."""

    operation_type = type(operation)
    if operation_type is Resize:
        with pytest.raises(ValueError, match="positive"):
            Resize(0, "bilinear")
    elif operation_type is CenterCrop:
        with pytest.raises(ValueError, match="positive"):
            CenterCrop([0, 2])
    else:
        with pytest.raises(ValueError, match="positive"):
            LetterBox([0, 2])


@pytest.mark.parametrize("size", [[1], [1, 2, 3], [1.5, 2], "2"])
def test_center_crop_rejects_invalid_size_values(size: object) -> None:
    """Reject invalid crop types and dimensions before processing."""

    with pytest.raises((TypeError, ValueError)):
        CenterCrop(size)  # type: ignore[arg-type]


def test_set_order_rejects_ambiguous_channel_layout() -> None:
    """Do not guess between CHW and HWC when both ends have three channels."""

    with pytest.raises(ValueError, match="ambiguous"):
        SetOrder("HWC")(np.zeros((3, 5, 3), dtype=np.uint8))


def test_normalize_rejects_ambiguous_channel_layout() -> None:
    """Do not silently apply HWC statistics to potentially CHW data."""

    with pytest.raises(ValueError, match="ambiguous"):
        Normalize("torch")(np.zeros((3, 640, 3), dtype=np.uint8))


def test_normalize_supports_channel_first_set_order_output() -> None:
    """Broadcast normalization statistics per channel after a CHW conversion."""

    image = np.array(
        [
            [[255, 0, 127], [0, 255, 127], [127, 127, 255], [64, 32, 16], [1, 2, 3]],
            [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30], [31, 32, 33]],
            [[34, 35, 36], [37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
        ],
        dtype=np.uint8,
    )

    normalized = Normalize("torch")(SetOrder("CHW")(image))
    expected = (
        image.transpose(2, 0, 1) / 255.0
        - np.array([0.485, 0.456, 0.406])[:, None, None]
    ) / np.array([0.229, 0.224, 0.225])[:, None, None]

    assert normalized.shape == (3, 4, 5)
    np.testing.assert_allclose(
        normalized, expected.astype(np.float32), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("operation", "output_type"),
    [
        (Reader("numpy"), np.ndarray),
        (Reader("pil"), Image.Image),
        (Normalize("cv"), np.ndarray),
        (CenterCrop([2, 3]), np.ndarray),
        (LetterBox([2, 3]), torch.Tensor),
    ],
)
def test_preprocessors_accept_grad_tracking_tensors(
    operation: object, output_type: type[object]
) -> None:
    """Detach tensors before preprocessing converts them to NumPy arrays."""

    image = torch.ones((4, 5, 3), requires_grad=True)

    assert isinstance(operation(image), output_type)  # type: ignore[operator]


def test_reader_pil_scales_normalized_float_arrays() -> None:
    """Convert normalized float RGB inputs deliberately instead of truncating them."""

    image = np.full((2, 3, 3), 0.5, dtype=np.float32)

    converted = np.asarray(Reader("pil")(image))

    np.testing.assert_array_equal(converted, np.full((2, 3, 3), 128, dtype=np.uint8))


def test_letterbox_scales_normalized_float_arrays() -> None:
    """Do not truncate normalized float RGB input before LetterBox resizing."""

    image = np.full((2, 2, 3), 0.5, dtype=np.float32)

    converted = LetterBox([4, 4])(image)

    assert converted.dtype == torch.uint8
    assert torch.equal(converted, torch.full((4, 4, 3), 128, dtype=torch.uint8))


@pytest.mark.parametrize(
    "image",
    [
        np.full((2, 3, 3), -0.1, dtype=np.float32),
        np.full((2, 3, 3), 256.0, dtype=np.float32),
        np.full((2, 3, 3), np.nan, dtype=np.float32),
    ],
)
def test_reader_pil_rejects_invalid_float_image_arrays(image: np.ndarray) -> None:
    """Do not silently cast invalid float RGB values to byte images."""

    with pytest.raises(ValueError, match=r"finite RGB values|\[0, 1\] or \[0, 255\]"):
        Reader("pil")(image)


@pytest.mark.parametrize(
    "image",
    [
        np.full((2, 3, 3), -0.1, dtype=np.float32),
        np.full((2, 3, 3), 256.0, dtype=np.float32),
        np.full((2, 3, 3), np.nan, dtype=np.float32),
    ],
)
def test_letterbox_rejects_invalid_float_image_arrays(image: np.ndarray) -> None:
    """Do not wrap out-of-range or non-finite float pixels to byte RGB."""

    with pytest.raises(ValueError, match=r"finite RGB values|\[0, 1\] or \[0, 255\]"):
        LetterBox([4, 4])(image)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: Reader("unsupported"), ValueError),
        (lambda: Normalize("unsupported"), ValueError),
        (lambda: SetOrder("unsupported"), ValueError),
    ],
)
def test_preprocessing_styles_raise_explicit_errors(
    factory, expected: type[Exception]
) -> None:
    """Use explicit configuration errors instead of runtime assertions."""

    with pytest.raises(expected):
        factory()


@pytest.mark.parametrize("interpolation", ["box", "hamming", "lanczos"])
@pytest.mark.parametrize(
    "image", [np.zeros((2, 3, 3), dtype=np.uint8), torch.zeros((3, 2, 3))]
)
def test_resize_rejects_pil_only_modes_for_tensor_backed_inputs(
    interpolation: str, image: np.ndarray | torch.Tensor
) -> None:
    """Fail clearly instead of passing unsupported modes to torch.interpolate."""

    with pytest.raises(ValueError, match="supported only for PIL images"):
        Resize([4, 6], interpolation)(image)


@pytest.mark.parametrize("interpolation", ["box", "hamming", "lanczos"])
def test_resize_keeps_pil_only_modes_available_for_pil_images(
    interpolation: str,
) -> None:
    """Retain the documented PIL resize modes for PIL callers."""

    image = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))

    resized = Resize([4, 6], interpolation)(image)

    assert isinstance(resized, Image.Image)
    assert resized.size == (6, 4)


def test_runtime_validation_survives_optimized_python() -> None:
    """Keep configuration checks active under ``python -O``."""

    code = """
from mblt_vision.utils.postprocess import build_postprocess
try:
    build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {
            "task": "object_detection",
            "dataset": "coco",
            "nl": 3,
            "reg_max": 16,
            "conf_thres": 0,
        },
    )
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True)

"""Tests for vision model ONNX inference on image classification."""

from __future__ import annotations

import pytest

from mblt_vision.image_classification import AlexNet, CAFormer_B36, YOLO26sCls

pytestmark = pytest.mark.requires_network


@pytest.mark.parametrize(
    "model_cls",
    [
        AlexNet,
        CAFormer_B36,
        YOLO26sCls,
    ],
)
def test_onnx_classification(model_cls, synthetic_image_path) -> None:
    """Run ONNX inference for representative classification models."""

    model = model_cls(framework="onnx")

    try:
        input_img = model.preprocess(str(synthetic_image_path))
        output = model(input_img)
        result = model.postprocess(output)

        assert result is not None
        assert result.task == "image_classification"
        assert result.acc is not None
        assert result.output is not None
    finally:
        model.dispose()

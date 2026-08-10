"""Optional end-to-end MXQ inference coverage for the standalone package."""

from __future__ import annotations

from pathlib import Path

import pytest

from mblt_npu.pytest_plugin import NpuParams
from mblt_vision import MBLT_Engine

pytestmark = [pytest.mark.requires_network, pytest.mark.requires_npu]


def test_mxq_classification_runs_with_shared_npu_options(npu_params: NpuParams) -> None:
    """Load a representative MXQ model using the shared NPU test options."""

    image_path = Path(__file__).parent / "rc" / "tabby.jpg"
    model = MBLT_Engine(model_cls="resnet50", **npu_params.base)
    try:
        result = model.postprocess(model(model.preprocess(str(image_path))))
        assert result.task == "image_classification"
        assert result.acc is not None
    finally:
        model.dispose()

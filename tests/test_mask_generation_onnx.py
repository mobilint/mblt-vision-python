"""Opt-in ONNX Runtime coverage for SAM2HieraLarge.

Requires network access (to download the two ONNX exports and the small
prompt-encoder weights bundle from ``mobilint/sam2-hiera-large``) and the
``onnxruntime`` optional extra -- but no NPU hardware, unlike
``tests/test_mask_generation_hardware.py``. Skipped by default, matching
"do not require ... downloads ... for ordinary unit tests" (AGENTS.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from mblt_vision.mask_generation import SAM2HieraLarge

pytestmark = pytest.mark.requires_network


def _synthetic_rectangle_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A simple synthetic image with one solid rectangle to segment."""

    image = np.full((480, 640, 3), 30, dtype=np.uint8)
    box = (150, 100, 400, 300)  # x0, y0, x1, y1
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = (200, 60, 60)
    return image, box


def test_sam2_onnx_predicts_a_precise_mask_for_a_synthetic_rectangle() -> None:
    """A single point inside a flat-colored rectangle should segment it precisely."""

    image, (x0, y0, x1, y1) = _synthetic_rectangle_image()

    with SAM2HieraLarge(framework="onnx") as engine:
        result = engine.predict(
            image, points=[[(x0 + x1) // 2, (y0 + y1) // 2]], labels=[1]
        )
        assert result.task == "mask_generation"
        assert result.masks is not None
        assert result.iou_predictions is not None
        assert result.masks.shape == (3, 480, 640)
        assert result.selected == int(np.argmax(result.iou_predictions))

        ground_truth = np.zeros((480, 640), dtype=bool)
        ground_truth[y0:y1, x0:x1] = True
        predicted = result.masks[result.selected]
        intersection = np.logical_and(predicted, ground_truth).sum()
        union = np.logical_or(predicted, ground_truth).sum()
        assert intersection / union > 0.9

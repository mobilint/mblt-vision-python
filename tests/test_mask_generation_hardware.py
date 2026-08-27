"""Opt-in real-hardware coverage for SAM2HieraLarge.

Requires real Aries NPU hardware and network access (to download the two MXQ
artifacts and the small prompt-encoder weights bundle from
``mobilint/sam2-hiera-large``, unless overridden with explicit local paths).
No external ``sam2`` package or manually cloned repository is needed -- see
``mblt_vision/mask_generation/_sam2_host.py``. Skipped by default, matching
"do not require hardware ... for ordinary unit tests" (AGENTS.md).

Override artifact paths with ``--encoder-mxq-path`` / ``--decoder-mxq-path``
(shared NPU pytest options, see ``mblt_npu.pytest_plugin``); omitted paths
fall back to downloading from the ``mobilint/sam2-hiera-large`` Hugging Face
repository.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mblt_npu.pytest_plugin import NpuParams
from mblt_vision.mask_generation import SAM2HieraLarge

pytestmark = [pytest.mark.requires_network, pytest.mark.requires_npu]


def _synthetic_rectangle_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A simple synthetic image with one solid rectangle to segment."""

    image = np.full((480, 640, 3), 30, dtype=np.uint8)
    box = (150, 100, 400, 300)  # x0, y0, x1, y1
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = (200, 60, 60)
    return image, box


def test_sam2_predicts_a_precise_mask_for_a_synthetic_rectangle(
    npu_params: NpuParams,
) -> None:
    """A single point inside a flat-colored rectangle should segment it precisely."""

    image, (x0, y0, x1, y1) = _synthetic_rectangle_image()

    engine = SAM2HieraLarge(**npu_params.encoder, **npu_params.decoder)
    try:
        result = engine.predict(
            image, points=[[(x0 + x1) // 2, (y0 + y1) // 2]], labels=[1]
        )
        assert result.task == "mask_generation"
        assert result.masks.shape == (3, 480, 640)
        assert result.selected == int(np.argmax(result.iou_predictions))

        ground_truth = np.zeros((480, 640), dtype=bool)
        ground_truth[y0:y1, x0:x1] = True
        predicted = result.masks[result.selected]
        intersection = np.logical_and(predicted, ground_truth).sum()
        union = np.logical_or(predicted, ground_truth).sum()
        iou = intersection / union
        assert (
            iou > 0.95
        ), f"Expected a near-perfect mask for a flat rectangle, got IoU={iou:.4f}."
    finally:
        engine.close()


def test_sam2_resolves_artifacts_from_huggingface_hub(npu_params: NpuParams) -> None:
    """Construct with no explicit paths and confirm all three artifacts resolve from Hub."""

    if npu_params.encoder or npu_params.decoder:
        pytest.skip(
            "This test specifically exercises Hub resolution with no explicit paths."
        )

    engine = SAM2HieraLarge()
    try:
        assert engine._encoder_backend.mxq_path.endswith("sam2_hiera_large_encoder.mxq")
        assert engine._decoder_backend.mxq_path.endswith("sam2_hiera_large_decoder.mxq")
        assert os.path.isfile(engine._encoder_backend.mxq_path)
        assert os.path.isfile(engine._decoder_backend.mxq_path)
        assert engine.weights is not None
    finally:
        engine.close()

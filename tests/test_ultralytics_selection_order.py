"""Candidate ranking reproduces Ultralytics' validation order on every device.

Ultralytics validates on CUDA, where ``torchvision.ops.nms``, the ``argsort`` it runs above
``max_nms`` candidates and an end-to-end head's ``torch.topk`` all order tied scores by
ascending index. Quantized MXQ scores tie often, so these tests pin that order on CPU too,
and the end-to-end selection's row count, against a direct reading of
``Detect.get_topk_index``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

import mblt_vision
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.postprocess.common import descending_order, dual_topk

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


def _model_cfg(name: str) -> tuple[dict, dict]:
    path = Path(mblt_vision.__file__).parent / "models" / f"{name}.yaml"
    config = yaml.safe_load(path.read_text())["DEFAULT"]
    return config["pre_cfg"], config["post_cfg"]


@pytest.mark.parametrize("device", DEVICES)
def test_descending_order_keeps_tied_scores_in_index_order(device: str) -> None:
    # Large enough that an unstable CPU sort reorders the ties.
    values = (torch.arange(5000) % 7 == 0).float().to(device)

    order = descending_order(values).cpu()

    expected = torch.cat(
        (torch.arange(0, 5000, 7), torch.tensor([i for i in range(5000) if i % 7]))
    )
    torch.testing.assert_close(order, expected)


def _ultralytics_end2end(rows: torch.Tensor, nc: int, conf: float) -> torch.Tensor:
    """``Detect.get_topk_index`` plus the validator's threshold, with CUDA's tie order."""
    scores = rows[:, 4 : 4 + nc]
    k = min(300, rows.shape[0])
    anchors = descending_order(scores.amax(dim=-1))[:k]
    flat = scores[anchors].reshape(-1)
    top = descending_order(flat)[:k]
    keep = flat[top] > conf
    top = top[keep]
    return torch.stack(
        (flat[top], (top % nc).to(rows.dtype), anchors[top // nc].to(rows.dtype)), dim=1
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dual_topk_keeps_every_class_of_the_anchors_that_clear_the_threshold(
    device: str,
) -> None:
    """100 anchors with three classes each keep 300 rows, as Ultralytics does, not 100."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.zeros((8400, 4 + 80))
    rows[:, :4] = torch.rand((8400, 4), generator=generator) * 640
    hot = torch.randperm(8400, generator=generator)[:100]
    for anchor in hot:
        classes = torch.randperm(80, generator=generator)[:3]
        rows[anchor, 4 + classes] = torch.rand(3, generator=generator) * 0.9 + 0.01

    selected = dual_topk(rows.to(device), 80, 0, max_det=300, conf_thres=0.001).cpu()

    expected = _ultralytics_end2end(rows, 80, 0.001)
    assert len(selected) == 300
    torch.testing.assert_close(selected[:, 4], expected[:, 0])
    torch.testing.assert_close(selected[:, 5], expected[:, 1])
    torch.testing.assert_close(selected[:, :4], rows[expected[:, 2].long(), :4])


@pytest.mark.parametrize("device", DEVICES)
def test_dual_topk_orders_tied_scores_by_index(device: str) -> None:
    rows = torch.zeros((8400, 4 + 80))
    rows[:, 4:] = 0.5

    selected = dual_topk(rows.to(device), 80, 0, max_det=300, conf_thres=0.001).cpu()

    # Every (anchor, class) ties, so both rankings keep index order: anchor 0's
    # 80 classes first, then anchor 1's.
    expected = torch.arange(300) % 80
    torch.testing.assert_close(selected[:, 5], expected.float())


@pytest.mark.parametrize("device", DEVICES)
def test_nms_keeps_the_lowest_index_of_tied_overlapping_candidates(device: str) -> None:
    """Greedy NMS keeps whichever tied candidate it ranks first; Ultralytics ranks index 0."""
    post = build_postprocess(*_model_cfg("YOLOv8n"))
    rows = torch.zeros((8400, 84))
    shift = torch.arange(2000, dtype=torch.float32) * 1e-3
    rows[:2000, 0], rows[:2000, 2] = 100.0 + shift, 200.0 + shift
    rows[:2000, 1], rows[:2000, 3] = 100.0, 200.0
    rows[:2000, 4] = 0.5

    (detections,) = post.nms_multilabel([rows.to(device)])

    assert detections.shape[0] == 1
    assert detections[0, 0].item() == pytest.approx(100.0)

"""Focused WiderFace evaluator validation tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

eval_widerface_module = importlib.import_module(
    "mblt_vision.utils.evaluation.eval_widerface"
)


def test_widerface_evaluation_rejects_wrong_model_taxonomy() -> None:
    """Do not evaluate another detector taxonomy using WiderFace metadata."""

    with pytest.raises(ValueError, match="post_cfg.dataset to be 'widerface'"):
        eval_widerface_module.eval_widerface(
            SimpleNamespace(post_cfg={"task": "face_detection", "dataset": "coco"}),
            "/dataset",
            batch_size=1,
        )


def test_widerface_evaluation_rejects_truncated_postprocess_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one decoded face prediction list for every submitted image."""

    class _Dataset:
        samples = [("unused", "event", "first.jpg"), ("unused", "event", "second.jpg")]

        def __len__(self) -> int:
            return 2

    class _Progress:
        def __init__(self, iterable, **kwargs) -> None:
            del kwargs
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix_str(self, value: str) -> None:
            del value

        def close(self) -> None:
            return None

    class _Model:
        post_cfg = {"task": "face_detection", "dataset": "widerface"}
        postprocessor = SimpleNamespace(
            nmsout2eval=lambda *_args, **_kwargs: ([], [[]], [[]])
        )
        preprocess_with_metadata = object()

        def set_postprocess_thresholds(self, **kwargs) -> None:
            del kwargs

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def postprocess(self, outputs: torch.Tensor) -> SimpleNamespace:
            del outputs
            return SimpleNamespace(output=[])

    batch = (
        torch.zeros((2, 8, 8, 3)),
        np.array([[8, 8], [8, 8]]),
        [None, None],
        ("event", "event"),
        ("first.jpg", "second.jpg"),
    )
    monkeypatch.setattr(eval_widerface_module, "CustomWiderface", lambda _: _Dataset())
    monkeypatch.setattr(
        eval_widerface_module, "get_widerface_loader", lambda *_: [batch]
    )
    monkeypatch.setattr(eval_widerface_module, "tqdm", _Progress)

    with pytest.raises(ValueError, match="WiderFace evaluation batch length mismatch"):
        eval_widerface_module.eval_widerface(_Model(), "/dataset", batch_size=2)

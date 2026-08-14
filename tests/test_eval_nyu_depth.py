"""Regression tests for NYU depth-evaluation output validation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

eval_nyu_depth_module = importlib.import_module(
    "mblt_vision.utils.evaluation.eval_nyu_depth"
)


def test_nyu_depth_evaluation_rejects_surplus_output_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not silently discard depth maps from a malformed backend batch."""

    class _Model:
        post_cfg = {"dataset": "nyu-depth"}
        pre_cfg = {"LetterBox": {"img_size": [4, 4]}}

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def postprocess(self, output: torch.Tensor) -> SimpleNamespace:
            del output
            return SimpleNamespace(depth=torch.zeros((2, 4, 4)))

    batch = (
        torch.zeros((1, 4, 4, 3)),
        np.ones((1, 4, 4), dtype=np.float32),
        [(4, 4)],
        [None],
        ("sample",),
    )
    monkeypatch.setattr(eval_nyu_depth_module, "CustomNYUDepth", lambda _: object())
    monkeypatch.setattr(
        eval_nyu_depth_module, "get_nyu_depth_loader", lambda *_args, **_kwargs: [batch]
    )
    monkeypatch.setattr(eval_nyu_depth_module, "build_preprocess", lambda _: object())

    with pytest.raises(
        ValueError, match=r"output batch length mismatch: maps=2, targets=1"
    ):
        eval_nyu_depth_module.eval_nyu_depth(_Model(), "/dataset", batch_size=1)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nyu_depth_metrics_reject_nonfinite_targets(invalid_value: float) -> None:
    """Keep direct metric callers from silently excluding corrupt depth targets."""

    target = np.array([[1.0, invalid_value]], dtype=np.float32)
    with pytest.raises(ValueError, match="target contains non-finite"):
        eval_nyu_depth_module.calculate_nyu_depth_metrics(np.ones((1, 2)), target)


@pytest.mark.parametrize("dtype", [np.complex64, np.dtype("U4")])
def test_nyu_depth_metrics_reject_nonreal_or_nonnumeric_inputs(dtype: np.dtype) -> None:
    """Direct metric callers must not lose source-dtype corruption during casting."""

    values = np.ones((1, 2), dtype=dtype)

    with pytest.raises(ValueError, match="real numeric dtype"):
        eval_nyu_depth_module.calculate_nyu_depth_metrics(values, values)


def test_nyu_depth_evaluation_rejects_wrong_model_taxonomy() -> None:
    """Require the model's declared depth taxonomy before loading a dataset."""

    with pytest.raises(ValueError, match="post_cfg.dataset to be 'nyu-depth'"):
        eval_nyu_depth_module.eval_nyu_depth(
            SimpleNamespace(post_cfg={"dataset": "ade20k"}), "/dataset", batch_size=1
        )

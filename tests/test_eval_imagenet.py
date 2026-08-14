"""Focused ImageNet evaluator regression tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

eval_imagenet_module = importlib.import_module(
    "mblt_vision.utils.evaluation.eval_imagenet"
)


def test_imagenet_evaluation_rejects_truncated_classification_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not broadcast one classification row across a multi-image label batch."""

    class _FakeDataset:
        def __len__(self) -> int:
            return 2

    class _FakeProgress:
        def __init__(self, iterable, **kwargs) -> None:
            del kwargs
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, value: str) -> None:
            del value

        def close(self) -> None:
            return None

    class _FakeModel:
        def preprocess(self, value: object) -> object:
            return value

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def postprocess(self, outputs: torch.Tensor) -> SimpleNamespace:
            del outputs
            return SimpleNamespace(output=torch.zeros((1, 5), dtype=torch.float32))

    batch = (torch.zeros((2, 3, 8, 8)), torch.tensor([0, 1]))
    monkeypatch.setattr(
        eval_imagenet_module, "CustomImageFolder", lambda _: _FakeDataset()
    )
    monkeypatch.setattr(
        eval_imagenet_module, "get_imagenet_loader", lambda *args: [batch]
    )
    monkeypatch.setattr(eval_imagenet_module, "tqdm", _FakeProgress)

    with pytest.raises(
        ValueError,
        match="got 1 outputs for 2 labels",
    ):
        eval_imagenet_module.eval_imagenet_metrics(
            _FakeModel(), "/dataset", batch_size=2
        )

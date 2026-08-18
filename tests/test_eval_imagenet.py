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
        classes = ["n00000000", "n00000001"]
        class_to_idx = {"n00000000": 0, "n00000001": 1}

        def make_dataset(self) -> None:
            return None

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
        post_cfg = {"dataset": "imagenet"}

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
    monkeypatch.setattr(
        eval_imagenet_module, "IMAGENET_SYNSET_ORDER", ("n00000000", "n00000001")
    )
    monkeypatch.setattr(
        eval_imagenet_module, "IMAGENET_SYNSETS", frozenset({"n00000000", "n00000001"})
    )

    with pytest.raises(
        ValueError,
        match="got 1 outputs for 2 labels",
    ):
        eval_imagenet_module.eval_imagenet_metrics(
            _FakeModel(), "/dataset", batch_size=2
        )


def test_imagenet_evaluation_preserves_canonical_synset_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not renumber a partial ImageNet directory tree by local sort order."""

    class _Dataset(SimpleNamespace):
        def __len__(self) -> int:
            return 1

    dataset = _Dataset(
        classes=["n00000001"],
        class_to_idx={"n00000001": 0},
        make_dataset=lambda: None,
    )
    monkeypatch.setattr(eval_imagenet_module, "CustomImageFolder", lambda _: dataset)
    monkeypatch.setattr(
        eval_imagenet_module, "IMAGENET_SYNSET_ORDER", ("n00000000", "n00000001")
    )
    monkeypatch.setattr(
        eval_imagenet_module, "IMAGENET_SYNSETS", frozenset({"n00000000", "n00000001"})
    )

    class _StopAfterClassMapping(Exception):
        pass

    def _stop_after_mapping(*args: object) -> object:
        del args
        raise _StopAfterClassMapping

    monkeypatch.setattr(
        eval_imagenet_module, "get_imagenet_loader", _stop_after_mapping
    )

    with pytest.raises(_StopAfterClassMapping):
        eval_imagenet_module.eval_imagenet_metrics(
            SimpleNamespace(post_cfg={"dataset": "imagenet"}, preprocess=object()),
            "/dataset",
            batch_size=1,
        )

    assert dataset.class_to_idx == {"n00000001": 1}


def test_imagenet_evaluation_rejects_empty_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before loader construction when no supported samples are available."""

    class _EmptyDataset:
        classes = ["n00000000"]
        class_to_idx = {"n00000000": 0}

        def make_dataset(self) -> None:
            return None

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(
        eval_imagenet_module, "CustomImageFolder", lambda _: _EmptyDataset()
    )
    monkeypatch.setattr(eval_imagenet_module, "IMAGENET_SYNSET_ORDER", ("n00000000",))
    monkeypatch.setattr(
        eval_imagenet_module, "IMAGENET_SYNSETS", frozenset({"n00000000"})
    )
    monkeypatch.setattr(
        eval_imagenet_module,
        "get_imagenet_loader",
        lambda *_: pytest.fail("empty datasets must not create an ImageNet loader"),
    )

    with pytest.raises(ValueError, match="contains no supported images"):
        eval_imagenet_module.eval_imagenet_metrics(
            SimpleNamespace(post_cfg={"dataset": "imagenet"}, preprocess=object()),
            "/empty",
            batch_size=1,
        )


def test_imagenet_evaluation_rejects_wrong_model_taxonomy() -> None:
    """Do not score another classification taxonomy against ImageNet labels."""

    with pytest.raises(ValueError, match="post_cfg.dataset to be 'imagenet'"):
        eval_imagenet_module.eval_imagenet_metrics(
            SimpleNamespace(post_cfg={"dataset": "coco"}), "/dataset", batch_size=1
        )

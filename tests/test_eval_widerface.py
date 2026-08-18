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
        eval_widerface_module,
        "_load_widerface_image_names",
        lambda _: {"event": {"first.jpg", "second.jpg"}},
    )
    monkeypatch.setattr(
        eval_widerface_module,
        "_widerface_difficulty_metadata_ready",
        lambda *_, **__: True,
    )
    monkeypatch.setattr(
        eval_widerface_module,
        "_widerface_image_shapes",
        lambda *_: [[(8, 8), (8, 8)]],
    )
    monkeypatch.setattr(
        eval_widerface_module, "get_widerface_loader", lambda *_: [batch]
    )
    monkeypatch.setattr(eval_widerface_module, "tqdm", _Progress)

    with pytest.raises(ValueError, match="WiderFace evaluation batch length mismatch"):
        eval_widerface_module.eval_widerface(_Model(), "/dataset", batch_size=2)


def test_widerface_evaluation_rejects_malformed_difficulty_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate metadata before direct evaluation constructs a dataset or loader."""

    model = SimpleNamespace(post_cfg={"task": "face_detection", "dataset": "widerface"})
    monkeypatch.setattr(
        eval_widerface_module,
        "_load_widerface_image_names",
        lambda _: {"0--Parade": {"sample.jpg"}},
    )
    monkeypatch.setattr(
        eval_widerface_module,
        "_widerface_difficulty_metadata_ready",
        lambda *_, **__: False,
    )
    monkeypatch.setattr(
        eval_widerface_module,
        "CustomWiderface",
        lambda _: pytest.fail(
            "invalid metadata must not construct a WiderFace dataset"
        ),
    )

    with pytest.raises(ValueError, match="metadata is malformed"):
        eval_widerface_module.eval_widerface(model, "/dataset", batch_size=1)


def test_widerface_evaluation_rejects_image_tree_mismatching_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exact event and filename identities before direct inference."""

    class _Dataset:
        samples = [("unused", "0--Parade", "sample.png")]

    model = SimpleNamespace(post_cfg={"task": "face_detection", "dataset": "widerface"})
    monkeypatch.setattr(
        eval_widerface_module,
        "_load_widerface_image_names",
        lambda _: {"0--Parade": {"sample.jpg"}},
    )
    monkeypatch.setattr(
        eval_widerface_module,
        "_widerface_difficulty_metadata_ready",
        lambda *_, **__: True,
    )
    monkeypatch.setattr(
        eval_widerface_module, "_widerface_image_shapes", lambda *_: [[(8, 8)]]
    )
    monkeypatch.setattr(eval_widerface_module, "CustomWiderface", lambda _: _Dataset())
    monkeypatch.setattr(
        eval_widerface_module,
        "get_widerface_loader",
        lambda *_: pytest.fail("mismatched image trees must not create a loader"),
    )

    with pytest.raises(ValueError, match="does not match the validation metadata"):
        eval_widerface_module.eval_widerface(model, "/dataset", batch_size=1)


def test_widerface_evaluation_rejects_unequal_box_and_score_counts() -> None:
    """Do not fabricate or drop face detections during result conversion."""

    with pytest.raises(ValueError, match="unequal box and score counts"):
        eval_widerface_module._boxes_scores_to_prediction([[0, 0, 1, 1]], [])

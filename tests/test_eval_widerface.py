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


def test_widerface_result_uses_hard_ap_as_primary() -> None:
    """Rank by Hard AP and retain Medium then Easy as secondary metrics."""

    result = eval_widerface_module.WiderFaceResult(0.8, 0.7, 0.6)

    assert result.primary_score == 0.6
    assert result.secondary_score == 0.7
    assert result.secondary_scores == (0.7, 0.8)


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


def test_widerface_evaluation_normalizes_no_face_sentinel() -> None:
    """Do not score the official all-zero no-face marker as a ground-truth box."""

    boxes = eval_widerface_module._normalize_ground_truth_boxes(
        np.zeros((1, 4), dtype=np.float32)
    )

    assert boxes.shape == (0, 4)


def test_widerface_evaluation_counts_predictions_on_no_face_image() -> None:
    """Predictions on an official no-face image must contribute false positives."""

    contribution = eval_widerface_module._empty_ground_truth_prediction_contribution(
        2, np.array([[0, 0, 1, 1, 0.9]], dtype=np.float32)
    )

    np.testing.assert_array_equal(contribution[:, 0], np.ones(2))
    np.testing.assert_array_equal(contribution[:, 1], np.zeros(2))


def test_widerface_image_eval_handles_empty_prediction_and_ground_truth() -> None:
    """The vectorised match must keep the degenerate cases the loop allowed.

    `argmax` raises on an empty axis where the per-prediction loop it replaces
    simply never ran, so an image with no predictions, or with no annotated
    faces, has to short-circuit rather than reduce an empty overlap matrix.
    """

    empty_predictions = np.zeros((0, 5), dtype=np.float32)
    empty_faces = np.zeros((0, 4), dtype=np.float32)
    predictions = np.array(
        [[10.0, 10.0, 20.0, 20.0, 0.9], [50.0, 50.0, 10.0, 10.0, 0.4]],
        dtype=np.float32,
    )
    faces = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)

    for pred, gt in ((empty_predictions, empty_faces), (empty_predictions, faces)):
        recall, proposals = eval_widerface_module.image_eval(
            pred, gt, np.ones(len(gt)), 0.5
        )
        assert recall.shape == (0,) and proposals.shape == (0,)
        assert eval_widerface_module.img_pr_info(4, pred, proposals, recall).sum() == 0

    # No annotated faces: every prediction stays a proposal, nothing is recalled.
    recall, proposals = eval_widerface_module.image_eval(
        predictions, empty_faces, np.ones(0), 0.5
    )
    np.testing.assert_array_equal(proposals, np.ones(2))
    np.testing.assert_array_equal(recall, np.zeros(2))


def test_widerface_cutoffs_are_compared_in_the_score_dtype() -> None:
    """A score sitting exactly on a cut-off must reach it, as float32.

    The official loop compares a float32 score array against a Python float,
    which numpy resolves in float32 by value-based casting, so `float32(0.9)`
    reaches the 0.9 cut-off. Widening the scores to float64 first makes the
    same score 0.899999976... and drops it, which would shift the proposal and
    recall counts at every cut-off a score lands on exactly — and `norm_score`
    maps the highest score to exactly 1.0, so that is the common case.
    """

    for thresh_num in (10, 200, 1000):
        cutoffs = (1 - np.arange(1, thresh_num + 1) / thresh_num).astype(np.float32)
        for scores in (
            cutoffs,
            cutoffs[::-1].copy(),
            np.concatenate([cutoffs, cutoffs]),
        ):
            predictions = np.zeros((len(scores), 5), dtype=np.float32)
            predictions[:, 4] = scores
            last = eval_widerface_module.score_threshold_index(predictions, thresh_num)
            expected = []
            for index in range(thresh_num):
                cutoff = 1 - (index + 1) / thresh_num
                qualifying = np.flatnonzero(scores >= cutoff)
                expected.append(int(qualifying[-1]) if len(qualifying) else -1)
            np.testing.assert_array_equal(last, expected)
        # Every cut-off is reached by the score built from it, which is the
        # property a widened comparison breaks.
        predictions = np.zeros((len(cutoffs), 5), dtype=np.float32)
        predictions[:, 4] = cutoffs
        assert (
            eval_widerface_module.score_threshold_index(predictions, thresh_num) >= 0
        ).all()

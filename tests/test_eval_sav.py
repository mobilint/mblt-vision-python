"""Tests for SA-V point-prompted mask generation evaluation."""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mblt_vision.utils.datasets import CustomSAV
from mblt_vision.utils.evaluation.eval_sav import (
    SAVMetricAccumulator,
    build_prompt,
    calculate_sav_sample_ious,
    eval_sav,
    iter_selected_samples,
    mask_iou,
)

eval_sav_module = importlib.import_module("mblt_vision.utils.evaluation.eval_sav")


def _disk_mask(height: int = 96, width: int = 128, radius: int = 20) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[:height, :width]
    mask[(yy - height // 2) ** 2 + (xx - width // 2) ** 2 <= radius**2] = True
    return mask


def test_mask_iou_matches_hand_computed_overlap() -> None:
    """Score partial overlap exactly and treat two empty masks as identical."""

    left = np.zeros((4, 4), dtype=bool)
    right = np.zeros((4, 4), dtype=bool)
    left[:2, :2] = True
    right[:2, :] = True
    assert mask_iou(left, right) == pytest.approx(4 / 8)
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0


def test_calculate_sav_sample_ious_rejects_shape_mismatch() -> None:
    """Never score candidates against a ground truth of a different geometry."""

    with pytest.raises(ValueError, match="shapes must match"):
        calculate_sav_sample_ious(np.zeros((3, 4, 4), bool), np.zeros((5, 5), bool))
    with pytest.raises(ValueError, match=r"shaped \(N, H, W\)"):
        calculate_sav_sample_ious(np.zeros((4, 4), bool), np.zeros((4, 4), bool))


def test_calculate_sav_sample_ious_rejects_non_binary_candidates() -> None:
    """Logits, probabilities, and NaN must be reported, not silently binarized.

    `astype(bool)` treats every nonzero value as foreground, including negative
    logits, which would yield a plausible but meaningless IoU.
    """

    gt = np.zeros((4, 4), dtype=bool)
    gt[:2, :2] = True

    logits = np.full((1, 4, 4), -3.5, dtype=np.float32)
    logits[0, :2, :2] = 4.0
    with pytest.raises(ValueError, match="must be a binary map"):
        calculate_sav_sample_ious(logits, gt)

    probabilities = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 4, 4)
    with pytest.raises(ValueError, match="must be a binary map"):
        calculate_sav_sample_ious(probabilities, gt)

    non_finite = np.zeros((1, 4, 4), dtype=np.float32)
    non_finite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        calculate_sav_sample_ious(non_finite, gt)

    # A single positive value is not sufficient: a thresholded probability map
    # and a degenerate uniform one both have exactly one, and the latter would
    # silently become an all-foreground mask.
    thresholded = np.where(gt, 0.5, 0.0).astype(np.float32)[None]
    with pytest.raises(ValueError, match="must be a binary map"):
        calculate_sav_sample_ious(thresholded, gt)

    uniform = np.full((1, 4, 4), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="must be a binary map"):
        calculate_sav_sample_ious(uniform, gt)

    with pytest.raises(ValueError, match="unsupported dtype"):
        calculate_sav_sample_ious(np.array([[["a", "b"], ["c", "d"]]]), gt[:2, :2])


@pytest.mark.parametrize(
    "encoding",
    [
        np.array([[[True, False], [False, False]]]),
        np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
        np.array([[[255, 0], [0, 0]]], dtype=np.uint8),
        np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    ],
)
def test_calculate_sav_sample_ious_accepts_conventional_binary_encodings(
    encoding: np.ndarray,
) -> None:
    """bool, {0, 1}, {0, 255} and float 0/1 all denote the same binary mask."""

    gt = np.array([[True, False], [False, False]])
    assert calculate_sav_sample_ious(encoding, gt) == [pytest.approx(1.0)]


@pytest.mark.parametrize(
    ("degenerate", "expected_iou"),
    [
        (np.zeros((1, 2, 2), dtype=np.uint8), 0.0),
        (np.ones((1, 2, 2), dtype=np.uint8), 0.25),
    ],
)
def test_calculate_sav_sample_ious_accepts_empty_and_full_masks(
    degenerate: np.ndarray, expected_iou: float
) -> None:
    """An all-background or all-foreground prediction is valid, just wrong."""

    gt = np.array([[True, False], [False, False]])
    assert calculate_sav_sample_ious(degenerate, gt) == [pytest.approx(expected_iou)]


@pytest.mark.parametrize(
    ("num_points", "expected_labels"),
    [(1, [1]), (2, [1, 0]), (3, [1, 1, 0])],
)
def test_build_prompt_points_lie_on_the_correct_side(
    num_points: int, expected_labels: list[int]
) -> None:
    """Place positive points inside the mask and negative points outside it."""

    mask = _disk_mask()
    prompt = build_prompt(mask, random.Random(0), num_points)
    assert prompt is not None
    points, labels = prompt
    assert labels.tolist() == expected_labels
    for (x, y), label in zip(points, labels):
        assert bool(mask[int(y), int(x)]) == bool(label)


def test_build_prompt_returns_none_for_empty_mask() -> None:
    """Skip masks that cannot anchor a positive point."""

    assert build_prompt(np.zeros((32, 32), bool), random.Random(0), 1) is None


def test_negative_point_avoids_the_dilated_mask() -> None:
    """Sample negatives outside the safety-dilated foreground, not merely outside it."""

    mask = _disk_mask()
    dilated = cv2.dilate(
        mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(bool)
    for seed in range(5):
        point = eval_sav_module._negative_point(
            mask, mask.shape[0], mask.shape[1], random.Random(seed)
        )
        assert point is not None
        assert not dilated[int(point[1]), int(point[0])]


def _write_sav_root(root: Path, videos: int = 3, frames_per_video: int = 4) -> None:
    """Write a tiny organized SA-V root with one masklet per video."""

    for video_index in range(videos):
        video_id = f"sav_{video_index:06d}"
        image_dir = root / "images" / video_id
        object_dir = root / "annotations" / video_id / "000"
        image_dir.mkdir(parents=True)
        object_dir.mkdir(parents=True)
        for frame_index in range(frames_per_video):
            stem = f"{frame_index * 4:05d}"
            Image.new("RGB", (64, 48), color=(video_index, 0, 0)).save(
                image_dir / f"{stem}.jpg"
            )
            mask = Image.new("L", (64, 48), color=0)
            mask.paste(255, (8, 8, 40, 40))
            mask.save(object_dir / f"{stem}.png")
    (root / "video_ids.txt").write_text(
        "\n".join(f"sav_{index:06d}" for index in range(videos)) + "\n",
        encoding="utf-8",
    )


def test_iter_selected_samples_is_deterministic_per_seed(tmp_path: Path) -> None:
    """Reproduce the exact sample selection for a fixed seed."""

    _write_sav_root(tmp_path)
    dataset = CustomSAV(str(tmp_path))
    first = list(iter_selected_samples(dataset, seed=3, per_video=2, min_mask_area=10))
    second = list(iter_selected_samples(dataset, seed=3, per_video=2, min_mask_area=10))
    other_seed = list(
        iter_selected_samples(dataset, seed=4, per_video=2, min_mask_area=10)
    )
    assert first and first == second
    assert first != other_seed


def test_iter_selected_samples_caps_per_video(tmp_path: Path) -> None:
    """Never draw more than per_video samples from one video."""

    _write_sav_root(tmp_path, videos=2, frames_per_video=6)
    dataset = CustomSAV(str(tmp_path))
    selected = list(
        iter_selected_samples(dataset, seed=0, per_video=2, min_mask_area=10)
    )
    videos = [dataset.samples[index][2] for index in selected]
    for video_id in set(videos):
        assert videos.count(video_id) <= 2


def test_iter_selected_samples_skips_small_masks(tmp_path: Path) -> None:
    """Exclude ground-truth masks below the minimum area."""

    _write_sav_root(tmp_path, videos=1)
    dataset = CustomSAV(str(tmp_path))
    assert not list(
        iter_selected_samples(dataset, seed=0, per_video=4, min_mask_area=10_000)
    )


def test_accumulator_computes_selection_and_oracle_means() -> None:
    """Aggregate own-selection and best-of-3 IoUs independently."""

    accumulator = SAVMetricAccumulator()
    accumulator.update([0.2, 0.9, 0.5], selected=1, video_id="a")
    accumulator.update([0.8, 0.1, 0.4], selected=2, video_id="b")
    result = accumulator.result()
    assert result.miou == pytest.approx((0.9 + 0.4) / 2)
    assert result.miou_best_of_3 == pytest.approx((0.9 + 0.8) / 2)
    assert result.num_samples == 2
    assert result.distinct_videos == 2
    assert result.primary_score == result.miou
    assert result.secondary_score == result.miou_best_of_3


def test_accumulator_rejects_wrong_candidate_count() -> None:
    """Enforce the three-candidate SAM2 output contract."""

    with pytest.raises(ValueError, match="Expected 3 candidate IoUs"):
        SAVMetricAccumulator().update([0.5, 0.5], selected=0, video_id="a")


def test_eval_sav_rejects_wrong_taxonomy() -> None:
    """Refuse to score a model configured for a different dataset."""

    # _PerfectModel (defined below) gives a real predict() the taxonomy check
    # never reaches, unlike SimpleNamespace's synthesized-but-statically-invisible
    # attributes.
    model = _PerfectModel()
    model.post_cfg = {"task": "mask_generation", "dataset": "coco"}
    with pytest.raises(ValueError, match="post_cfg.dataset to be 'sa-v'"):
        eval_sav(model, "/dataset")


@dataclass
class _MockPrediction:
    """Structurally matches ``eval_sav``'s ``PromptedPrediction`` -- unlike
    ``SimpleNamespace``, a dataclass's fields are visible to static typing."""

    masks: np.ndarray
    selected: int | None


class _PerfectModel:
    """Mock engine returning the ground truth as its best-selected candidate."""

    post_cfg = {"task": "mask_generation", "dataset": "sa-v"}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame, points, labels):
        del points, labels
        self.calls += 1
        height, width = frame.shape[:2]
        gt = np.zeros((height, width), dtype=bool)
        gt[8:40, 8:40] = True
        masks = np.stack([np.zeros_like(gt), gt, np.ones_like(gt)])
        return _MockPrediction(masks=masks, selected=1)


def test_eval_sav_scores_a_mocked_engine_exactly(tmp_path: Path) -> None:
    """Produce exact metrics for crafted candidates against the fixture masks."""

    _write_sav_root(tmp_path, videos=3, frames_per_video=4)
    model = _PerfectModel()
    result = eval_sav(
        model, str(tmp_path), num_samples=4, num_points=1, seed=0, per_video=2
    )
    assert result.miou == pytest.approx(1.0)
    assert result.miou_best_of_3 == pytest.approx(1.0)
    assert result.num_samples == 4
    assert model.calls == 4


def test_eval_sav_rejects_prediction_without_selected_index(tmp_path: Path) -> None:
    """Report a missing selection as ValueError, not a bare assert.

    The PromptedPrediction protocol permits `selected=None`, and an assert
    would vanish under `python -O` and resurface as an opaque comparison
    TypeError inside the accumulator.
    """

    _write_sav_root(tmp_path, videos=3, frames_per_video=4)

    class _NoSelectionModel(_PerfectModel):
        def predict(self, frame, points, labels):
            prediction = super().predict(frame, points, labels)
            return _MockPrediction(masks=prediction.masks, selected=None)

    with pytest.raises(ValueError, match="missing a selected mask index"):
        eval_sav(_NoSelectionModel(), str(tmp_path), num_samples=1, per_video=2)


def test_eval_sav_raises_when_samples_run_out(tmp_path: Path) -> None:
    """Fail loudly instead of silently reporting fewer samples than requested."""

    _write_sav_root(tmp_path, videos=1, frames_per_video=2)
    with pytest.raises(ValueError, match="only .* valid samples were available"):
        eval_sav(_PerfectModel(), str(tmp_path), num_samples=50, per_video=2)

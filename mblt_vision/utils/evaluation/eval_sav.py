"""SA-V point-prompted mask generation evaluation.

Protocol ported from the validated ``sam2-mxq-pipeline`` reference (its
Aries2 run measured MXQ mIoU 0.7757 on 200 sav_train samples): deterministic
area-balanced sampling of (video, object, frame) masklets, synthetic point
prompts derived from the ground-truth mask (distance-transform peak plus
optional negative/second-positive points), and per-candidate mask IoU against
the ground truth. The primary metric is the mean IoU of each sample's
own-selected candidate (``argmax`` of the model's predicted IoUs).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from tqdm import tqdm

from ..datasets import CustomSAV


class PromptedPrediction(Protocol):
    """The two fields ``eval_sav`` actually reads off a ``predict()`` result.

    A real ``SAM2HieraLarge.predict()`` call returns a full ``Results``, but
    only ``masks`` and ``selected`` are used here -- narrower than ``Results``
    so the lightweight test doubles in ``tests/test_eval_sav.py`` (structural
    stand-ins, not real ``Results`` instances) satisfy it too.
    """

    # Read-only (never written), so declared as properties rather than plain
    # attributes: a Protocol's plain attributes are matched invariantly, which
    # would reject a real implementer's narrower concrete type (e.g. a mock's
    # `selected: int` against `int | None`) even though it satisfies every
    # actual read here.
    @property
    def masks(self) -> Any: ...
    @property
    def selected(self) -> int | None: ...


class PointPromptedEngine(Protocol):
    """Structural contract ``eval_sav`` needs from a mask generation engine.

    ``SAM2HieraLarge`` satisfies this, but so do the lightweight test doubles
    in ``tests/test_eval_sav.py`` (no real backend or download); the concrete
    class is deliberately not required here.
    """

    post_cfg: dict[str, Any]

    def predict(self, image: Any, points: Any, labels: Any, /) -> PromptedPrediction:
        """Positional-only: every real caller here invokes this positionally,
        and implementations use varying parameter names (``image`` vs.
        ``frame``)."""
        ...


# Relative-mask-area bins used to balance sampling (reference dataset.py).
AREA_BINS = ((0.0, 0.005), (0.005, 0.02), (0.02, 0.08), (0.08, 1.01))
CANDIDATES_PER_PROMPT = 3
FRAMES_CONSIDERED_PER_MASKLET = 6


@dataclass(frozen=True)
class SAVResult:
    """Point-prompted SA-V mask generation metrics."""

    miou: float
    miou_ci95: float
    miou_best_of_3: float
    num_samples: int
    distinct_videos: int

    @property
    def primary_score(self) -> float:
        """Return the own-selection mean IoU."""

        return self.miou

    @property
    def secondary_score(self) -> float:
        """Return the best-of-3 (oracle) mean IoU."""

        return self.miou_best_of_3


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return the IoU of two boolean masks; both-empty counts as 1.0."""

    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)
    return float(intersection) / float(union) if union else 1.0


def calculate_sav_sample_ious(
    candidate_masks: np.ndarray, gt_mask: np.ndarray
) -> list[float]:
    """Return per-candidate IoUs of predicted binary masks against ground truth."""

    candidates = np.asarray(candidate_masks)
    gt = np.asarray(gt_mask).astype(bool)
    if candidates.ndim != 3:
        raise ValueError(
            f"Expected candidate masks shaped (N, H, W), got {candidates.shape}."
        )
    if candidates.shape[1:] != gt.shape:
        raise ValueError(
            "Candidate masks and ground truth shapes must match: "
            f"candidates {candidates.shape[1:]}, ground truth {gt.shape}."
        )
    return [mask_iou(candidate.astype(bool), gt) for candidate in candidates]


def _ci95(values: Sequence[float]) -> float:
    """Return the 95% confidence half-width of the mean."""

    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def _distance_peak(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the interior point of the mask farthest from its boundary."""

    distance = cv2.distanceTransform(mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    if float(distance.max()) < 1.0:
        return None
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return float(x), float(y)


def _second_positive(
    mask: np.ndarray, first: tuple[float, float], rng: random.Random
) -> tuple[float, float] | None:
    """Sample a second positive point far from the first."""

    ys, xs = np.where(mask)
    if not len(xs):
        return None
    best: tuple[float, float] | None = None
    best_distance = -1.0
    for _ in range(64):
        index = rng.randint(0, len(xs) - 1)
        point = float(xs[index]), float(ys[index])
        distance = (point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2
        if distance > best_distance:
            best, best_distance = point, distance
    return best if best_distance > 9 else None


def _negative_point(
    mask: np.ndarray, height: int, width: int, rng: random.Random
) -> tuple[float, float] | None:
    """Sample a background point outside the dilated mask, near it when possible."""

    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dilated = cv2.dilate(
        mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(bool)
    pad = max(8, int(0.15 * max(y1 - y0, x1 - x0)))
    bounds = (
        max(0, y0 - pad),
        min(height, y1 + pad),
        max(0, x0 - pad),
        min(width, x1 + pad),
    )
    for global_search in (False, True):
        for _ in range(200):
            if global_search:
                x, y = rng.uniform(0, width - 1), rng.uniform(0, height - 1)
            else:
                y_min, y_max, x_min, x_max = bounds
                x, y = rng.uniform(x_min, x_max - 1), rng.uniform(y_min, y_max - 1)
            if not dilated[int(y), int(x)]:
                return x, y
    return None


def build_prompt(
    mask: np.ndarray, rng: random.Random, num_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a deterministic point prompt from a ground-truth mask.

    1 point: positive distance-transform peak. 2 points: peak + negative.
    3 points: peak + far second positive + negative. Returns ``None`` when a
    required point cannot be constructed for this mask.
    """

    height, width = mask.shape
    first = _distance_peak(mask)
    if first is None:
        return None
    if num_points == 1:
        return np.asarray([first], np.float32), np.asarray([1], np.int64)
    negative = _negative_point(mask, height, width, rng)
    if negative is None:
        return None
    if num_points == 2:
        return np.asarray([first, negative], np.float32), np.asarray([1, 0], np.int64)
    if num_points != 3:
        raise ValueError(f"num_points must be 1, 2, or 3; got {num_points}")
    second = _second_positive(mask, first, rng)
    if second is None:
        return None
    return (
        np.asarray([first, second, negative], np.float32),
        np.asarray([1, 1, 0], np.int64),
    )


def _area_bin(mask: np.ndarray) -> int:
    """Return the relative-area bin index of a boolean mask."""

    fraction = int(mask.sum()) / float(mask.size)
    return next(
        (
            index
            for index, (low, high) in enumerate(AREA_BINS)
            if low <= fraction < high
        ),
        len(AREA_BINS) - 1,
    )


def _load_gt_mask(dataset: CustomSAV, index: int) -> np.ndarray | None:
    """Load only the boolean ground-truth mask for one dataset sample."""

    mask_path = dataset.samples[index][1]
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        return None
    return mask_image > 0


def iter_selected_samples(
    dataset: CustomSAV,
    *,
    seed: int,
    per_video: int,
    min_mask_area: int,
) -> Iterator[int]:
    """Yield dataset indices with deterministic, globally area-balanced sampling.

    Videos are visited in seed-shuffled order; per (video, object) masklet up
    to six evenly spaced annotated frames are considered; per video up to
    ``per_video`` samples are greedily chosen preferring the globally
    least-represented relative-area bin (reference ``_select_masks``).
    """

    by_video: dict[str, dict[str, list[int]]] = {}
    for index, (_, _, video_id, object_id, _) in enumerate(dataset.samples):
        by_video.setdefault(video_id, {}).setdefault(object_id, []).append(index)

    video_ids = sorted(by_video)
    random.Random(seed).shuffle(video_ids)
    rng = random.Random(seed + 7)
    bin_counts = [0] * len(AREA_BINS)

    for video_id in video_ids:
        candidates: list[tuple[int, int]] = []
        for object_id in sorted(by_video[video_id]):
            frame_indices = by_video[video_id][object_id]
            count = len(frame_indices)
            positions = sorted(
                {
                    int(
                        round(
                            position * (count - 1) / (FRAMES_CONSIDERED_PER_MASKLET - 1)
                        )
                    )
                    for position in range(FRAMES_CONSIDERED_PER_MASKLET)
                }
            )
            for position in positions:
                index = frame_indices[position]
                mask = _load_gt_mask(dataset, index)
                if mask is None or int(mask.sum()) < min_mask_area:
                    continue
                candidates.append((index, _area_bin(mask)))
        rng.shuffle(candidates)
        selected: list[int] = []
        used: set[int] = set()
        while len(selected) < per_video:
            remaining = [item for item in candidates if item[0] not in used]
            if not remaining:
                break
            target_bin = min(
                {item[1] for item in remaining}, key=lambda value: bin_counts[value]
            )
            index, chosen_bin = next(
                item for item in remaining if item[1] == target_bin
            )
            used.add(index)
            bin_counts[chosen_bin] += 1
            selected.append(index)
        yield from selected


class SAVMetricAccumulator:
    """Accumulates per-sample candidate IoUs into SA-V summary metrics."""

    def __init__(self) -> None:
        self._own_selection_ious: list[float] = []
        self._best_ious: list[float] = []
        self._videos: set[str] = set()

    @property
    def count(self) -> int:
        """Return the number of accumulated samples."""

        return len(self._own_selection_ious)

    def update(
        self, candidate_ious: Sequence[float], selected: int, video_id: str
    ) -> None:
        """Record one sample's per-candidate IoUs and the model's selection."""

        if len(candidate_ious) != CANDIDATES_PER_PROMPT:
            raise ValueError(
                f"Expected {CANDIDATES_PER_PROMPT} candidate IoUs, "
                f"got {len(candidate_ious)}."
            )
        if not 0 <= selected < len(candidate_ious):
            raise ValueError(
                f"Selected candidate index {selected} is out of range for "
                f"{len(candidate_ious)} candidates."
            )
        self._own_selection_ious.append(float(candidate_ious[selected]))
        self._best_ious.append(float(max(candidate_ious)))
        self._videos.add(video_id)

    def result(self) -> SAVResult:
        """Return the accumulated SA-V metrics."""

        if not self._own_selection_ious:
            raise ValueError("SA-V evaluation received no valid samples.")
        return SAVResult(
            miou=float(np.mean(self._own_selection_ious)),
            miou_ci95=_ci95(self._own_selection_ious),
            miou_best_of_3=float(np.mean(self._best_ious)),
            num_samples=len(self._own_selection_ious),
            distinct_videos=len(self._videos),
        )


def eval_sav(
    model: PointPromptedEngine,
    data_path: str,
    num_samples: int = 200,
    num_points: int = 1,
    seed: int = 0,
    per_video: int = 4,
    min_mask_area: int = 400,
) -> SAVResult:
    """Evaluate a point-prompted mask generation model on organized SA-V val.

    Args:
        model: Loaded mask generation engine exposing ``predict``.
        data_path: Organized SA-V dataset root.
        num_samples: Number of prompted samples to evaluate.
        num_points: Points per prompt (1, 2, or 3).
        seed: Sampling and prompt-synthesis seed.
        per_video: Maximum samples drawn from one video.
        min_mask_area: Minimum ground-truth mask area in pixels.

    Returns:
        Accumulated :class:`SAVResult` metrics.

    Raises:
        ValueError: If the model taxonomy mismatches, a prediction violates
            the three-candidate contract, or too few valid samples exist.
    """

    dataset_name = model.post_cfg.get("dataset")
    if not isinstance(dataset_name, str) or dataset_name.lower() != "sa-v":
        raise ValueError(
            "SA-V evaluation requires model post_cfg.dataset to be 'sa-v', "
            f"got {dataset_name!r}."
        )
    if num_points not in (1, 2, 3):
        raise ValueError(f"num_points must be 1, 2, or 3; got {num_points}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    dataset = CustomSAV(data_path)
    prompt_rng = random.Random(seed + 29)
    accumulator = SAVMetricAccumulator()
    progress = tqdm(total=num_samples, desc="Evaluating SA-V")
    try:
        for index in iter_selected_samples(
            dataset, seed=seed, per_video=per_video, min_mask_area=min_mask_area
        ):
            if accumulator.count >= num_samples:
                break
            frame, gt_mask, video_id, _, _ = dataset[index]
            prompt = build_prompt(gt_mask, prompt_rng, num_points)
            if prompt is None:
                continue
            points, labels = prompt
            result = model.predict(frame, points, labels)
            candidate_ious = calculate_sav_sample_ious(
                np.asarray(result.masks), gt_mask
            )
            # Results.selected is Optional at the shared-class level (most tasks
            # never set it), but a mask generation engine must populate it. Raise
            # rather than assert: this is a public evaluator documented to report
            # prediction-contract violations as ValueError, and an assert would
            # vanish under `python -O` and resurface as an opaque comparison
            # TypeError inside accumulator.update().
            if result.selected is None:
                raise ValueError(
                    "Mask generation prediction is missing a selected mask index; "
                    "predict() must report which candidate it chose."
                )
            accumulator.update(candidate_ious, result.selected, video_id)
            progress.update(1)
    finally:
        progress.close()

    outcome = accumulator.result()
    if outcome.num_samples < num_samples:
        raise ValueError(
            f"Requested {num_samples} SA-V evaluation samples but only "
            f"{outcome.num_samples} valid samples were available; lower "
            "--num-samples or relax the sampling constraints."
        )
    return outcome

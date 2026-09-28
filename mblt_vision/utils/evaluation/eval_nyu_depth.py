"""NYU Depth V2 evaluation for monocular depth-estimation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from mblt_vision.utils.preprocess import build_preprocess
from tqdm import tqdm

from ..datasets import CustomNYUDepth, get_nyu_depth_loader
from ._pipeline import map_batched_inference

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


@dataclass(frozen=True)
class NYUDepthResult:
    """Median-aligned NYU Depth V2 metrics."""

    delta1: float
    abs_rel: float
    rmse: float

    @property
    def primary_score(self) -> float:
        """Return the primary NYU Depth validation metric."""

        return self.delta1

    @property
    def secondary_score(self) -> float:
        """Return abs_rel for singular-score compatibility."""

        return self.abs_rel

    @property
    def secondary_scores(self) -> tuple[float, float]:
        """Return abs_rel and RMSE in secondary-metric order."""

        return self.abs_rel, self.rmse


class NYUDepthMetricAccumulator:
    """Accumulate median-aligned metrics with equal weight for each image."""

    MIN_DEPTH = 0.001
    MAX_DEPTH = 100.0

    def __init__(self) -> None:
        """Initialize zero-valued per-image metric sums."""

        self.delta1_sum = 0.0
        self.abs_rel_sum = 0.0
        self.rmse_sum = 0.0
        self.valid_sample_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Median-align one prediction and add its per-image metric values."""

        self.add(self.image_metrics(prediction, target))

    def add(self, metrics: tuple[float, float, float]) -> None:
        """Add one image's ``(delta1, abs_rel, rmse)`` from ``image_metrics``."""

        delta1, abs_rel, rmse = metrics
        self.delta1_sum += delta1
        self.abs_rel_sum += abs_rel
        self.rmse_sum += rmse
        self.valid_sample_count += 1

    @classmethod
    def image_metrics(
        cls, prediction: np.ndarray, target: np.ndarray
    ) -> tuple[float, float, float]:
        """Median-align one prediction and return its ``(delta1, abs_rel, rmse)``.

        This reads no accumulator state, so it may run on any thread; ``add``
        the results in sample order to reproduce ``update`` exactly.
        """

        prediction = _as_real_float32(prediction, "prediction")
        target = _as_real_float32(target, "target")
        if prediction.shape != target.shape:
            raise ValueError(
                f"NYU Depth prediction and target shapes must match, got {prediction.shape} and {target.shape}."
            )
        if not np.isfinite(target).all():
            raise ValueError("NYU Depth target contains non-finite values.")
        if (target < 0).any():
            raise ValueError("NYU Depth target contains negative values.")
        valid = (
            np.isfinite(target) & (target > cls.MIN_DEPTH) & (target < cls.MAX_DEPTH)
        )
        if not valid.any():
            raise ValueError(
                "NYU Depth sample has no valid pixels in the (0.001, 100.0) range."
            )

        predicted, actual = prediction[valid], target[valid]
        invalid_prediction_count = int((~np.isfinite(predicted)).sum())
        if invalid_prediction_count:
            raise ValueError(
                f"NYU Depth prediction contains {invalid_prediction_count} non-finite value(s) at valid target pixels."
            )
        median_prediction = np.median(np.maximum(predicted, cls.MIN_DEPTH))
        median_target = np.median(actual)
        aligned = predicted * (median_target / median_prediction)
        aligned = np.clip(aligned, cls.MIN_DEPTH, cls.MAX_DEPTH)
        ratio = np.maximum(actual / aligned, aligned / actual)
        # Match Ultralytics' depth validator: median-align and calculate all
        # metrics per image, then average validation images equally.
        return (
            float(np.mean(ratio < 1.25)),
            float(np.mean(np.abs(actual - aligned) / actual)),
            float(np.sqrt(np.mean((actual - aligned) ** 2))),
        )

    def result(self) -> NYUDepthResult:
        """Return mean per-image metrics."""

        if self.valid_sample_count == 0:
            raise ValueError("NYU Depth evaluation received no valid pixels.")
        return NYUDepthResult(
            delta1=self.delta1_sum / self.valid_sample_count,
            abs_rel=self.abs_rel_sum / self.valid_sample_count,
            rmse=self.rmse_sum / self.valid_sample_count,
        )


def _as_real_float32(values: np.ndarray, name: str) -> np.ndarray:
    """Validate metric input dtype before converting it to float32."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            f"NYU Depth {name} must use a real numeric dtype, got {array.dtype}."
        )
    return np.asarray(array, dtype=np.float32)


def calculate_nyu_depth_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> NYUDepthResult:
    """Calculate median-aligned NYU metrics for one sample."""

    accumulator = NYUDepthMetricAccumulator()
    accumulator.update(prediction, target)
    return accumulator.result()


def eval_nyu_depth(
    model: MBLT_Engine, data_path: str, batch_size: int
) -> NYUDepthResult:
    """Evaluate a depth model on paired NYU validation images and depth maps."""

    dataset_name = model.post_cfg.get("dataset")
    if not isinstance(dataset_name, str) or dataset_name.lower() != "nyu-depth":
        raise ValueError(
            "NYU Depth evaluation requires model post_cfg.dataset to be 'nyu-depth', "
            f"got {dataset_name!r}."
        )
    dataset = CustomNYUDepth(data_path)
    letterbox_cfg = model.pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, dict) or "img_size" not in letterbox_cfg:
        raise ValueError(
            "NYU Depth validation requires a LetterBox img_size in the model preprocessing config."
        )
    image_size = letterbox_cfg["img_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(
            "NYU Depth validation img_size must be a two-item [height, width] list."
        )

    validation_pre_cfg = {
        name: config for name, config in model.pre_cfg.items() if name != "LetterBox"
    }
    validation_preprocessor = build_preprocess(validation_pre_cfg)
    loader = get_nyu_depth_loader(
        dataset,
        batch_size,
        validation_preprocessor,
        image_size=(int(image_size[0]), int(image_size[1])),
    )

    def decode(batch: Any, output: Any) -> list[tuple[float, float, float]]:
        targets = batch[1]
        result = model.postprocess(output)
        depth = result.depth
        if depth is None:
            raise ValueError("Depth postprocessor returned no depth maps.")
        if isinstance(depth, list):
            maps = depth
        elif len(targets) == 1 and depth.ndim == 2:
            maps = [depth]
        else:
            if depth.ndim < 3 or depth.shape[0] != len(targets):
                raise ValueError(
                    "Depth postprocessor output batch length mismatch: "
                    f"maps={depth.shape[0] if depth.ndim else 0}, "
                    f"targets={len(targets)}."
                )
            maps = [depth[index] for index in range(len(targets))]
        if len(maps) != len(targets):
            raise ValueError(
                f"Depth postprocessor returned {len(maps)} maps for {len(targets)} targets."
            )
        metrics = []
        for prediction, target in zip(maps, targets):
            array = (
                prediction.detach().cpu().numpy()
                if hasattr(prediction, "detach")
                else np.asarray(prediction)
            )
            metrics.append(NYUDepthMetricAccumulator.image_metrics(array, target))
        return metrics

    accumulator = NYUDepthMetricAccumulator()
    batches = map_batched_inference(loader, lambda batch: model(batch[0]), decode)
    for image_metrics in tqdm(batches, total=len(loader), desc="Evaluating NYU Depth"):
        for metrics in image_metrics:
            accumulator.add(metrics)
    return accumulator.result()

"""Fold-based face verification, pinned to ops ec537764 protocol arithmetic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..datasets.face_pairs import FacePair, PairImage
from .retrieval import extract_embeddings

# Source grid includes both endpoints. Predictions use >= and ties choose highest t.
THRESHOLDS = np.linspace(-1.0, 1.0, 4001)


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Source float64 cosine scores; the norm-product floor is 1e-12."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape or min(a.shape) == 0:
        raise ValueError("Cosine inputs require matching nonempty (N,D) matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cosine embeddings must be finite")
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(norms, 1e-12)


def fold_accuracies(
    scores: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit threshold on all other folds and score the held-out official fold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    if (
        scores.ndim != 1
        or scores.size == 0
        or labels.shape != scores.shape
        or folds.shape != scores.shape
    ):
        raise ValueError("Scores, labels and folds must be matching nonempty vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Verification requires finite scores and binary labels")
    if (
        not np.issubdtype(folds.dtype, np.integer)
        or not np.isin(folds, np.arange(1, 11)).all()
    ):
        raise ValueError("Verification fold IDs must be integers in 1..10")
    labels = np.asarray(labels, dtype=bool)
    fold_ids = np.unique(folds)
    if len(fold_ids) < 2:
        raise ValueError("Verification protocol needs at least two folds")
    accuracies = []
    thresholds = []
    for fold in fold_ids:
        test = folds == fold
        train_scores, train_labels = scores[~test], labels[~test]
        # Keep the source accumulation, threshold grid and strictest-tie policy.
        fitted = np.array(
            [float(np.mean((train_scores >= t) == train_labels)) for t in THRESHOLDS]
        )
        best = float(THRESHOLDS[len(fitted) - 1 - int(np.argmax(fitted[::-1]))])
        thresholds.append(best)
        accuracies.append(float(np.mean((scores[test] >= best) == labels[test])))
    return np.asarray(accuracies), np.asarray(thresholds)


@dataclass(frozen=True)
class VerificationResult:
    """Unrounded fold results, with the source benchmark's rounded score view."""

    accuracies: np.ndarray
    thresholds: np.ndarray
    sample_count: int
    metric: str

    @property
    def primary_score(self) -> float:
        return float(np.mean(self.accuracies))

    @property
    def secondary_score(self) -> float:
        return float(np.std(self.accuracies))

    @property
    def threshold(self) -> float:
        return float(np.mean(self.thresholds))

    def as_dict(self) -> dict:
        return {
            "primary": {self.metric: round(self.primary_score, 5)},
            "secondary": {self.metric + "_std": round(self.secondary_score, 5)},
            "sample_count": self.sample_count,
        }


class VerificationEvaluator:
    """One embedding per aligned image, no implicit flip/fusion or alignment.

    Source evaluate_embeddings decodes one L2-normalized vector per crop;
    source score_pairs then computes cosine similarity, without flip fusion.
    ``extract_embeddings`` is reused unchanged from retrieval evaluation.
    """

    metric = "accuracy"
    extract_embeddings = staticmethod(extract_embeddings)

    def evaluate(
        self,
        embeddings: np.ndarray,
        images: Sequence[PairImage],
        pairs: Sequence[FacePair],
    ) -> VerificationResult:
        """Score embeddings in canonical image manifest order against source pairs."""
        values = np.asarray(embeddings)
        if (
            values.ndim != 2
            or min(values.shape) == 0
            or values.shape[0] != len(images)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Embeddings require one finite nonempty row per image")
        names = [image.name for image in images]
        if len(set(names)) != len(names):
            raise ValueError("Image manifest names must be unique")
        if not pairs:
            raise ValueError("No verification pairs supplied")
        by_name = dict(zip(names, values))
        try:
            a = np.stack([by_name[p.image_a] for p in pairs])
            b = np.stack([by_name[p.image_b] for p in pairs])
        except KeyError as exc:
            raise ValueError(
                f"Pair image missing from embedding manifest: {exc}"
            ) from exc
        accuracy, thresholds = fold_accuracies(
            cosine_scores(a, b),
            np.asarray([p.is_same for p in pairs]),
            np.asarray([p.fold for p in pairs]),
        )
        return VerificationResult(accuracy, thresholds, len(images), self.metric)


class LFWEvaluator(VerificationEvaluator):
    """LFW official fold accuracy."""

    metric = "lfw_accuracy"


class AgeDB30Evaluator(VerificationEvaluator):
    """AgeDB-30 prepared mirror fold accuracy."""

    metric = "agedb_30_accuracy"


class CFPEvaluator(VerificationEvaluator):
    """CFP-FP prepared mirror fold accuracy (see source YAML caveat)."""

    metric = "cfp_fp_accuracy"

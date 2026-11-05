# Verbatim datasets/_shared/face_verification.py at ec537764e36ca10da963fcb7afab157ea60ca022
"""Pair-based face verification protocol shared by LFW-style benchmarks.

A benchmark lists pairs of images with a same/different label and a fold id. The Model
provides one L2-normalized embedding per image. For every fold, the decision threshold on
cosine similarity is chosen to maximize accuracy on the other folds and then applied to
the held-out fold; the benchmark reports the mean and standard deviation over folds
(Huang et al., "Labeled Faces in the Wild", unrestricted protocol as implemented by
InsightFace / facenet ``evaluate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLDS = np.linspace(-1.0, 1.0, 4001)


@dataclass(frozen=True)
class Pair:
    image_a: str
    image_b: str
    is_same: bool
    fold: int


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(norms, 1e-12)


def _accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return float(np.mean((scores >= threshold) == labels))


def fold_accuracies(
    scores: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out accuracy per fold with the threshold fitted on the remaining folds."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    folds = np.asarray(folds)
    fold_ids = np.unique(folds)
    if len(fold_ids) < 2:
        raise ValueError("Verification protocol needs at least two folds")
    accuracies = []
    thresholds = []
    for fold in fold_ids:
        test = folds == fold
        train_scores, train_labels = scores[~test], labels[~test]
        fitted = np.array(
            [_accuracy(train_scores, train_labels, t) for t in THRESHOLDS]
        )
        # Ties on training accuracy resolve to the strictest threshold (the highest
        # similarity), matching the reference implementations that search distance
        # thresholds in ascending order and keep the first maximum.
        best = float(THRESHOLDS[len(fitted) - 1 - int(np.argmax(fitted[::-1]))])
        thresholds.append(best)
        accuracies.append(_accuracy(scores[test], labels[test], best))
    return np.asarray(accuracies), np.asarray(thresholds)


def verification_result(
    scores: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    accuracies, thresholds = fold_accuracies(scores, labels, folds)
    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "threshold": float(np.mean(thresholds)),
    }

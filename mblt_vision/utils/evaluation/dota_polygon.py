"""Original-image DOTA polygon VOC2007 AP; never an official server result.

Matching follows CAPTAIN-WHU/DOTA_devkit dota_evaluation_task1.py: strict
polygon IoU > .5, nonzero difficult targets ignored, eleven recall thresholds.
The continuous convex clipper is independent Python, not the devkit polyiou C++
extension. Numerical equivalence to that extension must be measured separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _vertices(values: Sequence[float], *, allow_degenerate: bool = False) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("DOTA polygon coordinates must be finite")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if allow_degenerate and np.all(crosses == 0):
        # Finite collinear/point predictions can arise from zero-width OBBs or
        # Task1 rounding. They are false positives, not malformed ground truth.
        return points
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise ValueError("DOTA polygon must be strictly convex and perimeter ordered")
    return points if crosses[0] > 0 else points[::-1]


def _area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    points = points - points[0]
    return (
        abs(
            float(
                np.sum(
                    points[:, 0] * np.roll(points[:, 1], -1)
                    - points[:, 1] * np.roll(points[:, 0], -1)
                )
            )
        )
        * 0.5
    )


def polygon_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Continuous convex IoU; a finite zero-area prediction has overlap zero.

    The second polygon is a ground-truth target and remains strictly validated.
    """
    subject, clip = _vertices(first, allow_degenerate=True), _vertices(second)
    subject_area = _area(subject)
    if subject_area == 0:
        return 0.0
    total = subject_area + _area(clip)
    for start, end in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        if not len(subject):
            return 0.0
        edge = end - start
        previous = subject[-1]
        delta = previous - start
        previous_side = float(edge[0] * delta[1] - edge[1] * delta[0])
        clipped = []
        for current in subject:
            delta = current - start
            side = float(edge[0] * delta[1] - edge[1] * delta[0])
            if (side >= 0) != (previous_side >= 0):
                clipped.append(
                    previous
                    + (current - previous) * previous_side / (previous_side - side)
                )
            if side >= 0:
                clipped.append(current)
            previous, previous_side = current, side
        subject = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    intersection = _area(subject)
    return min(1.0, max(0.0, intersection / (total - intersection)))


def score_original_polygons(
    predictions: list[dict[str, Any]],
    image_ids: Sequence[str],
    labels: Path,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Score merged Task1-rounded predictions against original official labels.

    All classes enter the macro mean, including classes absent from a selected
    subset (AP zero). Nonzero difficulty is ignored. Score ties use the devkit's
    NumPy default argsort, not a new deterministic ordering.
    """
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate original image IDs")
    known = set(image_ids)
    targets: dict[str, dict[int, list[tuple[list[float], bool]]]] = {}
    for image_id in image_ids:
        classes: dict[int, list[tuple[list[float], bool]]] = {}
        path = labels / f"{image_id}.txt"
        for number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            fields = line.split()
            if not fields or fields[0].lower().startswith(("imagesource", "gsd")):
                continue
            if (
                len(fields) not in {9, 10}
                or fields[8] not in class_names
                or (len(fields) == 10 and fields[9] not in {"0", "1", "2"})
            ):
                raise ValueError(f"Malformed original DOTA label at {path}:{number}")
            polygon = [float(value) for value in fields[:8]]
            _vertices(polygon)
            classes.setdefault(class_names.index(fields[8]), []).append(
                (polygon, len(fields) == 10 and fields[9] != "0")
            )
        targets[image_id] = classes
    for prediction in predictions:
        if prediction["image_id"] not in known:
            raise ValueError("Prediction names an unknown original image")
        if not 0 <= prediction["category_id"] < len(class_names):
            raise ValueError("Prediction class is outside DOTA taxonomy")
        if not np.isfinite(prediction["score"]) or not 0 <= prediction["score"] <= 1:
            raise ValueError("Invalid prediction confidence")
        _vertices(prediction["poly"], allow_degenerate=True)
    scores = []
    for class_id, name in enumerate(class_names):
        positives = sum(
            not difficult
            for classes in targets.values()
            for _, difficult in classes.get(class_id, [])
        )
        detections = [row for row in predictions if row["category_id"] == class_id]
        order = np.argsort(-np.asarray([row["score"] for row in detections]))
        matched = set()
        tp, fp = [], []
        for index in order:
            detection = detections[index]
            candidates = targets[detection["image_id"]].get(class_id, [])
            overlaps = [
                polygon_iou(detection["poly"], polygon) for polygon, _ in candidates
            ]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if best >= 0 and overlaps[best] > 0.5:
                if candidates[best][1]:
                    continue
                key = (detection["image_id"], best)
                duplicate = key in matched
                tp.append(float(not duplicate))
                fp.append(float(duplicate))
                matched.add(key)
            else:
                tp.append(0.0)
                fp.append(1.0)
        cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cumulative_tp / max(positives, 1)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(float).eps
        )
        ap = (
            sum(
                float(np.max(precision[recall >= threshold], initial=0.0))
                for threshold in np.linspace(0, 1, 11)
            )
            / 11
            if positives
            else 0.0
        )
        scores.append(
            {
                "name": name,
                "positives": positives,
                "detections": len(detections),
                "ap50": ap,
            }
        )
    return {
        "protocol": "dotav1-original-polygon-voc07-map50.v1",
        "score_kind": "local-original-image-polygon-voc07",
        "map50": float(np.mean([s["ap50"] for s in scores])),
        "classes": scores,
        "official_test_server": False,
    }

"""Evaluation script for WiderFace face detection."""

from __future__ import annotations

import math
import os
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from mblt_vision.utils.postprocess.base import YOLODetectionPostBase
from scipy.io import loadmat
from tqdm import tqdm

from ..datasets import CustomWiderFaceDataset, get_widerface_loader
from ..datasets.readiness import (
    _load_widerface_image_names,
    _widerface_image_shapes,
    _widerface_difficulty_metadata_ready,
)

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine

CustomWiderface = CustomWiderFaceDataset


class WiderFaceResult(NamedTuple):
    """WiderFace AP metrics."""

    easy_ap: float
    medium_ap: float
    hard_ap: float

    @property
    def primary_score(self) -> float:
        """Return Hard-set AP as the primary WiderFace metric."""

        return self.hard_ap

    @property
    def secondary_score(self) -> float:
        """Return Medium-set AP for singular-score compatibility."""

        return self.medium_ap

    @property
    def secondary_scores(self) -> tuple[float, float]:
        """Return Medium- and Easy-set AP in secondary-metric order."""

        return self.medium_ap, self.easy_ap


def _empty_prediction() -> np.ndarray:
    """Return an empty WiderFace prediction array."""

    return np.zeros((0, 5), dtype=np.float32)


def _initialize_predictions(
    dataset: CustomWiderFaceDataset,
) -> dict[str, dict[str, np.ndarray]]:
    """Initialize empty predictions for every WiderFace sample."""

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for _, event_name, file_name in dataset.samples:
        predictions.setdefault(event_name, {})[os.path.splitext(file_name)[0]] = (
            _empty_prediction()
        )
    return predictions


def _boxes_scores_to_prediction(
    boxes: list[list[float]], scores: list[float]
) -> np.ndarray:
    """Convert xywh boxes and scores to a WiderFace prediction array."""

    if len(boxes) != len(scores):
        raise ValueError(
            "WiderFace postprocess returned unequal box and score counts: "
            f"boxes={len(boxes)}, scores={len(scores)}."
        )
    if not boxes:
        return _empty_prediction()
    prediction = np.zeros((len(boxes), 5), dtype=np.float32)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        prediction[index, :4] = np.asarray(box, dtype=np.float32)
        prediction[index, 4] = float(score)
    return prediction


def eval_widerface(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> WiderFaceResult:
    """Evaluate a face-detection model on WiderFace validation data.

    Args:
        model: The face-detection engine to evaluate.
        data_path: Organized WiderFace dataset root.
        batch_size: Validation batch size.
        conf_thres: Optional confidence threshold override.
        iou_thres: Optional IoU threshold override.

    Returns:
        WiderFace Easy, Medium, and Hard AP metrics.
    """

    if model.post_cfg["task"] != "face_detection":
        raise NotImplementedError(
            f"Task {model.post_cfg['task']} is not supported for WiderFace evaluation."
        )
    dataset_name = model.post_cfg.get("dataset")
    if not isinstance(dataset_name, str) or dataset_name.lower() != "widerface":
        raise ValueError(
            "WiderFace evaluation requires model post_cfg.dataset to be 'widerface', "
            f"got {dataset_name!r}."
        )

    dataset_root = Path(data_path)
    expected_images = _load_widerface_image_names(dataset_root / "wider_face_val.mat")
    image_shapes = (
        _widerface_image_shapes(dataset_root, expected_images)
        if expected_images is not None
        else None
    )
    if (
        expected_images is None
        or image_shapes is None
        or not _widerface_difficulty_metadata_ready(
            dataset_root, expected_images, image_shapes=image_shapes
        )
    ):
        raise ValueError(
            "WiderFace evaluation metadata is malformed or has inconsistent difficulty indices."
        )

    dataset = CustomWiderface(os.path.join(data_path, "images"))
    actual_images: dict[str, set[str]] = {}
    for _, event_name, file_name in dataset.samples:
        actual_images.setdefault(event_name, set()).add(file_name)
    if actual_images != expected_images or sum(
        len(file_names) for file_names in actual_images.values()
    ) != len(dataset.samples):
        raise ValueError(
            "WiderFace image tree does not match the validation metadata identities."
        )
    dataloader = get_widerface_loader(
        dataset, batch_size, model.preprocess_with_metadata
    )
    model.set_postprocess_thresholds(conf_thres=conf_thres, iou_thres=iou_thres)

    predictions = _initialize_predictions(dataset)
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating WiderFace")
    inference_time = 0.0
    cum_num_data = 0

    for input_npu, org_shape, ratio_pad, target_classes, fnames in pbar:
        cum_num_data += len(fnames)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic
        nms_outs = model.postprocess(out_npu)
        input_shape = (int(input_npu.shape[1]), int(input_npu.shape[2]))
        img0_shapes = [(int(shape[0]), int(shape[1])) for shape in org_shape.tolist()]
        postprocessor = cast(YOLODetectionPostBase, model.postprocessor)
        _, boxes_list, scores_list = postprocessor.nmsout2eval(
            nms_outs.output,
            input_shape,
            img0_shapes,
            ratio_pad=ratio_pad,
        )
        batch_lengths = {
            "input batch": int(input_npu.shape[0]),
            "original shapes": len(org_shape),
            "ratio pads": len(ratio_pad),
            "target classes": len(target_classes),
            "file names": len(fnames),
            "boxes": len(boxes_list),
            "scores": len(scores_list),
        }
        if len(set(batch_lengths.values())) != 1:
            details = ", ".join(
                f"{name}={length}" for name, length in batch_lengths.items()
            )
            raise ValueError(f"WiderFace evaluation batch length mismatch: {details}.")

        for event_name, file_name, boxes, scores in zip(
            target_classes, fnames, boxes_list, scores_list, strict=True
        ):
            predictions[event_name][os.path.splitext(file_name)[0]] = (
                _boxes_scores_to_prediction(boxes, scores)
            )

        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    aps = evaluation(norm_score(predictions), data_path)
    print("WiderFace evaluation completed")
    return WiderFaceResult(*aps)


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU overlaps between boxes and query boxes."""

    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)

    boxes = boxes[:, None, :]
    query_boxes = query_boxes[None, :, :]

    iw = (
        np.minimum(boxes[..., 2], query_boxes[..., 2])
        - np.maximum(boxes[..., 0], query_boxes[..., 0])
        + 1
    )
    ih = (
        np.minimum(boxes[..., 3], query_boxes[..., 3])
        - np.maximum(boxes[..., 1], query_boxes[..., 1])
        + 1
    )
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    inter = iw * ih

    box_area = (boxes[..., 2] - boxes[..., 0] + 1) * (boxes[..., 3] - boxes[..., 1] + 1)
    query_area = (query_boxes[..., 2] - query_boxes[..., 0] + 1) * (
        query_boxes[..., 3] - query_boxes[..., 1] + 1
    )
    union = box_area + query_area - inter

    return inter / union


def get_gt_boxes(gt_dir: str) -> tuple[Any, ...]:
    """Load WiderFace evaluation `.mat` files from the organized dataset."""

    gt_mat = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]
    hard_gt_list = hard_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    easy_gt_list = easy_mat["gt_list"]

    return (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    )


def _normalize_ground_truth_boxes(boxes: np.ndarray) -> np.ndarray:
    """Convert WiderFace's all-zero no-face sentinel into an empty box table."""

    if boxes.shape == (1, 4) and not bool(np.any(boxes)):
        return np.empty((0, 4), dtype=boxes.dtype)
    return boxes


def _empty_ground_truth_prediction_contribution(
    thresh_num: int, pred_info: np.ndarray
) -> np.ndarray:
    """Return precision-recall counts for predictions on a no-face image."""

    return img_pr_info(
        thresh_num,
        pred_info,
        np.ones(len(pred_info), dtype=np.float32),
        np.zeros(len(pred_info), dtype=np.float32),
    )


def norm_score(pred: dict[str, Any]) -> dict[str, Any]:
    """Normalize WiderFace prediction scores to ``[0, 1]``."""

    max_score = -1e9
    min_score = 1e9
    found = False

    for _, event_predictions in pred.items():
        for _, image_predictions in event_predictions.items():
            if len(image_predictions) == 0:
                continue
            found = True
            _min = float(np.min(image_predictions[:, -1]))
            _max = float(np.max(image_predictions[:, -1]))
            if _max > max_score:
                max_score = _max
            if _min < min_score:
                min_score = _min

    if not found:
        return pred

    diff = max_score - min_score
    if diff <= 0:
        return pred

    for _, event_predictions in pred.items():
        for _, image_predictions in event_predictions.items():
            if len(image_predictions) == 0:
                continue
            image_predictions[:, -1] = (image_predictions[:, -1] - min_score) / diff

    return pred


def image_matches(
    pred: np.ndarray, gt: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Best-matching face per prediction, independent of the difficulty setting.

    Which faces a setting ignores is the only per-setting input to
    :func:`image_eval`, so the IoU matrix and the arg-max over it are computed
    here once per image and reused for Easy, Medium and Hard instead of three
    times over.
    """

    if len(pred) == 0 or len(gt) == 0:
        # Nothing to reduce over: `argmax` raises on an empty axis, while the
        # per-prediction loop this replaces simply never ran for an empty
        # prediction table, and an empty ground-truth table matches nothing
        # either. Both return "no prediction matched anything", which is what
        # the caller's own no-face branch assumes.
        return (
            np.zeros(len(pred), dtype=np.int64),
            np.zeros(len(pred), dtype=bool),
        )
    _pred = pred.copy()
    _gt = gt.copy()
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    best = overlaps.argmax(1)
    return best, overlaps[np.arange(_pred.shape[0]), best] >= iou_thresh


def matched_image_eval(
    best: np.ndarray, matched: np.ndarray, ignore: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn shared per-image matches into this setting's recall and proposals.

    Equivalent to the official per-prediction loop, without the loop: a
    prediction only leaves the proposal list when it matched a face this
    setting ignores, and a face is recalled once, at the first prediction that
    matched it, so the running count is a cumulative sum over first
    occurrences.
    """

    if len(ignore) == 0:
        # No annotated faces, so nothing can be matched or ignored: every
        # prediction stays a proposal and nothing is recalled, which is what
        # the caller's own no-face branch counts.
        return np.zeros(len(best), dtype=np.float64), np.ones(len(best))
    ignored = ignore[best] == 0
    proposal_list = np.where(matched & ignored, -1.0, 1.0)
    hits = np.flatnonzero(matched & ~ignored)
    first = np.zeros(len(best), dtype=np.int64)
    if hits.size:
        first[hits[np.unique(best[hits], return_index=True)[1]]] = 1
    return np.cumsum(first).astype(np.float64), proposal_list


def image_eval(
    pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one image worth of WiderFace predictions."""

    best, matched = image_matches(pred, gt, iou_thresh)
    return matched_image_eval(best, matched, ignore)


def score_threshold_index(pred_info: np.ndarray, thresh_num: int) -> np.ndarray:
    """Last prediction index at or above each score threshold, ``-1`` if none.

    The official curve walks ``thresh_num`` evenly spaced cut-offs and needs
    only that index at each one. Predictions are not assumed to be sorted, so
    the running suffix maximum is what makes the qualifying indices a prefix
    and lets one ``searchsorted`` replace the per-threshold scan.
    """

    scores = np.asarray(pred_info[:, 4], dtype=np.float64)
    thresholds = 1 - np.arange(1, thresh_num + 1) / thresh_num
    suffix_maximum = np.maximum.accumulate(scores[::-1])[::-1]
    return np.searchsorted(-suffix_maximum, -thresholds, side="right") - 1


def img_pr_info(
    thresh_num: int,
    pred_info: np.ndarray,
    proposal_list: np.ndarray,
    pred_recall: np.ndarray,
) -> np.ndarray:
    """Compute precision and recall contributions for one image."""

    pr_info = np.zeros((thresh_num, 2), dtype=np.float32)
    last_index = score_threshold_index(pred_info, thresh_num)
    reached = last_index >= 0
    if not reached.any():
        return pr_info
    index = last_index[reached]
    proposals = np.cumsum(proposal_list == 1)
    pr_info[reached, 0] = proposals[index]
    pr_info[reached, 1] = np.asarray(pred_recall)[index]
    return pr_info


def dataset_pr_info(
    thresh_num: int, pr_curve: np.ndarray, count_face: int
) -> np.ndarray:
    """Normalize a WiderFace precision-recall accumulator."""

    _pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
    for threshold_index in range(thresh_num):
        proposals = pr_curve[threshold_index, 0]
        matched = pr_curve[threshold_index, 1]
        _pr_curve[threshold_index, 0] = matched / proposals if proposals > 0 else 0.0
        _pr_curve[threshold_index, 1] = matched / count_face if count_face > 0 else 0.0
    return _pr_curve


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute VOC-style average precision."""

    mrec = np.concatenate((np.array([0.0]), rec, np.array([1.0])))
    mpre = np.concatenate((np.array([0.0]), prec, np.array([0.0])))

    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])

    recall_change_index = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum(
        (mrec[recall_change_index + 1] - mrec[recall_change_index])
        * mpre[recall_change_index + 1]
    )
    return float(ap)


def evaluation(
    pred: dict[str, Any], gt_path: str, iou_thresh: float = 0.5
) -> list[float]:
    """Evaluate WiderFace predictions against Easy, Medium, and Hard settings."""

    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = (
        get_gt_boxes(gt_path)
    )
    event_num = len(event_list)
    thresh_num = 1000
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    count_faces = [0, 0, 0]
    pr_curves = [np.zeros((thresh_num, 2), dtype=np.float32) for _ in range(3)]
    # One pass over the images scoring all three settings, because the IoU
    # matrix is the expensive part and does not depend on the setting.
    pbar = tqdm(range(event_num))
    for event_index in pbar:
        pbar.set_description("Processing easy/medium/hard")
        event_name = str(event_list[event_index][0][0])
        img_list = file_list[event_index][0]
        pred_list = pred[event_name]
        gt_bbx_list = facebox_list[event_index][0]
        for image_index, img_info in enumerate(img_list):
            pred_info = pred_list[str(img_info[0][0])]
            gt_boxes = _normalize_ground_truth_boxes(
                np.array(gt_bbx_list[image_index][0], dtype=np.float32)
            )
            keep_indices = [
                np.array(setting_gt[event_index][0][image_index][0], dtype=np.int64)
                for setting_gt in setting_gts
            ]
            for setting_id, keep_index in enumerate(keep_indices):
                count_faces[setting_id] += len(keep_index)
            if len(gt_boxes) == 0:
                if len(pred_info) != 0:
                    contribution = _empty_ground_truth_prediction_contribution(
                        thresh_num, pred_info
                    )
                    for pr_curve in pr_curves:
                        pr_curve += contribution
                continue
            if len(pred_info) == 0:
                continue
            best, matched = image_matches(pred_info, gt_boxes, iou_thresh)
            last_index = score_threshold_index(pred_info, thresh_num)
            reached = last_index >= 0
            index = last_index[reached]
            for setting_id, keep_index in enumerate(keep_indices):
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = matched_image_eval(best, matched, ignore)
                if not reached.any():
                    continue
                pr_curves[setting_id][reached, 0] += np.cumsum(proposal_list == 1)[
                    index
                ]
                pr_curves[setting_id][reached, 1] += pred_recall[index]
    pbar.close()
    aps = []
    for setting_id in range(3):
        pr_curve = dataset_pr_info(
            thresh_num, pr_curves[setting_id], count_faces[setting_id]
        )
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    print("==================== Results ====================")
    print(f"Easy   Val AP: {aps[0]}")
    print(f"Medium Val AP: {aps[1]}")
    print(f"Hard   Val AP: {aps[2]}")
    print("=================================================")
    return aps

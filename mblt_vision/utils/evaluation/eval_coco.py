"""Evaluation script for COCO dataset."""

from __future__ import annotations

import logging
import math
import os
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple

from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from ..._tasks import normalize_vision_task
from ...datasets import get_dataset_category_ids
from ..datasets import CustomCOCODataset, get_coco_loader
from ..datasets.readiness import _coco_task_annotations_valid

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine
    from ..results import Results

logger = logging.getLogger(__name__)


class COCOResult(NamedTuple):
    """COCO mAP metrics."""

    map5095: float
    map50: float

    @property
    def primary_score(self) -> float:
        """Return mAP50-95."""

        return self.map5095

    @property
    def secondary_score(self) -> float:
        """Return mAP50."""

        return self.map50


def _require_batch_cardinality(expected: int, **values: Any) -> None:
    """Reject postprocessing metadata that cannot represent every input image."""

    invalid = {
        name: len(value) for name, value in values.items() if len(value) != expected
    }
    if invalid:
        details = ", ".join(f"{name}={length}" for name, length in invalid.items())
        raise ValueError(
            "COCO evaluation batch cardinality mismatch: "
            f"expected {expected}, got {details}."
        )


def _validate_coco_dataset_taxonomy(dataset: CustomCOCODataset, task: str) -> None:
    """Ensure direct COCO evaluation uses only task-compatible category IDs."""

    raw_annotation = getattr(dataset, "raw_annotation", None)
    if raw_annotation is not None and not isinstance(raw_annotation, dict):
        raise ValueError("COCO evaluation dataset has invalid raw annotation data.")
    raw_images = (
        raw_annotation.get("images") if isinstance(raw_annotation, dict) else None
    )
    raw_categories = (
        raw_annotation.get("categories") if isinstance(raw_annotation, dict) else None
    )
    raw_annotations = (
        raw_annotation.get("annotations") if isinstance(raw_annotation, dict) else None
    )
    raw_image_records: list[Any] = []
    raw_category_records: list[Any] = []
    raw_annotation_records: list[Any] = []
    if raw_annotation is not None:
        if not isinstance(raw_images, list):
            raise ValueError("COCO evaluation dataset has malformed raw image table.")
        if not isinstance(raw_categories, list):
            raise ValueError(
                "COCO evaluation dataset has malformed raw category table."
            )
        if not isinstance(raw_annotations, list):
            raise ValueError(
                "COCO evaluation dataset has malformed raw annotation table."
            )
        raw_image_records = raw_images
        raw_category_records = raw_categories
        raw_annotation_records = raw_annotations

    categories = getattr(dataset.coco, "cats", None)
    if not isinstance(categories, dict) or not categories:
        raise ValueError("COCO evaluation dataset must define at least one category.")
    category_ids = set(categories)
    if any(
        not isinstance(category_id, int) or isinstance(category_id, bool)
        for category_id in category_ids
    ):
        raise ValueError("COCO evaluation dataset contains invalid category IDs.")
    expected_ids = (
        {1} if task == "pose_estimation" else set(get_dataset_category_ids("coco"))
    )
    unsupported_ids = category_ids - expected_ids
    if unsupported_ids:
        raise ValueError(
            "COCO evaluation dataset contains unsupported category IDs: "
            f"{sorted(unsupported_ids)}."
        )
    annotations = getattr(dataset.coco, "anns", None)
    if not isinstance(annotations, dict):
        raise ValueError("COCO evaluation dataset must define an annotation table.")
    images = getattr(dataset.coco, "imgs", None)
    if not isinstance(images, dict):
        raise ValueError("COCO evaluation dataset must define an image table.")
    image_shapes: dict[int, tuple[int, int] | None] = {}
    for image_id, image in images.items():
        if (
            not isinstance(image_id, int)
            or isinstance(image_id, bool)
            or not isinstance(image, dict)
        ):
            raise ValueError("COCO evaluation dataset contains invalid image metadata.")
        height, width = image.get("height"), image.get("width")
        if not (
            isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
            and isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
        ):
            raise ValueError(
                "COCO evaluation dataset contains invalid image dimensions."
            )
        image_shapes[image_id] = (height, width)

    annotation_records = (
        raw_annotation_records
        if raw_annotation is not None
        else list(annotations.values())
    )
    if not annotation_records:
        raise ValueError("COCO evaluation dataset must define at least one annotation.")
    if raw_annotation is not None:
        raw_image_ids = [
            record.get("id") if isinstance(record, dict) else None
            for record in raw_image_records
        ]
        raw_category_ids = [
            record.get("id") if isinstance(record, dict) else None
            for record in raw_category_records
        ]
        raw_annotation_ids = [
            record.get("id") if isinstance(record, dict) else None
            for record in raw_annotation_records
        ]
        if (
            any(
                not isinstance(record_id, int) or isinstance(record_id, bool)
                for record_id in (
                    *raw_image_ids,
                    *raw_category_ids,
                    *raw_annotation_ids,
                )
            )
            or len(raw_image_ids) != len(set(raw_image_ids))
            or len(raw_category_ids) != len(set(raw_category_ids))
            or len(raw_annotation_ids) != len(set(raw_annotation_ids))
        ):
            raise ValueError(
                "COCO evaluation dataset has duplicate or invalid raw IDs."
            )
        if (
            set(raw_image_ids) != set(images)
            or set(raw_category_ids) != category_ids
            or set(raw_annotation_ids) != set(annotations)
        ):
            raise ValueError(
                "COCO evaluation dataset raw and indexed records disagree."
            )
    if not _coco_task_annotations_valid(
        annotation_records,
        image_ids=set(images),
        category_ids=category_ids,
        image_shapes=image_shapes,
        task=task,
    ):
        raise ValueError(
            "COCO evaluation dataset contains invalid task-specific annotations."
        )


def format_coco_results(
    task: str,
    nms_outs: Results,
    input_shape: tuple[int, ...],
    org_shape: tuple[int, ...],
    ratio_pad: list[Any],
    idx: list[int],
    dataset_ids: list[int],
    postprocess: Any,
) -> list[dict[str, Any]]:
    """Format the results for COCO evaluation.

    Args:
        task (str): The task to evaluate.
        nms_outs (Results): The output of the postprocessing.
        input_shape (tuple): The shape of the input tensor.
        org_shape (tuple): The original shape of the image.
        idx (list): The indices of the images in the batch.
        dataset_ids (list): The list of image IDs in the dataset.
        postprocess: The postprocessing instance.
    Returns:
        list: The formatted results.
    """
    results = []
    if task == "object_detection":
        labels_list, boxes_list, scores_list = postprocess.nmsout2eval(
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
        )
        _require_batch_cardinality(
            len(idx),
            org_shape=org_shape,
            ratio_pad=ratio_pad,
            labels=labels_list,
            boxes=boxes_list,
            scores=scores_list,
        )
        for i, labels, boxes, scores in zip(
            idx, labels_list, boxes_list, scores_list, strict=True
        ):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                    }
                    for box, score, label in zip(boxes, scores, labels, strict=True)
                ]
            )
    elif task == "instance_segmentation":
        labels_list, boxes_list, scores_list, extra_list = postprocess.nmsout2eval(
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
        )
        _require_batch_cardinality(
            len(idx),
            org_shape=org_shape,
            ratio_pad=ratio_pad,
            labels=labels_list,
            boxes=boxes_list,
            scores=scores_list,
            extra=extra_list,
        )
        for i, labels, boxes, scores, extra in zip(
            idx, labels_list, boxes_list, scores_list, extra_list, strict=True
        ):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                        "segmentation": extra,
                    }
                    for box, score, label, extra in zip(
                        boxes, scores, labels, extra, strict=True
                    )
                ]
            )
    elif task == "pose_estimation":
        labels_list, boxes_list, scores_list, extra_list = postprocess.nmsout2eval(
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
        )
        _require_batch_cardinality(
            len(idx),
            org_shape=org_shape,
            ratio_pad=ratio_pad,
            labels=labels_list,
            boxes=boxes_list,
            scores=scores_list,
            extra=extra_list,
        )
        for i, labels, boxes, scores, extra in zip(
            idx, labels_list, boxes_list, scores_list, extra_list, strict=True
        ):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                        "keypoints": extra,
                    }
                    for box, score, label, extra in zip(
                        boxes, scores, labels, extra, strict=True
                    )
                ]
            )
    else:
        raise NotImplementedError(
            f"Only object detection, instance segmentation, and pose estimation are supported, but we got {task}"
        )
    return results


def eval_coco(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> float:
    """Evaluate a model on COCO and return the legacy numeric mAP50-95 score."""

    return eval_coco_metrics(
        model, data_path, batch_size, conf_thres, iou_thres
    ).primary_score


def eval_coco_metrics(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> COCOResult:
    """Evaluate a model on COCO and return structured mAP metrics.

    Args:
        model (MBLT_Engine): The model engine to evaluate.
        data_path (str): Path to the COCO dataset.
        batch_size (int): Batch size for evaluation.
        conf_thres (float | None): Optional confidence threshold override.
        iou_thres (float | None): Optional IoU threshold override.

    Returns:
        Structured mAP50-95 primary and mAP50 secondary metrics.
    """
    task = normalize_vision_task(
        model.post_cfg["task"],
        supported=("object_detection", "instance_segmentation", "pose_estimation"),
    )
    dataset_name = model.post_cfg.get("dataset")
    if not isinstance(dataset_name, str) or dataset_name.lower() != "coco":
        raise ValueError(
            "COCO evaluation requires model post_cfg.dataset to be 'coco', "
            f"got {dataset_name!r}."
        )
    if task in {"object_detection", "instance_segmentation"}:
        dataset = CustomCOCODataset(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "instances_val2017.json"),
        )
    else:
        dataset = CustomCOCODataset(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "person_keypoints_val2017.json"),
        )
    _validate_coco_dataset_taxonomy(dataset, task)

    dataloader = get_coco_loader(dataset, batch_size, model.preprocess_with_metadata)
    model.set_postprocess_thresholds(conf_thres=conf_thres, iou_thres=iou_thres)

    results = []
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating COCO")

    inference_time = 0.0
    cum_num_data = 0

    for input_npu, org_shape, ratio_pad, idx in pbar:
        cum_num_data += len(idx)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        nms_outs = model.postprocess(out_npu, multi_label=True)
        results.extend(
            format_coco_results(
                task,
                nms_outs,
                input_npu.shape[1:-1],
                org_shape,
                ratio_pad,
                idx,
                dataset.ids,
                model.postprocessor,
            )
        )

        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    res = evaluate_predictions_on_coco(dataset.coco, results, task, img_ids=dataset.ids)

    print("COCO evaluation completed")
    return COCOResult(
        map5095=float(res.stats[0].item()), map50=float(res.stats[1].item())
    )


def evaluate_predictions_on_coco(
    coco_gt: COCO,
    coco_results: list[dict[str, Any]],
    task: str,
    img_ids: list[int] | None = None,
) -> COCOeval_faster:
    """Evaluates predictions using the COCO API.

    Args:
        coco_gt (COCO): Ground truth COCO object.
        coco_results (list): Predictions in COCO format.
        task (str): Task type ('object_detection', 'instance_segmentation', or 'pose_estimation').
        img_ids: Optional image IDs to include in evaluation.

    Returns:
        COCOeval_faster: The COCO evaluation object containing results.
    """
    normalized_task = normalize_vision_task(
        task,
        supported=("object_detection", "instance_segmentation", "pose_estimation"),
    )

    if coco_results:
        coco_dt = coco_gt.loadRes(coco_results)
    else:
        coco_dt = COCO()

    if normalized_task == "object_detection":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "bbox", print_function=logger.info
        )
    elif normalized_task == "instance_segmentation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "segm", print_function=logger.info
        )
    elif normalized_task == "pose_estimation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "keypoints", print_function=logger.info
        )
    else:
        raise RuntimeError(f"Unexpected validated COCO task: {normalized_task}")

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

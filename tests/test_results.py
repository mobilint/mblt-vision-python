"""Tests for vision plotting helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import mblt_vision.utils.results as results_module
from mblt_vision.utils.datasets import get_dotav1_palette
from mblt_vision.utils.results import Results
from mblt_vision.utils.types import NestedListTensorLike


def test_image_classification_plot_saves_without_gui_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Save classification results without requiring OpenCV GUI support."""

    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "result.jpg"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(source_path), image)

    def _raise_destroy_all_windows() -> None:
        raise cv2.error("cvDestroyAllWindows is unavailable")

    monkeypatch.setattr(cv2, "destroyAllWindows", _raise_destroy_all_windows)

    output = torch.zeros(1000)
    output[980] = 0.9
    result = Results({}, {"task": "image_classification"}, output)

    plotted = result.plot(str(source_path), str(save_path), topk=1)

    assert plotted is not None
    assert save_path.is_file()


def test_image_classification_plot_rejects_missing_output() -> None:
    """Raise a runtime validation error when classification output is absent."""

    result = Results({}, {"task": "image_classification"}, torch.zeros(1))
    result.acc = None

    with pytest.raises(ValueError, match="No accuracy output found"):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), topk=1)


def test_instance_segmentation_plot_supports_nonzero_coco_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plot segmentation results when detections use regular COCO class ids."""

    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "segmentation.jpg"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(source_path), image)

    monkeypatch.setattr(results_module, "scale_boxes", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(results_module, "scale_masks", lambda mask, img0_shape: mask)
    monkeypatch.setattr(results_module, "crop_mask", lambda mask, boxes: mask)

    box_cls = torch.tensor([[4.0, 6.0, 20.0, 24.0, 0.9, 45.0]], dtype=torch.float32)
    mask = torch.ones((1, 32, 32), dtype=torch.float32)
    result = Results(
        {"LetterBox": {"img_size": (32, 32)}},
        {"task": "instance_segmentation"},
        [[box_cls, mask]],
    )

    plotted = result.plot(str(source_path), str(save_path))

    assert plotted is not None
    assert save_path.is_file()


def test_object_detection_plot_preserves_raw_output_coordinates() -> None:
    """Inverse scaling for plotting must not mutate stored postprocess boxes."""

    box_cls = torch.tensor([[40.0, 60.0, 160.0, 180.0, 0.9, 0.0]], dtype=torch.float32)
    result = Results(
        {"LetterBox": {"img_size": (200, 200)}},
        {"task": "object_detection"},
        [box_cls],
    )
    expected = box_cls.clone()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    first = result.plot(image)
    second = result.plot(image)

    torch.testing.assert_close(box_cls, expected)
    torch.testing.assert_close(result._box_cls_tensor(), expected)
    assert first is not None
    assert second is not None
    assert np.array_equal(first, second)


def test_plot_converts_rgb_array_source_to_bgr() -> None:
    """Return OpenCV-order output when callers provide an RGB NumPy image."""

    result = Results(
        {"LetterBox": {"img_size": (1, 1)}},
        {"task": "object_detection"},
        [torch.zeros((0, 6), dtype=torch.float32)],
    )
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)

    plotted = result.plot(rgb)

    assert plotted is not None
    assert np.array_equal(plotted, np.array([[[0, 0, 255]]], dtype=np.uint8))
    assert np.array_equal(rgb, np.array([[[255, 0, 0]]], dtype=np.uint8))


def test_plot_normalizes_float_rgb_array_source_to_uint8_bgr() -> None:
    """Plot normalized RGB arrays without rendering a near-black background."""

    result = Results(
        {"LetterBox": {"img_size": (1, 1)}},
        {"task": "object_detection"},
        [torch.zeros((0, 6), dtype=torch.float32)],
    )
    rgb = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)

    plotted = result.plot(rgb)

    assert plotted is not None
    assert np.array_equal(plotted, np.array([[[0, 0, 128]]], dtype=np.uint8))
    assert np.array_equal(rgb, np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32))


def test_pose_plot_hides_low_visibility_keypoints_and_limbs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw pose elements only when every required keypoint is visible."""

    circles: list[tuple[int, int]] = []
    limbs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def capture_circle(
        image: np.ndarray, center: tuple[int, int], *args: object, **kwargs: object
    ) -> np.ndarray:
        del args, kwargs
        circles.append(center)
        return image

    def capture_line(
        image: np.ndarray,
        point1: tuple[int, int],
        point2: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        del args, kwargs
        limbs.append((point1, point2))
        return image

    monkeypatch.setattr(cv2, "circle", capture_circle)
    monkeypatch.setattr(cv2, "line", capture_line)
    keypoints = torch.zeros((17, 3), dtype=torch.float32)
    keypoints[0] = torch.tensor([20.0, 20.0, 0.9])
    keypoints[1] = torch.tensor([40.0, 20.0, 0.9])
    box_cls = torch.cat(
        [
            torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]]),
            keypoints.reshape(1, -1),
        ],
        dim=1,
    )
    result = Results(
        {"LetterBox": {"img_size": (100, 100)}},
        {"task": "pose_estimation", "n_extra": 51},
        [box_cls],
        conf_thres=0.5,
    )

    result.plot(np.zeros((100, 100, 3), dtype=np.uint8))

    assert circles == [(20, 20), (40, 20)]
    assert limbs == [((20, 20), (40, 20))]


def test_obb_plot_uses_dotav1_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot DOTAv1 boxes without consulting the COCO palette."""

    def _reject_coco_palette(label_idx: int) -> tuple[int, int, int]:
        raise AssertionError(
            f"Unexpected COCO palette lookup for DOTAv1 class {label_idx}."
        )

    monkeypatch.setattr(results_module, "get_coco_det_palette", _reject_coco_palette)
    box_cls = torch.tensor(
        [[16.0, 16.0, 10.0, 10.0, 0.9, 2.0, 0.0]], dtype=torch.float32
    )
    result = Results(
        {"LetterBox": {"img_size": (32, 32)}},
        {"task": "obb"},
        [box_cls],
    )

    plotted = result.plot(np.zeros((32, 32, 3), dtype=np.uint8))

    assert plotted is not None
    assert np.any(np.all(plotted == get_dotav1_palette(2), axis=2))


def test_dotav1_palette_wraps_class_indices() -> None:
    """Match the modulo behavior used by the other visualization palettes."""

    assert get_dotav1_palette(15) == get_dotav1_palette(0)


def test_results_accept_path_and_basename_save_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accept pathlib inputs and avoid creating an empty parent directory."""

    source_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source_path), np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.chdir(tmp_path)
    result = Results(
        {}, {"task": "image_classification"}, torch.arange(3, dtype=torch.float32)
    )

    result.plot(source_path, Path("result.jpg"), topk=10)

    assert (tmp_path / "result.jpg").is_file()


def test_results_report_failed_image_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise OSError when OpenCV cannot encode or write the requested image."""

    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    result = Results({}, {"task": "image_classification"}, torch.ones(2))

    with pytest.raises(OSError, match="Failed to write"):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), "result.jpg", topk=1)


@pytest.mark.parametrize("topk", [0, -1, 1.5, True])
def test_results_validate_classification_topk(topk: object) -> None:
    """Reject invalid classification Top-K values."""

    result = Results({}, {"task": "image_classification"}, torch.ones(2))
    with pytest.raises((TypeError, ValueError)):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), topk=topk)


@pytest.mark.parametrize(
    ("task", "output"),
    [
        ("object_detection", []),
        ("instance_segmentation", []),
        ("instance_segmentation", [[]]),
    ],
)
def test_results_reject_empty_structured_outputs(
    task: str, output: NestedListTensorLike
) -> None:
    """Validate structured result containers before indexing."""

    with pytest.raises(ValueError):
        Results({}, {"task": task}, output)


def test_results_normalize_obb_task_and_semantic_taxonomy_case() -> None:
    """Normalize the OBB task and semantic palette taxonomy casing."""

    obb = Results({}, {"task": "obb"}, [torch.zeros((0, 7))])
    semantic = Results(
        {},
        {"task": "semantic_segmentation", "dataset": "CityScapes"},
        np.zeros((1, 4, 4), dtype=np.uint8),
    )

    assert obb.task == "obb"
    plotted = semantic.plot(np.zeros((4, 4, 3), dtype=np.uint8))
    assert plotted is not None
    assert plotted.shape == (4, 4, 3)

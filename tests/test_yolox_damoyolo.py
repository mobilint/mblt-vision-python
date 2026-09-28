"""YOLOX and DAMO-YOLO geometry, preprocessing, and decode checks."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from mblt_vision.utils.letterbox import LetterBoxGeometry, letterbox_center
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.postprocess.damoyolo_post import DAMOYOLODetectionPost
from mblt_vision.utils.postprocess.yolox_post import YOLOXDetectionPost
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.utils.preprocess.reader import Reader
from mblt_vision.wrapper import MBLT_Engine

TOP_LEFT_640 = {"LetterBox": {"img_size": [640, 640], "center": False}}
YOLOX_POST = {"task": "object_detection", "dataset": "coco", "head": "yolox", "nl": 3}
DAMO_POST = {
    "task": "object_detection",
    "dataset": "coco",
    "head": "damoyolo",
    "nl": 3,
    "reg_max": 16,
}


def _reference_yolox_decode(raw: torch.Tensor, size: int) -> torch.Tensor:
    """YOLOX ``YOLOXHead.decode_outputs`` followed by objectness x class."""

    grids, strides = [], []
    for stride in (8, 16, 32):
        cells = size // stride
        yv, xv = torch.meshgrid(torch.arange(cells), torch.arange(cells), indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        strides.append(torch.full((*grid.shape[:2], 1), stride))
    grid = torch.cat(grids, 1).float()
    stride = torch.cat(strides, 1).float()
    xy = (raw[..., :2] + grid) * stride
    wh = torch.exp(raw[..., 2:4]) * stride
    x1y1 = xy - wh / 2
    return torch.cat([x1y1, x1y1 + wh, raw[..., 4:5] * raw[..., 5:]], -1)


def _reference_damo_decode(
    classes: list[torch.Tensor], boxes: list[torch.Tensor], bins: int = 17
) -> torch.Tensor:
    """DAMO-YOLO's GFL ``Integral`` projection and ``distance2bbox`` from cell priors."""

    rows = []
    for stride, cls_map, box_map in zip((8, 16, 32), classes, boxes):
        batch, _, height, width = box_map.shape
        yv, xv = torch.meshgrid(
            torch.arange(height) * stride, torch.arange(width) * stride, indexing="ij"
        )
        priors = torch.stack((xv.flatten(), yv.flatten()), -1).float()
        dist = box_map.flatten(2).transpose(1, 2).reshape(batch, -1, 4, bins)
        dist = (dist.softmax(-1) * torch.linspace(0, bins - 1, bins)).sum(-1) * stride
        xyxy = torch.cat([priors - dist[..., :2], priors + dist[..., 2:]], -1)
        rows.append(torch.cat([xyxy, cls_map.flatten(2).transpose(1, 2)], -1))
    return torch.cat(rows, 1)


def _damo_heads(batch: int = 1, seed: int = 0) -> tuple[list, list]:
    generator = torch.Generator().manual_seed(seed)
    classes = [
        torch.rand((batch, 80, 640 // s, 640 // s), generator=generator) * 0.02
        for s in (8, 16, 32)
    ]
    boxes = [
        torch.randn((batch, 68, 640 // s, 640 // s), generator=generator)
        for s in (8, 16, 32)
    ]
    return classes, boxes


# --- Geometry and preprocessing -------------------------------------------------


def test_top_left_geometry_pads_only_bottom_and_right() -> None:
    geometry = LetterBoxGeometry.from_shapes((640, 640), (480, 640), center=False)

    assert geometry.pad == (0, 0)
    assert geometry.borders == (0, 160, 0, 0)
    assert LetterBoxGeometry.from_shapes((640, 640), (480, 640)).pad == (0, 80)


def test_letterbox_anchors_top_left_with_configured_padding() -> None:
    preprocess = build_preprocess(
        {"LetterBox": {"img_size": [64, 64], "center": False, "padding_value": 0}}
    )
    image = np.full((32, 64, 3), 200, dtype=np.uint8)

    output, metadata = preprocess.with_metadata(image)

    assert metadata["ratio_pad"] == ((1.0, 1.0), (0, 0))
    assert bool((output[:32] == 200).all())
    assert bool((output[32:] == 0).all())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"center": "no"}, TypeError),
        ({"padding_value": 1.5}, TypeError),
        ({"padding_value": True}, TypeError),
        ({"padding_value": 256}, ValueError),
    ],
)
def test_letterbox_rejects_invalid_anchoring_options(kwargs, error) -> None:
    with pytest.raises(error):
        build_preprocess({"LetterBox": {"img_size": [64, 64], **kwargs}})


def test_letterbox_center_reads_model_config() -> None:
    assert letterbox_center({"LetterBox": {"img_size": [64, 64]}}) is True
    assert letterbox_center(TOP_LEFT_640) is False
    with pytest.raises(TypeError, match="center"):
        letterbox_center({"LetterBox": {"center": 0}})


def test_reader_bgr_flips_arrays_into_a_copy_and_keeps_cv2_file_order(tmp_path) -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    reader = Reader("numpy", color_mode="BGR")

    bgr = reader(rgb)

    assert bgr[0, 0].tolist() == [0, 0, 255]
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert bgr.flags["C_CONTIGUOUS"]

    path = tmp_path / "red.png"
    cv2.imwrite(str(path), rgb[..., ::-1])
    assert reader(path)[0, 0].tolist() == [0, 0, 255]
    assert Reader("numpy")(path)[0, 0].tolist() == [255, 0, 0]


def test_reader_rejects_bgr_from_the_rgb_only_pil_style() -> None:
    with pytest.raises(ValueError, match="RGB images only"):
        Reader("pil", color_mode="BGR")
    with pytest.raises(ValueError, match="color_mode"):
        Reader("numpy", color_mode="YUV")


def _onnx_engine(graph_type: str) -> MBLT_Engine:
    engine = object.__new__(MBLT_Engine)
    engine.input_name = "images"
    graph_input = SimpleNamespace(type=graph_type, shape=[1, 3, 64, 64])
    engine._onnx_session = SimpleNamespace(get_inputs=lambda: [graph_input])
    engine._require_onnx_session = lambda: engine._onnx_session
    return engine


@pytest.mark.parametrize(
    ("input_dtype", "graph_type", "expected_dtype"),
    [
        # No Normalize step: LetterBox's bytes into a float graph.
        (torch.uint8, "tensor(float)", np.float32),
        # Normalize's float32 into reduced- or double-precision graphs.
        (torch.float32, "tensor(float16)", np.float16),
        (torch.float32, "tensor(double)", np.float64),
        (torch.float64, "tensor(float)", np.float32),
        (torch.float32, "tensor(float)", np.float32),
        # A graph declaring no float type keeps the old rule: bytes pass through.
        (torch.uint8, "tensor(uint8)", np.uint8),
    ],
)
def test_onnx_inputs_are_cast_to_the_declared_float_dtype(
    input_dtype: torch.dtype, graph_type: str, expected_dtype: type
) -> None:
    engine = _onnx_engine(graph_type)

    inputs = engine._prepare_onnx_inputs(torch.full((64, 64, 3), 7, dtype=input_dtype))

    assert inputs["images"].dtype == expected_dtype
    assert inputs["images"].shape == (1, 3, 64, 64)
    assert float(inputs["images"].max()) == 7.0


# --- YOLOX ---------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["rows", "channels_first", "extra_axis"])
def test_yolox_decode_matches_upstream(layout: str) -> None:
    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    assert isinstance(post, YOLOXDetectionPost)
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn((2, 8400, 85), generator=generator)
    raw[..., 4:] = raw[..., 4:].sigmoid()
    shaped = {
        "rows": raw,
        "channels_first": raw.transpose(1, 2),
        "extra_axis": raw[:, None],
    }[layout]

    decoded = post.non_e2e(post.check_input(shaped))

    assert decoded.shape == (2, 84, 8400)
    torch.testing.assert_close(
        decoded.transpose(1, 2), _reference_yolox_decode(raw, 640)
    )


def test_yolox_detects_a_planted_box_through_nms() -> None:
    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    raw = torch.zeros((1, 8400, 85))
    raw[..., 2:4] = -10.0
    # Stride-8 cell (x=10, y=5): offsets 0.5 put the centre at (84, 44), and
    # log(4) makes the box 32 px wide on that level.
    anchor = 5 * 80 + 10
    raw[0, anchor, :4] = torch.tensor([0.5, 0.5, np.log(4.0), np.log(4.0)])
    raw[0, anchor, 4] = 0.9
    raw[0, anchor, 5 + 17] = 0.8

    (detections,) = post(raw.numpy())

    assert detections.shape == (1, 6)
    torch.testing.assert_close(
        detections[0, :4], torch.tensor([68.0, 28.0, 100.0, 60.0])
    )
    assert detections[0, 4].item() == pytest.approx(0.72)
    assert detections[0, 5].item() == 17


def test_yolox_rejects_a_mismatched_anchor_count() -> None:
    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    with pytest.raises(ValueError, match=r"\(batch, 8400, 85\)"):
        post(torch.zeros((1, 3549, 85)))


def test_yolox_rejects_strides_that_disagree_with_nl() -> None:
    with pytest.raises(ValueError, match="strides"):
        build_postprocess(TOP_LEFT_640, {**YOLOX_POST, "strides": [8, 16]})


# --- DAMO-YOLO -------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_damoyolo_decode_matches_upstream_in_any_output_order(layout: str) -> None:
    post = build_postprocess(TOP_LEFT_640, DAMO_POST)
    assert isinstance(post, DAMOYOLODetectionPost)
    classes, boxes = _damo_heads(batch=2)
    outputs = [*reversed(boxes), *classes]
    if layout == "nhwc":
        # The stride-8 class map is 80x80x80 here: only the distribution maps
        # can say which axis holds the channels.
        outputs = [head.permute(0, 2, 3, 1) for head in outputs]

    decoded = post.non_e2e(post.check_input(outputs))

    assert decoded.shape == (2, 84, 8400)
    # The expectation is a matrix product here and a multiply-and-sum upstream:
    # float32 rounding reaches ~6e-5 px on 640-px coordinates.
    torch.testing.assert_close(
        decoded.transpose(1, 2),
        _reference_damo_decode(classes, boxes),
        atol=1e-4,
        rtol=0,
    )


def test_damoyolo_rejects_heads_in_mixed_layouts() -> None:
    post = build_postprocess(TOP_LEFT_640, DAMO_POST)
    classes, boxes = _damo_heads()
    outputs = [boxes[0], boxes[1].permute(0, 2, 3, 1), boxes[2], *classes]

    with pytest.raises(ValueError, match="one channel layout"):
        post(outputs)


def test_damoyolo_detects_a_planted_box_through_nms() -> None:
    post = build_postprocess(TOP_LEFT_640, DAMO_POST)
    classes = [torch.zeros((1, 80, 640 // s, 640 // s)) for s in (8, 16, 32)]
    boxes = [torch.full((1, 68, 640 // s, 640 // s), -20.0) for s in (8, 16, 32)]
    # Stride-16 cell (x=7, y=3) sits at (112, 48); put every side's mass on bin 2.
    boxes[1][0, :, 3, 7] = -20.0
    boxes[1][0, [2, 19, 36, 53], 3, 7] = 20.0
    classes[1][0, 3, 3, 7] = 0.9

    (detections,) = post([*classes, *boxes])

    assert detections.shape == (1, 6)
    torch.testing.assert_close(
        detections[0, :4], torch.tensor([80.0, 16.0, 144.0, 80.0]), atol=1e-3, rtol=0
    )
    assert detections[0, 4].item() == pytest.approx(0.9)
    assert detections[0, 5].item() == 3


def test_damoyolo_rejects_level_sizes_for_another_input() -> None:
    post = build_postprocess(
        {"LetterBox": {"img_size": [416, 416], "center": False}}, DAMO_POST
    )
    classes, boxes = _damo_heads()
    with pytest.raises(ValueError, match="level sizes"):
        post([*classes, *boxes])


# --- Dispatch and inverse geometry -----------------------------------------------


def test_head_dispatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unsupported post_cfg.head"):
        build_postprocess(TOP_LEFT_640, {**YOLOX_POST, "head": "yolov6"})
    with pytest.raises(ValueError, match="object_detection only"):
        build_postprocess(
            TOP_LEFT_640,
            {**YOLOX_POST, "task": "instance_segmentation", "n_extra": 32},
        )


def test_missing_ratio_pad_uses_the_models_own_anchoring() -> None:
    """A top-left model must not be restored as if it were centered."""

    detections = [torch.tensor([[0.0, 0.0, 64.0, 32.0, 0.9, 0.0]])]
    top_left = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    centered = build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {"task": "object_detection", "dataset": "coco", "nl": 3, "reg_max": 16},
    )

    _, (tl_boxes,), _ = top_left.nmsout2eval(detections, (640, 640), [(320, 640)])
    _, (c_boxes,), _ = centered.nmsout2eval(detections, (640, 640), [(320, 640)])

    assert tl_boxes == [[0.0, 0.0, 64.0, 32.0]]
    assert c_boxes == [[0.0, 0.0, 64.0, 0.0]]


# --- Backend dtypes --------------------------------------------------------------


def test_damoyolo_decodes_float16_heads_like_float32() -> None:
    """Float16 heads must not fail the bin projection, and decode in float32."""

    post = build_postprocess(TOP_LEFT_640, DAMO_POST)
    classes, boxes = _damo_heads()
    classes[1][0, 3, 3, 7] = 0.9
    half = [head.half() for head in [*classes, *boxes]]

    detections_half = post(half)
    detections_full = post([head.float() for head in half])

    assert len(detections_half) == 1 and detections_half[0].shape[0] > 0
    assert detections_half[0].dtype == torch.float32
    torch.testing.assert_close(detections_half[0], detections_full[0])


def test_yolox_decodes_float16_output_without_overflowing_exp() -> None:
    """``exp`` of a float16 size regression above ~11 is inf unless decoded in float32."""

    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    raw = torch.zeros((1, 8400, 85))
    raw[..., 2:4] = -10.0
    anchor = 8000  # the first stride-32 cell
    raw[0, anchor, 2:4] = 2.0
    raw[0, anchor, 4] = 0.9
    raw[0, anchor, 5] = 0.8
    raw[0, 0, :4] = torch.tensor([0.0, 0.0, 11.5, 11.5])
    raw[0, 0, 4] = 0.9
    raw[0, 0, 6] = 0.8

    (detections,) = post(raw.half())

    assert bool(torch.isfinite(detections).all())
    torch.testing.assert_close(detections, post(raw.half().float())[0])
    assert sorted(detections[:, 5].tolist()) == [0.0, 1.0]


@pytest.mark.parametrize(
    ("img0_shapes", "ratio_pad"),
    [
        # One shared shape for the whole batch, with and without metadata.
        ((320, 640), None),
        ((320, 640), ((1.0, 1.0), (0, 0))),
        # Per-image shapes, and per-image metadata over a shared shape.
        ([(320, 640), (320, 640)], None),
        ((320, 640), [None, ((1.0, 1.0), (0, 0))]),
    ],
)
def test_top_left_eval_conversion_keeps_the_nms_batch(img0_shapes, ratio_pad) -> None:
    """A shared image shape must cover every NMS output, as it does for centered models."""

    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    raw = torch.zeros((2, 8400, 85))
    raw[..., 2:4] = -10.0
    raw[:, 5 * 80 + 10, :4] = torch.tensor([0.5, 0.5, np.log(4.0), np.log(4.0)])
    raw[:, 5 * 80 + 10, 4] = 0.9
    raw[:, 5 * 80 + 10, 5] = 0.8

    detections = post(raw)
    _, boxes, _ = post.nmsout2eval(detections, (640, 640), img0_shapes, ratio_pad)

    # The top-left letterbox of a 320x640 image into 640x640 is a 1x resize with
    # no padding, so the xywh box keeps its input-space coordinates.
    assert boxes == [[[68.0, 28.0, 32.0, 32.0]], [[68.0, 28.0, 32.0, 32.0]]]


def test_top_left_eval_conversion_scales_each_image_by_its_own_shape() -> None:
    post = build_postprocess(TOP_LEFT_640, YOLOX_POST)
    detection = torch.tensor([[0.0, 0.0, 64.0, 32.0, 0.9, 0.0]])

    _, boxes, _ = post.nmsout2eval(
        [detection, detection.clone()], (640, 640), [(320, 640), (640, 1280)]
    )

    assert boxes == [[[0.0, 0.0, 64.0, 32.0]], [[0.0, 0.0, 128.0, 64.0]]]


# --- Top-left masks ----------------------------------------------------------------


def test_top_left_mask_crop_drops_only_the_bottom_padding() -> None:
    """A 480x640 source in a top-left 640x640 letterbox pads only 160 bottom rows."""

    from mblt_vision.utils.postprocess.common import scale_masks

    mask = torch.zeros((1, 640, 640))
    mask[0, :480] = 1.0  # exactly the image region

    restored = scale_masks(
        mask, (480, 640), ratio_pad=((1.0, 1.0), (0, 0)), center=False
    )

    assert restored.shape == (1, 480, 640)
    assert bool((restored > 0.5).all())
    # The centered crop keeps the padding and squeezes it into the image.
    centered = scale_masks(mask, (480, 640), ratio_pad=((1.0, 1.0), (0, 0)))
    assert float((centered > 0.5).float().mean()) == pytest.approx(0.75)


def test_centered_mask_crop_is_unchanged_by_the_anchoring_option() -> None:
    from mblt_vision.utils.postprocess.common import scale_masks

    masks = torch.rand((3, 640, 640), generator=torch.Generator().manual_seed(0))
    for ratio_pad in (None, ((1.0, 1.0), (0, 80)), ((0.5, 0.5), (0, 107))):
        torch.testing.assert_close(
            scale_masks(masks, (480, 640), ratio_pad=ratio_pad, center=True),
            scale_masks(masks, (480, 640), ratio_pad=ratio_pad),
            rtol=0,
            atol=0,
        )


def test_top_left_segmentation_eval_restores_a_non_square_mask() -> None:
    post = build_postprocess(
        TOP_LEFT_640,
        {"task": "instance_segmentation", "dataset": "coco", "nl": 3, "reg_max": 16},
    )
    detection = torch.tensor([[0.0, 0.0, 640.0, 480.0, 0.9, 0.0]])
    mask = torch.zeros((1, 640, 640))
    mask[0, :480] = 1.0

    _, _, _, (encoded,) = post.nmsout2eval(
        [(detection, mask)], (640, 640), [(480, 640)]
    )

    from faster_coco_eval.core import mask as mask_utils

    assert mask_utils.area(encoded[0]) == 480 * 640


def test_segmentation_eval_accepts_one_shared_shape_for_the_batch() -> None:
    from mblt_vision.utils.postprocess.common import nmsout2eval_seg

    detection = torch.tensor([[0.0, 0.0, 64.0, 32.0, 0.9, 0.0]])
    mask = torch.ones((1, 640, 640))

    labels, _, _, masks = nmsout2eval_seg(
        [(detection, mask), (detection.clone(), mask.clone())], (640, 640), (480, 640)
    )

    assert len(labels) == len(masks) == 2


def test_top_left_segmentation_plot_tints_the_whole_non_square_image(tmp_path) -> None:
    """Plotting must restore the mask over the image rows, not squeeze in the padding."""

    from mblt_vision.utils.results import Results

    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full((24, 32, 3), 255, dtype=np.uint8))
    box_cls = torch.tensor([[0.0, 0.0, 32.0, 24.0, 0.9, 0.0]])
    mask = torch.zeros((1, 32, 32))
    mask[0, :24] = 1.0
    result = Results(
        {"LetterBox": {"img_size": [32, 32], "center": False}},
        {"task": "instance_segmentation"},
        [[box_cls, mask]],
    )

    plotted = result.plot(str(source))

    # Sample the mask interior, away from the box outline drawn on the image
    # border. The old centered crop squeezed the 8 padding rows into the image,
    # so its bottom quarter stayed white.
    assert (plotted[20, 16] != 255).any()
    assert (plotted[10, 16] != 255).any()


def test_dense_fallback_follows_the_models_top_left_anchoring() -> None:
    """Without recorded metadata, a top-left depth map is restored from its own rows."""

    from mblt_vision.utils.postprocess.depth_post import DepthPost

    post = DepthPost(TOP_LEFT_640, {"task": "depth_estimation", "dataset": "nyu-depth"})
    depth = torch.zeros((1, 640, 640))
    depth[0, :480] = 1.0  # the image region; the 160 bottom rows are padding

    restored = post(depth, img0_shape=[(480, 640)])  # one image: a bare map

    assert restored.shape == (480, 640)
    assert float(restored.min()) == pytest.approx(1.0)

"""YOLOX postprocessing.

YOLOX is anchor-free with an objectness channel, so it belongs to neither the
anchor-based family nor the Ultralytics anchorless one: its head emits a single
``(batch, anchors, 5 + classes)`` tensor of ``[dx, dy, dw, dh, objectness, classes]``
with objectness and the class scores already through sigmoid and the box left
undecoded (``head.decode_in_inference = False``, which is what
``mblt-model-ops/models/YOLOX-*/compile/export_onnx.py`` exports).

Only the conversion from that tensor to candidates differs from the anchorless
family, so everything after it -- confidence filtering, ``xywh2xyxy``, NMS and the
inverse letterbox -- is inherited rather than rewritten.
"""

from __future__ import annotations

import torch

from ..types import ListTensorLike, TensorLike
from .common import YOLOFaceDetectionMixin
from .yolo_anchorless_post import YOLOAnchorlessDetectionPost


class YOLOXDetectionPost(YOLOAnchorlessDetectionPost):
    """Postprocessing for YOLOX object detection."""

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        """Initialize the YOLOX postprocessor.

        Args:
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)
        # YOLOX's grid is `torch.arange(size)` with no half-cell shift, while
        # `make_anchors` defaults to the Ultralytics 0.5 offset. Half a stride is 4
        # pixels on the coarsest level, enough to move every box, so rebuild the
        # anchor grid on YOLOX's own convention rather than inherit that offset.
        self.make_anchors(offset=0.0)

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode YOLOX's raw head into Ultralytics-style candidates.

        Applies upstream's ``decode_outputs``: the centre is the grid cell plus the
        predicted offset in stride units, the size is the exponential of the predicted
        log-size in stride units, and a candidate's per-class score is objectness times
        the class score, which is what its own demo thresholds on.

        Args:
            x: Exactly one raw output tensor.

        Returns:
            Candidates as ``(batch, 4 + classes, anchors)`` with ``xywh`` boxes in
            letterboxed pixels and scores in ``[0, 1]``.

        Raises:
            ValueError: If the tensor is not one YOLOX head of the configured shape.
        """
        if len(x) != 1:
            raise ValueError(
                f"YOLOX postprocessing expects one output tensor, got {len(x)}."
            )
        raw = x[0]
        if raw.ndim == 2:
            raw = raw[None]
        # `check_input` promotes a single output to a batch-of-one of a batch-of-one;
        # drop the leading singletons the same way `filter_conversion` does.
        while raw.ndim == 4 and 1 in (raw.shape[0], raw.shape[1]):
            raw = raw.squeeze(0) if raw.shape[0] == 1 else raw.squeeze(1)
        if raw.ndim != 3:
            raise ValueError(
                f"Expected a (batch, anchors, 5 + classes) tensor, got shape "
                f"{tuple(raw.shape)}."
            )
        channels = 5 + self.nc
        if raw.shape[-1] != channels:
            if raw.shape[1] != channels:
                raise ValueError(
                    f"Expected {channels} channels for {self.nc} classes, got shape "
                    f"{tuple(raw.shape)}."
                )
            raw = raw.transpose(1, 2)

        anchors = self.anchors_as_tensor().to(raw.device)  # (2, anchors)
        stride = self.stride_as_tensor().to(raw.device)  # (1, anchors)
        if raw.shape[1] != anchors.shape[-1]:
            raise ValueError(
                f"Output carries {raw.shape[1]} anchors but this configuration has "
                f"{anchors.shape[-1]}; check pre_cfg.LetterBox.img_size and post_cfg.nl."
            )
        grid = anchors.transpose(0, 1).unsqueeze(0)  # (1, anchors, 2)
        strides = stride.transpose(0, 1).unsqueeze(0)  # (1, anchors, 1)

        centre = (raw[..., 0:2] + grid) * strides
        size = torch.exp(raw[..., 2:4]) * strides
        scores = raw[..., 4:5] * raw[..., 5:]
        return torch.cat([centre, size, scores], dim=-1).permute(0, 2, 1)

    def extract_final_outputs(
        self, x: TensorLike | ListTensorLike
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.Tensor | None]:
        """Never treat a YOLOX head as an already-decoded detection tensor.

        The inherited detector reads a trailing width of six as "xyxy, confidence,
        class" from an end-to-end export. A single-class YOLOX head is exactly six
        channels wide (four box, objectness, one class) and is not decoded at all, so
        the raw path has to own it.
        """
        return None, None


class YOLOXFaceDetectionPost(YOLOFaceDetectionMixin, YOLOXDetectionPost):
    """Single-class WiderFace postprocessing over the YOLOX decode."""

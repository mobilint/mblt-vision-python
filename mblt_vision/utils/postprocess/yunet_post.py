"""YuNet postprocessing.

YuNet is anchor-free with a separate objectness, like YOLOX, but its head is exported
per level *and* per branch: twelve tensors, in upstream's own order and naming
(``cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, ..., kps_8, ...``), with the
class score and objectness already through sigmoid and the box and landmarks
undecoded. A candidate's score is the class score times the objectness -- a product,
not the square root of one that OpenCV's C++ demo uses -- and its box is
``xy * stride + prior`` and ``exp(wh) * stride`` against a prior grid with no half-cell
offset (`yunet_train/tasks/face/codec.py`, `engine/priors.py`).

The five landmark points are decoded by upstream and dropped here, because
``face_detection`` in this package is a box-and-score contract: the anchor-based
`YOLOv5*-face` and `YOLOv7*-face` families have their landmark channels stripped in
`mblt-model-ops` for the same reason.
"""

from __future__ import annotations

import torch

from ..types import ListTensorLike, TensorLike
from .common import YOLOFaceDetectionMixin
from .yolo_anchorless_post import YOLOAnchorlessDetectionPost, _AnchorlessNMSInput

# Channel width of each head, in upstream's export order. Class score and objectness are
# both one channel wide, so nothing but that order tells them apart -- which is why the
# exporter pins it and why this reads the tensors positionally.
HEAD_WIDTHS = (1, 1, 4, 10)


class YuNetFaceDetectionPost(YOLOFaceDetectionMixin, YOLOAnchorlessDetectionPost):
    """Postprocessing for YuNet face detection."""

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        """Initialize the YuNet postprocessor.

        Args:
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)
        # `MlvlPointGenerator(strides, offset=0)`: the prior is the top-left corner of the
        # cell, not its centre, so the inherited Ultralytics half-cell offset is wrong here.
        self.make_anchors(offset=0.0)

    def nms(
        self,
        x: object,
        max_det: int = 1000,
        max_nms: int = 30000,
        max_wh: int = 7680,
        multi_label: bool = False,
    ) -> list[torch.Tensor]:
        """Suppress with a cap high enough for the split the primary metric comes from.

        WiderFace's Hard AP is decided on crowded images: fifteen validation images carry
        more than the inherited 300 faces and one carries 709, and upstream evaluates with
        no cap at all (`max_detections=-1`). Keeping 300 truncates exactly those images.
        """
        return super().nms(
            x,  # type: ignore[arg-type]
            max_det=max_det,
            max_nms=max_nms,
            max_wh=max_wh,
            multi_label=multi_label,
        )

    def _pre_process(
        self, x: list[torch.Tensor]
    ) -> tuple[_AnchorlessNMSInput, torch.Tensor | None]:
        """Convert the twelve raw head tensors into filtered candidates."""
        candidates = self.to_candidates(x)
        return (
            _AnchorlessNMSInput(self.filter_conversion(candidates), "candidates_first"),
            None,
        )

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return export-style candidates without confidence filtering."""
        return self.to_candidates(x)

    def to_candidates(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode the raw head into Ultralytics-style candidates.

        Args:
            x: The twelve head tensors in upstream's export order.

        Returns:
            Candidates as ``(batch, 4 + classes, anchors)`` with ``xywh`` boxes in
            letterboxed pixels and scores in ``[0, 1]``.

        Raises:
            ValueError: If the tensors are not this model's twelve heads.
        """
        heads = self._grouped_heads(x)
        scores = torch.cat(
            [
                (cls_score * objectness).transpose(1, 2)
                for cls_score, objectness in zip(heads[0], heads[1])
            ],
            dim=-1,
        )  # (batch, classes, anchors)
        boxes = torch.cat(
            [prediction.transpose(1, 2) for prediction in heads[2]], dim=-1
        )  # (batch, 4, anchors)

        anchors = self.anchors_as_tensor().to(boxes.device)  # (2, anchors)
        stride = self.stride_as_tensor().to(boxes.device)  # (1, anchors)
        if boxes.shape[-1] != anchors.shape[-1]:
            raise ValueError(
                f"Head carries {boxes.shape[-1]} anchors but this configuration has "
                f"{anchors.shape[-1]}; check pre_cfg.LetterBox.img_size and post_cfg.nl."
            )
        priors = (anchors * stride).unsqueeze(0)  # (1, 2, anchors), in pixels
        centre = boxes[:, :2] * stride + priors
        size = torch.exp(boxes[:, 2:]) * stride
        return torch.cat([centre, size, scores], dim=1)

    def _grouped_heads(self, x: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """Split the flat output list into its four heads, each in stride order.

        Every tensor is ``(batch, anchors, channels)``; a level is identified by its
        anchor count, which is what allows a runtime to hand the levels over in any
        order, while the four heads are identified by position, because class score and
        objectness are indistinguishable by shape.
        """
        expected = self.nl * len(HEAD_WIDTHS)
        if len(x) != expected:
            raise ValueError(
                f"YuNet postprocessing expects {expected} head tensors "
                f"({len(HEAD_WIDTHS)} heads x {self.nl} levels), got {len(x)}."
            )
        grouped: list[list[torch.Tensor]] = []
        for index, width in enumerate(HEAD_WIDTHS):
            level_tensors = []
            for tensor in x[index * self.nl : (index + 1) * self.nl]:
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.ndim == 2:
                    tensor = tensor[None]
                if tensor.ndim != 3:
                    raise ValueError(
                        f"Expected a (batch, anchors, channels) head, got shape "
                        f"{tuple(tensor.shape)}."
                    )
                if tensor.shape[-1] != width:
                    if tensor.shape[1] != width:
                        raise ValueError(
                            f"Head {index} is {tensor.shape[-1]} channels wide, expected "
                            f"{width}."
                        )
                    tensor = tensor.transpose(1, 2)
                level_tensors.append(tensor)
            grouped.append(
                sorted(level_tensors, key=lambda head: head.shape[1], reverse=True)
            )
        return grouped

    def extract_final_outputs(
        self, x: TensorLike | ListTensorLike
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.Tensor | None]:
        """Never treat these raw heads as already-decoded detections."""
        return None, None

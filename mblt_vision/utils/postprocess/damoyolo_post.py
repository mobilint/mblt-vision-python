"""DAMO-YOLO postprocessing.

DAMO-YOLO's ZeroHead is anchor-free with a distribution-based box, like the
Ultralytics anchorless family, but three things differ and each one moves boxes:

* the head is exported per level and per branch, six tensors rather than one -- three
  class maps ``(batch, classes, h, w)`` **already through sigmoid**, and three
  distribution maps ``(batch, 4 * (reg_max + 1), h, w)``;
* ``reg_max`` counts the largest distance in the distribution rather than the number of
  bins, so a ``reg_max`` of 16 means 17 bins, not 16;
* the centre priors are ``arange(size) * stride`` with no half-cell offset, where
  Ultralytics uses ``(arange(size) + 0.5) * stride``.

Everything after the conversion to candidates -- confidence filtering, ``xywh2xyxy``,
NMS and the inverse letterbox -- is the inherited anchorless path.
"""

from __future__ import annotations

import torch

from ..types import ListTensorLike, TensorLike
from .common import YOLOFaceDetectionMixin
from .yolo_anchorless_post import YOLOAnchorlessDetectionPost, _AnchorlessNMSInput


class DAMOYOLODetectionPost(YOLOAnchorlessDetectionPost):
    """Postprocessing for DAMO-YOLO object detection."""

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        """Initialize the DAMO-YOLO postprocessor.

        Args:
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.

        Raises:
            ValueError: If ``reg_max`` is missing or not positive.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)
        if self.reg_max <= 0:
            raise ValueError(
                "DAMO-YOLO postprocessing requires a positive reg_max in post_cfg."
            )
        # Upstream's `Integral` projects onto `linspace(0, reg_max, reg_max + 1)`, so
        # the distribution has one more bin than `reg_max` -- the opposite of the
        # Ultralytics convention this base class assumes.
        self.bins = self.reg_max + 1
        self.project = torch.arange(self.bins, dtype=torch.float32, device=self.device)
        # `get_single_level_center_priors` uses `arange(size) * stride`; no half cell.
        self.make_anchors(offset=0.0)

    def _pre_process(
        self, x: list[torch.Tensor]
    ) -> tuple[_AnchorlessNMSInput, torch.Tensor | None]:
        """Convert the six raw head tensors into filtered candidates.

        Args:
            x: Checked model output tensors.

        Returns:
            Layout-aware detections and ``None``, since detection has no prototypes.
        """
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
            x: The six raw head tensors, in any order.

        Returns:
            Candidates as ``(batch, 4 + classes, anchors)`` with ``xywh`` boxes in
            letterboxed pixels and scores in ``[0, 1]``.

        Raises:
            ValueError: If the tensors are not this model's class and distribution maps.
        """
        class_maps, distribution_maps = self._split_heads(x)
        self.validate_split_head_counts(
            classification=class_maps, detection=distribution_maps
        )

        scores, distances = [], []
        for class_map, distribution_map in zip(class_maps, distribution_maps):
            if class_map.shape[-2:] != distribution_map.shape[-2:]:
                raise ValueError(
                    "Class and distribution maps disagree on feature size: "
                    f"{tuple(class_map.shape[-2:])} and "
                    f"{tuple(distribution_map.shape[-2:])}."
                )
            batch = distribution_map.shape[0]
            cells = distribution_map.shape[-2] * distribution_map.shape[-1]
            # (b, 4 * bins, h, w) -> (b, 4, bins, h * w), then the distribution's
            # expected value over its bins, in stride units.
            bins = distribution_map.reshape(batch, 4, self.bins, cells).softmax(dim=2)
            distances.append(
                torch.einsum("bkna,n->bka", bins, self.project.to(bins.device))
            )
            scores.append(class_map.reshape(batch, self.nc, cells))

        distance = torch.cat(distances, dim=-1)  # (b, 4, anchors), stride units
        score = torch.cat(scores, dim=-1)  # (b, classes, anchors)

        anchors = self.anchors_as_tensor().to(distance.device)  # (2, anchors)
        stride = self.stride_as_tensor().to(distance.device)  # (1, anchors)
        if distance.shape[-1] != anchors.shape[-1]:
            raise ValueError(
                f"Head carries {distance.shape[-1]} anchors but this configuration has "
                f"{anchors.shape[-1]}; check pre_cfg.LetterBox.img_size and post_cfg.nl."
            )
        # Priors and distances are both in pixels once multiplied by the level's stride.
        priors = (anchors * stride).unsqueeze(0)  # (1, 2, anchors)
        distance = distance * stride
        top_left = priors - distance[:, :2]
        bottom_right = priors + distance[:, 2:]
        centre = (top_left + bottom_right) / 2
        size = bottom_right - top_left
        return torch.cat([centre, size, score], dim=1)

    def _split_heads(
        self, x: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sort the raw tensors into class maps and distribution maps, coarsest last.

        Channel count identifies the branch and cell count orders the levels, so the
        runtime is free to hand the tensors over in any order -- which the NPU backend
        does not promise to preserve.
        """
        class_maps: list[torch.Tensor] = []
        distribution_maps: list[torch.Tensor] = []
        distribution_channels = 4 * self.bins
        for tensor in x:
            if tensor.ndim == 3:
                tensor = tensor[None]
            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected a (batch, channels, h, w) head, got shape "
                    f"{tuple(tensor.shape)}."
                )
            # A channels-last artifact carries the channel count in the last axis.
            if tensor.shape[1] not in (self.nc, distribution_channels) and tensor.shape[
                -1
            ] in (self.nc, distribution_channels):
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.shape[1] == self.nc:
                class_maps.append(tensor)
            elif tensor.shape[1] == distribution_channels:
                distribution_maps.append(tensor)
            else:
                raise ValueError(
                    f"Head with {tensor.shape[1]} channels is neither {self.nc} class "
                    f"scores nor {distribution_channels} distribution bins."
                )

        def by_cells(head: torch.Tensor) -> int:
            return int(head.shape[-2] * head.shape[-1])

        return (
            sorted(class_maps, key=by_cells, reverse=True),
            sorted(distribution_maps, key=by_cells, reverse=True),
        )

    def extract_final_outputs(
        self, x: TensorLike | ListTensorLike
    ) -> tuple[list[torch.Tensor] | torch.Tensor | None, torch.Tensor | None]:
        """Never treat these raw heads as already-decoded detections."""
        return None, None


class DAMOYOLOFaceDetectionPost(YOLOFaceDetectionMixin, DAMOYOLODetectionPost):
    """Single-class WiderFace postprocessing over the DAMO-YOLO decode."""

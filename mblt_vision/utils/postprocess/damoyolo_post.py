"""Postprocessing for DAMO-YOLO detectors (tinyvision)."""

from __future__ import annotations

import torch

from .common import dist2bbox, make_grid_points
from .yolo_anchorless_post import YOLOAnchorlessDetectionPost, _AnchorlessNMSInput
from .yolox_post import head_strides


class DAMOYOLODetectionPost(YOLOAnchorlessDetectionPost):
    """Decode DAMO-YOLO's raw heads, then reuse the anchorless filter and NMS.

    The export stops at the head convolutions: per level, one sigmoid class map
    ``(batch, nc, H, W)`` and one distribution map ``(batch, 4 * (reg_max + 1), H, W)``.
    ``reg_max`` counts the largest distance, not the bins, so the default 16 means
    17 bins -- one more than Ultralytics' DFL with the same number. Each side's
    distance is the softmax expectation over ``0..reg_max`` times the stride,
    measured from the cell's top-left corner. Channels-last maps, as an NPU
    returns them, and any output order are accepted; levels are told apart by
    channel count and ordered by spatial size.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        reg_max = post_cfg.get("reg_max", 16)
        if isinstance(reg_max, bool) or not isinstance(reg_max, int) or reg_max < 1:
            raise ValueError(
                f"DAMO-YOLO post_cfg.reg_max must be a positive integer, got {reg_max!r}."
            )
        # The anchorless base reads ``reg_max`` as a bin count; keep its DFL out of
        # this head's different convention.
        super().__init__(pre_cfg, {**post_cfg, "reg_max": 0}, **kwargs)
        if self.n_extra:
            raise ValueError("DAMO-YOLO postprocessing supports object detection only.")
        self.bins = reg_max + 1
        if self.nc == 4 * self.bins:
            raise ValueError(
                f"DAMO-YOLO cannot tell {self.nc}-class maps from 4 x {self.bins}-bin "
                "distribution maps by channel count."
            )
        self.bin_values = torch.arange(
            self.bins, dtype=torch.float32, device=self.device
        )
        self.head_strides = head_strides(post_cfg, self.nl)
        self.grid, self.grid_stride = make_grid_points(
            self.imh, self.imw, self.head_strides, self.device
        )

    def _levels(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened ``(batch, anchors, nc)`` scores and ``(batch, anchors, 4 * bins)``."""

        if len(x) != 2 * self.nl:
            raise ValueError(
                f"DAMO-YOLO exports {2 * self.nl} tensors for {self.nl} levels, got {len(x)}."
            )
        box_channels = 4 * self.bins
        channels = {self.nc, box_channels}
        heads = []
        for output in x:
            while output.ndim > 4 and output.shape[0] == 1:
                output = output.squeeze(0)
            if output.ndim != 4:
                raise ValueError(
                    f"DAMO-YOLO head tensors must be 4D, got shape {tuple(output.shape)}."
                )
            heads.append(output)
        # One layout for the whole set. A 640 input's stride-8 class map is
        # 80x80x80, so its own shape cannot say which axis holds the classes; the
        # distribution maps (4 * bins channels) always can.
        layouts = {
            "nchw" if head.shape[1] in channels else "nhwc"
            for head in heads
            if (head.shape[1] in channels) != (head.shape[-1] in channels)
        }
        if len(layouts) != 1:
            raise ValueError(
                "Could not infer one channel layout for DAMO-YOLO heads "
                f"{[tuple(head.shape) for head in heads]}."
            )
        channels_last = layouts == {"nhwc"}
        classes, boxes = [], []
        for head in heads:
            head = head.permute(0, 3, 1, 2) if channels_last else head
            if head.shape[1] not in channels:
                raise ValueError(
                    f"DAMO-YOLO head has {head.shape[1]} channels, expected "
                    f"{self.nc} (classes) or {box_channels} (4 x {self.bins} bins)."
                )
            (classes if head.shape[1] == self.nc else boxes).append(head)
        self.validate_split_head_counts(detection=boxes, classification=classes)
        expected = [(self.imh // s, self.imw // s) for s in self.head_strides]
        flattened = []
        for group in (classes, boxes):
            group.sort(key=lambda head: head.shape[-2] * head.shape[-1], reverse=True)
            shapes = [tuple(head.shape[-2:]) for head in group]
            if shapes != expected:
                raise ValueError(
                    f"DAMO-YOLO level sizes {shapes} do not match strides "
                    f"{self.head_strides} for a {self.imh}x{self.imw} input."
                )
            flattened.append(
                torch.cat([head.flatten(2) for head in group], 2).transpose(1, 2)
            )
        return flattened[0], flattened[1]

    def decode_levels(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode every anchor to ``(batch, anchors, 4 + nc)`` xyxy boxes and scores."""

        scores, distributions = self._levels(x)
        batch, anchors = distributions.shape[:2]
        probabilities = distributions.reshape(batch, anchors, 4, self.bins).softmax(-1)
        distances = (probabilities @ self.bin_values) * self.grid_stride
        boxes = dist2bbox(distances, self.grid * self.grid_stride, xywh=False)
        return torch.cat([boxes, scores], -1)

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[_AnchorlessNMSInput, None]:
        decoded = self.decode_levels(x)
        keep = decoded[..., 4:].amax(-1) > self.conf_thres
        return (
            _AnchorlessNMSInput(
                [rows[mask] for rows, mask in zip(decoded, keep)], "candidates_first"
            ),
            None,
        )

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return every decoded anchor as ``(batch, 4 + nc, anchors)``, unfiltered."""

        return self.decode_levels(x).transpose(1, 2)

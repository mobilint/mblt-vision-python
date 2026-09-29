"""Postprocessing for YOLOX detectors (Megvii-BaseDetection)."""

from __future__ import annotations

import torch

from .common import make_grid_points, xywh2xyxy
from .yolo_anchorless_post import YOLOAnchorlessDetectionPost, _AnchorlessNMSInput


def head_strides(post_cfg: dict, nl: int) -> list[int]:
    """Return ``post_cfg.strides``, or the ``8, 16, 32, ...`` default for ``nl`` levels.

    Raises:
        ValueError: If the configured strides disagree with ``nl``.
    """

    strides = post_cfg.get("strides")
    if strides is None:
        return [2 ** (3 + level) for level in range(nl)]
    if (
        not isinstance(strides, list)
        or len(strides) != nl
        or not all(
            isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in strides
        )
    ):
        raise ValueError(
            f"post_cfg.strides must list {nl} positive integers, got {strides!r}."
        )
    return list(strides)


class YOLOXDetectionPost(YOLOAnchorlessDetectionPost):
    """Decode YOLOX's deploy output, then reuse the anchorless filter and NMS.

    The export stops where upstream's own does (``head.decode_in_inference =
    False``): one ``(batch, anchors, 5 + nc)`` tensor holding the raw box
    regression, a sigmoid objectness, and sigmoid class scores. A box is
    ``xy = (raw + grid) * stride`` and ``wh = exp(raw) * stride`` from the cell's
    top-left corner, and a class score is objectness times the class probability.
    Channels-first ``(batch, 5 + nc, anchors)`` and extra singleton axes, as an
    NPU may return them, are accepted too.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        super().__init__(pre_cfg, post_cfg, **kwargs)
        if self.n_extra:
            raise ValueError("YOLOX postprocessing supports object detection only.")
        self.head_strides = head_strides(post_cfg, self.nl)
        self.grid, self.grid_stride = make_grid_points(
            self.imh, self.imw, self.head_strides, self.device
        )

    def _raw_rows(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return the single output as float32 ``(batch, anchors, 5 + nc)``.

        Decoding runs in float32 whatever the backend returns: ``exp`` of a float16
        size regression overflows to ``inf`` above about 11, before any promotion
        against the float32 grid could help.
        """

        if len(x) != 1:
            raise ValueError(f"YOLOX exports one output tensor, got {len(x)}.")
        raw = x[0]
        # ``check_dim`` adds a leading axis to every 3D output, and an NPU may add
        # its own; drop singleton axes ahead of the (anchors, channels) pair.
        while raw.ndim > 3 and 1 in raw.shape[:2]:
            raw = raw.squeeze(0 if raw.shape[0] == 1 else 1)
        width, anchors = 5 + self.nc, self.grid.shape[0]
        if raw.ndim == 3 and raw.shape[1:] == (anchors, width):
            return raw.float()
        if raw.ndim == 3 and raw.shape[1:] == (width, anchors):
            return raw.transpose(1, 2).float()
        raise ValueError(
            f"YOLOX output must be (batch, {anchors}, {width}) for a "
            f"{self.imh}x{self.imw} input, got {tuple(x[0].shape)}."
        )

    def decode_rows(self, raw: torch.Tensor) -> torch.Tensor:
        """Decode ``(batch, anchors, 5 + nc)`` rows to ``(batch, anchors, 4 + nc)`` xyxy."""

        xy = (raw[..., :2] + self.grid) * self.grid_stride
        wh = torch.exp(raw[..., 2:4]) * self.grid_stride
        scores = raw[..., 4:5] * raw[..., 5:]
        return torch.cat([xywh2xyxy(torch.cat([xy, wh], -1)), scores], -1)

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[_AnchorlessNMSInput, None]:
        decoded = self.decode_rows(self._raw_rows(x))
        keep = decoded[..., 4:].amax(-1) > self.conf_thres
        return (
            _AnchorlessNMSInput(
                [rows[mask] for rows, mask in zip(decoded, keep)], "candidates_first"
            ),
            None,
        )

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return every decoded anchor as ``(batch, 4 + nc, anchors)``, unfiltered."""

        return self.decode_rows(self._raw_rows(x)).transpose(1, 2)

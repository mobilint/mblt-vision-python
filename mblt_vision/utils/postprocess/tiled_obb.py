"""Tiled OBB restore/merge helpers and thin subclasses of existing YOLO heads.

Decoding, per-tile NMS and evaluation box conversion remain inherited. Only
native tiling orchestration, restoration and cross-tile merging are added here.
See the package README for the pinned Ultralytics source and callback contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ...datasets import get_dataset_class_names
from ..datasets.dota_tiling import (
    _DEFAULT_CONFIG,
    _DEFAULT_PREPROCESSING,
    ULTRALYTICS_VERSION,
    Tile,
    TiledConfig,
    _image_id,
    _validate_merge_max_candidates,
    get_eval_protocol,
    iter_tiles,
    prepare_tile,
    validate_preprocessing,
)
from ..letterbox import LetterBoxGeometry
from .common import rotated_nms, xywhr2xyxyxyxy
from .yolo_anchorless_post import YOLOAnchorlessOBBPost
from .yolo_dflfree_post import YOLODFLFreeOBBPost


class TilePredictor(Protocol):
    """Backend seam: post-NMS native-tile rows, NOT network-space rows.

    Return Nx7 ``[cx, cy, w, h, angle_radians, score, class0]`` as numpy or torch.
    Boxes must already have inverse letterboxing applied once, with no clipping
    of rotated centers/sizes. Empty output is shape (0, 7). Model preprocessing,
    precision, confidence filtering and per-tile NMS belong to the callback.
    """

    def __call__(self, tile: Tile) -> np.ndarray | torch.Tensor: ...


def _rows(rows: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(rows, (np.ndarray, torch.Tensor)):
        raise TypeError("predictor must return a numpy array or torch tensor")
    result = (
        torch.as_tensor(rows).detach().to(device="cpu", dtype=torch.float32).clone()
    )
    if result.ndim != 2 or result.shape[1] != 7:
        raise ValueError("Expected Nx7 [cx,cy,w,h,angle,score,class0] rows")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Prediction rows must be finite")
    if bool((result[:, 2:4] < 0).any()):
        raise ValueError("Box widths/heights must be non-negative")
    if bool(((result[:, 5] < 0) | (result[:, 5] > 1)).any()):
        raise ValueError("Scores must be in [0, 1]")
    if bool(((result[:, 6] < 0) | (result[:, 6] != result[:, 6].floor())).any()):
        raise ValueError("Classes must be non-negative integers")
    return result


def merge_tile_predictions(
    predictions: Iterable[tuple[Tile, np.ndarray | torch.Tensor]],
    protocol: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Restore and merge one original image's tiles, returning owned CPU float32 Nx7.

    Reproduce pred_to_json's Python round(rbox,3)/round(score,5) BEFORE adding
    offsets, then eval_json's dynamic class offsets and Fast-NMS at 0.3. There is
    no rate division, polygon clipping, additional conf filter or merged max_det.
    Stable score ties follow this package's device-independent selection policy.
    Over-budget candidate streams raise before NMS rather than silently truncating;
    explicit trusted-input overrides belong to protocol.merge.max_candidates.
    """
    merge = (get_eval_protocol() if protocol is None else protocol)["merge"]
    max_candidates = _validate_merge_max_candidates(merge.get("max_candidates"))
    restored = []
    image_id = None
    candidate_count = 0
    for tile, rows in predictions:
        if image_id is not None and image_id != tile.image_id:
            raise ValueError("merge_tile_predictions accepts only one original image")
        image_id = tile.image_id
        # Count before copying/rounding this tile and before quadratic merge work.
        # Never top-k/drop rows: rejection leaves all accepted numerical semantics intact.
        if not isinstance(rows, (np.ndarray, torch.Tensor)):
            raise TypeError("predictor must return a numpy array or torch tensor")
        if rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError("Expected Nx7 [cx,cy,w,h,angle,score,class0] rows")
        candidate_count += len(rows)
        if max_candidates is not None and candidate_count > max_candidates:
            raise ValueError(
                f"DOTA merge has {candidate_count} candidates, exceeding "
                f"max_candidates={max_candidates}; raise only for trusted inputs"
            )
        local = _rows(rows)
        # Python round, not torch.round: JSON serialisation is part of the reference.
        values = [
            [*[round(v, 3) for v in row[:5]], round(row[5], 5), row[6]]
            for row in local.tolist()
        ]
        # val.py adds integer offsets to JSON's Python floats before constructing
        # a float32 tensor. Adding to an already-float32 tensor can round twice.
        for row in values:
            row[0] += tile.window[0]
            row[1] += tile.window[1]
        restored.append(torch.tensor(values, dtype=torch.float32).reshape(-1, 7))
    if not restored:
        return torch.empty((0, 7), dtype=torch.float32)
    boxes = torch.cat(restored)
    if not len(boxes):
        return boxes
    # Deliberately no clamp to 1: val.py uses exactly max(center) * 2.
    class_offset = (
        boxes[:, 6:7] * (torch.max(boxes[:, :2]).item() * 2)
        if merge["classwise"]
        else torch.zeros_like(boxes[:, 6:7])
    )
    shifted = boxes[:, :5].clone()
    shifted[:, :2] += class_offset
    return boxes[rotated_nms(shifted, boxes[:, 5], merge["iou_thres"])]


def predict_tiled(
    image_bgr: np.ndarray,
    image_id: str,
    predictor: TilePredictor,
    config: TiledConfig = _DEFAULT_CONFIG,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Stream crops and merge one image, optionally using a trusted protocol override."""
    return merge_tile_predictions(
        ((tile, predictor(tile)) for tile in iter_tiles(image_bgr, image_id, config)),
        protocol,
    )


def restore_tile_boxes(
    rows: np.ndarray | torch.Tensor, geometry: LetterBoxGeometry
) -> torch.Tensor:
    """Undo shared letterbox once for Nx7 network-space rows; no offset or clipping.

    An NPU callback can use this after decoding and per-tile rotated NMS. Angles,
    scores and classes are unchanged; only centers and sizes are transformed.
    """
    result = _rows(rows)
    result[:, 0] -= geometry.pad[0]
    result[:, 1] -= geometry.pad[1]
    result[:, :4] /= geometry.ratio
    return result


class UltralyticsTilePredictor:
    """Optional pinned CPU PyTorch reference adapter with validation-style tile NMS.

    Loads only an existing local .pt checkpoint (never downloads). Requires exactly
    ultralytics==8.3.228. Fuses the eval-mode model like validation AutoBackend's
    fuse=True default, then uses upstream multi_label=True rotated NMS, not
    YOLO.predict's single-label defaults. All inference is CPU FP32, without
    augmentation. Construction loads/fuses weights; __call__ performs inference.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        imgsz: int = 1024,
        conf: float = 0.01,
        iou: float = 0.7,
        max_det: int = 300,
        preprocessing: str = _DEFAULT_PREPROCESSING,
    ) -> None:
        self.preprocessing = validate_preprocessing(preprocessing)
        if not Path(weights).is_file() or Path(weights).suffix.lower() != ".pt":
            raise ValueError("weights must be an existing trusted local .pt checkpoint")
        if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if (
            not 0 <= conf <= 1
            or not 0 < iou <= 1
            or type(max_det) is not int
            or max_det <= 0
        ):
            raise ValueError(
                "Require conf in [0,1], iou in (0,1], and positive integer max_det"
            )
        try:
            import ultralytics
            from ultralytics import YOLO
            from ultralytics.utils.nms import non_max_suppression
        except ImportError as exc:
            raise ImportError(
                "Reference adapter requires optional ultralytics==8.3.228"
            ) from exc
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"Require ultralytics=={ULTRALYTICS_VERSION}, got {ultralytics.__version__}"
            )
        wrapper = YOLO(str(Path(weights).resolve()), task="obb")
        if wrapper.task != "obb":
            raise ValueError("weights must contain an OBB model")
        # Validation's AutoBackend defaults to fuse=True (pinned autobackend.py
        # lines 145, 200-209). Unfused Conv+BN arithmetic can cross Task1's 0.001
        # rounding boundary even with otherwise identical pixels and detections.
        self.model = wrapper.model.cpu().float().eval().fuse(verbose=False)
        self.names = tuple(wrapper.names[i] for i in range(len(wrapper.names)))
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self._nms = non_max_suppression

    def __call__(self, tile: Tile) -> torch.Tensor:
        tensor, geometry = prepare_tile(
            tile, self.imgsz, preprocessing=self.preprocessing
        )
        with torch.inference_mode():
            raw = self.model(tensor)
            pred = self._nms(
                raw,
                conf_thres=self.conf,
                iou_thres=self.iou,
                multi_label=True,
                max_det=self.max_det,
                nc=len(self.names),
                rotated=True,
            )[0]
        return restore_tile_boxes(pred[:, [0, 1, 2, 3, 6, 4, 5]], geometry)


def write_task1(
    predictions: Mapping[str, np.ndarray | torch.Tensor],
    output_dir: str | Path,
    class_names: Sequence[str] | None = None,
) -> Path:
    """Write merged original-image Task1 files, replacing rather than appending.

    Input rows are already merged native-original Nx7. Polygon corner order and
    Python round(poly,3)/round(score,3), with normal str(float), match val.py.
    All class files are emitted, including empty ones. Invalid data fails before
    writing files; use a dedicated output directory for a submission.
    """
    names = tuple(
        get_dataset_class_names("dotav1") if class_names is None else class_names
    )
    names = tuple(name.replace(" ", "-") for name in names)
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "class_names must be non-empty and unique after replacing spaces"
        )
    for name in names:
        if (
            not name
            or any(c.isspace() for c in name)
            or any(c in name for c in "/\\")
            or name in {".", ".."}
        ):
            raise ValueError("class_names must be safe filename tokens")
    lines: list[list[str]] = [[] for _ in names]
    for image_id, rows in predictions.items():
        _image_id(image_id)
        data = _rows(rows)
        if bool((data[:, 6] >= len(names)).any()):
            raise ValueError("Class ID outside class_names taxonomy")
        polygons = xywhr2xyxyxyxy(data[:, :5]).reshape(-1, 8).tolist()
        for row, polygon in zip(data.tolist(), polygons, strict=True):
            values = [
                image_id,
                str(round(row[5], 3)),
                *[str(round(v, 3)) for v in polygon],
            ]
            lines[int(row[6])].append(" ".join(values) + "\n")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, contents in zip(names, lines, strict=True):
        (root / f"Task1_{name}.txt").write_text("".join(contents), encoding="utf-8")
    return root


class DOTATiledOBBPostMixin:
    """Add dataset protocol selection and tile orchestration, not another decoder.

    ``__call__`` and ``nmsout2eval`` are inherited unchanged. Raw engine inference
    and the single-tile output order therefore keep their established contract.
    Model-level threshold overrides win over dataset inference defaults.
    """

    def __init__(
        self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any], **kwargs: Any
    ) -> None:
        self.eval_protocol = get_eval_protocol(
            post_cfg["dataset"], post_cfg["eval_protocol"]
        )
        resolved = dict(self.eval_protocol["inference"])
        resolved.update(post_cfg)
        self.tile_config = TiledConfig(
            **{
                **self.eval_protocol["tiling"],
                "rates": tuple(self.eval_protocol["tiling"]["rates"]),
            }
        )
        self.tile_preprocessing = validate_preprocessing(resolved["preprocessing"])
        self.tile_max_det = resolved["max_det"]
        super().__init__(pre_cfg, resolved, **kwargs)

    def iter_tiles(
        self, image_bgr: np.ndarray, image_id: str, config: TiledConfig | None = None
    ) -> Iterable[Tile]:
        """Yield dataset-native crops without changing model preprocessing."""
        return iter_tiles(
            image_bgr, image_id, self.tile_config if config is None else config
        )

    def merge_tiles(
        self, predictions: Iterable[tuple[Tile, np.ndarray | torch.Tensor]]
    ) -> torch.Tensor:
        """Merge existing tile-pixel OBB rows under the selected dataset protocol."""
        return merge_tile_predictions(predictions, self.eval_protocol)

    def nms(self, x: Any, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        """Reuse the head's NMS with the dataset's per-tile detection cap."""
        if not args:
            kwargs.setdefault("max_det", self.tile_max_det)
        return super().nms(x, *args, **kwargs)


class YOLOAnchorlessTiledOBBPost(DOTATiledOBBPostMixin, YOLOAnchorlessOBBPost):
    """DOTA tiling over the existing YOLOv8/YOLO11 OBB postprocessor."""


class YOLODFLFreeTiledOBBPost(DOTATiledOBBPostMixin, YOLODFLFreeOBBPost):
    """DOTA tiling over the existing DFL-free OBB postprocessor."""

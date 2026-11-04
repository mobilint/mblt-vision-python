"""Opt-in DOTA native-crop inference, pinned to Ultralytics v8.3.228.

See ``mblt_vision/obb/TILED.md`` for provenance, numerical differences and the
callback contract. Import this module explicitly; ordinary predict/val is unchanged.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from ..datasets import get_dataset_class_names
from ..utils.letterbox import LetterBoxGeometry
from ..utils.postprocess.common import rotated_nms, xywhr2xyxyxyxy
from ..utils.preprocess.letterbox import _apply_letterbox

TILED_API_VERSION = 1
ULTRALYTICS_VERSION = "8.3.228"
ULTRALYTICS_COMMIT = "f8e6fddc89ec754b3a62eab87d5221e83058005a"


@dataclass(frozen=True)
class TiledConfig:
    """Native-window configuration; nominal rates change crop size, not pixels.

    ``jpeg_roundtrip`` reproduces split_test's default OpenCV JPEG write/read.
    Disable it explicitly for lossless in-memory slicing (a documented difference).
    Filtering is applied jointly across all rates, as in the reference.
    """

    crop_size: int = 1024
    gap: int = 500
    rates: tuple[float, ...] = (0.5, 1.0, 1.5)
    im_rate_thr: float = 0.6
    eps: float = 0.01
    jpeg_roundtrip: bool = True

    def __post_init__(self) -> None:
        if type(self.crop_size) is not int or type(self.gap) is not int:
            raise TypeError("crop_size and gap must be integers")
        if not 0 <= self.gap < self.crop_size:
            raise ValueError("Require 0 <= gap < crop_size")
        if not self.rates or any(not math.isfinite(r) or r <= 0 for r in self.rates):
            raise ValueError("rates must be non-empty, finite and positive")
        pairs = self.crop_pairs
        if any(not 0 <= gap < size for size, gap in pairs):
            raise ValueError("Each int(crop_size / rate) must exceed int(gap / rate)")
        if len({size for size, _ in pairs}) != len(pairs):
            raise ValueError(
                "Rates must produce distinct crop sizes (reference filenames otherwise collide)"
            )
        if (
            not 0 <= self.im_rate_thr < 1
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError("Require 0 <= im_rate_thr < 1 and finite eps > 0")
        if type(self.jpeg_roundtrip) is not bool:
            raise TypeError("jpeg_roundtrip must be a boolean")

    @property
    def crop_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return truncated crop/gap pairs in rate order."""
        return tuple((int(self.crop_size / r), int(self.gap / r)) for r in self.rates)


_DEFAULT_CONFIG = TiledConfig()


def get_windows(
    im_size: tuple[int, int], config: TiledConfig = _DEFAULT_CONFIG
) -> np.ndarray:
    """Return reference int64 ``[x1,y1,x2,y2]`` windows, x-major within each rate.

    Stops can exceed a small image's extent. Python slicing, not padding, clips
    them. The strict area comparison and near-maximum fallback are intentional.
    """
    if len(im_size) != 2 or any(type(v) is not int or v <= 0 for v in im_size):
        raise ValueError("im_size must contain positive integer (height, width)")
    h, w = im_size
    windows = []
    for size, gap in config.crop_pairs:
        step = size - gap
        xn = 1 if w <= size else math.ceil((w - size) / step + 1)
        yn = 1 if h <= size else math.ceil((h - size) / step + 1)
        xs, ys = [step * i for i in range(xn)], [step * i for i in range(yn)]
        if len(xs) > 1 and xs[-1] + size > w:
            xs[-1] = w - size
        if len(ys) > 1 and ys[-1] + size > h:
            ys[-1] = h - size
        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        windows.append(np.concatenate((start, start + size), axis=1))
    result = np.concatenate(windows, axis=0)
    clipped = result.copy()
    clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, w)
    clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, h)
    area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    win_area = (result[:, 2] - result[:, 0]) * (result[:, 3] - result[:, 1])
    rates = area / win_area
    if not (rates > config.im_rate_thr).any():
        rates[abs(rates - rates.max()) < config.eps] = 1
    return result[rates > config.im_rate_thr]


def _image_id(image_id: str) -> None:
    if not isinstance(image_id, str) or not image_id or "__" in image_id:
        raise ValueError("image_id must be a non-empty original stem without '__'")
    if (
        any(c.isspace() for c in image_id)
        or any(c in image_id for c in "/\\")
        or image_id in {".", ".."}
    ):
        raise ValueError("image_id must be a whitespace-free filename stem")


@dataclass(frozen=True)
class Tile:
    """Native tile and explicit original-image metadata, without inferred filename parsing.

    ``image`` is an owned, contiguous uint8 HWC BGR copy, never resized or padded.
    ``window`` holds nominal bounds; actual dimensions may be smaller. Callbacks
    may modify their owned image, but must not mutate the metadata.
    """

    image_id: str
    name: str
    rate: float
    crop_size: int
    gap: int
    window: tuple[int, int, int, int]
    image: np.ndarray


def iter_tiles(
    image_bgr: np.ndarray, image_id: str, config: TiledConfig = _DEFAULT_CONFIG
) -> Iterator[Tile]:
    """Yield native BGR crops, optionally JPEG round-tripped like split_test.

    Input must be non-empty uint8 HWC BGR with three channels. It is never mutated
    or retained by the yielded owned images. EXIF rotation is the caller's job;
    dimensions must describe the same decoded raster that is being sliced.
    """
    _image_id(image_id)
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must be a uint8 numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) <= 0:
        raise ValueError("image_bgr must be non-empty HWC BGR with three channels")
    pairs = {
        size: (rate, gap)
        for rate, (size, gap) in zip(config.rates, config.crop_pairs, strict=True)
    }
    for coords in get_windows(tuple(image_bgr.shape[:2]), config):
        x1, y1, x2, y2 = (int(c) for c in coords)
        size = x2 - x1
        rate, gap = pairs[size]
        patch = image_bgr[y1:y2, x1:x2].copy(order="C")
        if config.jpeg_roundtrip:
            ok, encoded = cv2.imencode(".jpg", patch)
            if not ok:
                raise RuntimeError("OpenCV failed to encode native DOTA crop")
            patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if patch is None:
                raise RuntimeError("OpenCV failed to decode native DOTA crop")
        yield Tile(
            image_id,
            f"{image_id}__{size}__{x1}___{y1}",
            rate,
            size,
            gap,
            (x1, y1, x2, y2),
            patch,
        )


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
) -> torch.Tensor:
    """Restore and merge one original image's tiles, returning owned CPU float32 Nx7.

    Reproduce pred_to_json's Python round(rbox,3)/round(score,5) BEFORE adding
    offsets, then eval_json's dynamic class offsets and Fast-NMS at 0.3. There is
    no rate division, polygon clipping, additional conf filter or merged max_det.
    Stable score ties follow this package's device-independent selection policy.
    """
    restored = []
    image_id = None
    for tile, rows in predictions:
        if image_id is not None and image_id != tile.image_id:
            raise ValueError("merge_tile_predictions accepts only one original image")
        image_id = tile.image_id
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
    class_offset = boxes[:, 6:7] * (torch.max(boxes[:, :2]).item() * 2)
    shifted = boxes[:, :5].clone()
    shifted[:, :2] += class_offset
    return boxes[rotated_nms(shifted, boxes[:, 5], 0.3)]


def predict_tiled(
    image_bgr: np.ndarray,
    image_id: str,
    predictor: TilePredictor,
    config: TiledConfig = _DEFAULT_CONFIG,
) -> torch.Tensor:
    """Stream native crops through a backend callback, restore and merge one image."""
    return merge_tile_predictions(
        (tile, predictor(tile)) for tile in iter_tiles(image_bgr, image_id, config)
    )


def prepare_tile(
    tile: Tile, imgsz: int = 1024
) -> tuple[torch.Tensor, LetterBoxGeometry]:
    """Prepare fixed-square validation input: BGR->RGB, NCHW float32 /255 on CPU.

    Uses shared centered letterbox, INTER_LINEAR, pad=114 and scaleup=True.
    Returns owned contiguous tensor and exact forward metadata for NPU adapters.
    No inference occurs. Rectangular auto-padding and TTA are deliberately off.
    """
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    geometry = LetterBoxGeometry.from_shapes(
        (imgsz, imgsz), tuple(tile.image.shape[:2])
    )
    image, _ = _apply_letterbox(tile.image, [imgsz, imgsz], cv2.INTER_LINEAR, 114)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)))
        .float()
        .unsqueeze(0)
        / 255
    )
    return tensor, geometry


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
    ultralytics==8.3.228. Uses raw eval-mode model + upstream multi_label=True
    rotated NMS, not YOLO.predict's single-label defaults. All inference is CPU,
    FP32, no augmentation. Construction loads weights; __call__ performs inference.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        imgsz: int = 1024,
        conf: float = 0.01,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> None:
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
        self.model = wrapper.model.cpu().float().eval()
        self.names = tuple(wrapper.names[i] for i in range(len(wrapper.names)))
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self._nms = non_max_suppression

    def __call__(self, tile: Tile) -> torch.Tensor:
        tensor, geometry = prepare_tile(tile, self.imgsz)
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

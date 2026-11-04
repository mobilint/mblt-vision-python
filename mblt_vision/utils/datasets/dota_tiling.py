"""Dataset-owned DOTA native windows/crops and reference input preparation.

The versioned protocol lives in datasets/dotav1.yaml. See the package README's
DOTA multiscale native tiles section for immutable source references.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ...datasets import get_dataset_config
from ..letterbox import LetterBoxGeometry
from ..preprocess.letterbox import _apply_letterbox

TILED_API_VERSION = 1
ULTRALYTICS_VERSION = "8.3.228"
ULTRALYTICS_COMMIT = "f8e6fddc89ec754b3a62eab87d5221e83058005a"


def get_eval_protocol(
    dataset: str = "dotav1", selector: str | None = None
) -> dict[str, Any]:
    """Resolve the dataset-owned protocol and reject unsupported selectors/methods.

    Returns an owned mapping; model YAMLs select its versioned id rather than
    duplicating geometry, inference defaults or merge settings.
    """
    protocol = get_dataset_config(dataset).get("eval_protocol")
    if not isinstance(protocol, dict) or not isinstance(protocol.get("id"), str):
        raise ValueError(f"Dataset {dataset!r} does not define an eval_protocol")
    if selector is not None and selector != protocol["id"]:
        raise ValueError(
            f"Unsupported eval_protocol {selector!r} for dataset {dataset!r}"
        )
    for section in ("tiling", "inference", "merge"):
        if not isinstance(protocol.get(section), dict):
            raise ValueError(f"eval_protocol requires a {section} mapping")
    merge = protocol["merge"]
    if merge.get("method") != "rotated_probiou_fastnms":
        raise ValueError("DOTA tiled merge requires rotated_probiou_fastnms")
    if (
        not isinstance(merge.get("classwise"), bool)
        or not 0 < merge.get("iou_thres", 0) <= 1
    ):
        raise ValueError(
            "DOTA merge requires boolean classwise and iou_thres in (0, 1]"
        )
    inference = protocol["inference"]
    if (
        not 0 <= inference.get("conf_thres", -1) <= 1
        or not 0 < inference.get("iou_thres", 0) <= 1
        or type(inference.get("max_det")) is not int
        or inference["max_det"] <= 0
    ):
        raise ValueError(
            "DOTA inference requires valid conf_thres, iou_thres and max_det"
        )
    return protocol


_TILING = get_eval_protocol()["tiling"]


@dataclass(frozen=True)
class TiledConfig:
    """Native-window configuration; nominal rates change crop size, not pixels.

    ``jpeg_roundtrip`` reproduces split_test's default OpenCV JPEG write/read.
    Disable it explicitly for lossless in-memory slicing (a documented difference).
    Filtering is applied jointly across all rates, as in the reference.
    """

    crop_size: int = _TILING["crop_size"]
    gap: int = _TILING["gap"]
    rates: tuple[float, ...] = tuple(_TILING["rates"])
    im_rate_thr: float = _TILING["im_rate_thr"]
    eps: float = _TILING["eps"]
    jpeg_roundtrip: bool = _TILING["jpeg_roundtrip"]

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

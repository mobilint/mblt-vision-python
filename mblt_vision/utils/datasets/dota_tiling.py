"""Dataset-owned DOTA native windows/crops and reference input preparation.

Hyperparameters live directly in datasets/dotav1.yaml. See the package README's
DOTA multiscale native tiles section for immutable source references.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ...datasets import get_dataset_config
from ..letterbox import LetterBoxGeometry, RatioPad
from ..preprocess.letterbox import _apply_letterbox

ULTRALYTICS_VERSION = "8.3.228"
ULTRALYTICS_COMMIT = "f8e6fddc89ec754b3a62eab87d5221e83058005a"


def get_dota_settings(
    dataset: str = "dotav1", overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read owned tiling/inference/merge YAML settings, with optional model overrides.

    Each section is a simple mapping. Nested overrides merge by key; established
    flat inference keys (conf_thres/iou_thres/max_det/preprocessing) win when given.
    No protocol ID, version marker or source hash is required.
    """
    source = get_dataset_config(dataset)
    settings = {}
    for section in ("tiling", "inference", "merge"):
        value = source.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"DOTA settings require a {section} mapping")
        settings[section] = deepcopy(dict(value))
        if overrides is not None and section in overrides:
            update = overrides[section]
            if not isinstance(update, Mapping):
                raise ValueError(f"DOTA {section} overrides must be a mapping")
            settings[section].update(deepcopy(dict(update)))
    if overrides is not None:
        for key in ("conf_thres", "iou_thres", "max_det", "preprocessing"):
            if key in overrides:
                settings["inference"][key] = deepcopy(overrides[key])
    for name in ("max_pixels", "max_tiles"):
        value = settings["tiling"].get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"DOTA tiling requires a positive integer {name}")
    merge = settings["merge"]
    _validate_merge_max_candidates(merge.get("max_candidates"))
    if (
        not isinstance(merge.get("classwise"), bool)
        or not 0 < merge.get("iou_thres", 0) <= 1
    ):
        raise ValueError(
            "DOTA merge requires boolean classwise and iou_thres in (0, 1]"
        )
    inference = settings["inference"]
    if (
        not 0 <= inference.get("conf_thres", -1) <= 1
        or not 0 < inference.get("iou_thres", 0) <= 1
        or type(inference.get("max_det")) is not int
        or inference["max_det"] <= 0
    ):
        raise ValueError(
            "DOTA inference requires valid conf_thres, iou_thres and max_det"
        )
    validate_preprocessing(inference.get("preprocessing"))
    return settings


def _validate_merge_max_candidates(value: Any) -> int | None:
    """Validate an opt-in merge budget; None retains official uncapped merging."""
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("DOTA merge max_candidates must be null or a positive integer")
    return value


def validate_preprocessing(preprocessing: str) -> str:
    """Require an explicit supported validation input methodology."""
    if preprocessing not in ("ultralytics-val", "direct-letterbox"):
        raise ValueError(
            "DOTA preprocessing must be 'ultralytics-val' or 'direct-letterbox'"
        )
    return preprocessing


_SETTINGS = get_dota_settings()
_TILING = _SETTINGS["tiling"]
_DEFAULT_PREPROCESSING = _SETTINGS["inference"]["preprocessing"]


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
    max_pixels: int = _TILING["max_pixels"]
    max_tiles: int = _TILING["max_tiles"]

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
        for name in ("max_pixels", "max_tiles"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def crop_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return truncated crop/gap pairs in rate order."""
        return tuple((int(self.crop_size / r), int(self.gap / r)) for r in self.rates)


_DEFAULT_CONFIG = TiledConfig()


def validate_image_budget(
    im_size: tuple[int, int], config: TiledConfig = _DEFAULT_CONFIG
) -> int:
    """Check pixels and pre-filter candidate count without allocating window arrays.

    Returns the candidate count, including windows later rejected by area filtering.
    Budgets are per original image; explicitly raise them only for trusted inputs.
    Integer arithmetic prevents huge shape values from overflowing float geometry.
    """
    if len(im_size) != 2 or any(type(v) is not int or v <= 0 for v in im_size):
        raise ValueError("im_size must contain positive integer (height, width)")
    h, w = im_size
    pixels = h * w
    if pixels > config.max_pixels:
        raise ValueError(
            f"DOTA image has {pixels} pixels, exceeding max_pixels={config.max_pixels}"
        )
    count = 0
    for size, gap in config.crop_pairs:
        step = size - gap
        xn = 1 if w <= size else (w - size + step - 1) // step + 1
        yn = 1 if h <= size else (h - size + step - 1) // step + 1
        count += xn * yn
        if count > config.max_tiles:
            raise ValueError(
                f"DOTA candidate windows exceed max_tiles={config.max_tiles}"
            )
    return count


def validate_image_source(
    source: str | Path, config: TiledConfig = _DEFAULT_CONFIG
) -> tuple[int, int]:
    """Read header only, reject over-budget/non-normal-EXIF input before decoding.

    Returns raster (height, width). Array entry points recheck actual dimensions;
    applications handling untrusted uploads still need process/time/memory limits.
    """
    with Image.open(source) as header:
        shape = (header.height, header.width)
        validate_image_budget(shape, config)
        if header.getexif().get(274, 1) != 1:
            raise ValueError(
                "DOTA tiled inference requires EXIF-normalized original images"
            )
    return shape


def get_windows(
    im_size: tuple[int, int], config: TiledConfig = _DEFAULT_CONFIG
) -> np.ndarray:
    """Return reference int64 ``[x1,y1,x2,y2]`` windows, x-major within each rate.

    Stops can exceed a small image's extent. Python slicing, not padding, clips
    them. The strict area comparison and near-maximum fallback are intentional.
    """
    validate_image_budget(im_size, config)
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


@dataclass(frozen=True)
class _TileWindow:
    """Lightweight native bounds; no source image, crop copy or JPEG work."""

    image_id: str
    name: str
    rate: float
    crop_size: int
    gap: int
    window: tuple[int, int, int, int]


def _iter_tile_windows(
    image_bgr: np.ndarray, image_id: str, config: TiledConfig = _DEFAULT_CONFIG
) -> Iterator[_TileWindow]:
    """Validate input/budgets and enumerate metadata before materializing crops."""
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
        yield _TileWindow(
            image_id,
            f"{image_id}__{size}__{x1}___{y1}",
            rate,
            size,
            gap,
            (x1, y1, x2, y2),
        )


def _materialize_tile(
    image_bgr: np.ndarray, window: _TileWindow, config: TiledConfig
) -> Tile:
    """Copy one validated window and optionally JPEG round-trip its owned pixels."""
    x1, y1, x2, y2 = window.window
    patch = image_bgr[y1:y2, x1:x2].copy(order="C")
    if config.jpeg_roundtrip:
        ok, encoded = cv2.imencode(".jpg", patch)
        if not ok:
            raise RuntimeError("OpenCV failed to encode native DOTA crop")
        patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if patch is None:
            raise RuntimeError("OpenCV failed to decode native DOTA crop")
    return Tile(
        window.image_id,
        window.name,
        window.rate,
        window.crop_size,
        window.gap,
        window.window,
        patch,
    )


def iter_tiles(
    image_bgr: np.ndarray, image_id: str, config: TiledConfig = _DEFAULT_CONFIG
) -> Iterator[Tile]:
    """Yield native BGR crops, optionally JPEG round-tripped like split_test.

    Input must be non-empty uint8 HWC BGR with three channels. It is never mutated
    or retained by the yielded owned images. EXIF rotation is the caller's job;
    dimensions must describe the same decoded raster that is being sliced.
    """
    for window in _iter_tile_windows(image_bgr, image_id, config):
        yield _materialize_tile(image_bgr, window, config)


class _ValidationGeometry(LetterBoxGeometry):
    """Preserve upstream's (height, width) ratio metadata and scalar height gain.

    BaseDataset records ratios in hw order, and scale_boxes consumes only the
    first ratio for all four coordinates. Do not substitute a nominal long-side
    gain or an anisotropic inverse: either differs on ceil-resized rectangles.
    """

    @property
    def ratio_pad(self) -> RatioPad:
        return (
            (
                self.resized_shape[0] / self.original_shape[0],
                self.resized_shape[1] / self.original_shape[1],
            ),
            self.pad,
        )


def prepare_tile_image(
    tile: Tile, imgsz: int = 1024, *, preprocessing: str = _DEFAULT_PREPROCESSING
) -> tuple[np.ndarray, LetterBoxGeometry]:
    """Return owned square uint8 BGR input and exact validation restoration metadata.

    ``ultralytics-val`` matches BaseDataset.load_image's long-side ceil resize,
    followed by centered LetterBox(scaleup=False), INTER_LINEAR and pad 114.
    ``direct-letterbox`` preserves the historical one-stage rounded resize.
    The validation geometry's ratio is upstream's scalar height gain, not a
    claim that the ceil resize is exactly isotropic. No auto-padding or TTA.
    """
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    validate_preprocessing(preprocessing)
    image = tile.image
    original_shape = tuple(image.shape[:2])
    if preprocessing == "ultralytics-val":
        height, width = original_shape
        ratio = imgsz / max(height, width)
        if ratio != 1:
            resized_shape = (
                min(math.ceil(height * ratio), imgsz),
                min(math.ceil(width * ratio), imgsz),
            )
            image = cv2.resize(
                image, resized_shape[::-1], interpolation=cv2.INTER_LINEAR
            )
        resized_shape = tuple(image.shape[:2])
        # The long side already equals imgsz, so the shared letterbox has gain
        # exactly one: this is the scaleup=False validation transform.
        padding = LetterBoxGeometry.from_shapes((imgsz, imgsz), resized_shape)
        geometry = _ValidationGeometry(
            (imgsz, imgsz),
            original_shape,
            resized_shape[0] / height,
            resized_shape,
            padding.pad,
        )
    else:
        geometry = LetterBoxGeometry.from_shapes((imgsz, imgsz), original_shape)
    image, _ = _apply_letterbox(image, [imgsz, imgsz], cv2.INTER_LINEAR, 114)
    return image, geometry


def prepare_tile(
    tile: Tile, imgsz: int = 1024, *, preprocessing: str = _DEFAULT_PREPROCESSING
) -> tuple[torch.Tensor, LetterBoxGeometry]:
    """Prepare CPU contiguous RGB NCHW float32 /255 plus restoration geometry.

    Defaults to the dataset's ``ultralytics-val`` two-stage input methodology.
    Pass ``preprocessing='direct-letterbox'`` for the frozen historical method.
    """
    image, geometry = prepare_tile_image(tile, imgsz, preprocessing=preprocessing)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)))
        .float()
        .unsqueeze(0)
        / 255
    )
    return tensor, geometry

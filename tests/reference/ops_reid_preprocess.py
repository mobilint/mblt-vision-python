# Verbatim source: models/person_reid/_shared/preprocess.py
# mblt-model-ops ec537764e36ca10da963fcb7afab157ea60ca022
"""Default person_reid preprocess: the CLIP-ReID / TransReID test transform.

Upstream evaluates with ``Resize((H, W)) -> ToTensor -> Normalize(mean, std)``; there is no
crop, and the two backbone families differ only in the normalization statistics, so the
statistics are explicit ``mean``/``std`` lists in ``pre_cfg`` rather than a named style.

Operations are applied in a fixed canonical order, not in ``config`` iteration order:
``predict.input.json`` is written with sorted keys, so relying on dict order would run
``Normalize`` before ``Resize``.

A camera-aware Model (SIE) declares ``CameraOneHot``. The camera index is the ``_c<N>``
token of the crop's file name, which is how every ReID dataset and the upstream loader
encode it; it is a Model input, not ground truth. The network then takes two inputs, and
the returned tensor is a tuple ``(image, camera_one_hot)`` in that order.

The return value is ``(source, network_input, camera_index)`` so that the shared
``inference.tensor_of`` finds the network input in the second slot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PIPELINE_ORDER = ("Reader", "Resize", "SetOrder", "Normalize", "CameraOneHot")
REQUIRED = ("Resize", "Normalize")
_RESAMPLING = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}
DEFAULT_CAMERA_PATTERN = r"_c(\d+)"

NetworkInput = np.ndarray | tuple[np.ndarray, np.ndarray]


def camera_index(source: Path, config: dict[str, Any]) -> int:
    """The 0-based camera index of a crop, from its ``_c<N>`` file-name token."""
    pattern = str(config.get("pattern", DEFAULT_CAMERA_PATTERN))
    match = re.search(pattern, Path(source).name)
    if match is None:
        raise ValueError(f"{source}: file name carries no camera token {pattern!r}")
    cameras = int(config["num_cameras"])
    camera = int(match.group(1)) - 1
    if not 0 <= camera < cameras:
        raise ValueError(f"{source}: camera {camera + 1} is outside 1..{cameras}")
    return camera


def preprocess(
    source: Path, config: dict[str, Any]
) -> tuple[Path, NetworkInput, int | None]:
    """Return ``(source, network input, camera index or None)``."""
    unknown = set(config) - set(PIPELINE_ORDER)
    if unknown:
        raise ValueError(f"unsupported preprocess operations: {sorted(unknown)}")
    missing = [name for name in REQUIRED if name not in config]
    if missing:
        raise ValueError(f"pre_cfg is missing required operations: {missing}")
    source = Path(source)
    value: Any = Image.open(source).convert("RGB")
    camera: int | None = None
    for operation in PIPELINE_ORDER:
        if operation not in config:
            continue
        options = config[operation]
        if operation == "Reader":
            continue
        if operation == "Resize":
            height, width = (int(v) for v in options["size"])
            mode = _RESAMPLING[str(options.get("interpolation", "bilinear"))]
            value = value.resize((width, height), mode)
        elif operation == "SetOrder":
            if options.get("shape", "HWC") != "HWC":
                raise ValueError("person_reid preprocess produces HWC tensors only")
        elif operation == "Normalize":
            mean = np.asarray(options["mean"], dtype=np.float32)
            std = np.asarray(options["std"], dtype=np.float32)
            value = (np.asarray(value, dtype=np.float32) / 255.0 - mean) / std
        elif operation == "CameraOneHot":
            camera = camera_index(source, options)
    tensor = np.ascontiguousarray(value, dtype=np.float32)
    if camera is None:
        return source, tensor, None
    one_hot = np.zeros(int(config["CameraOneHot"]["num_cameras"]), dtype=np.float32)
    one_hot[camera] = 1.0
    return source, (tensor, one_hot), camera

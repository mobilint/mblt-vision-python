# Verbatim models/face_recognition/_shared/preprocess.py at ec537764e36ca10da963fcb7afab157ea60ca022
"""Default face_recognition preprocess: an aligned face crop, normalized.

The benchmarks of this task hand the Model already aligned 112x112 crops (five-point
similarity alignment is part of the prepared evaluation layout), so the Model-side
transform is a guard resize to the network input and an explicit ``mean``/``std``
normalization, in HWC order. Operations run in canonical order regardless of the
sorted-key order of ``pre_cfg`` on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PIPELINE_ORDER = ("Reader", "Resize", "SetOrder", "Normalize")
REQUIRED = ("Resize", "Normalize")
_RESAMPLING = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def preprocess(source: Path, config: dict[str, Any]) -> tuple[Path, np.ndarray, None]:
    """Return ``(source, HWC float32 tensor, None)``."""
    unknown = set(config) - set(PIPELINE_ORDER)
    if unknown:
        raise ValueError(f"unsupported preprocess operations: {sorted(unknown)}")
    missing = [name for name in REQUIRED if name not in config]
    if missing:
        raise ValueError(f"pre_cfg is missing required operations: {missing}")
    source = Path(source)
    value: Any = Image.open(source).convert("RGB")
    for operation in PIPELINE_ORDER:
        if operation not in config:
            continue
        options = config[operation]
        if operation == "Reader":
            continue
        if operation == "Resize":
            height, width = (int(v) for v in options["size"])
            if value.size != (width, height):
                mode = _RESAMPLING[str(options.get("interpolation", "bilinear"))]
                value = value.resize((width, height), mode)
        elif operation == "SetOrder":
            if options.get("shape", "HWC") != "HWC":
                raise ValueError(
                    "face_recognition preprocess produces HWC tensors only"
                )
        elif operation == "Normalize":
            mean = np.asarray(options["mean"], dtype=np.float32)
            std = np.asarray(options["std"], dtype=np.float32)
            value = (np.asarray(value, dtype=np.float32) / 255.0 - mean) / std
    return source, np.ascontiguousarray(value, dtype=np.float32), None

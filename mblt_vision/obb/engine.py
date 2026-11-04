"""High-level default DOTA tiled image prediction over an MBLT raw backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch
from PIL import Image

from .tiled import TiledConfig

if TYPE_CHECKING:
    from ..utils.results import Results
    from ..wrapper import MBLT_Engine


def predict_dota_image(
    model: MBLT_Engine,
    source: str | Path,
    *,
    config: TiledConfig | None = None,
    batch_size: int = 1,
) -> Results:
    """Return ordinary Results in original pixels after three-scale tiled merge.

    This high-level path does not change ``MBLT_Engine.__call__``: that remains
    raw tensor inference. The source is decoded as BGR and each JPEG-native crop
    enters the same engine preprocessing/postprocessing used for validation.
    """
    from ..utils.evaluation.eval_dota_tiled import predict_engine_tiles
    from ..utils.results import Results

    if (
        model.post_cfg.get("task") != "obb"
        or str(model.post_cfg.get("dataset", "")).lower() != "dotav1"
    ):
        raise ValueError("Tiled DOTA prediction requires an OBB DOTAv1 model")
    with Image.open(source) as header:
        if header.getexif().get(274, 1) != 1:
            raise ValueError(
                "DOTA tiled inference requires EXIF-normalized original images"
            )
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)
    rows = predict_engine_tiles(
        model, iter([("image", image)]), batch_size, config or TiledConfig()
    )["image"]
    # Results' normal OBB representation is cx,cy,w,h,score,class,angle. The
    # prediction is already original-pixel, so plotting must use identity scale.
    output = torch.cat((rows[:, :4], rows[:, 5:7], rows[:, 4:5]), dim=1)
    pre_cfg = deepcopy(model.pre_cfg)
    pre_cfg["LetterBox"] = {"img_size": list(image.shape[:2])}
    return Results(pre_cfg, model.post_cfg, [output])


__all__ = ["predict_dota_image"]

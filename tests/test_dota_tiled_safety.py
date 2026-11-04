"""DOTA per-image resource checks run before raster/window allocation or inference."""

from __future__ import annotations

import struct
import zlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mblt_vision.utils.datasets import dota_tiling
from mblt_vision.utils.datasets.dota_tiling import (
    TiledConfig,
    get_windows,
    iter_tiles,
    validate_image_budget,
    validate_image_source,
)
from mblt_vision.utils.postprocess.tiled_obb import DOTATiledOBBPostMixin


@pytest.mark.parametrize("name", ["max_pixels", "max_tiles"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_budget_config_requires_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        TiledConfig(**{name: value})


def test_window_budget_checked_before_array_materialization(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("window array allocation before budget rejection")

    monkeypatch.setattr(dota_tiling.np, "array", forbidden)
    with pytest.raises(ValueError, match="max_pixels"):
        get_windows((8192, 8192))
    with pytest.raises(ValueError, match="max_tiles"):
        get_windows((8192, 8192), TiledConfig(max_pixels=100_000_000))
    with pytest.raises(ValueError, match="max_pixels"):
        get_windows((10**200, 10**200))


def test_budget_count_boundary_and_explicit_override():
    # Exact pre-filter count is 803, even though later area filtering may remove rows.
    trusted = TiledConfig(max_pixels=100_000_000, max_tiles=803)
    assert validate_image_budget((8192, 8192), trusted) == 803
    assert len(get_windows((8192, 8192), trusted)) == 803
    with pytest.raises(ValueError, match="max_tiles"):
        validate_image_budget((8192, 8192), replace(trusted, max_tiles=802))
    assert validate_image_budget((8000, 8000), TiledConfig(max_tiles=1000)) > 0


def test_array_guard_precedes_crop_and_jpeg(monkeypatch):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        dota_tiling.cv2, "imencode", lambda *_args: pytest.fail("JPEG encoded")
    )
    with pytest.raises(ValueError, match="max_pixels"):
        next(iter_tiles(image, "P", TiledConfig(max_pixels=11)))
    with pytest.raises(ValueError, match="max_tiles"):
        next(iter_tiles(image, "P", TiledConfig(max_tiles=2)))


def oversized_png_header(tmp_path):
    """Change a tiny PNG's IHDR only; never allocate its advertised large raster."""
    path = tmp_path / "large.png"
    Image.new("RGB", (1, 1)).save(path)
    data = bytearray(path.read_bytes())
    data[16:24] = struct.pack(">II", 8192, 8192)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]) & 0xFFFFFFFF)
    path.write_bytes(data)
    return path


def test_header_guard_rejects_before_raster_load(tmp_path, monkeypatch):
    source = oversized_png_header(tmp_path)
    monkeypatch.setattr(
        Image.Image, "load", lambda *_args: pytest.fail("raster loaded")
    )
    with pytest.raises(ValueError, match="max_pixels"):
        validate_image_source(source)


@pytest.mark.parametrize("entry", ["predict", "eval"])
def test_high_level_rejects_header_before_decode_or_inference(
    tmp_path, monkeypatch, entry
):
    import mblt_vision.utils.evaluation.eval_dota_tiled as tiled

    source = oversized_png_header(tmp_path)
    monkeypatch.setattr(
        tiled.cv2, "imread", lambda *_args: pytest.fail("raster decoded")
    )
    model = SimpleNamespace(
        post_cfg={"task": "obb", "dataset": "dotav1"},
        postprocessor=object(),
        set_postprocess_thresholds=lambda **_kwargs: None,
    )
    with pytest.raises(ValueError, match="max_pixels"):
        if entry == "predict":
            tiled.predict_dota_image(model, source)
        else:
            monkeypatch.setattr(
                tiled,
                "CustomDOTAv1",
                lambda _path: SimpleNamespace(ids=["P"], image_paths=[str(source)]),
            )
            tiled.eval_dota_tiled(model, str(tmp_path), 1)


def test_prediction_uses_selected_processor_budget(tmp_path, monkeypatch):
    import mblt_vision.utils.evaluation.eval_dota_tiled as tiled

    source = tmp_path / "small.png"
    Image.new("RGB", (2, 2)).save(source)
    processor = DOTATiledOBBPostMixin.__new__(DOTATiledOBBPostMixin)
    processor.tile_config = TiledConfig(max_pixels=3)
    model = SimpleNamespace(
        post_cfg={"task": "obb", "dataset": "dotav1"}, postprocessor=processor
    )
    monkeypatch.setattr(
        tiled.cv2, "imread", lambda *_args: pytest.fail("raster decoded")
    )
    with pytest.raises(ValueError, match="max_pixels=3"):
        tiled.predict_dota_image(model, source)

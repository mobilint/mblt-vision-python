"""DOTA per-image resource checks run before raster/window allocation or inference."""

from __future__ import annotations

import struct
import zlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "30000"])
def test_merge_budget_strict_in_dataset_and_explicit_override(monkeypatch, value):
    from mblt_vision.utils.postprocess.tiled_obb import merge_tile_predictions

    settings = dota_tiling.get_dota_settings()
    settings["merge"]["max_candidates"] = value
    with pytest.raises(ValueError, match="max_candidates"):
        merge_tile_predictions([], settings)
    monkeypatch.setattr(dota_tiling, "get_dataset_config", lambda _dataset: settings)
    with pytest.raises(ValueError, match="max_candidates"):
        dota_tiling.get_dota_settings()


def test_opt_in_merge_budget_rejects_maximum_tile_stream_before_nms(monkeypatch):
    from mblt_vision.utils.datasets.dota_tiling import Tile
    from mblt_vision.utils.postprocess import tiled_obb

    settings = dota_tiling.get_dota_settings()
    assert settings["merge"]["max_candidates"] is None
    settings["merge"]["max_candidates"] = 30000
    seen = []
    # Sparse non-overlapping outputs would avoid suppression and force worst-case
    # quadratic work at 768 tiles * 300 rows. Allocate only one reusable tile.
    rows = torch.zeros((300, 7))
    rows[:, 0] = torch.arange(300) * 100
    rows[:, 1:4] = torch.tensor([100, 1, 1])
    rows[:, 5] = 0.9
    tile = Tile(
        "P",
        "P__1024__0___0",
        1,
        1024,
        500,
        (0, 0, 1024, 1024),
        np.empty((0, 0, 3), np.uint8),
    )

    def predictions():
        for index in range(TiledConfig().max_tiles):
            seen.append(index)
            yield tile, rows

    monkeypatch.setattr(
        tiled_obb, "rotated_nms", lambda *_args: pytest.fail("over-budget NMS ran")
    )
    with pytest.raises(ValueError, match="30300 candidates.*max_candidates=30000"):
        tiled_obb.merge_tile_predictions(predictions(), settings)
    assert len(seen) == 101  # Abort the stream at the first over-budget tile.


def test_official_default_uncapped_merge_passes_all_rows_to_nms(monkeypatch):
    from mblt_vision.utils.datasets.dota_tiling import Tile
    from mblt_vision.utils.postprocess import tiled_obb

    settings = dota_tiling.get_dota_settings()
    assert settings["merge"]["max_candidates"] is None
    rows = torch.zeros((30300, 7))
    rows[:, :4] = torch.tensor([2, 3, 1, 1])
    rows[:, 5] = 0.9
    tile = Tile(
        "P", "P__8__0___0", 1, 8, 0, (0, 0, 8, 8), np.empty((0, 0, 3), np.uint8)
    )
    calls = []

    def nms(boxes, scores, iou):
        # Do not execute quadratic comparisons for this default-policy proof.
        calls.append(len(boxes))
        return torch.arange(len(boxes))

    monkeypatch.setattr(tiled_obb, "rotated_nms", nms)
    torch.testing.assert_close(tiled_obb.merge_tile_predictions([(tile, rows)]), rows)
    assert calls == [30300]


def test_merge_budget_boundary_and_trusted_override_keep_every_row(monkeypatch):
    from mblt_vision.utils.postprocess import tiled_obb

    image = np.zeros((8, 8, 3), np.uint8)
    config = TiledConfig(crop_size=8, gap=0, rates=(1.0,), jpeg_roundtrip=False)
    tile = next(iter_tiles(image, "P", config))
    rows = torch.tensor([[2, 3, 1, 1, 0, 0.9, 0], [6, 7, 1, 1, 0, 0.8, 1]])
    settings = dota_tiling.get_dota_settings()
    calls = []

    def nms(boxes, scores, iou):
        calls.append(len(boxes))
        return torch.arange(len(boxes))

    monkeypatch.setattr(tiled_obb, "rotated_nms", nms)
    settings["merge"]["max_candidates"] = 1
    with pytest.raises(ValueError, match="max_candidates=1"):
        tiled_obb.predict_tiled(
            image, "P", lambda _tile: rows, config, settings=settings
        )
    assert calls == []
    settings["merge"]["max_candidates"] = 2
    torch.testing.assert_close(
        tiled_obb.merge_tile_predictions([(tile, rows)], settings), rows
    )
    torch.testing.assert_close(
        tiled_obb.predict_tiled(
            image, "P", lambda _tile: rows, config, settings=settings
        ),
        rows,
    )
    # An explicit higher trusted budget admits the complete input; no top-k.
    settings["merge"]["max_candidates"] = 3
    extra = torch.cat((rows, rows[:1]))
    torch.testing.assert_close(
        tiled_obb.merge_tile_predictions([(tile, extra)], settings), extra
    )
    assert calls == [2, 2, 3]

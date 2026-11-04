"""Published ReID metadata, offline processing and source parity (no weights)."""

from pathlib import Path

import mblt_vision
import numpy as np
import pytest
import torch
from mblt_vision.utils.postprocess import PersonReIDPost, build_postprocess
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.wrapper import resolve_model_config
from PIL import Image

MODELS = [
    f"CLIP_ReID_{backbone}{baseline}_{dataset}"
    for backbone in ("RN50", "ViT_B_16")
    for baseline in ("", "_Baseline")
    for dataset in ("Market1501", "DukeMTMC", "OccDuke")
]
DATASETS = {
    "Market1501": "market1501",
    "DukeMTMC": "dukemtmc-reid",
    "OccDuke": "occluded-duke",
}


@pytest.mark.parametrize("model", MODELS)
def test_published_reid_config_and_discovery(model):
    cfg = resolve_model_config(model)
    dim = 1280 if "ViT" in model else 3072
    assert cfg["file_cfg"] == {
        "repo_id": "mobilint/" + model,
        "filename": model.lower() + ".mxq",
        "revision": "main",
        "onnx_local_artifact_only": True,
        "onnx_filename": model.lower() + ".onnx",
    }
    assert cfg["post_cfg"] == {
        "task": "person_reid",
        "dataset": DATASETS[model.split("_")[-1]],
        "normalize": True,
        "embedding_dim": dim,
    }
    assert cfg["source"]["approved_hf_revision"] == "not recorded"
    assert cfg["source"]["revision"] == "ec537764e36ca10da963fcb7afab157ea60ca022"
    assert model in mblt_vision.list_models("person_reid")["person_reid"]
    assert getattr(mblt_vision, model) is getattr(mblt_vision.person_reid, model)
    post = build_postprocess(cfg["pre_cfg"], cfg["post_cfg"])
    assert isinstance(post, PersonReIDPost)
    raw = np.ones((1, dim), dtype=np.float32)
    out = post([raw])
    assert tuple(out.shape) == (1, dim)
    torch.testing.assert_close(torch.linalg.vector_norm(out, dim=1), torch.ones(1))


@pytest.mark.parametrize("model", MODELS)
def test_published_reid_preprocess_source_numeric_parity(tmp_path, model):
    cfg = resolve_model_config(model)
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (53, 31, 3), dtype=np.uint8)
    path = tmp_path / "0001_c1_synthetic.png"
    Image.fromarray(pixels).save(path)
    actual = build_preprocess(cfg["pre_cfg"])(path)
    if "ViT" in model:
        mean, std = [0.5] * 3, [0.5] * 3
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # ops models/person_reid/_shared/preprocess.py at ec537764, canonical order.
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((128, 256), Image.Resampling.BILINEAR)
    expected = (
        np.asarray(resized, dtype=np.float32) / 255.0
        - np.asarray(mean, dtype=np.float32)
    ) / np.asarray(std, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)  # atol=0 for every concrete config


def test_face_publish_pending_models_remain_inventory_only():
    root = Path(__file__).parents[1] / "mblt_vision/models"
    assert not list(root.glob("ArcFace_IResNet_*.yaml"))
    assert not list(root.glob("CosFace_IResNet_*.yaml"))


def test_top_level_reid_exports_are_model_only():
    assert set(mblt_vision.person_reid.MODEL_EXPORTS) == set(MODELS)
    for name in (
        "EmbeddingPost",
        "ReIDSample",
        "Market1501Dataset",
        "RetrievalEvaluator",
        "extract_embeddings",
    ):
        assert name not in mblt_vision.__all__
        assert name not in mblt_vision._LEGACY_MODEL_EXPORTS
        assert hasattr(mblt_vision.person_reid, name)


def test_reid_compile_is_explicitly_unsupported():
    from mblt_vision.compile.vision import _normalize_task

    with pytest.raises(ValueError, match="Unsupported Vision task"):
        _normalize_task("person_reid")

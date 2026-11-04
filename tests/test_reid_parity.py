"""Offline numeric differential tests against pinned, verbatim ops functions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mblt_vision._config import load_model_variant
from mblt_vision.utils.evaluation.retrieval import (
    RetrievalEvaluator,
    euclidean_distance,
)
from mblt_vision.utils.preprocess import build_preprocess
from PIL import Image


def reference(name):
    path = Path(__file__).parent / "reference" / f"ops_reid_{name}.py"
    spec = importlib.util.spec_from_file_location(f"ops_reid_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", range(10))
def test_ops_evaluator_exact_numeric_parity(seed):
    ops = reference("metrics")
    rng = np.random.default_rng(seed)
    # Quantized integer vectors produce numerous exact distance ties.
    query = rng.integers(-2, 3, size=(17, 8)).astype(np.float32)
    gallery = rng.integers(-2, 3, size=(61, 8)).astype(np.float32)
    q_ids = rng.integers(1, 6, 17)
    g_ids = rng.integers(0, 6, 61)  # zero-valued distractors remain ordinary rows
    q_cam = rng.integers(0, 3, 17)
    g_cam = rng.integers(0, 3, 61)
    expected_distances = ops.euclidean_distance(query, gallery)
    actual_distances = euclidean_distance(query, gallery)
    np.testing.assert_array_equal(actual_distances, expected_distances)
    for max_rank in (1, 5, 50, 100):
        expected_cmc, expected_map = ops.evaluate(
            expected_distances, q_ids, g_ids, q_cam, g_cam, max_rank
        )
        actual_cmc, actual_map = RetrievalEvaluator.evaluate_distances(
            actual_distances, q_ids, g_ids, q_cam, g_cam, max_rank
        )
        np.testing.assert_array_equal(actual_cmc, expected_cmc)
        assert actual_map == expected_map


@pytest.mark.parametrize("shape", [(19, 11), (128, 256), (383, 191)])
@pytest.mark.parametrize("style", ["torch", "tf"])
def test_ops_preprocessing_parity(tmp_path, shape, style):
    ops = reference("preprocess")
    cfg = load_model_variant(
        Path(__file__).parents[1] / "mblt_vision/models/bases/person_reid.yaml",
        "DEFAULT",
    )
    config = cfg["pre_cfg"]
    if style == "tf":
        # CLIP_ReID_ViT_B_16_Baseline_DukeMTMC/distribution.yaml at ec537764.
        config["Normalize"] = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    rng = np.random.default_rng(0)
    path = tmp_path / "0001_c2_synthetic.png"
    Image.fromarray(rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)).save(path)
    expected = ops.preprocess(path, config)[1]
    actual = build_preprocess(config)(path)
    # Both use PIL bilinear + explicit float32 stats; no tolerance needed.
    np.testing.assert_array_equal(actual, expected)
    assert np.max(np.abs(actual - expected)) == 0.0

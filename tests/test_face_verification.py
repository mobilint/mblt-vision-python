"""CPU prepared face-pair, engine and source-exact numeric contracts."""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from mblt_vision._config import load_model_variant
from mblt_vision.face_recognition import (
    AgeDB30Dataset,
    AgeDB30Evaluator,
    CFPEvaluator,
    CFPFPDataset,
    EmbeddingPost,
    FacePair,
    FaceRecognitionPost,
    LFWDataset,
    LFWEvaluator,
    PairImage,
    cosine_scores,
    fold_accuracies,
)
from mblt_vision.utils.datasets import face_pairs, readiness
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.wrapper import MBLT_Engine
from PIL import Image


def reference(name):
    path = Path(__file__).parent / "reference" / f"ops_face_{name}.py"
    spec = importlib.util.spec_from_file_location(f"ops_face_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_pairs(root, folder, rows):
    (root / folder).mkdir(exist_ok=True)
    for name in {r[3] for r in rows} | {r[4] for r in rows}:
        if "/" not in name:
            Image.new("RGB", (112, 112)).save(root / folder / name)
    with (root / "pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "fold_id", "is_same", "image_a", "image_b"])
        writer.writerows(rows)


@pytest.mark.parametrize(
    "loader,folder",
    [(LFWDataset, "aligned"), (AgeDB30Dataset, "images"), (CFPFPDataset, "images")],
)
def test_prepared_pairs_preserve_order_folds_and_alignment(tmp_path, loader, folder):
    rows = [(9, 3, 0, "b.png", "a.png"), (2, 1, 1, "a.png", "a.png")]
    write_pairs(tmp_path, folder, rows)
    data = loader(tmp_path, validate_counts=False)
    assert [p.fold for p in data.pairs] == [3, 1]
    assert [p.is_same for p in data.pairs] == [False, True]
    assert [s.name for s in data.samples] == ["a.png", "b.png"]
    assert isinstance(data[0], PairImage) and len(data) == 2
    with pytest.raises(ValueError, match="expected"):
        loader(tmp_path)
    Image.new("RGB", (113, 112)).save(tmp_path / folder / "a.png")
    with pytest.raises(ValueError, match="112x112"):
        loader(tmp_path, validate_counts=False)


@pytest.mark.parametrize(
    "bad",
    [
        (1, 0, 1, "a.png", "b.png"),
        (1, 11, 1, "a.png", "b.png"),
        (1, 1, 2, "a.png", "b.png"),
        (1, 1, 1, "../a.png", "b.png"),
    ],
)
def test_invalid_pair_metadata(tmp_path, bad):
    write_pairs(tmp_path, "aligned", [bad])
    with pytest.raises(ValueError):
        LFWDataset(tmp_path, validate_counts=False)


def test_duplicate_missing_and_empty_pairs(tmp_path):
    with pytest.raises(FileNotFoundError):
        LFWDataset(tmp_path)
    write_pairs(tmp_path, "aligned", [(1, 1, 1, "a.png", "b.png")] * 2)
    with pytest.raises(ValueError, match="unique"):
        LFWDataset(tmp_path, validate_counts=False)
    write_pairs(tmp_path, "aligned", [(1, 1, 1, "a.png", "b.png")])
    (tmp_path / "aligned/b.png").unlink()
    with pytest.raises(FileNotFoundError):
        LFWDataset(tmp_path, validate_counts=False)


def test_engine_embedding_path():
    assert issubclass(FaceRecognitionPost, EmbeddingPost)
    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.pre_cfg = {}
    engine.post_cfg = {
        "task": "face_recognition",
        "dataset": "lfw",
        "normalize": True,
        "embedding_dim": 2,
    }
    engine.postprocessor = build_postprocess(engine.pre_cfg, engine.post_cfg)
    assert isinstance(engine.postprocessor, FaceRecognitionPost)
    result = engine.postprocess([np.array([[3.0, 4.0]], dtype=np.float32)])
    torch.testing.assert_close(result.embeddings, torch.tensor([[0.6, 0.8]]))
    assert result.output is result.embeddings and result.acc is None
    with pytest.raises(ValueError):
        FaceRecognitionPost({}, {"dataset": "market1501"})
    with pytest.raises(ValueError):
        FaceRecognitionPost({}, {})


@pytest.mark.parametrize(
    "loader,name,folder",
    [
        (LFWDataset, "lfw", "aligned"),
        (AgeDB30Dataset, "agedb-30", "images"),
        (CFPFPDataset, "cfp-fp", "images"),
    ],
)
def test_readiness(tmp_path, monkeypatch, loader, name, folder):
    original = face_pairs.get_dataset_config_for_task

    def tiny(task, dataset):
        cfg = original(task, dataset)
        cfg["pairs"] = 10
        return cfg

    monkeypatch.setattr(face_pairs, "get_dataset_config_for_task", tiny)
    assert not readiness.dataset_ready(tmp_path, "face_recognition", name)
    rows = [(i, i, 1, "a.png", "b.png") for i in range(1, 11)]
    write_pairs(tmp_path, folder, rows)
    assert readiness.dataset_ready(tmp_path, "face_recognition", name)
    assert not readiness.dataset_ready(tmp_path, "face_recognition", "market1501")
    (tmp_path / folder / "a.png").write_bytes(b"broken")
    assert not readiness.dataset_ready(tmp_path, "face_recognition", name)


@pytest.mark.parametrize("seed", range(5))
def test_ops_metric_exact_parity(seed):
    ops = reference("verification")
    rng = np.random.default_rng(seed)
    a = rng.integers(-2, 3, (80, 8)).astype(np.float32)
    b = rng.integers(-2, 3, (80, 8)).astype(np.float32)
    labels = rng.integers(0, 2, 80).astype(bool)
    folds = np.tile(np.arange(1, 11), 8)
    expected_scores = ops.cosine_scores(a, b)
    scores = cosine_scores(a, b)
    np.testing.assert_array_equal(scores, expected_scores)
    expected_acc, expected_thresholds = ops.fold_accuracies(
        expected_scores, labels, folds
    )
    acc, thresholds = fold_accuracies(scores, labels, folds)
    np.testing.assert_array_equal(acc, expected_acc)
    np.testing.assert_array_equal(thresholds, expected_thresholds)
    expected = ops.verification_result(scores, labels, folds)
    assert float(np.mean(acc)) == expected["accuracy"]
    assert float(np.std(acc)) == expected["accuracy_std"]
    assert float(np.mean(thresholds)) == expected["threshold"]


def test_threshold_ties_and_inclusive_boundary():
    acc, thresholds = fold_accuracies(
        np.zeros(4), np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2])
    )
    np.testing.assert_array_equal(thresholds, [1.0, 1.0])
    np.testing.assert_array_equal(acc, [0.5, 0.5])
    # Exactly equal to threshold counts as same.
    scores = cosine_scores(np.ones((2, 2)), np.ones((2, 2)))
    np.testing.assert_allclose(scores, 1.0)
    with pytest.raises(ValueError):
        fold_accuracies(np.ones(2), np.ones(2), np.ones(2, dtype=int))


@pytest.mark.parametrize("evaluator", [LFWEvaluator, AgeDB30Evaluator, CFPEvaluator])
def test_evaluator_mapping_and_no_implicit_fusion(evaluator):
    images = [PairImage(Path(n), n) for n in ["a", "b", "c"]]
    pairs = [
        FacePair("a", "b", True, 1),
        FacePair("a", "c", False, 1),
        FacePair("a", "b", True, 2),
        FacePair("a", "c", False, 2),
    ]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    result = evaluator().evaluate(embeddings, images, pairs)
    assert result.primary_score == 1.0 and result.secondary_score == 0.0
    assert result.as_dict()["primary"][evaluator.metric] == 1.0
    assert result.sample_count == 3
    with pytest.raises(ValueError):
        evaluator().evaluate(embeddings[:2], images, pairs)
    with pytest.raises(ValueError):
        evaluator().evaluate(embeddings, images, [FacePair("a", "missing", True, 1)])


@pytest.mark.parametrize("shape", [(112, 112), (35, 51), (241, 179)])
def test_ops_preprocessing_byte_parity(tmp_path, shape):
    ops = reference("preprocess")
    cfg = load_model_variant(
        Path(__file__).parents[1] / "mblt_vision/models/bases/face_recognition.yaml",
        "DEFAULT",
    )
    path = tmp_path / "synthetic.png"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    ).save(path)
    expected = ops.preprocess(path, cfg["pre_cfg"])[1]
    actual = build_preprocess(cfg["pre_cfg"])(path)
    np.testing.assert_array_equal(
        actual, expected
    )  # atol=0, including no-op 112x112 resize
    assert np.max(np.abs(actual - expected)) == 0.0


@pytest.mark.parametrize(
    "name,loader,folder",
    [
        ("lfw", LFWDataset, "aligned"),
        ("agedb-30", AgeDB30Dataset, "images"),
        ("cfp-fp", CFPFPDataset, "images"),
    ],
)
def test_rejects_correct_total_with_unbalanced_official_folds(
    tmp_path, monkeypatch, name, loader, folder
):
    original = face_pairs.get_dataset_config_for_task

    def tiny(task, dataset):
        cfg = original(task, dataset)
        cfg["pairs"] = 20
        return cfg

    monkeypatch.setattr(face_pairs, "get_dataset_config_for_task", tiny)
    rows = [(i, i // 2 + 1, i % 2, "a.png", "b.png") for i in range(20)]
    write_pairs(tmp_path, folder, rows)
    assert readiness.dataset_ready(tmp_path, "face_recognition", name)
    rows[0] = (
        0,
        2,
        0,
        "a.png",
        "b.png",
    )  # 20 total, all ten folds, but 1/3 instead of 2/2.
    write_pairs(tmp_path, folder, rows)
    with pytest.raises(ValueError, match="official fold"):
        loader(tmp_path)
    assert not readiness.dataset_ready(tmp_path, "face_recognition", name)
    assert len(loader(tmp_path, validate_counts=False).pairs) == 20

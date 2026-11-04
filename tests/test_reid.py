"""CPU-only embedding, prepared loader and retrieval protocol contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
from mblt_vision._config import load_model_variant
from mblt_vision.datasets import get_dataset_config
from mblt_vision.utils.datasets.reid import (
    DukeMTMCReIDDataset,
    Market1501Dataset,
    OccludedDukeDataset,
    ReIDSample,
)
from mblt_vision.utils.evaluation.retrieval import (
    DukeMTMCReIDEvaluator,
    Market1501Evaluator,
    OccludedDukeEvaluator,
    RetrievalEvaluator,
    extract_embeddings,
)
from mblt_vision.utils.postprocess import build_postprocess
from mblt_vision.utils.postprocess.embedding_post import EmbeddingPost, PersonReIDPost
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.utils.preprocess.normalize import Normalize


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 1, 1), (2, 1, 1, 3)])
def test_embedding_shapes(shape):
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32).reshape(shape)
    post = build_postprocess(
        {},
        {
            "task": "person_reid",
            "dataset": "market1501",
            "normalize": True,
            "embedding_dim": 3,
        },
    )
    assert isinstance(post, (EmbeddingPost, PersonReIDPost))
    np.testing.assert_array_equal(
        post([raw]), np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(raw.reshape(2, 3)[0], [3.0, 4.0, 0.0])


@pytest.mark.parametrize(
    "raw",
    [
        np.empty((0, 3)),
        np.ones(3),
        np.ones((2, 3, 2, 2)),
        np.ones((2, 3, 1)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_embedding_rejects_invalid(raw):
    with pytest.raises(ValueError):
        EmbeddingPost({}, {})(raw)


def test_embedding_validation_and_optional_normalization():
    post = EmbeddingPost({}, {})
    raw = torch.tensor([[3.0, 4.0]])
    assert torch.equal(post(raw), raw)
    with pytest.raises(TypeError):
        post(np.ones((1, 2), dtype=np.int32))
    with pytest.raises(ValueError):
        post([raw, raw])
    with pytest.raises(ValueError):
        EmbeddingPost({}, {"embedding_dim": 3})(raw)
    with pytest.raises(ValueError):
        PersonReIDPost({}, {"dataset": "coco"})
    with pytest.raises(ValueError):
        PersonReIDPost({}, {})
    with pytest.raises(TypeError):
        EmbeddingPost({}, {"normalize": "yes"})


@pytest.mark.parametrize(
    "dataset,loader",
    [
        ("market1501", Market1501Dataset),
        ("dukemtmc-reid", DukeMTMCReIDDataset),
        ("occluded-duke", OccludedDukeDataset),
    ],
)
def test_loader_metadata_and_source(tmp_path, dataset, loader):
    for folder in ("query", "bounding_box_test"):
        (tmp_path / folder).mkdir()
    for filename in ("0002_c2_b.jpg", "0001_c1s1_a.jpg"):
        (tmp_path / "query" / filename).touch()
    (tmp_path / "bounding_box_test" / "0001_c2_a.jpg").touch()
    if dataset == "market1501":
        (tmp_path / "query" / "-1_c1_junk.jpg").touch()
        (tmp_path / "bounding_box_test" / "0000_c1_background.jpg").touch()
    data = loader(tmp_path, validate_counts=False)
    assert [(s.pid, s.camid, s.split) for s in data][:2] == [
        (1, 0, "query"),
        (2, 1, "query"),
    ]
    assert data[-1].split == "gallery"
    assert all(s.pid != -1 for s in data)
    if dataset == "market1501":
        assert data[2].pid == 0
    with pytest.raises(ValueError, match="expected"):
        loader(tmp_path)
    cfg = get_dataset_config(dataset)
    assert cfg["source"]["revision"] == "ec537764e36ca10da963fcb7afab157ea60ca022"
    assert cfg["eval_protocol"]["id"] == cfg["evaluation"]["protocol"]


@pytest.mark.parametrize("filename", ["bad.jpg", "1_c9_bad.jpg", "-1_c1_junk.jpg"])
def test_duke_invalid_names(tmp_path, filename):
    (tmp_path / "query").mkdir()
    (tmp_path / "bounding_box_test").mkdir()
    (tmp_path / "query" / filename).touch()
    with pytest.raises(ValueError):
        DukeMTMCReIDDataset(tmp_path, validate_counts=False)


def test_missing_and_empty_splits(tmp_path):
    with pytest.raises(FileNotFoundError):
        Market1501Dataset(tmp_path)
    (tmp_path / "query").mkdir()
    with pytest.raises(ValueError, match="no images"):
        Market1501Dataset(tmp_path)


def test_handbuilt_exclusions_ties_and_missing_identity():
    # same-camera positive is removed; tied background stays before the positive.
    distances = np.array([[0.0, 1.0, 1.0, 2.0], [4.0, 3.0, 2.0, 1.0]])
    cmc, mean_ap = RetrievalEvaluator.evaluate_distances(
        distances,
        np.array([1, 99]),
        np.array([1, 0, 1, 1]),
        np.array([0, 0]),
        np.array([0, 1, 1, 2]),
    )
    np.testing.assert_array_equal(cmc, [0.0, 1.0, 1.0, 1.0])
    assert mean_ap == (1 / 2 + 2 / 3) / 2
    # Stable ties are manifest order, not identity order.
    cmc, mean_ap = RetrievalEvaluator.evaluate_distances(
        np.zeros((1, 2)),
        np.array([1]),
        np.array([1, 0]),
        np.array([0]),
        np.array([1, 1]),
    )
    assert mean_ap == 1.0
    np.testing.assert_array_equal(cmc, [1.0, 1.0])


@pytest.mark.parametrize(
    "evaluator", [Market1501Evaluator, DukeMTMCReIDEvaluator, OccludedDukeEvaluator]
)
def test_score_and_rank_saturation(evaluator):
    samples = [
        ReIDSample(Path("q"), 1, 0, "query"),
        ReIDSample(Path("g"), 1, 1, "gallery"),
    ]
    result = evaluator().evaluate(np.array([[1.0, 0.0], [1.0, 0.0]]), samples)
    assert result.as_dict() == {
        "primary": {"map": 1.0},
        "secondary": {"rank1": 1.0, "rank5": 1.0, "rank10": 1.0},
        "sample_count": 2,
    }
    assert result.primary_score == result.secondary_score == result.rank(50) == 1.0
    with pytest.raises(ValueError):
        result.rank(0)


def test_no_valid_queries():
    with pytest.raises(ValueError, match="No query"):
        RetrievalEvaluator.evaluate_distances(
            np.zeros((1, 1)), np.array([1]), np.array([1]), np.array([0]), np.array([0])
        )


def test_extraction_order():
    batches = [np.full((2, 3), i, dtype=np.float32) for i in range(5)]
    actual = extract_embeddings(batches, lambda x: [x], EmbeddingPost({}, {}))
    np.testing.assert_array_equal(actual, np.concatenate(batches))
    with pytest.raises(ValueError):
        extract_embeddings([], lambda x: x, lambda x: x)


@pytest.mark.parametrize(
    "options",
    [
        {"mean": [0, 0, 0]},
        {"mean": [0, 0, 0], "std": [1, 0, 1]},
        {"mean": [0, 0], "std": [1, 1]},
        {"style": "torch", "mean": [0, 0, 0], "std": [1, 1, 1]},
    ],
)
def test_explicit_normalization_validation(options):
    with pytest.raises(ValueError):
        Normalize(**options)


def test_family_base_inheritance(tmp_path):
    base = Path(__file__).parents[1] / "mblt_vision/models/bases/person_reid.yaml"
    for variant, name in [
        ("DEFAULT", "dukemtmc-reid"),
        ("MARKET1501", "market1501"),
        ("OCCLUDED_DUKE", "occluded-duke"),
    ]:
        cfg = load_model_variant(base, variant)
        assert cfg["post_cfg"]["dataset"] == name
        assert isinstance(
            build_postprocess(cfg["pre_cfg"], cfg["post_cfg"]), PersonReIDPost
        )
        assert build_preprocess(cfg["pre_cfg"])

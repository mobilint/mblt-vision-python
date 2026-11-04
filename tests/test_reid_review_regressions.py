"""Retrieval cutoffs and prepared-cache boundary regressions."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mblt_vision.utils.datasets import readiness, reid
from mblt_vision.utils.datasets.reid import ReIDSample
from mblt_vision.utils.evaluation.retrieval import RetrievalEvaluator


def test_truncated_cmc_omits_uncomputed_rank_metrics():
    samples = [ReIDSample(Path("query"), 1, 0, "query")]
    samples += [
        ReIDSample(Path(str(i)), pid, 1, "gallery")
        for i, pid in enumerate([2, 1, 3, 4, 5])
    ]
    vectors = np.array(
        [[0, 0], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32
    )
    result = RetrievalEvaluator().evaluate(vectors, samples, max_rank=1)
    assert result.as_dict()["secondary"] == {"rank1": 0.0}
    with pytest.raises(ValueError, match="cutoff"):
        result.rank(5)
    complete = RetrievalEvaluator().evaluate(vectors, samples, max_rank=50)
    assert complete.as_dict()["secondary"]["rank5"] == 1.0
    assert complete.as_dict()["secondary"]["rank10"] == 1.0


@pytest.mark.parametrize("component", ["root", "query", "bounding_box_test"])
def test_reid_readiness_rejects_symlink_components(tmp_path, monkeypatch, component):
    original = reid.get_dataset_config_for_task

    def tiny_config(task, dataset):
        config = original(task, dataset)
        for split in config["retrieval_splits"].values():
            split["count"] = 1
        return config

    monkeypatch.setattr(reid, "get_dataset_config_for_task", tiny_config)
    root = tmp_path / "cache"
    for folder in ("query", "bounding_box_test"):
        (root / folder).mkdir(parents=True)
        Image.new("RGB", (4, 8)).save(root / folder / "0001_c1_sample.jpg")
    assert readiness.dataset_ready(root, "person_reid", "market1501")
    if component == "root":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    else:
        target = tmp_path / "outside"
        (root / component).rename(target)
        (root / component).symlink_to(target, target_is_directory=True)
    assert not readiness.dataset_ready(root, "person_reid", "market1501")

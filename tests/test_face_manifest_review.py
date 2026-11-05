"""Official fold label balance and malformed CSV readiness regressions."""

import csv

import pytest
from PIL import Image

from mblt_vision.utils.datasets import face_pairs, readiness


@pytest.mark.parametrize(
    "name,folder,loader",
    [
        ("lfw", "aligned", face_pairs.LFWDataset),
        ("agedb-30", "images", face_pairs.AgeDB30Dataset),
        ("cfp-fp", "images", face_pairs.CFPFPDataset),
    ],
)
def test_same_different_balance_and_csv_readiness(
    tmp_path, monkeypatch, name, folder, loader
):
    original = face_pairs.get_dataset_config_for_task

    def tiny(task, dataset):
        config = original(task, dataset)
        config["pairs"] = 20
        return config

    monkeypatch.setattr(face_pairs, "get_dataset_config_for_task", tiny)
    (tmp_path / folder).mkdir()
    for image in ["a.png", "b.png"]:
        Image.new("RGB", (112, 112)).save(tmp_path / folder / image)
    manifest = tmp_path / "pairs.csv"

    def write(labels):
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pair_id", "fold_id", "is_same", "image_a", "image_b"])
            writer.writerows(
                (i, i // 2 + 1, label, "a.png", "b.png")
                for i, label in enumerate(labels)
            )

    write([i % 2 for i in range(20)])
    assert readiness.dataset_ready(tmp_path, "face_recognition", name)
    write([1] * 20)
    with pytest.raises(ValueError, match="same/different"):
        loader(tmp_path)
    assert not readiness.dataset_ready(tmp_path, "face_recognition", name)
    assert len(loader(tmp_path, validate_counts=False).pairs) == 20
    write([i % 2 for i in range(20)])
    old = csv.field_size_limit()
    try:
        csv.field_size_limit(4)
        assert not readiness.dataset_ready(tmp_path, "face_recognition", name)
    finally:
        csv.field_size_limit(old)

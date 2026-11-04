"""CPU proofs for legacy resolution and opt-in local YAML inheritance."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from mblt_vision._config import load_model_variant
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.wrapper import MODEL_CONFIG_DIR, resolve_model_config


def legacy_resolve(document: dict, variant: str) -> dict:
    """Independent pre-refactor resolution, including ONNX artifact derivation."""

    part = document[variant]
    if isinstance(part, str):
        return legacy_resolve(document, part)
    child = copy.deepcopy(part)
    base = child.pop("update", None)
    if base is not None:
        resolved = legacy_resolve(document, base)
        for key, value in child.items():
            if isinstance(resolved.get(key), dict) and isinstance(value, dict):
                resolved[key].update(value)
            else:
                resolved[key] = value
        return resolved
    return child


def ordered(value):
    """Deep equality alone misses insertion order, which defines pre-op order."""

    if isinstance(value, dict):
        return [(key, ordered(item)) for key, item in value.items()]
    if isinstance(value, list):
        return [ordered(item) for item in value]
    return value


def legacy_config(document: dict, variant: str) -> dict:
    config = legacy_resolve(document, variant)
    files = config.get("file_cfg", {})
    if files.get("filename") and "onnx_filename" not in files:
        files["onnx_filename"] = str(Path(files["filename"]).with_suffix(".onnx"))
    return config


CONFIG_VARIANTS = [
    (path, variant)
    for path in sorted(MODEL_CONFIG_DIR.glob("*.yaml"))
    for variant, definition in yaml.safe_load(path.read_text(encoding="utf-8")).items()
    # Only pre-existing alias/update configurations have a legacy counterpart.
    if not isinstance(definition, dict) or "extends" not in definition
]


@pytest.mark.parametrize("path,variant", CONFIG_VARIANTS, ids=lambda value: str(value))
def test_every_legacy_variant_is_order_identical(path: Path, variant: str) -> None:
    expected = legacy_config(yaml.safe_load(path.read_text(encoding="utf-8")), variant)
    actual = resolve_model_config(str(path), variant)
    assert actual == expected
    assert ordered(actual) == ordered(expected)


def write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_recursive_extends_keeps_order_and_independent_results(tmp_path: Path) -> None:
    write_config(
        tmp_path / "bases/base.yaml",
        {
            "DEFAULT": {
                "pre_cfg": {
                    "Reader": {"style": "pil", "color_mode": "RGB"},
                    "Resize": {"size": [32, 32], "interpolation": "bilinear"},
                },
                "post_cfg": {"nested": {"labels": [1, 2], "other": True}},
            }
        },
    )
    child = write_config(
        tmp_path / "child.yaml",
        {
            "DEFAULT": {
                "extends": "bases/base.yaml",
                "pre_cfg": {"Resize": {"size": [16, 16]}, "SetOrder": {"shape": "HWC"}},
                "post_cfg": {"nested": {"labels": [3]}},
            }
        },
    )
    result = load_model_variant(child)
    assert list(result["pre_cfg"]) == ["Reader", "Resize", "SetOrder"]
    assert result["pre_cfg"]["Resize"] == {
        "size": [16, 16],
        "interpolation": "bilinear",
    }
    assert result["post_cfg"]["nested"] == {"labels": [3], "other": True}
    result["post_cfg"]["nested"]["labels"].append(4)
    assert load_model_variant(child)["post_cfg"]["nested"]["labels"] == [3]


def test_update_stays_shallow_with_extends_base(tmp_path: Path) -> None:
    write_config(
        tmp_path / "base.yaml",
        {"DEFAULT": {"pre_cfg": {"Resize": {"size": 32, "interpolation": "bilinear"}}}},
    )
    path = write_config(
        tmp_path / "child.yaml",
        {
            "BASE": {"extends": "base.yaml"},
            "DEFAULT": {"update": "BASE", "pre_cfg": {"Resize": {"size": 16}}},
        },
    )
    assert load_model_variant(path)["pre_cfg"]["Resize"] == {"size": 16}


@pytest.mark.parametrize(
    "reference",
    [None, "", 123, "#DEFAULT", "base.yaml#", "https://host/base.yaml", "a#b#c"],
)
def test_invalid_extends(tmp_path: Path, reference) -> None:
    path = write_config(tmp_path / "child.yaml", {"DEFAULT": {"extends": reference}})
    with pytest.raises(ValueError):
        load_model_variant(path)


def test_cross_file_cycle(tmp_path: Path) -> None:
    a = write_config(tmp_path / "a.yaml", {"DEFAULT": {"extends": "b.yaml"}})
    write_config(tmp_path / "b.yaml", {"DEFAULT": {"extends": "./a.yaml"}})
    with pytest.raises(ValueError, match="Circular"):
        load_model_variant(a)


def test_missing_base_and_conflicting_modes(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "child.yaml", {"DEFAULT": {"extends": "absent.yaml"}}
    )
    with pytest.raises(FileNotFoundError):
        load_model_variant(path)
    write_config(
        path, {"BASE": {}, "DEFAULT": {"extends": "absent.yaml", "update": "BASE"}}
    )
    with pytest.raises(ValueError, match="cannot combine"):
        load_model_variant(path)


@pytest.mark.parametrize("model", ["ResNet50", "YOLOv8n", "YOLOX-s", "DAMO-YOLO-T"])
def test_cpu_preprocessing_numeric_identity(model: str) -> None:
    path = MODEL_CONFIG_DIR / f"{model}.yaml"
    expected = legacy_config(
        yaml.safe_load(path.read_text(encoding="utf-8")), "DEFAULT"
    )
    actual = resolve_model_config(str(path))
    image = np.random.default_rng(0).integers(0, 256, (71, 103, 3), dtype=np.uint8)
    before = build_preprocess(expected["pre_cfg"])(image.copy())
    after = build_preprocess(actual["pre_cfg"])(image.copy())
    np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

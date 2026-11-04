"""Intentional family-sibling configs are discoverable but never auto-download."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import mblt_vision
from mblt_vision._hub_validation import check_hub_artifacts
from mblt_vision.compile import vision as compile_module
from mblt_vision.wrapper import MBLT_Engine, resolve_model_config


@pytest.mark.parametrize(
    "name", ["EfficientViT_B0", "EfficientViT_B1", "EfficientViT_B2"]
)
def test_official_sibling_pipeline_and_pending_metadata(name) -> None:
    config = resolve_model_config(name)
    reference = resolve_model_config("EfficientViT_B3")
    assert config["pre_cfg"] == reference["pre_cfg"]
    assert config["pre_cfg"]["Resize"] == {"size": 235, "interpolation": "bicubic"}
    assert config["file_cfg"]["artifact_status"] == "pending_hf_family_sibling"
    assert (
        name in mblt_vision.list_models("image_classification")["image_classification"]
    )
    assert getattr(mblt_vision, name) is getattr(mblt_vision.image_classification, name)
    api = Mock()
    checks = check_hub_artifacts(config, api=api)
    assert all(item.status == "pending_hf_family_sibling" for item in checks)
    assert not api.mock_calls


@pytest.mark.parametrize("framework", ["mxq", "onnx"])
def test_pending_engine_never_downloads(framework) -> None:
    engine = object.__new__(MBLT_Engine)
    engine.framework = framework
    engine.file_cfg = resolve_model_config("EfficientViT_B0")["file_cfg"]
    engine._download_hub_artifact = Mock()
    with pytest.raises(RuntimeError, match="pending HF"):
        engine.file_config_cleansing()
    engine._download_hub_artifact.assert_not_called()


@pytest.mark.parametrize("framework", ["mxq", "onnx"])
def test_pending_engine_accepts_trusted_local_artifact(framework, tmp_path) -> None:
    engine = object.__new__(MBLT_Engine)
    engine.framework = framework
    engine.file_cfg = resolve_model_config("EfficientViT_B0")["file_cfg"]
    source = tmp_path / f"trusted.{framework}"
    source.write_bytes(b"fixture")
    engine.file_cfg[f"{framework}_path"] = str(source)
    engine._download_hub_artifact = Mock()
    engine.file_config_cleansing()
    assert "artifact_status" not in engine._mxq_backend_kwargs()
    engine._download_hub_artifact.assert_not_called()


def test_pending_compile_never_downloads_and_accepts_local(
    monkeypatch, tmp_path
) -> None:
    config = resolve_model_config("EfficientViT_B0")["file_cfg"]
    download = Mock()
    monkeypatch.setattr(compile_module, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="pending HF"):
        compile_module._resolve_compile_onnx_path(config, None)
    local = tmp_path / "trusted.onnx"
    local.write_bytes(b"fixture")
    assert compile_module._resolve_compile_onnx_path(config, local) == local.resolve()
    download.assert_not_called()

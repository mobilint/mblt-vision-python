"""Positive and negative CPU tests for framework-specific local-only artifacts."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mblt_vision._hub_validation import check_hub_artifacts
from mblt_vision.compile import vision as compile_module
from mblt_vision.wrapper import MBLT_Engine, resolve_model_config


def engine_config(framework):
    engine = object.__new__(MBLT_Engine)
    engine.framework = framework
    engine.file_cfg = resolve_model_config("EfficientViT_B3")["file_cfg"]
    engine._download_hub_artifact = Mock(return_value="/downloaded/artifact")
    return engine


def test_marked_remote_onnx_fails_without_download() -> None:
    engine = engine_config("onnx")
    with pytest.raises(RuntimeError, match="trusted local"):
        engine.file_config_cleansing()
    engine._download_hub_artifact.assert_not_called()


def test_marked_local_onnx_still_works(tmp_path) -> None:
    engine = engine_config("onnx")
    source = tmp_path / "trusted.onnx"
    source.write_bytes(b"local fixture; runtime not constructed")
    engine.file_cfg["onnx_path"] = str(source)
    engine.file_config_cleansing()
    assert engine.file_cfg["onnx_path"] == str(source)
    engine._download_hub_artifact.assert_not_called()


def test_marked_mxq_keeps_hub_download() -> None:
    engine = engine_config("mxq")
    engine.file_config_cleansing()
    engine._download_hub_artifact.assert_called_once_with(
        repo_id="mobilint/EfficientViT_B3",
        filename="efficientvit_b3.mxq",
        revision="main",
        subfolders=["aries-rb"],
    )
    kwargs = engine._mxq_backend_kwargs()
    assert "onnx_local_artifact_only" not in kwargs
    assert kwargs["mxq_path"] == "/downloaded/artifact"


def test_unmarked_remote_onnx_keeps_hub_download() -> None:
    engine = engine_config("onnx")
    engine.file_cfg.pop("onnx_local_artifact_only")
    engine.file_config_cleansing()
    engine._download_hub_artifact.assert_called_once_with(
        repo_id="mobilint/EfficientViT_B3",
        filename="efficientvit_b3.onnx",
        revision="main",
    )


def test_verifier_skips_only_marked_framework() -> None:
    api = Mock()
    api.model_info.return_value.sha = "observed"
    api.list_repo_files.return_value = ["aries-rb/efficientvit_b3.mxq"]
    checks = check_hub_artifacts(resolve_model_config("EfficientViT_B3"), api=api)
    assert [item.status for item in checks] == ["resolved", "local_artifact_only"]
    assert checks[1].commit is None
    assert api.model_info.call_count == 1


def test_onnx_only_verification_stays_offline() -> None:
    api = Mock()
    checks = check_hub_artifacts(
        resolve_model_config("EfficientViT_B3"), frameworks=("onnx",), api=api
    )
    assert checks[0].status == "local_artifact_only"
    assert not api.mock_calls


def test_compile_resolver_requires_local_onnx(monkeypatch, tmp_path) -> None:
    download = Mock()
    monkeypatch.setattr(compile_module, "hf_hub_download", download)
    config = resolve_model_config("EfficientViT_B3")["file_cfg"]
    with pytest.raises(RuntimeError, match="trusted local"):
        compile_module._resolve_compile_onnx_path(config, None)
    download.assert_not_called()
    local = tmp_path / "trusted.onnx"
    local.write_bytes(b"fixture")
    assert compile_module._resolve_compile_onnx_path(config, local) == local.resolve()
    assert (
        compile_module._resolve_compile_onnx_path(
            {**config, "onnx_path": str(local)}, None
        )
        == local.resolve()
    )
    download.assert_not_called()


def test_compile_unmarked_config_retains_download(monkeypatch, tmp_path) -> None:
    local = tmp_path / "downloaded.onnx"
    local.write_bytes(b"fixture")
    download = Mock(return_value=str(local))
    monkeypatch.setattr(compile_module, "hf_hub_download", download)
    monkeypatch.setattr(compile_module, "get_mobilint_cache_dir", lambda: str(tmp_path))
    config = resolve_model_config("EfficientViT_B3")["file_cfg"]
    config.pop("onnx_local_artifact_only")
    assert compile_module._resolve_compile_onnx_path(config, None) == local.resolve()
    download.assert_called_once()

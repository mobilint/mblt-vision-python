"""Offline unit tests and explicit metadata-only Hub integration checks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mblt_vision._hub_validation import check_hub_artifacts
from mblt_vision.wrapper import resolve_model_config


def config(**overrides):
    return {
        "file_cfg": {
            "repo_id": "mobilint/example",
            "revision": "approved-commit",
            "filename": "model.mxq",
            "onnx_filename": "model.onnx",
            **overrides,
        }
    }


def api_with_files(files):
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="exact-live-commit")
    api.list_repo_files.return_value = files
    return api


def test_exact_paths_and_commit_snapshot() -> None:
    api = api_with_files(["aries-rb/model.mxq", "model.onnx"])
    checks = check_hub_artifacts(config(), api=api)
    assert [item.status for item in checks] == ["resolved", "resolved"]
    assert [item.path for item in checks] == ["aries-rb/model.mxq", "model.onnx"]
    assert all(item.commit == "exact-live-commit" for item in checks)
    api.model_info.assert_called_once_with(
        repo_id="mobilint/example", revision="approved-commit"
    )
    api.list_repo_files.assert_called_once_with(
        repo_id="mobilint/example", revision="exact-live-commit", repo_type="model"
    )


def test_wrong_board_never_falls_back() -> None:
    api = api_with_files(["aries-rb/model.mxq", "model.mxq", "model.onnx"])
    checks = check_hub_artifacts(config(), target_device="regulus-ra", api=api)
    assert checks[0].path == "regulus-ra/model.mxq"
    assert checks[0].status == "missing"
    assert checks[1].status == "resolved"


def test_local_artifact_only_does_not_contact_hub() -> None:
    api = Mock()
    checks = check_hub_artifacts(config(local_artifact_only=True), api=api)
    assert all(
        item.status == "local_artifact_only" and item.commit is None for item in checks
    )
    assert not api.mock_calls


def test_api_errors_remain_errors() -> None:
    api = Mock()
    api.model_info.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError, match="offline"):
        check_hub_artifacts(config(), api=api)


@pytest.mark.parametrize("frameworks", [(), ("other",)])
def test_invalid_frameworks(frameworks) -> None:
    with pytest.raises(ValueError, match="frameworks"):
        check_hub_artifacts(config(), frameworks=frameworks)


@pytest.mark.parametrize(
    "override",
    [{"repo_id": ""}, {"revision": None}, {"filename": None}, {"onnx_filename": ""}],
)
def test_missing_required_fields(override) -> None:
    with pytest.raises(ValueError):
        check_hub_artifacts(config(**override), api=Mock())


def test_framework_selection_does_not_require_other_filename() -> None:
    api = api_with_files(["model.onnx"])
    checks = check_hub_artifacts(config(filename=None), frameworks=("onnx",), api=api)
    assert len(checks) == 1
    assert checks[0].status == "resolved"


@pytest.mark.requires_network
@pytest.mark.parametrize("model", ["AlexNet", "ResNet50", "YOLOv8n"])
def test_representative_hub_metadata_resolves(model: str) -> None:
    checks = check_hub_artifacts(resolve_model_config(model))
    assert all(item.status == "resolved" for item in checks), checks

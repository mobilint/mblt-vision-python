"""Read-only Hub metadata checks for resolved model artifact configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi
from mblt_npu import normalize_target_device


@dataclass(frozen=True)
class HubArtifactCheck:
    """A metadata-only resolution result; a live commit is not approval provenance."""

    repo_id: str
    revision: str
    commit: str | None
    framework: str
    path: str
    status: str


def check_hub_artifacts(
    config: dict[str, Any],
    *,
    target_device: str | None = None,
    frameworks: tuple[str, ...] = ("mxq", "onnx"),
    api: HfApi | None = None,
) -> tuple[HubArtifactCheck, ...]:
    """Check exact runtime Hub paths without downloading model bytes.

    Accept a resolved configuration (``resolve_model_config`` derives the ONNX
    filename). MXQ is checked only under the normalized target board; ONNX is
    checked at the repository root, matching the engine. Never fall back to a
    different board, revision or filename. Local-only configs are reported as
    skipped, without contacting the Hub. Hub/API exceptions propagate so missing
    access or network failure cannot be mistaken for a missing file.

    The returned live commit is diagnostic, not evidence that a commit was
    approved. This function does not modify YAMLs or infer approved revisions.
    """

    if not frameworks or any(
        framework not in {"mxq", "onnx"} for framework in frameworks
    ):
        raise ValueError("frameworks must contain only 'mxq' and/or 'onnx'.")
    files = config.get("file_cfg")
    if not isinstance(files, dict):
        raise TypeError("Resolved model configuration requires a file_cfg mapping.")
    repo_id, revision = files.get("repo_id"), files.get("revision")
    if (
        not isinstance(repo_id, str)
        or not repo_id
        or not isinstance(revision, str)
        or not revision
    ):
        raise ValueError(
            "Hub verification requires nonempty repo_id and revision strings."
        )
    board = normalize_target_device(
        files.get("target_device", "aries-rb") if target_device is None else target_device
    )
    paths = []
    for framework in dict.fromkeys(frameworks):
        filename = files.get("filename" if framework == "mxq" else "onnx_filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Resolved configuration requires the {framework} artifact filename."
            )
        paths.append(
            (framework, f"{board}/{filename}" if framework == "mxq" else filename)
        )

    if files.get("artifact_status") == "pending_hf_family_sibling":
        return tuple(
            HubArtifactCheck(
                repo_id, revision, None, framework, path, "pending_hf_family_sibling"
            )
            for framework, path in paths
        )

    def is_local_only(framework: str) -> bool:
        return bool(files.get("local_artifact_only")) or (
            framework == "onnx" and bool(files.get("onnx_local_artifact_only"))
        )

    if all(is_local_only(framework) for framework, _ in paths):
        return tuple(
            HubArtifactCheck(
                repo_id, revision, None, framework, path, "local_artifact_only"
            )
            for framework, path in paths
        )
    client = HfApi() if api is None else api
    info = client.model_info(repo_id=repo_id, revision=revision)
    if not isinstance(info.sha, str) or not info.sha:
        raise ValueError("Hub model metadata did not return a commit SHA.")
    # Pin the file listing to the just-resolved commit: symbolic branches can move
    # between API calls, and results must describe a single repository snapshot.
    available = set(
        client.list_repo_files(repo_id=repo_id, revision=info.sha, repo_type="model")
    )
    return tuple(
        HubArtifactCheck(
            repo_id,
            revision,
            None if is_local_only(framework) else info.sha,
            framework,
            path,
            "local_artifact_only"
            if is_local_only(framework)
            else "resolved"
            if path in available
            else "missing",
        )
        for framework, path in paths
    )

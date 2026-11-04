"""Local YAML model inheritance, independent of inference backends."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(
    base: dict[str, Any], child: dict[str, Any], *, recursive: bool
) -> dict[str, Any]:
    """Merge independent mappings while retaining preprocessing operation order."""

    result = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            if recursive:
                result[key] = _merge(result[key], value, recursive=True)
            else:
                # Historical `update` merges exactly one section level.
                result[key].update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_model_variant(
    config_path: str | Path, variant: str = "DEFAULT"
) -> dict[str, Any]:
    """Resolve a local YAML variant, its aliases, and optional family inheritance.

    ``update`` retains the legacy same-file, one-section merge semantics.
    ``extends: bases/family.yaml#DEFAULT`` opts into recursive mapping inheritance;
    the fragment defaults to DEFAULT and paths are relative to the declaring file.
    Lists and scalars replace their base values. Mapping key order is retained, so
    inherited preprocessing operations retain their order. New operations append;
    replace a complete operation sequence with a standalone config when reordering.
    Only local files are read; each result is independently deep-copied.
    """

    documents: dict[Path, dict[str, Any]] = {}
    resolving: set[tuple[Path, str]] = set()

    def document(path: Path) -> dict[str, Any]:
        if path not in documents:
            with path.open(encoding="utf-8") as stream:
                parsed = yaml.safe_load(stream)
            if not isinstance(parsed, dict):
                raise TypeError(
                    f"Model configuration '{path}' should define a dictionary."
                )
            documents[path] = parsed
        return documents[path]

    def resolve(path: Path, key: str) -> dict[str, Any]:
        identity = (path, key)
        if identity in resolving:
            raise ValueError(
                f"Circular model configuration reference detected for '{path}#{key}'."
            )
        resolving.add(identity)
        try:
            full = document(path)
            part = full.get(key)
            if part is None:
                raise ValueError(f"Model type '{key}' not found in configuration.")
            if isinstance(part, str):
                if part not in full:
                    raise ValueError(
                        f"Model alias '{part}' not found in configuration."
                    )
                return resolve(path, part)
            if not isinstance(part, dict):
                raise TypeError(
                    f"Resolved model configuration for '{key}' is not a dictionary."
                )
            child = copy.deepcopy(part)
            base_key = child.pop("update", None)
            has_extends = "extends" in child
            reference = child.pop("extends", None)
            if base_key is not None and has_extends:
                raise ValueError(
                    "A model variant cannot combine 'update' and 'extends'."
                )
            if base_key is not None:
                if not isinstance(base_key, str) or base_key not in full:
                    raise ValueError(
                        f"Base configuration '{base_key}' not found for update."
                    )
                return _merge(resolve(path, base_key), child, recursive=False)
            if has_extends:
                if not isinstance(reference, str) or not reference:
                    raise ValueError(
                        "Model 'extends' must be a nonempty local YAML reference."
                    )
                filename, separator, base_variant = reference.partition("#")
                if (
                    not filename
                    or "://" in filename
                    or (separator and not base_variant)
                ):
                    raise ValueError(
                        f"Invalid local model inheritance reference: {reference!r}."
                    )
                if "#" in base_variant:
                    raise ValueError(
                        f"Invalid local model inheritance reference: {reference!r}."
                    )
                base_path = (path.parent / filename).resolve()
                return _merge(
                    resolve(base_path, base_variant or "DEFAULT"), child, recursive=True
                )
            return child
        finally:
            resolving.remove(identity)

    return resolve(Path(config_path).resolve(), variant)

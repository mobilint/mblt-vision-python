---
description: Guidance for coding agents working on the PyPI-distributed mblt-vision Python API.
paths:
  - "**"
---

# mblt-vision-python Agent Guide

## Mission

`mblt-vision-python` is the Python distribution and public compatibility layer for Mobilint
Vision. It binds to the supported native API in `mblt-vision`; it is not a second implementation
of the vision runtime. Its end-state is a drop-in replacement for the Vision API currently shipped
by `mblt-model-zoo`.

The ownership boundary is deliberate:

- `mblt-vision` owns inference, model loading, preprocessing, postprocessing, and native resource
  management.
- This package owns Python ergonomics, Python object conversion, package metadata, wheels, API
  documentation, and compatibility shims.
- Do not put GStreamer integration, C++ runtime logic, or duplicated numerical kernels in Python.

## Before Editing

- Run `git status --short` and preserve unrelated changes.
- Read `pyproject.toml`, `README.md`, package exports, binding sources, and relevant tests before
  changing a public API or packaging behavior.
- For a Model Zoo replacement item, inspect the matching behavior in
  `../mblt-model-zoo/mblt_model_zoo/vision`, including its tests and model YAML configuration.
  Treat it as the compatibility reference until the new package explicitly supersedes it.
- Coordinate native API changes with `../mblt-vision`. Do not bind private, undocumented, or
  build-tree-only native symbols.

## Python API Contract

- Make `mblt_vision` the only intended import namespace. Keep its exports intentional, documented,
  typed, and stable.
- Preserve established Model Zoo Vision user-facing behavior wherever practical: engine/model
  construction, task discovery, model aliases, supported arguments, result shapes/types, error
  classes, and default semantics. Deprecate rather than silently remove a compatible public name.
- Prefer a small idiomatic Python surface over mirroring every C++ implementation class. Convert
  native errors to specific, actionable Python exceptions while retaining the original context.
- Define numpy/image/tensor conversion rules precisely: accepted dtype, shape, layout, color order,
  contiguity, mutability, ownership, and copying behavior. Zero-copy paths must retain the Python
  buffer for as long as native code can access it.
- Never expose raw native pointers or require callers to manage native lifetime. Wrap resources in
  deterministic `close()`/context-manager behavior and safe finalization as appropriate.
- Maintain task vocabulary compatibility: canonical `obb`, with
  `oriented_bounding_boxes` accepted only as an input compatibility alias.

## Binding and Native Dependency Rules

- The binding must use only the documented, versioned `mblt-vision` interface. Add a native
  capability/version check when a Python feature needs a newer library.
- Keep ownership and the GIL explicit. Release the GIL only around blocking native work that does
  not access Python objects; reacquire it before callbacks, exceptions, or Python buffer access.
- Do not catch broad native errors and return empty or plausible-looking results. Fail loudly with
  an exception that identifies the invalid input, unsupported feature, or native-library problem.
- Keep native library discovery relocatable and diagnosable. Avoid hard-coded developer paths,
  `LD_LIBRARY_PATH` requirements for normal wheels, or importing an optional native backend at
  module import time if that prevents useful error reporting.

## PyPI and Wheel Packaging

- `pyproject.toml` is the source of truth for Python metadata, supported Python versions,
  dependencies, and build backend. Keep package versioning synchronized with the exposed API and
  native compatibility requirements.
- Distribute wheels that include or correctly depend on the matching native library according to
  the chosen packaging strategy. Do not publish an sdist/wheel whose import or basic diagnostics
  are broken without a locally installed development tree.
- Build and test each intended platform/architecture wheel in a clean environment. Verify wheel
  contents, package metadata, install-from-wheel, import, and a minimal inference-free native
  smoke test. Do not upload from a developer environment as the only validation.
- Keep optional dependencies genuinely optional and avoid importing them from package top level.
  Do not add model weights, caches, test assets, or compiled build artifacts to source control.

## Compatibility Migration and Tests

- Maintain an explicit, tested compatibility matrix for each migrated Model Zoo Vision feature:
  import/export, constructor arguments, preprocessing inputs, inference outputs, postprocessing
  results, errors, CLI behavior if provided, and deprecation status.
- Use deterministic differential tests against Model Zoo for shared behavior. Cover edge cases,
  not only successful end-to-end examples: invalid layouts/dtypes, empty detections, threshold
  boundaries, image geometry, model aliases, task aliases, and resource cleanup.
- Numerical semantics belong to the native layer, but binding tests must verify that Python
  conversion does not change values, layout, coordinates, ordering, dtype, or ownership.
- Avoid making hardware, downloaded models, or GStreamer a requirement for ordinary unit tests.
  Mark and document integration prerequisites; run the narrowest relevant suite first.

## Code Quality and Documentation

- Use four-space indentation, PEP 484 type annotations, clear docstrings for public APIs, and
  lines of at most 120 characters unless the repository tooling specifies otherwise.
- Keep imports ordered as standard library, third-party, then local. Catch specific exceptions.
- Update the README and API examples whenever installation, native-library discovery, supported
  platforms, imports, or migration compatibility changes.
- For documentation-only changes, run `git diff --check` and verify headings and links. Report
  skipped platform, hardware, or native-runtime checks clearly.

## Git Safety

- Do not alter `../mblt-vision` or `../mblt-model-zoo` as an incidental change in this repository.
- Keep commits focused. Do not commit virtual environments, wheelhouse contents, caches, native
  build directories, downloaded models, or generated coverage/benchmark files.

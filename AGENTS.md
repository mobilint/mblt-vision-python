---
description: Guidance for coding agents working on the PyPI-distributed mblt-vision Python API.
paths:
  - "**"
---

# mblt-vision-python Agent Guide

## Mission

`mblt-vision-python` is the Python distribution and public compatibility layer for Mobilint
Vision. The immediate plan is a pure-Python implementation that can replace the Vision API
currently shipped by `mblt-model-zoo`. C++ bindings are deferred until `mblt-vision` has a stable,
supported native API.

The current ownership boundary is deliberate:

- This package owns the public Python API, model loading, preprocessing, postprocessing, runtime
  integration, dataset organization, benchmark evaluation, model compilation, package metadata,
  wheels, documentation, and compatibility shims.
- Keep implementation code in Python; do not block Python API progress on C++ or GStreamer work.
- Design internal seams so a future optional `mblt-vision` backend can replace implementation
  details without changing the documented Python API or result contracts.

## Before Editing

- Run `git status --short` and preserve unrelated changes.
- Read `pyproject.toml`, `README.md`, package exports, binding sources, and relevant tests before
  changing a public API or packaging behavior.
- For a Model Zoo replacement item, inspect the matching behavior in
  `../mblt-model-zoo/mblt_model_zoo/vision`, including its tests and model YAML configuration.
  Treat it as the compatibility reference until the new package explicitly supersedes it.
- Do not make `../mblt-vision`, a compiled extension, or GStreamer a dependency of normal package
  development, installation, import, or unit tests. Revisit integration only after its native API
  is documented and versioned.

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
- Use `obb` as the sole oriented-bounding-box task name.

## Vision Models and Processing Contracts

- Use mblt_vision.MBLT_Engine for loading. Prefer model_path; retain mxq_path and
  onnx_path as compatibility aliases.
- Update task-package exports and lazy top-level exports together. Confirm that
  list_models() discovers every public model class.
- Keep each model YAML's file_cfg, pre_cfg, and post_cfg shape stable.
  file_cfg.filename is the canonical MXQ Hub artifact; derive the same-stem ONNX filename
  unless the Hub artifact requires an explicit onnx_filename.
- Require post_cfg.dataset in every model YAML and resolve output class counts using the
  dataset/task pair. Do not assume one output taxonomy for every model in a task.
- Preserve automatic .mxq/.onnx framework detection and the fail-fast error when a local
  suffix conflicts with an explicitly selected framework.
- Preserve anchorless decoded-output layout provenance through NMS. When a tensor is ambiguous
  and provenance is unavailable, normalize it as raw channels-first before candidates-first.
- Use the shared letterbox helpers for forward geometry and inverse output restoration. Detection
  postprocessors require pre_cfg.LetterBox; metadata-aware semantic preprocessing returns the
  original image shape and ratio_pad so logits can be restored before argmax.
- Normalize dense outputs before inverse letterboxing: upsample quarter-resolution depth maps by
  four, preserve baked-resize depth maps, convert Cityscapes NHWC logits to NCHW, and reject
  non-finite, fractional, or out-of-range baked semantic IDs before casting.
- Keep hardware-specific runtime access behind mblt-npu-python. Optional ONNX Runtime imports
  must remain lazy and report the appropriate package extra when unavailable.
- For WiderFace evaluation, rank results by Hard-set AP and retain Medium-set
  then Easy-set AP as secondary metrics. Do not compute a mean across splits.
- For NYU Depth evaluation, rank results by delta1 and retain abs_rel then
  RMSE (m) as secondary metrics. Median-align each image and average every
  metric per image, following Ultralytics' depth-validation convention.
- `mask_generation` (`SAM2HieraLarge`) is the precedent for a promptable, multi-artifact model:
  it loads two `MobilintNPUBackend` instances (encoder + decoder) and takes point prompts, so it
  bypasses `MBLT_Engine.__init__`, `build_preprocess`/`build_postprocess`, and
  `create_model_class`'s single-artifact legacy constructor entirely, implementing its own
  `preprocess`/`predict`/`predict_preprocessed` instead. Its `models/SAM2HieraLarge.yaml` is
  documentation/test-metadata only, not actually loaded. Reuse `wrapper.download_hub_artifact`
  (extracted from `MBLT_Engine._download_hub_artifact`) for any future model needing more than
  one Hub artifact, rather than duplicating Hub-resolution logic.
- Never add the PyPI `sam2` package as a dependency of `mblt_vision` (it is an unofficial
  third-party mirror, not Meta's) and never require a manually cloned
  `facebookresearch/sam2` checkout. Host-side prompt encoding is instead a from-scratch,
  dependency-free port of the official prompt encoder/mask-decoder token setup in
  `mask_generation/_sam2_prompt.py` and `_sam2_host.py`, backed by a small (~16KB) bundle of
  weight tensors extracted from the official checkpoint and hosted at
  `mobilint/sam2-hiera-large`'s Hub repo root (`sam2_hiera_large_prompt_weights.pt`) --
  downloaded the same way as the encoder/decoder MXQ artifacts, not shipped as package data.
  Any change to that port must stay numerically verified against the real
  `facebookresearch/sam2` predictor (`tests/test_mask_generation_prompt_encoding.py`, opt-in,
  skips without a real `sam2` install). No added dependency: the input resize/normalize step
  is plain `torch` (`F.interpolate` bilinear with `antialias=True` reproduces torchvision's
  tensor `Resize` bit-for-bit), so mask_generation runs on the package's existing dependencies.

## Python-First Architecture

- Keep the public layer independent from a particular backend. Define small internal interfaces
  for model execution and artifact resolution, but do not add speculative abstractions before a
  second backend exists.
- Put preprocessing, postprocessing, model configuration, and compatibility behavior in tested
  Python modules. Reuse Model Zoo semantics deliberately; do not copy code wholesale without
  understanding its public contract and license context.
- Use established Python runtime dependencies only when they materially support the package goals.
  Keep optional frameworks lazy-imported and raise a specific installation error when a requested
  backend is unavailable.
- Do not catch broad runtime errors and return empty or plausible-looking results. Fail loudly with
  an exception that identifies the invalid input, unsupported feature, or unavailable dependency.
- If/when a native backend is introduced, it must be optional, use a documented and versioned
  `mblt-vision` interface, and preserve the Python public API, exceptions, result values, layouts,
  and lifecycle behavior. Add native capability/version checks at that time.

## Benchmark and Compilation Tooling

- Keep executable benchmark organizers, the unified benchmark runner, and result comparison scripts
  directly under `benchmark/`. Put reusable benchmark reporting helpers in `mblt_vision.benchmark`.
- Keep executable compilation helpers and their guide directly under `compile/`. Do not recreate a
  Vision-only subdirectory under either tooling root.
- Use `~/.mblt_model_zoo` as the shared artifact and dataset cache root. Organizer defaults,
  dataset registry YAMLs, compilation defaults, and documented commands must agree on that root.
- Keep package imports free of cache-directory creation, write probes, downloads, and temporary
  directory allocation. Resolve a writable cache lazily only when an artifact or compilation output needs it.
- If the preferred cache is unavailable, use a stable, private, user-owned fallback cache. Do not
  create a new temporary cache per process or reuse an unsafe shared directory.
- Benchmark and compilation commands are development tools; do not package them as public CLI
  entry points without an explicit product decision. The supported end-user command is
  `mblt-vision`.
- Compile and artifact resolution must use normalized board identifiers (`aries-rb`, `regulus-ra`,
  or `regulus-rb`) and must not fall back to a different board folder.

## PyPI and Wheel Packaging

- `pyproject.toml` is the source of truth for Python metadata, supported Python versions,
  dependencies, and build backend. Keep package versioning synchronized with the exposed API and
  native compatibility requirements.
- Publish pure-Python wheels and sdists that install and import without a local C++ build, a native
  library, or GStreamer. Do not publish artifacts whose import or basic diagnostics require a
  developer environment.
- Build and test each intended platform/architecture wheel in a clean environment. Verify wheel
  contents, package metadata, install-from-wheel, import, and a minimal API smoke test. Do not
  upload from a developer environment as the only validation.
- Keep optional dependencies genuinely optional and avoid importing them from package top level.
  Do not add model weights, caches, test assets, or compiled build artifacts to source control.

## Compatibility Migration and Tests

- Maintain an explicit, tested compatibility matrix for each migrated Model Zoo Vision feature:
  import/export, constructor arguments, preprocessing inputs, inference outputs, postprocessing
  results, errors, CLI behavior if provided, and deprecation status.
- Use deterministic differential tests against Model Zoo for shared behavior. Cover edge cases,
  not only successful end-to-end examples: invalid layouts/dtypes, empty detections, threshold
  boundaries, image geometry, model aliases, task aliases, and resource cleanup.
- Verify that Python preprocessing and postprocessing preserve expected values, layouts,
  coordinates, ordering, dtype, and ownership. If a future backend is used, require the same
  parity from its conversion boundary.
- Avoid making hardware, downloaded models, compiled extensions, or GStreamer a requirement for
  ordinary unit tests.
  Mark and document integration prerequisites; run the narrowest relevant suite first.
- Use a deterministic default seed of 0 for any public API that samples or otherwise uses
  randomness.

## Code Quality and Documentation

- Use four-space indentation, PEP 484 type annotations, clear docstrings for public APIs, and
  lines of at most 120 characters unless the repository tooling specifies otherwise.
- Keep imports ordered as standard library, third-party, then local. Catch specific exceptions.
- Update the README and API examples whenever installation, native-library discovery, supported
  platforms, imports, or migration compatibility changes.
- Keep the root README focused on installation and navigation. Maintain the complete Vision API,
  model-family, runtime, and taxonomy reference in mblt_vision/README.md.
- Write documentation with ATX headings, one blank line between blocks, hyphen lists,
  language-tagged code fences, and concise paragraphs. Keep examples executable against the
  public mblt_vision namespace and do not document Model Zoo CLI commands as standalone features.
- When a durable public fact changes, update this guide, the matching agent skill, CLAUDE.md, and
  the relevant README in the same change. Treat a significant package change—public API,
  dependency/runtime, artifact layout, CLI, or tooling structure—as a required guide-and-skill
  synchronization point.
- For documentation-only changes, run `git diff --check` and verify headings and links. Report
  skipped platform, hardware, or native-runtime checks clearly.

## Git Safety

- Do not alter `../mblt-vision` or `../mblt-model-zoo` as an incidental change in this repository.
- Keep commits focused. Do not commit virtual environments, wheelhouse contents, caches, native
  build directories, downloaded models, or generated coverage/benchmark files.

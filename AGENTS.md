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
  it loads two backend instances (encoder + decoder) and takes point prompts, so it
  bypasses `MBLT_Engine.__init__`, `build_preprocess`/`build_postprocess`, and
  `create_model_class`'s single-artifact legacy constructor entirely, implementing its own
  `preprocess`/`predict`/`predict_preprocessed` instead. Its `models/SAM2HieraLarge.yaml` is
  documentation/test-metadata only, not actually loaded. Reuse `wrapper.download_hub_artifact`
  (extracted from `MBLT_Engine._download_hub_artifact`) for any future model needing more than
  one Hub artifact, rather than duplicating Hub-resolution logic.
- `mask_generation` supports both frameworks with `MBLT_Engine`-style semantics:
  `framework="mxq"` (default, two `MobilintNPUBackend` MXQs from the board folder) or
  `framework="onnx"` (two `ONNXBackend` sessions over the same-stem
  `sam2_hiera_large_{encoder,decoder}.onnx` exports at the Hub repo root, board-agnostic).
  Framework is inferred from explicit `encoder_onnx_path`/`decoder_onnx_path` vs
  `encoder_mxq_path`/`decoder_mxq_path` suffixes and conflicts fail fast; NPU-only arguments are
  ignored for ONNX. The two runtimes have different graph contracts, pinned in
  `_sam2_contracts.py` and validated at construction: MXQ takes the six flattened positional
  inputs (NHWC image and features), while the exported ONNX graphs are NCHW with five named
  decoder inputs (`src_plus_pos_src` stays inside the graph) and a dynamic token axis. The
  shared prompt-encoding host path and `classify_decoder_outputs` are framework-independent;
  only the encoder-feed layout and decoder-feed builders differ
  (`fpn_from_onnx`/`prepare_decoder_tensors_onnx` vs
  `fpn_from_runtime`/`prepare_decoder_tensors`). The ONNX pipeline is numerically verified
  against the official `facebookresearch/sam2` fp32 predictor (identical binary masks;
  opt-in `tests/test_mask_generation_onnx.py` covers it end-to-end without NPU hardware),
  and `eval_sav` works unchanged for both frameworks. Resolve the optional runtime before
  downloading any artifact, so a missing `onnxruntime` reports the package extra rather than a
  network failure. Build each backend so it disposes itself when `launch()`/graph validation
  fails after `create()`: the caller only assigns it on success, so the constructor's cleanup
  cannot otherwise reach it. Point prompts are validated as 1-3 points with finite
  coordinates and labels of exactly `1`/`0` (checked before the integer cast, which would
  otherwise truncate `0.5` into a valid label) before any backend call, and host prompt tensors
  are built on the weights' device so `device="cuda"` works. `classify_decoder_outputs` requires
  exactly three mask candidates and rejects non-finite decoder outputs, so a NaN cannot reach
  `argmax` over the IoU scores or the `> 0` mask threshold. Mask generation models reject `--model-path`/`--mxq-path`/`--onnx-path`
  in every CLI command that builds one, never only in `predict`.
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
- `mask_generation` evaluates on the SA-V validation split (155 videos, 293 masklets, 31967
  annotated masks; JPEG frames + per-object binary PNG masks, binarized as `> 0`). The registry
  entry `datasets/sa-v.yaml` downloads the unmodified official `sav_val.tar` from the Mobilint Hub
  mirror `datasets/mobilint/sa-v` (SA-V is CC BY 4.0 by Meta AI; keep the mirror README's
  attribution intact and pin the archive sha256 in the YAML and `PINNED_ARCHIVE_SHA256`). The
  organizer keeps only annotated frames. Readiness pins all three counts
  (`SAV_VALIDATION_MASK_COUNT` alongside the video/masklet counts), since video and masklet totals
  alone accept a source truncated to a few annotated frames per masklet and would silently
  evaluate a different corpus. Both the organizer's staged-mask check and the readiness check
  require every non-zero mask value to be one object ID: counting unique values alone accepts a
  `{1, 2}` mask that `> 0` binarization turns entirely into foreground. Readiness checks geometry
  and values for *every* cached mask, not the first per video. The official split is 1-bit
  bilevel, a format that cannot encode a non-zero background, so those masks are validated from
  the header and only non-bilevel masks are decoded -- complete validation at header cost
  (~3s for 31967 masks; decoding them all would be ~380s on every `val`).
- SA-V video ids come from `sav_val.txt` file contents rather than a directory listing, so
  `construct_sav` must reject ids failing `SAV_VIDEO_ID_PATTERN` and must confirm every staged
  path resolves inside the staging tree, both *before* any `makedirs`/`copy`. COCO applies the
  equivalent guard to its JSON-declared file names; organizers that build paths only from
  directory listings do not need it. The evaluation protocol (`eval_sav`) is ported from the
  validated `sam2-mxq-pipeline` reference: seed-deterministic area-balanced sampling, synthetic
  point prompts from the GT mask (distance-transform peak, dilated-mask negatives), and
  own-selection mean IoU as the primary metric with best-of-3 secondary. Numbers measured on
  SA-V val are a different protocol from that reference's sav_train-sampled 0.7757 and are not
  directly comparable.

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
  Do not add model weights, caches, or compiled build artifacts to source control. The one
  deliberate exception is `assets/`: a small fixed set of sample images kept in git as inputs for
  manual QA and the documented CLI examples. They are development-only and must never reach a
  distributed artifact -- only `mblt_vision*` packages are built, and `MANIFEST.in` prunes
  `assets` to pin that intent. Do not grow this directory for new one-off inputs, and do not
  reintroduce downloaded datasets, weights, or generated outputs under it.

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

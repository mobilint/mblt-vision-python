---
name: mblt-vision
description: >-
  Work on the standalone Mobilint Vision Python API, model registry, preprocessing,
  postprocessing, results, runtime integration, and package compatibility contracts.
---

# Mobilint Vision Python

## Start Here

1. Read AGENTS.md.
2. Run git status --short before changing files.
3. Read pyproject.toml, the affected package exports, matching model YAML, and relevant tests.
4. For a compatibility migration, compare against
   ../mblt-model-zoo/mblt_model_zoo/vision deliberately; do not make it a runtime dependency.

## Public API and Model Registry

- Use mblt_vision.MBLT_Engine and task subpackages as the public surface.
- Keep mblt_vision as the sole intended import namespace. Use obb as the sole
  oriented-bounding-box task name.
- Update a task package, top-level lazy exports, and list_models() discovery together.
- Preserve constructor arguments including model_path, mxq_path, onnx_path,
  model_type, and core-selection options unless intentionally changing the API.
- Keep .mxq/.onnx suffix routing and explicit-framework conflict errors intact.
- Every model YAML must define stable file_cfg, pre_cfg, and post_cfg mappings.
  Use file_cfg.filename for MXQ and derive the same-stem ONNX artifact unless
  onnx_filename is required.
- Every post_cfg declares dataset; resolve output taxonomy from the dataset/task pair.

## Processing and Results

- Reuse the shared letterbox geometry for both preprocessing and inverse coordinate restoration.
- Detection requires pre_cfg.LetterBox. Keep semantic metadata (img0_shape and
  ratio_pad) through postprocessing so logits restore to the original geometry before
  argmax.
- Preserve decoded-output layout provenance through NMS. For ambiguous tensors without
  provenance, prioritize channels-first raw-output normalization.
- Normalize dense depth and semantic outputs before inverse letterboxing. Validate baked semantic
  maps are finite, integral, and in-range before converting them to integer class IDs.
- Keep result shapes, ordering, coordinates, dtype, and empty-result behavior compatible with
  the Model Zoo reference.
- Rank WiderFace evaluation by Hard-set AP. Expose Medium-set then Easy-set AP
  as secondary metrics, and do not compute mean AP across difficulty splits.
- Rank NYU Depth evaluation by delta1. Expose abs_rel then RMSE (m) as
  secondary metrics, with median-aligned metrics averaged per image.

## Runtime and Packaging

- Route NPU runtime access through mblt-npu-python; do not copy backend classes into Vision.
- Use the shared `ONNXBackend` for ONNX inference. Keep ONNX Runtime optional and lazy-imported;
  raise a specific installation error when it is requested but unavailable.
- Normalize legacy `aries` and `regulus` target values through mblt-npu-python. MXQ artifacts and
  compilation metadata must resolve only from the selected board folder, never a core-mode path or
  a fallback board folder.
- Include model and dataset YAML files as package data. Build a wheel and inspect it after
  changing metadata or assets.
- Do not require native bindings, GStreamer, hardware, downloaded models, or caches for normal
  imports and unit tests.

## Tooling Layout and Documentation

- Keep all executable benchmark scripts directly in `benchmark/`; reusable reporting helpers belong
  in `mblt_vision.benchmark`.
- Keep all executable compile scripts and the compile guide directly in `compile/`.
- Use `~/.mblt_model_zoo` as the shared artifact and dataset cache root. Keep organizer defaults,
  dataset registry YAMLs, compilation defaults, and documented commands aligned to it.
- Keep imports free of cache-directory creation, write probes, downloads, and temporary-directory
  allocation; resolve a writable cache only when an artifact or compilation output needs it.
- Make fallback caches stable, private, and user-owned. Never use a new temporary directory per
  process or trust a shared fallback cache without validating it.
- For every significant package change (public API, CLI, runtime/dependency, artifact layout, or
  tooling structure), update `AGENTS.md`, this canonical skill, the Claude skill entry point when
  its workflow changes, and the relevant README in the same change.

## Validate Proportionately

- Begin with the smallest relevant test file or -k selection.
- Add deterministic differential tests for Model Zoo compatibility, including invalid inputs,
  empty detections, threshold boundaries, task discovery, and image geometry.
- Run pre-commit run --files <touched files> when available. For docs, run
  git diff --check.
- Report unavailable hardware, downloads, or optional dependencies rather than weakening tests.

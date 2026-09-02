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
- `face_detection` is a single-class WiderFace task, not an 80-class COCO one. Each YOLO head
  family reuses its own object-detection decode and NMS through a thin
  `YOLOFaceDetectionMixin` subclass (`YOLOAnchorFaceDetectionPost`,
  `YOLOAnchorlessFaceDetectionPost`, `YOLODFLFreeFaceDetectionPost`,
  `YOLONMSFreeFaceDetectionPost`); only evaluation-format conversion differs, because
  `nmsout2eval_face` labels every row `"face"` and rejects any class index other than `0`
  instead of routing indices through the COCO category-id table. `build_postprocess` therefore
  dispatches `face_detection` on its own branch, ahead of `object_detection`, using the same
  `anchors` / `dflfree` / `nmsfree` `post_cfg` keys. The anchor-based branch serves the
  `YOLOv5*-face` (deepcam-cn) and `YOLOv7*-face` (derronqi) families, whose published ONNX
  exports emit three raw `(batch, 3, H, W, 6)` heads with the original repositories' five
  landmark pairs stripped, so they decode through the shared anchor path with `nc = 1`.
- Take face-detection `pre_cfg`/`post_cfg` defaults from
  `../mblt-model-ops/models/<Model>/pipeline.yaml`, which is the source of truth for the
  compiled artifacts. Face-detection input geometry is `640x640` for every shipped model except
  `YOLOv8m-face` and `YOLOv8l-face`, which are `960x960` because those checkpoints' own embedded
  `train_args` record `imgsz: 960`. The anchor-based families additionally use `iou_thres: 0.5`
  where every other face model uses `0.7`. Do not normalize the exception away; a size change here is a
  durable model-behavior change requiring the guide, both skill copies, and
  `mblt_vision/README.md` to be updated in the same commit.
- `eval_sav` requires already-binarized candidate masks, enumerated per dtype (bool, integer
  `{0, 1}`/`{0, 255}`, float `{0.0, 1.0}`). A weaker "single positive value" rule still admits a
  probability map such as `{0.0, 0.5}`, or a uniform `0.5` candidate that `astype(bool)` turns
  entirely into foreground. This is deliberately stricter than the SA-V ground-truth mask check,
  where any single positive value is a legitimate object ID.
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
  ignored for ONNX. The runtimes have different graph contracts, pinned in
  `_sam2_contracts.py` and validated at construction. Two MXQ decoder generations exist and
  both are supported, identified from the loaded artifact's declared input shapes
  (`detect_decoder_contract`, run while the engine is built so a drifted artifact fails before
  any inference is spent): the *assembled* contract takes six host-flattened positional inputs
  (tokens pre-concatenated, `image_embeddings + dense` pre-summed; four outputs), while the
  *bridged* contract -- the SDK tutorial's legacy-parser decoder, whose MBLT carries that
  assembly as an in-graph host-bridge subgraph -- takes the prompt encoder's raw outputs
  (two outputs: `classify_decoder_outputs` keeps `masks`/`iou` required and treats
  `sam_tokens`/`object_score` as optional). The uploaded ONNX graphs use the same direct-MBLT boundary as the SDK MXQs: batched NHWC encoder input `input_image_0`, NHWC FPN outputs, and six named decoder inputs (`image_embeddings`, `dense_prompt_embeddings`, `image_pe`, `sparse_prompt_embeddings_0`, `high_res_features0_0`, `high_res_features1_0`) with a dynamic prompt axis. Both frameworks use the raw prompt-encoder feed; ONNX supplies it by name and MXQ supplies it positionally. The ONNX
  pipeline is numerically verified against the official `facebookresearch/sam2` fp32 predictor
  (identical binary masks; opt-in `tests/test_mask_generation_onnx.py` covers it end-to-end
  without NPU hardware); the bridged MXQ contract is numerically verified on real hardware by
  the opt-in `test_sam2_bridged_contract_predicts_numerically` (ground-truth mask IoU across
  1-3-point prompts, artifacts supplied via `MBLT_VISION_SAM2_BRIDGED_{ENCODER,DECODER}_MXQ`),
  and `eval_sav` works unchanged for both frameworks. Resolve the optional runtime before
  downloading any artifact, so a missing `onnxruntime` reports the package extra rather than a
  network failure. Build each backend so it disposes itself when `launch()`/graph validation
  fails after `create()`: the caller only assigns it on success, so the constructor's cleanup
  cannot otherwise reach it. Suppress disposal failures while unwinding (both in those builders
  and in the constructor, which uses `_close(suppress_errors=True)` like the base engine) so a
  failing `dispose()` cannot replace the original construction error. Validate every explicitly
  supplied artifact path -- prompt weights included -- with a fail-fast `FileNotFoundError` after
  the argument-coherence checks but before any download, and normalize the MXQ `target_device`
  there too so an unknown board reports as such rather than as a Hub failure. Lowercase an
  explicit `framework` before validating it, matching `_model_paths.resolve_framework`, so
  `framework="ONNX"` behaves as it does for every other model. Mask-generation-only CLI overrides
  (`--encoder-*-path`, `--decoder-*-path`, `--prompt-weights-path`) must be rejected by every
  command that builds a non-mask engine, since the generic engine never receives them and would
  silently run the downloaded default instead. ONNX graph validation treats `-1` as
  "this axis must be declared dynamic" (ONNX Runtime reports a dynamic axis as a `str`), not as a
  wildcard: a decoder frozen at one token count would otherwise pass construction and fail inside
  ONNX Runtime for two- and three-point prompts. Point prompts are validated as 1-3 points with finite
  coordinates and labels of exactly `1`/`0` (checked before the integer cast, which would
  otherwise truncate `0.5` into a valid label) before any backend call, and host prompt tensors
  are built on the weights' device so `device="cuda"` works. `original_hw` is likewise validated
  as exactly two positive whole numbers, and the source image as numeric and finite, before
  either backend runs: interpolation preserves NaN, and a zero dimension yields infinite prompt
  coordinates. Both FPN converters pin each level's complete `(C, H, W)` rather than only its
  channel count, since `build_backbone_features` `view()`s them and a same-element-count geometry
  would be silently rearranged into a corrupted FPN that still passes every downstream check. `classify_decoder_outputs` requires
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
  entry `datasets/sa-v.yaml` does not download it: Meta distributes SA-V through a form-gated
  portal, and it must never be mirrored on Mobilint infrastructure. Users supply the official
  `sav_val.tar` (or its extracted directory) via `--annotation-dir`/`--image-dir`, exactly as
  Cityscapes requires its manual archives; `_resolve_sav_source` fails with the portal link and
  the SAM 2 `sav_dataset/README.md` layout reference rather than falling back to a URL. The
  archive is deliberately absent from `PINNED_ARCHIVE_SHA256` because it is user-supplied rather
  than fetched from a URL this package controls, so identity is enforced on content by the
  readiness inventory instead. SA-V is CC BY 4.0 by Meta AI. The
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

- The unified benchmark runner's `TASK_CHOICES` is the subset of `VISION_TASKS` it can actually
  execute. `mask_generation` is excluded because `_run_target` builds a generic `MBLT_Engine` and
  `_evaluate` has no `eval_sav` branch; benchmark it with `mblt-vision val` instead. Do not wire a
  new canonical task into the runner's choices before its engine and evaluator paths exist.
- `classify_decoder_outputs` pins the decoder mask layout to `(N, 65536)` or `(N, 256, 256)`,
  optionally batched, before reshaping. A channels-last `(1, 256, 256, 3)` output has a matching
  element count and candidate count, so only the layout check catches it; without it the reshape
  interleaves the candidates into plausible but corrupted masks.
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

---
description: Guidance for coding agents working on the PyPI-distributed mblt-vision Python API.
paths:
  - "**"
---

# mblt-vision-python Agent Guide

This is the one guide for every coding agent: `CLAUDE.md` is a symlink to this file. For
focused model, preprocessing, postprocessing, and model-registry work, also read
`.claude/skills/mblt-vision/SKILL.md`.

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
- Keep a timm classifier's pre_cfg identical to the same model's pipeline.yaml in
  mblt-model-ops, which derives it from the timm id's default pretrained configuration.
  Resize.size there is floor(input_size / crop_pct) — timm's own transforms_factory
  floors, and rounding differs by a pixel at 224/0.9 and 224/0.95. Several names
  (ConvNext_Base) resolve in timm yet carry torchvision's transform here, so read the
  expected value from that model's source.yaml provenance rather than from the name.
- Require post_cfg.dataset in every model YAML and resolve output class counts using the
  dataset/task pair. Do not assume one output taxonomy for every model in a task.
- Preserve automatic .mxq/.onnx framework detection and the fail-fast error when a local
  suffix conflicts with an explicitly selected framework.
- Preserve anchorless decoded-output layout provenance through NMS. When a tensor is ambiguous
  and provenance is unavailable, normalize it as raw channels-first before candidates-first.
- Rank every NMS and end-to-end candidate with `common.descending_order`, a stable descending
  sort, never a plain `argsort(descending=True)` or `torch.topk`. That is Ultralytics' validation
  order: `non_max_suppression` passes up to 30000 candidates straight to `torchvision.ops.nms`,
  which sorts stably on CPU and CUDA, and the `argsort` above that cap and an end-to-end head's
  `torch.topk` (at `k = 300`) are stable on CUDA, where it validates. On CPU those calls order
  tied scores differently, and quantized MXQ scores tie often, so the unstable sort this rule
  used to require made mblt-vision's CPU results diverge from Ultralytics; the earlier
  measurement (YOLOv8m -0.00047, YOLOv8m-pose -0.0038, YOLOv8m-seg +0.00029 mAP50-95 for the
  stable sort) compared two tie orders, not either against Ultralytics.
  `tests/test_ultralytics_selection_order.py` pins the order. `non_max_suppression` suppresses
  only an IoU *above* `iou_thres`, as `torchvision.ops.nms` does: two zero-area boxes give a NaN
  IoU, which must keep the candidate, not drop it.
- `dual_topk` reproduces `Detect.get_topk_index`: it may rank only the anchors that clear the
  confidence threshold, but its second stage still returns up to `max_det` (anchor, class) pairs.
  Capping that stage at the surviving anchor count dropped every further class of those anchors
  whenever fewer than `max_det` of them cleared the threshold.
- Rank ImageNet predictions once with a stable sort and take top-1 as the head of top-5, so
  tied scores favour the higher class index as in mblt-model-ops' evaluator, and top-1 is
  always inside top-5.
- Decode-true pose MXQ parts carry keypoint visibility as logits, so the anchorless and
  DFL-free pose paths apply the sigmoid; ONNX rows already carry it in-graph and are left as
  is. Evaluation conversions such as `nmsout2eval_pose` must copy before rescaling in place,
  so the caller's NMS rows stay usable for rendering.
- Use the shared letterbox helpers for forward geometry and inverse output restoration. Detection
  postprocessors require pre_cfg.LetterBox; metadata-aware semantic preprocessing returns the
  original image shape and ratio_pad so logits can be restored before argmax.
- Normalize dense outputs before inverse letterboxing: upsample quarter-resolution depth maps by
  four, preserve baked-resize depth maps, convert Cityscapes NHWC logits to NCHW, and reject
  non-finite, fractional, or out-of-range baked semantic IDs before casting.
- Keep hardware-specific runtime access behind mblt-npu-python. Optional ONNX Runtime imports
  must remain lazy and report the appropriate package extra when unavailable.
- YOLOX and DAMO-YOLO are object-detection families with non-Ultralytics heads, selected by
  `post_cfg.head: yolox` / `damoyolo` (`build_postprocess` rejects any other value, and any
  `head` on another task). Both reuse the anchorless candidate filter, NMS and inverse
  letterbox; only the decode differs. YOLOX takes one `(batch, anchors, 5 + nc)` tensor
  (upstream `decode_in_inference = False`): `xy = (raw + grid) * stride`,
  `wh = exp(raw) * stride`, score = objectness x class. DAMO-YOLO takes six per-level maps --
  sigmoid class maps and `4 * (reg_max + 1)`-channel distributions, so `reg_max: 16` is 17
  bins -- decoded as the softmax expectation times stride. Neither adds Ultralytics' half-cell
  offset. A 640 input's stride-8 DAMO class map is 80x80x80, so the head set's layout (NCHW
  or NHWC) is resolved jointly from the unambiguous distribution maps and a mixed set fails.
- Their `pre_cfg` is upstream's own test transform, which is the source of truth (their
  `post_cfg` follows mblt-model-ops' `models/<Model>/pipeline.yaml` on branch `jm/temp`,
  commit `0368367e8`). YOLOX is `yolox/data/data_augment.py:preproc`: a cv2 BGR image
  (`Reader.color_mode: BGR`), resized to `int(w * r) x int(h * r)`, top-left on a 114 canvas,
  boxes restored by `/ r`. DAMO-YOLO is the December 2022 release's `Resize` and
  `to_image_list` (tinyvision/DAMO-YOLO `55ae14f`; not upstream HEAD's stretch): a PIL RGB
  decode (`Reader.style: pil`), the same truncated resize, top-left with zeros, and boxes
  restored per axis by `BoxList.resize`. COCO evaluation decodes with the same library as the
  model's `Reader` (`CustomCOCODataset(decoder="pil")` for `style: pil`), because the loader
  decodes before preprocessing. Hence `LetterBox.size_rounding: floor` on both and
  `per_axis_ratio: true` on DAMO-YOLO; tests compare both pipelines pixel-for-pixel with
  inlined copies of the upstream code. Neither declares `Normalize`: both take the unscaled
  0-255 image, and the ONNX path casts the byte tensor to the graph's float dtype. Every place
  that derives geometry from shapes alone (`PostBase.ratio_pads_for`, `Results` plotting,
  dense crops, semantic targets) must use the model's own `letterbox_layout(pre_cfg)`
  rather than Ultralytics' default. Instance segmentation rejects `size_rounding: floor` and
  `per_axis_ratio`, and OBB rejects `per_axis_ratio`, because their restorations cannot honour
  them. No Hub repository exists yet, so their YAMLs keep `file_cfg.local_artifact_only: true`.
- For WiderFace evaluation, rank results by Hard-set AP and retain Medium-set
  then Easy-set AP as secondary metrics. Do not compute a mean across splits.
- The YOLOv5/YOLOv7 face repositories do not yet publish project-pinned immutable revisions
  and artifact SHA-256 digests. Their YAMLs must keep `file_cfg.local_artifact_only: true`, and
  callers must supply a trusted local MXQ or ONNX path; never restore automatic Hub downloads
  using `main` or `TURBO`.
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
- For NYU Depth evaluation, rank results by delta1 and retain abs_rel then
  RMSE (m) as secondary metrics. Median-align each image and average every
  metric per image, following Ultralytics' depth-validation convention.
- For DOTAv1 evaluation, rank results by rotated mAP50 and retain mAP50-95 as the secondary
  metric, because rotated mAP50 is what Ultralytics publishes for its OBB models. Load difficult
  objects (flag `1` or `2`) as ordinary targets, as Ultralytics' validation does: its
  `convert_dota_to_yolo_obb` drops the flag. `evaluate_dota_predictions` still honours ignore
  regions a caller supplies explicitly (the DOTA devkit protocol), but the loader produces none.
- Run YOLOv7 models (`YOLOv7`, `-x`, `-w6`, `-e6`, `-d6`, `-e6e`) at `iou_thres: 0.65`, the
  value WongKinYiu/yolov7's `test.py --iou-thres` defaults to and its README reproduces every
  published COCO number with. The `iou_thres=0.6` in that script's `test()` signature is always
  overridden by the command line.
- Reuse `wrapper.download_hub_artifact` (extracted from `MBLT_Engine._download_hub_artifact`) for
  any future model needing more than one Hub artifact, rather than duplicating Hub-resolution
  logic.

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

- The unified benchmark runner's `TASK_CHOICES` must only list tasks it can actually execute. Do not
  wire a new canonical task into the runner's choices before its engine and evaluator paths exist.
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

## Postprocess and Scoring Performance

- Postprocessing and metric scoring are a real share of an evaluation run, and several
  hot paths here are shaped for cost rather than only for clarity. Keep those shapes:
  `rotated_nms` tiles pairwise probIoU in `ROTATED_NMS_BLOCK`-wide blocks and drops a
  column as soon as any higher-scored candidate suppresses it, which preserves the
  Fast-NMS verdict while bounding every intermediate at `ROTATED_NMS_BLOCK ** 2` — the
  single-matrix form needs tens of gigabytes at the candidate caps a dense OBB head
  reaches. `to_string` encodes all RLE counts in lockstep instead of one Python loop per
  count, which is 5x on a 300-mask image. `_match_predictions` sorts and filters once and
  compares the matched IoU against the threshold vector, because the candidate set at a
  higher threshold is a prefix of the sorted set at the lowest one. `eval_widerface`
  scores Easy, Medium and Hard from one pass, because the IoU matrix does not depend on
  which faces a setting ignores.
- A speed change to any of these must be shown to keep the metric, not argued to: run the
  old and the new implementation on the same randomized inputs, degenerate cases included
  (no predictions, no annotated targets, tied scores, unsorted scores, empty count lists),
  and require identical output. Then measure. Not every port lands: compacting the
  candidate tensors inside `non_max_suppression` is 1.8x in numpy and measured ~2x
  *slower* here, because six boolean index operations per iteration cost more in torch
  than one gather through a shrinking index; that measurement is recorded in the function.
- Every `eval_*` evaluator runs through
  `utils/evaluation/_pipeline.map_batched_inference`, which prepares batches and runs
  postprocessing on two separate bounded thread pools while inference stays on the consuming
  thread, following `mblt-model-ops`'s helper of the same name. Results come back in input
  order. Each evaluator's `decode` callback must be a pure function of its batch and raw
  outputs; accumulate metrics only in the consuming loop, and keep that accumulation
  order-exact (NYU adds per-image `image_metrics` in sample order; semantic segmentation
  sums per-batch integer confusion matrices). A single-process `DataLoader` is fetched per
  batch on the preparation pool, so datasets, collate functions, and preprocessing must be
  thread-safe: `PreBase.with_metadata` takes `ratio_pad` from `LetterBox.with_ratio_pad`'s
  return value, because the legacy `LetterBox.ratio_pad` attribute is shared per instance.
  Do not share one pool between the two stages, because a slow decode would starve the
  prefetch that feeds the NPU.
- `mblt-model-ops`'s `datasets/*/evaluator.py` are the counterparts of
  `mblt_vision/utils/evaluation/eval_*.py`, and nothing compares them automatically. A
  scoring change on one side is owed to the other, verified separately on each, since the
  two do not share conventions everywhere. When a release carries such work, bump
  `mblt_vision.__version__` so the other repository's parity pin can move to it.

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
- When a durable public fact changes, update this guide, the matching agent skill, and the
  relevant README in the same change. Each guide and skill has one real copy, the way
  mblt-model-ops lays them out: `CLAUDE.md` is a symlink to this `AGENTS.md`, and every
  `.agents/skills/<name>` is a symlink to `../../.claude/skills/<name>`, the real directory.
  Edit the real file, never replace a symlink with a copy, and link a new skill the same way
  (`ln -s ../../.claude/skills/<name> .agents/skills/<name>`). Treat a significant package change—public API,
  dependency/runtime, artifact layout, CLI, or tooling structure—as a required guide-and-skill
  synchronization point.
- For documentation-only changes, run `git diff --check` and verify headings and links. Report
  skipped platform, hardware, or native-runtime checks clearly.

## Git Safety

- Do not alter `../mblt-vision` or `../mblt-model-zoo` as an incidental change in this repository.
- Keep commits focused. Do not commit virtual environments, wheelhouse contents, caches, native
  build directories, downloaded models, or generated coverage/benchmark files.

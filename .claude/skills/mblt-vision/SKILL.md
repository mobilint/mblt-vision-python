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
- Reuse wrapper.download_hub_artifact for any model that needs more than one Hub artifact
  rather than duplicating Hub-resolution logic.
- Registering a new dataset requires readiness (`_*_ready` + `dataset_ready`
  map) before the organizer, since staged validation calls `dataset_ready`; also register the
  organizer in the test_dataset_organizer.py parametrize lists.

## Processing and Results

- Derive a timm classifier's pre_cfg from the timm id's default pretrained configuration,
  and size the pre-crop resize as floor(input_size / crop_pct) — timm's own
  transforms_factory floors, and rounding differs by a pixel at the two commonest
  settings (224/0.9, 224/0.95), which shifts every interpolated pixel. Several names
  (ConvNext_Base) resolve in timm yet carry torchvision's transform here, so read the
  expected value from that model's source.yaml provenance in mblt-model-ops rather than
  from the name. Keep the value identical to that model's pipeline.yaml.
- Reuse the shared letterbox geometry for both preprocessing and inverse coordinate restoration.
- Detection requires pre_cfg.LetterBox. Keep semantic metadata (img0_shape and
  ratio_pad) through postprocessing so logits restore to the original geometry before
  argmax.
- Preserve decoded-output layout provenance through NMS. For ambiguous tensors without
  provenance, prioritize channels-first raw-output normalization.
- Keep NMS candidate sorts on Ultralytics' unstable `argsort(descending=True)` (a stable sort
  shifts MXQ mAP through tied scores; see AGENTS.md), and suppress only IoU above the threshold
  so NaN from zero-area boxes keeps the box, as torchvision does. Rank ImageNet top-1/top-5 from
  one stable sort. Decode-true pose MXQ visibility is sigmoided; copy NMS rows before any
  in-place rescale for evaluation.
- Normalize dense depth and semantic outputs before inverse letterboxing. Validate baked semantic
  maps are finite, integral, and in-range before converting them to integer class IDs.
- Keep result shapes, ordering, coordinates, dtype, and empty-result behavior compatible with
  the Model Zoo reference.
- Treat the shape of the postprocess and scoring hot paths as load-bearing: `rotated_nms`
  tiles probIoU and drops suppressed columns (bounding every intermediate at
  `ROTATED_NMS_BLOCK ** 2`, which is what makes a 30000-candidate OBB head feasible),
  `to_string` encodes all RLE counts in lockstep, `_match_predictions` sorts once and
  compares the matched IoU against the threshold vector, and `eval_widerface` scores all
  three difficulty settings from one IoU pass. Prove any speed change output-identical
  against the implementation it replaces on randomized and degenerate inputs, then
  measure — and mirror it into `mblt-model-ops`'s matching `datasets/*/evaluator.py`,
  which no test compares against these. See AGENTS.md, Postprocess and Scoring Performance.
- Evaluators run through `map_batched_inference`: inference on the consuming thread,
  batch preparation and postprocessing on separate bounded pools, results in input order.
  Keep each `decode` callback pure, accumulate only in the consuming loop, and keep
  preprocessing thread-safe (read `ratio_pad` from `LetterBox.with_ratio_pad`, never from
  the shared instance attribute).
- YOLOX and DAMO-YOLO dispatch on `post_cfg.head` (`yolox` / `damoyolo`) and reuse the
  anchorless filter and NMS; only their decode is new (no half-cell offset; DAMO `reg_max: 16`
  is 17 bins; DAMO head layout resolved jointly). Upstream's test transform is the source of
  truth for `pre_cfg`: both letterbox top-left (`LetterBox.center: false`, padding 114 / 0)
  with upstream's truncated `int(w * r)` size (`size_rounding: floor`) and take unscaled input
  (no Normalize); YOLOX reads cv2 BGR (`Reader.color_mode`) and restores boxes by `/ r`;
  DAMO-YOLO reads PIL RGB (`Reader.style: pil`) and restores each axis separately
  (`per_axis_ratio: true`). Shape-only geometry must follow `letterbox_layout(pre_cfg)`.
  `post_cfg` follows mblt-model-ops `jm/temp` `pipeline.yaml`; YAMLs are `local_artifact_only`.
- Rank WiderFace evaluation by Hard-set AP. Expose Medium-set then Easy-set AP
  as secondary metrics, and do not compute mean AP across difficulty splits.
- Treat face_detection as a single-class WiderFace task. Each YOLO head family gets a thin
  YOLOFaceDetectionMixin subclass over its object-detection postprocessor, so only
  evaluation-format conversion differs: nmsout2eval_face labels every row "face" and rejects a
  class index other than 0 rather than using the COCO category-id table. build_postprocess
  dispatches face_detection before object_detection on the same anchors/dflfree/nmsfree keys.
  The anchor branch serves the YOLOv5*-face and YOLOv7*-face families, whose ONNX exports emit
  three raw (batch, 3, H, W, 6) heads with landmarks stripped and use iou_thres: 0.5.
- Keep YOLOv5/YOLOv7 face artifacts local-only until their YAMLs can pin both an immutable Hub
  revision and SHA-256 digest; do not automatically download those families from symbolic `main`
  or `TURBO` revisions.
- Source face-detection pre_cfg/post_cfg from ../mblt-model-ops/models/<Model>/pipeline.yaml.
  Every shipped face model is 640x640 except YOLOv8m-face and YOLOv8l-face at 960x960, whose
  checkpoints record imgsz: 960 in their own train_args. Changing an input size is a durable
  model-behavior change: update AGENTS.md, both SKILL.md copies, and mblt_vision/README.md
  together.
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
  changing metadata or assets. `assets/` holds development-only sample images: tracked in git by
  deliberate exception, pruned from distribution via MANIFEST.in, never grown for one-off inputs.
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

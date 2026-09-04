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
- A promptable or multi-artifact model (see mask_generation/SAM2HieraLarge) bypasses
  MBLT_Engine.__init__, build_preprocess/build_postprocess, and create_model_class entirely,
  implementing its own preprocess/predict methods; it still subclasses MBLT_Engine only for
  list_models() discovery. Reuse wrapper.download_hub_artifact for any additional Hub artifact
  rather than duplicating Hub-resolution logic.
- mask_generation supports framework="mxq" (default) and framework="onnx" with
  MBLT_Engine-style inference/conflict semantics (explicit encoder/decoder path suffixes infer
  the framework; NPU-only arguments are ignored for ONNX). The ONNX exports are same-stem
  sam2_hiera_large_{encoder,decoder}.onnx at the Hub repo root (board-agnostic). Keep the
  graph contracts pinned in _sam2_contracts.py and validated at construction: two MXQ decoder
  generations exist, identified from the artifact's declared input shapes at engine
  construction (detect_decoder_contract) -- assembled (six host-flattened inputs, four
  outputs) and bridged (prompt-encoder raw inputs, two outputs; sam_tokens/object_score are
  optional in classify_decoder_outputs); the ONNX graphs use the direct-MBLT NHWC encoder
  and six named decoder inputs with a dynamic token axis. Keep the prompt-encoding host path and
  classify_decoder_outputs framework-independent. ONNX uses fpn_from_runtime and
  prepare_decoder_tensors_bridged; MXQ uses fpn_from_runtime plus the detected decoder builder.
  eval_sav works unchanged for both.
  Load the optional runtime before downloading artifacts; validate explicit artifact paths
  (prompt weights included) with FileNotFoundError before any download; dispose a backend that
  fails after create() inside its builder (the caller assigns it only on success) while
  suppressing dispose failures so they cannot mask the original error; treat -1 in ONNX graph
  validation as "must be dynamic", not a wildcard; validate point labels as
  exactly 1/0; build host prompt tensors on the weights' device so device="cuda" works; and
  reject single-artifact path options in every CLI command that builds the engine, not just
  predict.
- Never depend on the PyPI `sam2` package (unofficial third-party mirror) or a manually cloned
  facebookresearch/sam2 checkout. mask_generation's host-side prompt encoding is a from-scratch
  port verified bit-for-bit against the real predictor, backed by a small Hub-hosted weights
  bundle, not package data.
- mask_generation validates on SA-V val via datasets/sa-v.yaml. SA-V is NOT auto-downloaded and
  must never be mirrored on Mobilint infrastructure: Meta gates it behind a download form, so the
  user supplies the official sav_val.tar (or its extracted directory) with
  --annotation-dir/--image-dir, like Cityscapes. Not sha256-pinned (user-supplied, not fetched);
  identity comes from the readiness inventory. CC BY 4.0 by Meta AI. Readiness pins all three
  inventory counts (155 videos / 293 masklets / 31967 masks); video and masklet totals alone
  accept a truncated source. Both the organizer and readiness require every non-zero mask value
  to be one object ID, since `{1, 2}` survives a unique-value count but `> 0` binarization makes
  it all foreground. Readiness validates every mask (not just the first per video); bilevel masks
  are validated from the header, since a 1-bit PNG cannot hold a non-zero background. Reject
  `sav_val.txt` ids that fail SAV_VIDEO_ID_PATTERN or escape staging, before any write. Registering a new dataset requires readiness (`_*_ready` + `dataset_ready`
  map) before the organizer, since staged validation calls `dataset_ready`; also register the
  organizer in the test_dataset_organizer.py parametrize lists.

## Processing and Results

- Reuse the shared letterbox geometry for both preprocessing and inverse coordinate restoration.
- LetterBox covers several conventions through `center`, `keep_ratio` and `padding_value`:
  Ultralytics centres and pads with 114 (default), YOLOX pads with 114 at the top-left
  (`center: false`), DAMO-YOLO pads with 0 at the top-left, and `keep_ratio: false` stretches
  with no padding (scaling the axes differently, so the geometry reports `ratio_xy`). Match a
  model's geometry to its checkpoint's era, not to upstream HEAD — for DAMO-YOLO that is worth
  two points of mAP. Derive an inverse from the model's own pre_cfg flags, never from image
  shapes alone.
- A pipeline may declare no Normalize step. Normalize always divides by 255, and YOLOX and
  DAMO-YOLO take unscaled 0-255 input; the ONNX path casts the byte tensor to the dtype the
  graph declares. Reader.color_mode picks the channel order (RGB default, BGR for YOLOX).
- YOLOX and DAMO-YOLO decode through `post_cfg.yolox` / `post_cfg.damoyolo`, dispatched ahead of
  anchors/dflfree/nmsfree. Both drop the Ultralytics half-cell anchor offset; DAMO-YOLO's
  `reg_max` counts the largest distance, so 16 means 17 distribution bins. Only the conversion
  to candidates is theirs — filtering, xywh2xyxy, NMS and the inverse letterbox are inherited.
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
- Treat face_detection as a single-class WiderFace task. Each YOLO head family gets a thin
  YOLOFaceDetectionMixin subclass over its object-detection postprocessor, so only
  evaluation-format conversion differs: nmsout2eval_face labels every row "face" and rejects a
  class index other than 0 rather than using the COCO category-id table. build_postprocess
  dispatches face_detection before object_detection on the same anchors/dflfree/nmsfree keys.
  The anchor branch serves the YOLOv5*-face and YOLOv7*-face families, whose ONNX exports emit
  three raw (batch, 3, H, W, 6) heads with landmarks stripped and use iou_thres: 0.5.
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

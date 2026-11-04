# Mobilint Vision API

`mblt-vision-python` provides Python access to pre-trained Mobilint Vision models:
image classification, depth estimation, face and object detection, oriented bounding
boxes (OBB), instance and semantic segmentation, and pose estimation. Each model
configuration includes the artifact, preprocessing, output taxonomy, and
postprocessing contract needed to produce task-specific results.

Breaking change in 0.0.5: SAM2 promptable mask generation, shipped in 0.0.3 and 0.0.4, has
been removed. The `mask_generation` task, `mblt_vision.mask_generation.SAM2HieraLarge`, the
`sam2-hiera-large` model, the `mblt-vision predict --point` and encoder/decoder path options,
and SA-V validation are no longer available. Pin `mblt-vision-python<0.0.5` to keep using them.

## Loading models

Use a task subpackage for a known model, or `MBLT_Engine` to choose a model and
artifact at runtime. These imports are both supported:

```python
from mblt_vision import ResNet50
from mblt_vision import YOLO11m
```

```python
from mblt_vision import MBLT_Engine

model = MBLT_Engine(model_cls="resnet50", model_type="DEFAULT", model_path="", core_mode="global8")
```

```python
from mblt_vision import MBLT_Engine

# Install the optional `onnxruntime` or `onnxruntime-gpu` extra first.
model = MBLT_Engine(model_cls="alexnet", framework="onnx")
```

When `framework="onnx"` is selected, the engine uses ONNX Runtime's `CPUExecutionProvider` by
default. Pass `onnx_providers` to select another provider order explicitly when needed. If
`model_path` or `file_cfg.model_path` ends
with `.mxq` or `.onnx`, the engine auto-detects the framework from that suffix when `framework` is
omitted.

For Hub-backed models, `file_cfg.filename` is the canonical MXQ artifact name. The engine derives
the corresponding ONNX artifact by replacing its `.mxq` suffix with `.onnx`, so every model uses
the same ONNX loading path. Set `file_cfg.onnx_filename` only when a Hub repository uses a
non-matching ONNX filename.

Every model configuration also declares `post_cfg.dataset`. This identifies the output taxonomy,
not merely the broad task: COCO detection uses 80 classes, Cityscapes semantic segmentation uses
19, and ADE20K semantic segmentation uses 150. YOLO postprocessing resolves shape-sensitive class
counts from the dataset and task together.

```python
from mblt_vision.image_classification import ResNet50
from mblt_vision.object_detection import YOLO11m
```

The task subpackages are the clearest import surface; `MBLT_Engine`, `list_tasks()`,
and `list_models()` are the preferred discovery and loading APIs for new code.
Top-level model imports remain available for compatibility. `obb` is the sole
oriented-bounding-box task name.

## DOTA multiscale native tiles

The dataset-owned tiled protocol and development CLI
`benchmark/dota_tiled.py` reproduce the native-window and merged-export protocol
of **Ultralytics v8.3.228**, commit
`f8e6fddc89ec754b3a62eab87d5221e83058005a`. The DOTA OBB `predict` CLI and `eval_dota` now use this tiled pipeline by default;
model constructors, preprocessing configuration shapes, and raw `MBLT_Engine.__call__`
remain compatible. `--legacy-single-image` (or `eval_dota(...,
legacy_single_image=True)`) retains the historical whole-image path explicitly.
`TILED_API_VERSION = 1` identifies the backend-neutral callback contract.

This implementation has synthetic, CPU-only differential tests. It is not a
claim of reproducing published DOTA accuracy or improving measured mAP. Such a
claim additionally requires identical checkpoint and digest, data split,
training history, runtime precision, preprocessing, inference configuration,
class mapping, scoring protocol, and benchmark-time TTA settings. A version pin
alone does not establish which version produced a published table.

### Configuration and inherited postprocessors

`datasets/dotav1.yaml` owns the versioned `eval_protocol` block with id
`dota-native-multirate-1024-500.v1`. Its `tiling` mapping defines native crop
geometry and JPEG policy; `inference` defines `conf_thres: 0.01`,
`iou_thres: 0.7`, `max_det: 300`; `merge` selects classwise
`rotated_probiou_fastnms` with `iou_thres: 0.3`. Model `post_cfg.eval_protocol`
selects this id. The existing `TURBO: {update: DEFAULT}` inheritance carries the
selector; model mappings contain only genuine deviations from dataset defaults.
The shared model loader and its shallow section-merge behavior are unchanged.

`build_postprocess` selects `YOLOAnchorlessTiledOBBPost` or
`YOLODFLFreeTiledOBBPost`. Their `DOTATiledOBBPostMixin` extends the existing OBB
postprocessors, retaining their decode, per-tile NMS and `nmsout2eval` box
conversion. It adds only dataset-native `iter_tiles` and `merge_tiles` orchestration
and forwards the dataset's per-tile cap to inherited NMS. Normal postprocess calls
still return the established tile/network-space result contract; raw engine tensor
inference does not silently become image tiling. Prediction and evaluation bridge
through `utils/evaluation/eval_dota_tiled.py`, using that same model pre/post.
Explicit model thresholds and CLI threshold overrides remain supported.

Windows, crops, `TiledConfig`, `Tile`, `prepare_tile`, version constants and
`get_eval_protocol` live in `utils/datasets/dota_tiling.py`; restore/merge,
Task1 export and optional reference adapter live in `utils/postprocess/tiled_obb.py`.
There is no standalone public OBB engine. Completed images are merged before
retaining the next image's candidate rows; bounded preparation and decoding pools
surround inference on the consuming thread. Scoring stays in `utils/evaluation`.

### Pinned reference

Upstream sources carry the notice **Ultralytics AGPL-3.0 License —
<https://ultralytics.com/license>**. Preserve that notice and assess upstream
licensing when redistributing upstream code or checkpoints. This feature reuses
the package's existing geometry/NMS helpers and adapts the reference algorithm;
the optional reference adapter loads upstream code supplied separately.

All source links below are immutable at the pinned commit:

- [`data/split_dota.py:101–150`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/data/split_dota.py#L101-L150):
  `get_windows`, edge adjustment, x-major ordering, joint area-rate filtering.
- [`data/split_dota.py:298–339`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/data/split_dota.py#L298-L339):
  `split_test`, truncated crop/gap sizes, naming, direct native slices and JPEG
  writes. There is **no resize or pad in split_test**. `split_images` is not a
  function in this version; `split_images_and_labels` is the labeled splitter
  (lines 218–257), with `get_window_obj` (153–164) selecting labels by IoF.
- [`models/yolo/obb/val.py:163–190`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/models/yolo/obb/val.py#L163-L190):
  pre-merge JSON serialization rounds rboxes/polygons to 3 and scores to 5 decimals.
- [`models/yolo/obb/val.py:219–289`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/models/yolo/obb/val.py#L219-L289):
  inverse letterbox, offset-only merge, dynamic class offsets, Fast-NMS at 0.3,
  merged polygon/score rounding to 3 decimals. The upstream comment explicitly
  warns its probIoU merge can differ from the official DOTA merging script.
- [`data/base.py:208–257`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/data/base.py#L208-L257)
  and [`376–395`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/data/base.py#L376-L395):
  validation-loader long-side pre-resize uses `ceil` dimensions; metadata records
  resized/original height and width ratios before letterboxing.
- [`data/dataset.py:206–235`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/data/dataset.py#L206-L235):
  non-augmented validation adds `LetterBox(..., scaleup=False)` after that resize.
- [`engine/validator.py:123–125`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/engine/validator.py#L123-L125):
  default OBB validation confidence is 0.01, distinct from detection's 0.001.
- [`models/yolo/detect/val.py:115–125`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/models/yolo/detect/val.py#L115-L125):
  validation uses multi-label NMS and the configured IoU/max_det.
- [`utils/nms.py:12–169`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/nms.py#L12-L169):
  tile confidence selection, multi-label switch, class offset 7680, rotated NMS,
  `max_nms=30000` and per-tile `max_det`.
- [`utils/nms.py:187–243`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/nms.py#L187-L243):
  Fast-NMS, suppression when **any** higher-scored row has probIoU **>=** threshold,
  including rows which are themselves suppressed. This is not greedy NMS.
- [`utils/metrics.py:187–209`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/metrics.py#L187-L209)
  and [`251–289`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/metrics.py#L251-L289):
  Gaussian covariance and `batch_probiou`; not polygon intersection IoU.
- [`utils/ops.py:102–139`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/ops.py#L102-L139)
  and [`405–437`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/ultralytics/utils/ops.py#L405-L437):
  inverse letterbox for xywh without axis-aligned clipping, corner ordering.

The pinned [`docs/en/tasks/obb.md`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/docs/en/tasks/obb.md#L47)
recommends OBB validation on DOTAv1 `split=test`, followed by DOTA-server scoring.
Its [`YOLO OBB performance macro`](https://github.com/ultralytics/ultralytics/blob/f8e6fddc89ec754b3a62eab87d5221e83058005a/docs/en/macros/yolo-obb-perf.md)
reports YOLO11 n/s/m/l/x mAP50 of 78.4/79.5/80.9/81.0/81.3 for single-model
multiscale evaluation. These are upstream published results, not measurements of
this package. The historical exact checkpoint revision, confidence/IoU, TTA and
preprocessing settings behind those numbers are not established by this pin.

Methodology references (living documentation, not immutable experiment evidence):
[Ultralytics OBB](https://docs.ultralytics.com/tasks/obb/),
[DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v1/),
[DOTA benchmark](https://captain-whu.github.io/DOTA/).

### Geometry and data contract

`TiledConfig()` uses `crop_size=1024`, `gap=500`,
`rates=(0.5, 1.0, 1.5)`, `im_rate_thr=0.6`, `eps=0.01`, and
`jpeg_roundtrip=True`. These multiscale/gap settings are an explicit experiment
configuration: upstream `split_test` itself defaults to gap 200 and rate 1.0.

| Nominal rate | Native crop `int(1024/r)` | Gap `int(500/r)` | Stride |
| --- | --- | --- | --- |
| 0.5 | 2048 | 1000 | 1048 |
| 1.0 | 1024 | 500 | 524 |
| 1.5 | 682 | 333 | 349 |

Window starts are `stride*i`. Last starts shift back to touch the far edge when
necessary. Cartesian product is x-major, then y, in rate order. All rates are
concatenated **before** filtering: retain `image_area_in_window / window_area >
0.6`. If none passes, retain those within strictly less than `eps` of the
maximum by setting their rate to 1. Stops can extend beyond a small image; numpy
slicing clips them. No synthetic pixels are added at this stage.

`iter_tiles(image_bgr, image_id, config)` accepts nonempty **uint8 HWC BGR** numpy
arrays with three channels. Each `Tile.image` is an owned contiguous copy; the
source is not modified. With the default JPEG option, the slice is encoded and
decoded using OpenCV's default JPEG options, matching upstream's `.jpg` tile
write/read within the same codec build. This is lossy compression, not resizing.
Tile names are `P0001__682__349___0` (native nominal crop size, not rate).
`Tile.window` stores nominal `(x1,y1,x2,y2)` while `Tile.image.shape` stores the
actual clipped raster dimensions. EXIF orientation must already be normalized;
the CLI rejects nontrivial EXIF orientation rather than mixing displayed and
raster dimensions. Original stems containing `__`, whitespace, or path separators
are rejected to avoid ambiguous upstream naming/Task1 parsing.

A predictor implements `__call__(tile: Tile) -> numpy.ndarray | torch.Tensor` and
returns **Nx7 `[cx, cy, w, h, angle_radians, score, class0]`** after per-tile NMS,
in **native tile pixels**, already inverse-letterboxed. Empty means shape
`(0,7)`. Rows must be finite, sizes nonnegative, scores in `[0,1]`, classes
nonnegative integers. The merge creates owned CPU float32 tensors and never
mutates callback rows. It rejects mixing original image IDs in one merge call.

`prepare_tile(tile, imgsz=1024)` supplies the frozen v1 direct-letterbox
fixed-square input helper: centered aspect-preserving letterbox with rounded resized size,
INTER_LINEAR, pad 114, scaleup enabled, then BGR->RGB, contiguous NCHW float32
`/255` on CPU. It returns the input and shared `LetterBoxGeometry` metadata.
`restore_tile_boxes(network_rows, geometry)` subtracts center padding and divides
centers/sizes by the actual letterbox ratio once. Angles, scores and classes are
unchanged. It deliberately does not clip rotated centers or vertices. A runtime
may change tensor layout/dtype at its own boundary while keeping this geometry.

Merge rounds native rboxes to 3 decimals and scores to 5 using Python `round`,
adds tile x/y offsets to the rounded Python floats, constructs float32 rows, then
applies dynamic class-center offsets `class * (max(center_coordinates) * 2)` and
Fast-NMS with `batch_probiou` at 0.3. It does **not** clamp that dynamic value,
divide by nominal rate, rerun confidence filtering, or cap merged detections.
The image was never resized by nominal rate; dividing by it would double-invert
scale. Polygons use the upstream four-corner order, not axis-aligned boxes.

### Python usage and NPU seam

```python
import cv2
from mblt_vision.utils.datasets.dota_tiling import TiledConfig
from mblt_vision.utils.postprocess.tiled_obb import predict_tiled, write_task1

# Your callback owns runtime invocation, decode, thresholds and tile NMS.
# It MUST return native-tile Nx7 rows, not network-space boxes.
def predictor(tile):
    return your_native_tile_predictor(tile)  # application-defined callable

image = cv2.imread("original/P0001.png")
rows = predict_tiled(image, "P0001", predictor, TiledConfig())
write_task1({"P0001": rows}, "submission")
```

For an NPU callback, `prepare_tile` and `restore_tile_boxes` provide the pre/post
boundary. Run the model through `MBLT_Engine` / `mblt-npu-python`, decode and
apply per-tile rotated NMS, convert into Nx7 order, then restore with the exact
metadata. Do not assume the engine's native output column order matches this
API (many existing NMS rows are `[cx,cy,w,h,score,class,angle]`). Do not run an
inverse transform both in the engine and again in the callback. Quantization,
head decoding and tied-score behavior still require numerical validation.
No device is opened by this tiled module itself.

The optional `UltralyticsTilePredictor(weights, *, imgsz=1024, conf=0.01,
iou=0.7, max_det=300)` is a real CPU FP32 reference adapter. It requires an
existing trusted local `.pt` checkpoint and exactly `ultralytics==8.3.228`,
checked at construction; it never resolves a model name or downloads weights.
It executes the raw eval-mode OBB model, then upstream `non_max_suppression`
with `multi_label=True`, `rotated=True`, and the checkpoint's class count.
This deliberately uses validation-style multi-label candidates, **not** the
single-label defaults of `YOLO.predict`. It does not augment, fuse explicitly,
use FP16, rectangular auto-padding, or select GPU/NPU. `names` exposes checkpoint
class names so callers can validate taxonomy. Fixed-square raw-model inference
is an explicit controlled configuration, not all of `model.val`'s dataloader,
fusing/AutoBackend, `rect` and preprocessing optimizations.

```python
from mblt_vision.utils.postprocess.tiled_obb import UltralyticsTilePredictor

predictor = UltralyticsTilePredictor("/trusted/local/yolo11n-obb.pt")
# Then pass predictor into predict_tiled as above.
```

### Development CLI

This is a standalone benchmark script, not a new installed public entry point.
The output directory must be new/empty; it writes all 15 DOTAv1 Task1 files plus
`pipeline.json` recording geometry, backend and explicit reference settings.
Only existing local input images and weights are accepted. No scoring or
network submission occurs.

```bash
python benchmark/dota_tiled.py --source /data/DOTA/images/test \
  --output /results/task1 --weights /trusted/local/yolo11n-obb.pt \
  --imgsz 1024 --conf 0.01 --iou 0.7 --max-det 300

# Trusted application module defines factory() -> TilePredictor.
python benchmark/dota_tiled.py --source /data/DOTA/images/test \
  --output /results/task1-npu --predictor-factory my_application:factory
```

Callbacks are trusted Python code; loading a factory is not a sandbox. Custom
callbacks configure their inference settings themselves. `--imgsz`, `--conf`,
`--iou` and `--max-det` apply only to the optional reference adapter; nondefault
values with a factory are rejected rather than silently ignored.
`--no-jpeg-roundtrip` opts into lossless in-memory crops, a deliberate numerical
difference from the JPEG-producing upstream splitter.

### Task1 and scoring distinctions

`write_task1` accepts already-merged original-image Nx7 rows. It writes
`Task1_<class-name>.txt`, replacing spaces in class names with hyphens. Each row
is `image_id score x1 y1 x2 y2 x3 y3 x4 y4`, with Python `round(value,3)` and
normal float string formatting (not fixed-width `%.3f`). Empty class files are
created. Repeated API writes replace files instead of appending duplicates.
The CLI refuses nonempty output directories. Use a dedicated directory and do
not combine taxonomies there.

These are different measurements and must never share an unlabeled mAP field:

- **Local tiled-validation mAP** scores predictions against tile-local labels.
  It counts tiles as samples, and requires the labeled splitter's object IoF and
  label rules; this module does not generate training/validation labels.
- **Local merged original-image mAP** scores merged detections against original
  annotations, not repeated tile targets. The default `eval_dota` primary
  `map50` is continuous polygon-IoU **VOC2007 eleven-point AP**, strict IoU > .5,
  with nonzero difficult objects ignored, macro-averaged over all fifteen classes.
  It requires `labels/val_original`, and scores Task1-rounded polygons/scores.
  The independent Python convex clipper is not the official C++ polyiou extension;
  full-devkit numerical parity is not yet claimed.
  `DOTAResult.map5095` is retained only as a **diagnostic probabilistic** AP field,
  not a DOTA server score. The unchanged `evaluate_dota_predictions` and explicit
  legacy whole-image path use probIoU, interpolated AP and difficult-as-ordinary
  targets. These two primary scores have different definitions and must not be
  compared as an accuracy improvement without re-scoring identical predictions.
- **Official DOTA test-server mAP** requires Task1 files for the official split
  evaluated by the server/devkit protocol, including its polygon IoU, difficulty
  handling and AP settings. This exporter does not compute or submit that score.
  The Ultralytics merge uses probIoU rather than the official merge algorithm;
  even identical inference can therefore differ. Do not infer published TTA or
  multiscale settings from the table alone.

The existing `save_dota_task1_predictions` is left unchanged. It does not apply
this new API's pre-merge JSON rounding and clamps its dynamic class spacing to
at least 1. It should not be substituted as the exact pinned merge oracle.

### Explicit differences and reproducibility limits

- The frozen v1 direct-letterbox helper and engine bridge are **not pixel-exact
  upstream validation-loader preprocessing on all rectangular tiles**. Upstream
  first resizes the longest side to imgsz using `ceil` for the other dimension,
  then applies scaleup-disabled letterbox; direct letterbox uses rounded resized
  dimensions once. A synthetic 7x31 input at imgsz64 yields upstream height15
  versus direct height14 before padding, with different pixels and ratio metadata.
  `test_dota_validation_preprocess_reference.py` runs the actual pinned loader,
  transform builder and letterbox definitions without model inference to pin this
  distinction (square controls match). This restructure intentionally preserves
  existing Task1 numerics; correcting validation preprocessing is a separate,
  explicit behavior/protocol change, not an invisible refactor.

- Tied merge scores use this package's stable `descending_order` on every device,
  as required by its selection contract. Upstream Fast-NMS uses an unstable
  `torch.argsort`; equal-score selection can differ on CPU, especially after
  score rounding or quantization. Unequal-score differential tests compare exact
  Task1 text. Tests also pin stable ties separately.
- Shared blocked Fast-NMS bounds pairwise memory rather than allocating a full
  NxN matrix; suppression still includes all higher-ranked candidates, even
  suppressed ones, and uses `>= 0.3`.
- The fixed-square reference adapter omits upstream validation dataloader's
  optional rectangular batches and AutoBackend/fusing. It enables multi-label
  tile NMS explicitly. No published accuracy claim is made for these choices.
- Decode-library/codec versions can change pixels. Default JPEG parity is tested
  against actual upstream `split_test` writes in the same OpenCV installation.
  EXIF-mismatched dimensions, colliding crop sizes/stems and unsafe filenames fail
  clearly rather than reproducing upstream's ambiguous overwrites/parsing.
- `write_task1` replaces outputs and emits empty classes; upstream appends and
  creates only encountered classes. Detection row content matches apart from
  the documented tie policy. The optional package version guard does not prove
  the installed wheel's contents/hash match the pinned Git commit.
- Only geometry, serialization and synthetic postprocessing are validated here.
  Runtime accuracy, real checkpoints, hardware, full validation and server
  submission must be validated separately with recorded provenance.

### Tests

Run ordinary CPU synthetic fixtures without optional Ultralytics or weights:

```bash
python -m pytest tests/test_dota_tiled.py -q
```

For differential tests, supply the separately obtained pinned source directory
containing `data/split_dota.py`, `utils/{ops,metrics,nms}.py` and
`models/yolo/obb/val.py`. Named definitions are AST-isolated; upstream package
imports are not executed and no model is loaded.

```bash
MBLT_DOTA_REFERENCE=/path/to/pinned/ultralytics \
  python -m pytest tests/test_dota_tiled.py -q
```

Fixtures cover window counts/order/boundaries, joint filtering/fallback,
actual native JPEG pixels/names, color/layout/normalization/padding, inverse
letterbox at each scale, unclipped rotated geometry, offset-only restoration,
JSON rounding before offsets, cross-tile/cross-scale duplicates, class offsets,
Fast-NMS versus greedy semantics, threshold equality, Task1 corner order and
rounding, empty cases, invalid inputs, mocked optional adapter contracts and a
synthetic callback CLI. No GPU/NPU or real model inference is needed.

The committed `tests/fixtures/dota_tiled_da3ef806_task1.json` freezes all Task1
file bytes for the seed-0 randomized multi-rate fixture from Vision commit
`da3ef806c613eb101d9d53e2c53767df07cb16d6`. The refactored helper output is
compared byte-for-byte both to this frozen result and to pinned upstream
`OBBValidator.eval_json`; no accuracy measurement is implied.

Benchmark outputs label the default primary score `original_polygon_voc07_map50`
and retain `diagnostic_probiou_map50_95`, `eval_protocol`, `score_protocol` and
`official_test_server: false`. `--legacy-single-image` labels its primary score
`legacy_single_image_probiou_map50`; these scores are not interchangeable.
Finite zero-width/height or rounded-collinear prediction polygons have IoU zero
and count as false positives. Ground-truth polygons remain strictly convex.

## Pre-Trained Vision Models

This section lists the publicly pre-trained models supported by the vision framework.

### Image Classification

| Model | Input Size<br>(H,W,C) | Acc<sup>Top1</sup><br>(NPU) | Acc<sup>Top1</sup><br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | (224,224,3) | 56.084 | 56.556 | 1.43 | 61.10 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) |  |
| CAFormer_S18 | (224,224,3) | 82.592 | 83.626 | 9.10 | 26.34 | [Link](https://huggingface.co/timm/caformer_s18.sail_in1k) | sail_in1k |
| CAFormer_B36 | (224,224,3) | 83.938 | 85.482 | 49.41 | 98.75 | [Link](https://huggingface.co/timm/caformer_b36.sail_in1k) | sail_in1k |
| CoAtNet_0_RW_224 | (224,224,3) | 81.842 | 82.418 | 10.06 | 30.46 | [Link](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k) | sw_in1k |
| CoAtNet_1_RW_224 | (224,224,3) | 83.506 | 83.600 | 18.17 | 47.91 | [Link](https://huggingface.co/timm/coatnet_1_rw_224.sw_in1k) | sw_in1k |
| CoAtNet_2_RW_224 | (224,224,3) | 86.084 | 86.534 | 32.80 | 82.11 | [Link](https://huggingface.co/timm/coatnet_2_rw_224.sw_in12k_ft_in1k) | sw_in12k_ft_in1k |
| ConvFormer S36 | (224,224,3) | 83.360 | 84.016 | 16.66 | 40.01 | [Link](https://huggingface.co/timm/convformer_s36.sail_in1k) | sail_in1k |
| ConvFormer M36 | (224,224,3) | 84.014 | 84.448 | 27.58 | 57.05 | [Link](https://huggingface.co/timm/convformer_m36.sail_in1k) | sail_in1k |
| ConvFormer B36 | (224,224,3) | 84.244 | 84.830 | 47.79 | 99.88 | [Link](https://huggingface.co/timm/convformer_b36.sail_in1k) | sail_in1k |
| ConvNeXt_Tiny | (224,224,3) | 82.354 | 82.458 | 9.11 | 28.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) |  |
| ConvNeXt_Small | (224,224,3) | 83.434 | 83.560 | 17.68 | 50.22 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html) |  |
| ConvNeXt_Base | (224,224,3) | 83.940 | 84.048 | 31.13 | 88.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html) |  |
| ConvNeXt_Large | (224,224,3) | 84.316 | 84.410 | 69.35 | 197.77 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_large.html) |  |
| DeiT_Tiny_Patch16_224 | (224,224,3) | 71.944 | 72.030 | 2.66 | 5.72 | [Link](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Small_Patch16_224 | (224,224,3) | 79.722 | 79.790 | 9.50 | 22.05 | [Link](https://huggingface.co/timm/deit_small_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Base_Patch16_224 | (224,224,3) | 81.932 | 81.980 | 35.73 | 86.57 | [Link](https://huggingface.co/timm/deit_base_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Base_Patch16_384 | (384,384,3) | 83.046 | 83.100 | 115.00 | 86.86 | [Link](https://huggingface.co/timm/deit_base_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Small_Patch16_224 | (224,224,3) | 81.368 | 81.390 | 9.50 | 22.06 | [Link](https://huggingface.co/timm/deit3_small_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Small_Patch16_384 | (384,384,3) | 83.350 | 83.420 | 33.01 | 22.21 | [Link](https://huggingface.co/timm/deit3_small_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Medium_Patch16_224 | (224,224,3) | 83.020 | 83.044 | 16.39 | 38.85 | [Link](https://huggingface.co/timm/deit3_medium_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Base_Patch16_224 | (224,224,3) | 83.708 | 83.766 | 35.74 | 86.59 | [Link](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Base_Patch16_384 | (384,384,3) | 84.986 | 85.064 | 115.02 | 86.88 | [Link](https://huggingface.co/timm/deit3_base_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Large_Patch16_224 | (224,224,3) | 84.734 | 84.736 | 124.73 | 304.37 | [Link](https://huggingface.co/timm/deit3_large_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Large_Patch16_384 | (384,384,3) | 85.806 | 85.826 | 392.94 | 304.76 | [Link](https://huggingface.co/timm/deit3_large_patch16_384.fb_in1k) | fb_in1k |
| ConvFormer S18 | (224,224,3) | 81.862 | 82.866 | 8.59 | 26.77 | [Link](https://huggingface.co/timm/convformer_s18.sail_in1k) | sail_in1k |
| EfficientFormer_L7 | (224,224,3) | 82.526 | 83.352 | 20.67 | 82.14 | [Link](https://huggingface.co/timm/efficientformer_l7.snap_dist_in1k) | snap_dist_in1k |
| DenseNet121 | (224,224,3) | 74.320 | 74.422 | 6.37 | 8.04 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) |  |
| DenseNet161 | (224,224,3) | 77.200 | 77.142 | 16.85 | 28.86 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html) |  |
| DenseNet169 | (224,224,3) | 75.554 | 75.568 | 7.62 | 14.28 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet169.html) |  |
| DenseNet201 | (224,224,3) | 76.742 | 76.882 | 9.82 | 20.21 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html) |  |
| FlexiViT_Small | (240,240,3) | 82.150 | 82.536 | 11.05 | 22.06 | [Link](https://huggingface.co/timm/flexivit_small.1200ep_in1k) | 1200ep_in1k |
| FlexiViT_Base | (240,240,3) | 84.484 | 84.664 | 41.30 | 86.59 | [Link](https://huggingface.co/timm/flexivit_base.1200ep_in1k) | 1200ep_in1k |
| FlexiViT_Large | (240,240,3) | 85.540 | 85.658 | 143.89 | 304.36 | [Link](https://huggingface.co/timm/flexivit_large.1200ep_in1k) | 1200ep_in1k |
| Inception_V3 | (299,299,3) | 77.246 | 77.278 | 11.51 | 23.82 | [Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html) |  |
| LeViT_Conv_128 | (224,224,3) | 77.282 | 78.488 | 0.88 | 9.19 | [Link](https://huggingface.co/timm/levit_conv_128.fb_dist_in1k) | fb_dist_in1k |
| LeViT_Conv_128S | (224,224,3) | 75.204 | 76.574 | 0.65 | 7.76 | [Link](https://huggingface.co/timm/levit_conv_128s.fb_dist_in1k) | fb_dist_in1k |
| LeViT_Conv_192 | (224,224,3) | 79.138 | 79.876 | 1.38 | 10.92 | [Link](https://huggingface.co/timm/levit_conv_192.fb_dist_in1k) | fb_dist_in1k |
| LeViT_Conv_256 | (224,224,3) | 80.960 | 81.538 | 2.34 | 18.86 | [Link](https://huggingface.co/timm/levit_conv_256.fb_dist_in1k) | fb_dist_in1k |
| LeViT_Conv_384 | (224,224,3) | 82.170 | 82.582 | 4.83 | 39.08 | [Link](https://huggingface.co/timm/levit_conv_384.fb_dist_in1k) | fb_dist_in1k |
| MNASNet1_0 | (224,224,3) | 72.806 | 73.416 | 0.65 | 4.36 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_0.html) |  |
| MobileNet_V2 | (224,224,3) | 71.728 | 72.142 | 0.64 | 3.49 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mobilenet_v2.html) | IMAGENET1K_V2 |
| RegNet_X_400MF | (224,224,3) | 72.530 | 72.908 | 0.84 | 5.48 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V1 |
| RegNet_X_400MF | (224,224,3) | 74.182 | 74.860 | 0.84 | 5.48 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V2 |
| RegNet_X_800MF | (224,224,3) | 74.972 | 75.210 | 1.62 | 7.24 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V1 |
| RegNet_X_800MF | (224,224,3) | 77.056 | 77.496 | 1.62 | 7.24 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V2 |
| RegNet_X_1_6GF | (224,224,3) | 76.900 | 77.084 | 3.24 | 9.17 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_X_1_6GF | (224,224,3) | 79.254 | 79.676 | 3.24 | 9.17 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_X_3_2GF | (224,224,3) | 78.142 | 78.342 | 6.40 | 15.27 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_X_3_2GF | (224,224,3) | 80.888 | 81.194 | 6.40 | 15.27 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_X_8GF | (224,224,3) | 79.338 | 79.372 | 16.05 | 39.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V1 |
| RegNet_X_8GF | (224,224,3) | 81.386 | 81.692 | 16.05 | 39.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V2 |
| RegNet_X_16GF | (224,224,3) | 79.932 | 80.092 | 31.99 | 54.22 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V1 |
| RegNet_X_16GF | (224,224,3) | 82.434 | 82.712 | 31.99 | 54.22 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V2 |
| RegNet_X_32GF | (224,224,3) | 80.550 | 80.592 | 63.63 | 107.73 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V1 |
| RegNet_X_32GF | (224,224,3) | 82.856 | 83.022 | 63.63 | 107.73 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V2 |
| RegNet_Y_400MF | (224,224,3) | 73.690 | 74.004 | 0.82 | 4.33 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V1 |
| RegNet_Y_400MF | (224,224,3) | 75.312 | 75.802 | 0.82 | 4.33 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V2 |
| RegNet_Y_800MF | (224,224,3) | 76.100 | 76.396 | 1.70 | 6.42 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V1 |
| RegNet_Y_800MF | (224,224,3) | 78.448 | 78.890 | 1.70 | 6.42 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V2 |
| RegNet_Y_1_6GF | (224,224,3) | 77.370 | 77.926 | 3.27 | 11.18 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_Y_1_6GF | (224,224,3) | 80.490 | 80.882 | 3.27 | 11.18 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_Y_3_2GF | (224,224,3) | 78.744 | 78.962 | 6.41 | 19.40 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_Y_3_2GF | (224,224,3) | 81.454 | 82.018 | 6.41 | 19.40 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_Y_8GF | (224,224,3) | 79.874 | 80.052 | 17.05 | 39.34 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V1 |
| RegNet_Y_8GF | (224,224,3) | 82.546 | 82.824 | 17.05 | 39.34 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V2 |
| RegNet_Y_16GF | (224,224,3) | 80.326 | 80.424 | 31.95 | 83.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html) | IMAGENET1K_V1 |
| RegNet_Y_16GF | (224,224,3) | 82.470 | 82.862 | 31.95 | 83.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html) | IMAGENET1K_V2 |
| RegNet_Y_32GF | (224,224,3) | 80.700 | 80.834 | 64.72 | 144.97 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V1 |
| RegNet_Y_32GF | (224,224,3) | 82.890 | 83.362 | 64.72 | 144.97 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V2 |
| ResNet18 | (224,224,3) | 69.558 | 69.778 | 3.64 | 11.68 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet18.html) |  |
| ResNet34 | (224,224,3) | 73.166 | 73.304 | 7.35 | 21.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet34.html) |  |
| ResNet50 | (224,224,3) | 75.980 | 76.116 | 8.23 | 25.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V1 |
| ResNet50 | (224,224,3) | 80.574 | 80.852 | 8.23 | 25.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V2 |
| ResNet101 | (224,224,3) | 77.076 | 77.350 | 15.69 | 44.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V1 |
| ResNet101 | (224,224,3) | 81.534 | 81.914 | 15.69 | 44.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V2 |
| ResNet152 | (224,224,3) | 78.044 | 78.306 | 23.14 | 60.12 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V1 |
| ResNet152 | (224,224,3) | 81.952 | 82.272 | 23.14 | 60.12 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V2 |
| ResNeXt50_32X4D | (224,224,3) | 77.568 | 77.634 | 8.53 | 24.99 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V1 |
| ResNeXt50_32X4D | (224,224,3) | 80.896 | 81.212 | 8.53 | 24.99 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V2 |
| ResNeXt101_32X8D | (224,224,3) | 79.234 | 79.290 | 32.97 | 88.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V1 |
| ResNeXt101_32X8D | (224,224,3) | 82.598 | 82.784 | 32.97 | 88.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V2 |
| ResNeXt101_64X4D | (224,224,3) | 82.980 | 83.234 | 31.06 | 83.35 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_64x4d.html) |  |
| ShuffleNet_V2_X1_0 | (224,224,3) | 68.734 | 69.312 | 0.30 | 2.27 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) |  |
| ShuffleNet_V2_X1_5 | (224,224,3) | 72.458 | 72.966 | 0.60 | 3.49 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html) |  |
| ShuffleNet_V2_X2_0 | (224,224,3) | 75.614 | 76.224 | 1.18 | 7.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x2_0.html) |  |
| Swin_B | (224,224,3) | 83.254 | 83.562 | 31.58 | 88.61 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html) |  |
| Swin_S | (224,224,3) | 82.848 | 83.180 | 18.02 | 50.24 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_s.html) |  |
| Swin_T | (224,224,3) | 81.076 | 81.444 | 9.32 | 28.60 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html) |  |
| VGG11 | (224,224,3) | 68.706 | 68.974 | 15.26 | 132.86 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11.html) |  |
| VGG11_BN | (224,224,3) | 70.074 | 70.328 | 15.26 | 132.86 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11_bn.html) |  |
| VGG13 | (224,224,3) | 69.742 | 69.888 | 22.68 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13.html) |  |
| VGG13_BN | (224,224,3) | 71.370 | 71.564 | 22.68 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13_bn.html) |  |
| VGG16 | (224,224,3) | 71.526 | 71.616 | 31.01 | 138.36 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16.html) |  |
| VGG16_BN | (224,224,3) | 73.276 | 73.406 | 31.01 | 138.36 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16_bn.html) |  |
| VGG19 | (224,224,3) | 72.284 | 72.386 | 39.34 | 143.67 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19.html) |  |
| VGG19_BN | (224,224,3) | 74.022 | 74.170 | 39.34 | 143.67 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19_bn.html) |  |
| ViT_B_16 | (224,224,3) | 81.002 | 81.040 | 35.73 | 86.57 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) | IMAGENET1K_V1 |
| ViT_B_16 | (384,384,3) | 85.118 | 85.276 | 115.01 | 86.86 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) | IMAGENET1K_SWAG_E2E_V1 |
| ViT_B_16 | (224,224,3) | 81.516 | 81.926 | 35.73 | 86.57 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) | IMAGENET1K_SWAG_LINEAR_V1 |
| ViT_B_32 | (224,224,3) | 75.680 | 75.908 | 8.90 | 88.22 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_32.html) |  |
| RepViT_M1 | (224,224,3) | 77.668 | 78.510 | 1.70 | 5.46 | [Link](https://huggingface.co/timm/repvit_m1.dist_in1k) | dist_in1k |
| RepViT_M1_0 | (224,224,3) | 79.210 | 80.198 | 2.39 | 7.29 | [Link](https://huggingface.co/timm/repvit_m1_0.dist_300e_in1k) | dist_300e_in1k |
| RepViT_M1_1 | (224,224,3) | 79.838 | 80.828 | 2.87 | 8.79 | [Link](https://huggingface.co/timm/repvit_m1_1.dist_300e_in1k) | dist_300e_in1k |
| RepViT_M1_5 | (224,224,3) | 81.532 | 82.374 | 4.88 | 14.62 | [Link](https://huggingface.co/timm/repvit_m1_5.dist_300e_in1k) | dist_300e_in1k |
| RepViT_M2 | (224,224,3) | 79.882 | 80.526 | 2.77 | 8.77 | [Link](https://huggingface.co/timm/repvit_m2.dist_in1k) | dist_in1k |
| RepViT_M2_3 | (224,224,3) | 82.960 | 83.466 | 9.59 | 23.66 | [Link](https://huggingface.co/timm/repvit_m2_3.dist_300e_in1k) | dist_300e_in1k |
| RepViT_M3 | (224,224,3) | 80.896 | 81.478 | 3.85 | 10.65 | [Link](https://huggingface.co/timm/repvit_m3.dist_in1k) | dist_in1k |
| VisFormer_Small | (224,224,3) | 81.586 | 82.106 | 10.05 | 40.24 | [Link](https://huggingface.co/timm/visformer_small.in1k) | in1k |
| VisFormer_Tiny | (224,224,3) | 77.564 | 78.262 | 2.68 | 10.33 | [Link](https://huggingface.co/timm/visformer_tiny.in1k) | in1k |
| ViT_Tiny_Patch16_224 | (224,224,3) | 74.568 | 75.456 | 2.66 | 5.72 | [Link](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Tiny_Patch16_384 | (384,384,3) | 77.312 | 78.470 | 10.37 | 5.79 | [Link](https://huggingface.co/timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch16_224 | (224,224,3) | 81.404 | 81.412 | 9.50 | 22.05 | [Link](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch16_384 | (384,384,3) | 83.560 | 83.782 | 33.00 | 22.20 | [Link](https://huggingface.co/timm/vit_small_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch32_224 | (224,224,3) | 75.792 | 75.922 | 2.32 | 22.88 | [Link](https://huggingface.co/timm/vit_small_patch32_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch32_384 | (384,384,3) | 80.092 | 80.452 | 7.07 | 22.92 | [Link](https://huggingface.co/timm/vit_small_patch32_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch8_224 | (224,224,3) | 86.302 | 86.268 | 163.48 | 86.58 | [Link](https://huggingface.co/timm/vit_base_patch8_224.augreg2_in21k_ft_in1k) | augreg2_in21k_ft_in1k |
| ViT_Base_Patch16_224 | (224,224,3) | 85.080 | 85.112 | 35.73 | 86.57 | [Link](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k) | augreg2_in21k_ft_in1k |
| ViT_Base_Patch16_384 | (384,384,3) | 85.860 | 86.018 | 115.00 | 86.86 | [Link](https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch32_224 | (224,224,3) | 80.612 | 80.698 | 8.89 | 88.22 | [Link](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch32_384 | (384,384,3) | 83.138 | 83.398 | 26.45 | 88.30 | [Link](https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch16_224 | (224,224,3) | 85.880 | 85.870 | 124.71 | 304.33 | [Link](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch16_384 | (384,384,3) | 86.980 | 87.086 | 392.88 | 304.72 | [Link](https://huggingface.co/timm/vit_large_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch32_384 | (384,384,3) | 81.122 | 81.512 | 91.52 | 306.63 | [Link](https://huggingface.co/timm/vit_large_patch32_384.orig_in21k_ft_in1k) | orig_in21k_ft_in1k |
| Wide_ResNet50_2 | (224,224,3) | 78.402 | 78.488 | 22.87 | 68.85 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V1 |
| Wide_ResNet50_2 | (224,224,3) | 81.268 | 81.630 | 22.87 | 68.85 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V2 |
| Wide_ResNet101_2 | (224,224,3) | 78.500 | 78.830 | 45.61 | 126.82 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V1 |
| Wide_ResNet101_2 | (224,224,3) | 82.358 | 82.510 | 45.61 | 126.82 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V2 |
| YOLOv5n-cls | (224,224,3) | 63.538 | 63.982 | 0.43 | 2.49 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s-cls | (224,224,3) | 70.698 | 70.854 | 1.42 | 5.45 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m-cls | (224,224,3) | 75.348 | 75.418 | 4.03 | 12.95 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l-cls | (224,224,3) | 77.536 | 77.528 | 8.82 | 26.54 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x-cls | (224,224,3) | 78.338 | 78.312 | 16.46 | 48.07 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv8s-cls | (224,224,3) | 72.630 | 73.774 | 1.67 | 6.36 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m-cls | (224,224,3) | 76.032 | 76.824 | 5.37 | 17.04 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l-cls | (224,224,3) | 77.708 | 78.276 | 12.53 | 37.47 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-cls | (224,224,3) | 78.440 | 78.936 | 19.38 | 57.40 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLO11s-cls | (224,224,3) | 74.532 | 75.244 | 1.63 | 6.72 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m-cls | (224,224,3) | 76.484 | 77.388 | 5.17 | 11.62 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l-cls | (224,224,3) | 77.658 | 78.284 | 6.51 | 14.10 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-cls | (224,224,3) | 78.666 | 79.426 | 14.20 | 29.61 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO26s-cls | (224,224,3) | 75.300 | 75.910 | 1.63 | 6.72 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26m-cls | (224,224,3) | 77.608 | 78.078 | 5.17 | 11.62 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26l-cls | (224,224,3) | 78.614 | 79.060 | 6.51 | 14.10 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26x-cls | (224,224,3) | 79.478 | 79.866 | 14.20 | 29.61 | [Link](https://docs.ultralytics.com/models/yolo26/) | |

<details>
<summary>Image Classification (ImageNet)</summary>

- Acc<sup>Top1</sup> values are the primary model accuracies on the
  [ImageNet](https://www.image-net.org/index.php) validation set. Validation also reports
  Acc<sup>Top5</sup> as the secondary metric.

</details>

### Face Detection

WiderFace validation uses Hard-set AP as the primary metric. Medium-set AP and
Easy-set AP are secondary metrics, in that order. Mean AP across the difficulty
splits is not computed.

Face detection is a single-class task: every detection is labeled `face`, and no
COCO category mapping is applied.

| Model | Input Size<br>(H,W,C) | Source | Note |
| --- | --- | --- | --- |
| YOLOv8n-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv8m-face | (960,960,3) | [Link](https://github.com/akanametov/yolo-face) | Trained at 960; see the note below. |
| YOLOv8l-face | (960,960,3) | [Link](https://github.com/akanametov/yolo-face) | Trained at 960; see the note below. |
| YOLO11n-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO11s-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO11m-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO11l-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO12n-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO12s-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO12m-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLO12l-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv10n-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv10s-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv10m-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv10l-face | (640,640,3) | [Link](https://github.com/akanametov/yolo-face) | |
| YOLOv5n-face | (640,640,3) | [Link](https://github.com/deepcam-cn/yolov5-face) | Anchor-based. |
| YOLOv5n-0.5-face | (640,640,3) | [Link](https://github.com/deepcam-cn/yolov5-face) | Anchor-based. |
| YOLOv5s-face | (640,640,3) | [Link](https://github.com/deepcam-cn/yolov5-face) | Anchor-based. |
| YOLOv5m-face | (640,640,3) | [Link](https://github.com/deepcam-cn/yolov5-face) | Anchor-based. |
| YOLOv7-lite-t-face | (640,640,3) | [Link](https://github.com/derronqi/yolov7-face) | Anchor-based. |
| YOLOv7-lite-s-face | (640,640,3) | [Link](https://github.com/derronqi/yolov7-face) | Anchor-based. |
| YOLOv7-tiny-face | (640,640,3) | [Link](https://github.com/derronqi/yolov7-face) | Anchor-based. |
| YOLOv7s-face | (640,640,3) | [Link](https://github.com/derronqi/yolov7-face) | Anchor-based. |
| YOLOv7-face | (640,640,3) | [Link](https://github.com/derronqi/yolov7-face) | Anchor-based. |

Input geometry follows each checkpoint's own training resolution. Every face
model is trained and served at 640x640 except `YOLOv8m-face` and `YOLOv8l-face`,
whose released weights record `imgsz: 960` in their embedded `train_args`; those
two use 960x960 and must not be normalized down to 640.

The `YOLOv5*-face` and `YOLOv7*-face` Hub repositories do not currently have
project-pinned immutable revisions and artifact SHA-256 digests. To prevent an
unverified remote artifact from reaching a native runtime, construct these models
with a trusted local `model_path`, `mxq_path`, or `onnx_path`. Automatic Hub
downloads are disabled for these families.

The `YOLOv5*-face` and `YOLOv7*-face` families are anchor-based and carry an
`anchors` list in `post_cfg` with `iou_thres: 0.5`; every other face model is
anchorless or NMS-free with `iou_thres: 0.7`. Their published ONNX exports emit
three raw detection heads of shape `(batch, 3, H, W, 6)` — the five landmark
pairs of the original repositories are stripped at export — so they decode
through the shared anchor detection path with `nc = 1`.

WiderFace accuracy columns are omitted until the full validation split has been
measured for each artifact.

### Depth Estimation

NYU Depth V2 validation uses median-aligned delta1 as the primary metric.
Absolute relative error (abs_rel) and RMSE in meters are the secondary metrics,
in that order. Each metric is calculated per image, then averaged across the
validation split, following Ultralytics' depth-validation convention.

### Object Detection

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv3 | (640,640,3) | 46.354 | 46.839 | 162.27 | 61.92 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv3u | (640,640,3) | 51.214 | 51.582 | 289.23 | 103.73 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv3-spp | (640,640,3) | 47.106 | 47.616 | 163.23 | 62.97 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv3-sppu | (640,640,3) | 51.710 | 51.754 | 290.20 | 104.78 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv5nu | (640,640,3) | 33.549 | 34.286 | 8.79 | 2.65 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5n6 | (1280,1280,3) | 35.021 | 35.892 | 22.10 | 3.24 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5n6u | (1280,1280,3) | 41.637 | 42.111 | 35.63 | 4.33 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s | (640,640,3) | 36.780 | 37.540 | 18.21 | 7.23 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5su | (640,640,3) | 42.414 | 42.877 | 25.93 | 9.14 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s6 | (1280,1280,3) | 43.878 | 44.512 | 74.33 | 12.61 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s6u | (1280,1280,3) | 48.363 | 48.636 | 105.38 | 15.29 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m | (640,640,3) | 44.446 | 45.238 | 52.14 | 21.17 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5mu | (640,640,3) | 48.388 | 48.910 | 67.70 | 25.09 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m6 | (1280,1280,3) | 50.637 | 51.078 | 212.83 | 35.70 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m6u | (1280,1280,3) | 53.259 | 53.476 | 275.41 | 41.19 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l | (640,640,3) | 48.128 | 48.914 | 114.20 | 46.53 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5lu | (640,640,3) | 51.824 | 52.172 | 140.50 | 53.19 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l6 | (1280,1280,3) | 52.892 | 53.376 | 466.00 | 76.73 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l6u | (1280,1280,3) | 55.344 | 55.466 | 571.74 | 86.05 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x | (640,640,3) | 49.986 | 50.554 | 213.04 | 86.71 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5xu | (640,640,3) | 52.792 | 53.090 | 254.38 | 97.23 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x6 | (1280,1280,3) | 54.197 | 54.706 | 869.05 | 140.73 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x6u | (1280,1280,3) | 56.516 | 56.488 | 1035.24 | 155.48 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv7 | (640,640,3) | 50.442 | 50.942 | 110.55 | 36.91 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7x | (640,640,3) | 52.389 | 52.706 | 197.67 | 71.31 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7w6 | (1280,1280,3) | 53.886 | 54.128 | 374.79 | 70.39 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7d6 | (1280,1280,3) | 55.510 | 55.799 | 730.75 | 133.76 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7e6 | (1280,1280,3) | 55.379 | 55.567 | 538.42 | 97.20 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7e6e | (1280,1280,3) | 55.932 | 56.298 | 878.43 | 151.69 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv8n | (640,640,3) | 36.650 | 37.328 | 9.78 | 3.15 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8s | (640,640,3) | 44.243 | 44.918 | 30.49 | 11.16 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m | (640,640,3) | 49.915 | 50.239 | 82.26 | 25.89 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l | (640,640,3) | 52.485 | 52.772 | 170.26 | 43.67 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x | (640,640,3) | 53.497 | 53.802 | 264.17 | 68.20 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| GELANs | (640,640,3) | 45.679 | 46.486 | 29.01 | 7.11 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9s | (640,640,3) | 45.637 | 46.777 | 29.01 | 7.11 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANm | (640,640,3) | 50.564 | 50.928 | 80.81 | 19.98 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9m | (640,640,3) | 50.902 | 51.191 | 80.81 | 19.98 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANc | (640,640,3) | 52.086 | 52.273 | 107.83 | 25.29 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9c | (640,640,3) | 52.598 | 52.917 | 107.83 | 25.29 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANe | (640,640,3) | 54.568 | 54.927 | 199.86 | 57.35 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9e | (640,640,3) | 55.466 | 55.528 | 199.86 | 57.35 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv10n | (640,640,3) | 37.432 | 38.382 | 8.06 | 2.30 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10s | (640,640,3) | 45.386 | 46.015 | 24.10 | 7.25 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10m | (640,640,3) | 50.075 | 50.838 | 63.51 | 15.36 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10b | (640,640,3) | 51.525 | 52.096 | 97.77 | 19.07 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10l | (640,640,3) | 52.246 | 52.816 | 127.32 | 24.37 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10x | (640,640,3) | 53.427 | 53.993 | 170.13 | 29.47 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLO11n | (640,640,3) | 38.567 | 39.298 | 7.76 | 2.62 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11s | (640,640,3) | 46.179 | 46.617 | 23.80 | 9.44 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m | (640,640,3) | 50.887 | 51.310 | 72.81 | 20.09 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l | (640,640,3) | 52.770 | 53.165 | 93.21 | 25.34 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x | (640,640,3) | 54.136 | 54.478 | 204.31 | 56.92 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO12n | (640,640,3) | 40.172 | 40.740 | 9.27 | 2.59 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12s | (640,640,3) | 47.124 | 47.687 | 26.73 | 9.26 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12m | (640,640,3) | 51.899 | 52.297 | 77.22 | 20.17 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12l | (640,640,3) | 53.200 | 53.508 | 105.07 | 26.40 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12x | (640,640,3) | 54.758 | 55.061 | 223.27 | 59.14 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLOX_Nano | (416,416,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_Tiny | (416,416,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_s | (640,640,3) | - | 40.679 | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_m | (640,640,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_l | (640,640,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_x | (640,640,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| YOLOX_Darknet53 | (640,640,3) | - | - | - | - | [Link](https://github.com/Megvii-BaseDetection/YOLOX) | Local artifact; see the note below. |
| DAMO_YOLO_T | (640,640,3) | - | 41.789 | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |
| DAMO_YOLO_TDistill | (640,640,3) | - | - | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |
| DAMO_YOLO_S | (640,640,3) | - | - | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |
| DAMO_YOLO_SDistill | (640,640,3) | - | - | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |
| DAMO_YOLO_M | (640,640,3) | - | - | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |
| DAMO_YOLO_MDistill | (640,640,3) | - | - | - | - | [Link](https://github.com/tinyvision/DAMO-YOLO) | Local artifact; see the note below. |

<details>
<summary>Object Detection (COCO)</summary>

- $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$ values are for single-model single-scale on the [COCO val2017](https://cocodataset.org/) dataset.
- YOLOX and DAMO-YOLO have no Hub artifact yet, so pass a local `model_path`. Their GPU
  scores are the ONNX exports of the upstream checkpoints, measured through this package
  (upstream publishes 40.5 for YOLOX-s and 41.8 for DAMO-YOLO-T). Neither preprocesses like
  Ultralytics: both reproduce upstream's own test transform, which letterboxes top-left with
  a truncated `int(w * r)` size (`LetterBox.size_rounding: floor`) and feeds the unscaled
  0-255 image. YOLOX reads cv2 BGR and restores boxes by the one resize ratio. DAMO-YOLO
  follows its December 2022 checkpoints, the only ones still downloadable: it decodes with
  PIL (`Reader.style: pil`) and restores each axis by its own ratio
  (`LetterBox.per_axis_ratio: true`).
- YOLOv7 models run NMS at `iou_thres` 0.65, the value WongKinYiu/yolov7's `test.py` defaults
  to and its README reproduces with. Their scores above were measured at 0.6 and are pending
  re-measurement.

</details>

### Instance Segmentation

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{mask}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{mask}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n-seg | (640,640,3) | 22.671 | 23.334 | 8.23 | 1.99 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s-seg | (640,640,3) | 31.164 | 31.592 | 28.47 | 7.61 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m-seg | (640,640,3) | 36.764 | 37.148 | 74.57 | 21.97 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l-seg | (640,640,3) | 39.685 | 39.942 | 153.53 | 47.89 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x-seg | (640,640,3) | 41.185 | 41.318 | 273.99 | 88.77 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv8n-seg | (640,640,3) | 29.951 | 30.464 | 13.91 | 3.40 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8s-seg | (640,640,3) | 36.534 | 36.691 | 44.86 | 11.81 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m-seg | (640,640,3) | 40.362 | 40.596 | 114.06 | 27.27 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l-seg | (640,640,3) | 42.316 | 42.462 | 226.27 | 45.97 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-seg | (640,640,3) | 43.048 | 43.155 | 351.31 | 71.80 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| GELANc-seg | (640,640,3) | 42.150 | 42.204 | 150.93 | 27.42 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9c-seg | (640,640,3) | 42.574 | 42.655 | 152.44 | 27.45 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9e-seg | (640,640,3) | 44.410 | 44.394 | 256.38 | 59.74 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLO11n-seg | (640,640,3) | 31.509 | 32.129 | 11.88 | 2.87 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11s-seg | (640,640,3) | 37.379 | 37.726 | 38.18 | 10.10 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m-seg | (640,640,3) | 41.552 | 41.683 | 128.82 | 22.40 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l-seg | (640,640,3) | 42.948 | 42.974 | 149.22 | 27.65 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-seg | (640,640,3) | 43.895 | 43.910 | 329.44 | 62.09 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO12n-seg | (640,640,3) | 32.516 | 32.941 | 12.94 | 2.80 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12s-seg | (640,640,3) | 38.470 | 38.787 | 39.26 | 9.76 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12m-seg | (640,640,3) | 42.270 | 42.433 | 125.74 | 21.94 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12l-seg | (640,640,3) | 43.429 | 43.456 | 154.99 | 28.76 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12x-seg | (640,640,3) | 44.187 | 44.418 | 334.55 | 64.51 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO26m-seg | (640,640,3) | 42.961 | 43.732 | 137.37 | 23.57 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26l-seg | (640,640,3) | 44.246 | 45.242 | 157.06 | 27.97 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26x-seg | (640,640,3) | 46.033 | 46.717 | 346.96 | 62.82 | [Link](https://docs.ultralytics.com/models/yolo26/) | |

<details>
<summary>Instance Segmentation (COCO)</summary>

- $\underset{50\text{–}95}{\text{mAP}_{\text{val}}^{\text{mask}}}$ values are for single-model single-scale on the [COCO val2017](https://cocodataset.org/) dataset.

</details>

### Pose Estimation

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8s-pose | (640,640,3) | 57.007 | 59.366 | 32.10 | 11.62 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m-pose | (640,640,3) | 62.153 | 64.451 | 84.37 | 26.45 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l-pose | (640,640,3) | 65.173 | 66.822 | 173.70 | 44.47 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-pose | (640,640,3) | 67.063 | 68.357 | 269.63 | 69.46 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-pose-p6 | (1280,1280,3) | 69.221 | 70.758 | 1092.51 | 99.14 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLO11s-pose | (640,640,3) | 55.731 | 57.846 | 25.41 | 9.90 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m-pose | (640,640,3) | 62.136 | 64.280 | 76.25 | 20.89 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l-pose | (640,640,3) | 63.174 | 65.414 | 96.65 | 26.14 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-pose | (640,640,3) | 67.218 | 68.600 | 212.26 | 58.75 | [Link](https://docs.ultralytics.com/models/yolo11/) | |

<details>
<summary>Pose Estimation (COCO)</summary>

- $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$ values are for single-model single-scale on the [COCO Keypoints val2017](https://cocodataset.org/) dataset.

</details>

### OBB

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50}}{\texttt{mAP}_{\texttt{val}}^{\texttt{obb}}}$<br>(NPU) | $\underset{\texttt{50}}{\texttt{mAP}_{\texttt{val}}^{\texttt{obb}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |

<details>
<summary>OBB (DOTA v1.0)</summary>

- Historical table values above are whole-image, single-scale probabilistic-IoU
  validation results; they have not been remeasured with the new tiled pipeline.
- Default validation now reports merged original-image **polygon VOC07 mAP50**, with
  nonzero difficult flags ignored. The compatible `map5095` field is a separately
  labeled probabilistic diagnostic. Neither is an official DOTA test-server score.
- `--legacy-single-image` keeps the prior whole-image probabilistic metric and
  difficult-as-ordinary target policy. Metric-definition changes alone are not
  evidence of improved model accuracy.

</details>

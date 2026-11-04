# DOTA multiscale native tiles

The explicit `mblt_vision.obb.tiled` API and development CLI
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

## Pinned reference

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

Methodology references (living documentation, not immutable experiment evidence):
[Ultralytics OBB](https://docs.ultralytics.com/tasks/obb/),
[DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v1/),
[DOTA benchmark](https://captain-whu.github.io/DOTA/).

## Geometry and data contract

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

`prepare_tile(tile, imgsz=1024)` supplies a reference-compatible fixed-square
input helper: centered aspect-preserving letterbox with rounded resized size,
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

## Python usage and NPU seam

```python
import cv2
from mblt_vision.obb.tiled import TiledConfig, predict_tiled, write_task1

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
from mblt_vision.obb.tiled import UltralyticsTilePredictor

predictor = UltralyticsTilePredictor("/trusted/local/yolo11n-obb.pt")
# Then pass predictor into predict_tiled as above.
```

## Development CLI

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

## Task1 and scoring distinctions

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

## Explicit differences and reproducibility limits

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

## Tests

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

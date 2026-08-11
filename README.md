# Mobilint Vision Python

Run pre-trained Mobilint Vision models from Python. `mblt-vision-python` provides
model configuration, artifact loading, preprocessing, inference integration, and
typed postprocessing results for image classification, depth estimation, face and
object detection, OBB, instance and semantic segmentation,
and pose estimation.

Version `0.0.1` is the initial standalone release.

## Installation

```bash
pip install mblt-vision-python
```

MXQ inference requires a supported Mobilint NPU environment. Model artifacts are
downloaded from the Mobilint Hugging Face organization when no local `model_path`
is supplied. For ONNX execution, install one of the optional extras:

```bash
pip install "mblt-vision-python[onnxruntime]"
# Or, on supported systems:
pip install "mblt-vision-python[onnxruntime-gpu]"
```

## Quick start

Each model includes its matching preprocess and postprocess behavior:

```python
from mblt_vision import ResNet50

model = ResNet50()
x = model.preprocess("image.jpg")
result = model.postprocess(model(x))
```

For configurable model selection and local MXQ or ONNX artifacts, use
`MBLT_Engine`:

```python
from mblt_vision import MBLT_Engine

model = MBLT_Engine(model_cls="resnet50", model_type="DEFAULT")
try:
    result = model.postprocess(model(model.preprocess("image.jpg")))
finally:
    model.dispose()
```

Discover supported tasks and models with `list_tasks()` and `list_models()`. New
code should use the task subpackages (for example,
`mblt_vision.object_detection`) or `MBLT_Engine`. Top-level model imports such as
`from mblt_vision import ResNet50` remain supported for convenience.

`obb` is the canonical oriented-bounding-box task name.

## Model Zoo migration

Vision is now maintained in this package. `mblt-model-zoo` retains
`mblt_model_zoo.vision` as a compatibility facade for existing applications; new
projects should import from `mblt_vision` directly. Its `mblt-model-zoo predict`,
`val`, and `compile` commands also delegate to this package.

## Command line

The standalone package provides the `mblt-vision` command with `predict`, `val`,
and `compile` subcommands:

```bash
mblt-vision predict --source image.jpg --model resnet50
```

`predict` is the single inference command for classification, depth estimation,
object and face detection, instance and semantic segmentation, OBB, and pose
estimation. The selected model determines its task and processing pipeline.
By default it downloads the model artifact and saves a plotted result under
`runs/vision/predict/`. Use `--output` to choose the result-image path,
`--topk` for classification labels, and `--conf-thres`/`--iou-thres` for
detection-style tasks. `--framework onnx` selects ONNX Runtime inference;
`--target-device` and `--core-mode` select the MXQ board/runtime mode.

```bash
mblt-vision predict --source image.jpg --model yolo11m --conf-thres 0.4 --output result.jpg
mblt-vision predict --source image.jpg --model yolo11m-pose --target-device regulus-ra --core-mode single
```

The corresponding `mblt-model-zoo` commands use the same standalone handlers for
backward compatibility.

## Documentation and tests

See [the Vision API guide](mblt_vision/README.md) for supported model families,
model details, artifact selection, and output taxonomy behavior. See the
[compilation guide](compile/README.md) for calibration-data preparation
and MXQ compilation. The [test guide](tests/TEST.md) explains offline, Hugging
Face, and NPU test runs.

## Support and issues

For installation, model, or runtime support, visit the
[Mobilint forum](https://discuss.mobilint.com/). Report reproducible package issues in the
[mblt-vision-python issue tracker](https://github.com/mobilint/mblt-vision-python/issues).

## License

Distributed under the [BSD 3-Clause License](LICENSE).

# Vision datasets and benchmarks

`mblt-vision-python` owns the utilities that organize validation datasets and
evaluate Vision models. The standardized benchmark runner writes JSON, CSV,
Markdown, and optional chart artifacts for one or more models.

Run scripts from the repository root, or invoke them as modules after installing
the package from source.

## Dataset organization

Most organizers can download their public source archives when paths are omitted;
pass local paths to reuse existing downloads. Organized datasets default to
`~/.mblt_vision/datasets`.

```bash
python benchmark/vision/organize_coco.py
python benchmark/vision/organize_ade20k.py
python benchmark/vision/organize_nyu_depth.py
```

Cityscapes is not downloaded automatically. Supply the official archives:

```bash
python benchmark/vision/organize_cityscapes.py \
  --image-dir benchmark/leftImg8bit_trainvaltest.zip \
  --annotation-dir benchmark/gtFine_trainvaltest.zip
```

The Cityscapes organizer selects only the 500 validation RGB images and their
`gtFine_labelIds` masks, installing them as `images/` and `annotations/` under the
chosen output directory.

ImageNet and DOTA require their source archives or directories because their terms
of use may restrict automatic retrieval. See each organizer's `--help` output for
required arguments.

## Running a benchmark

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models ResNet50 \
  --task image_classification \
  --data-path ~/.mblt_vision/datasets/imagenet
```

Use `--framework onnx` (or an `.onnx` `--model-path`) for ONNX Runtime. Use
`--core-mode all` to compare NPU core modes. The runner keeps evaluating remaining
models after an individual failure unless `--fail-fast` is set.

Compare two or more benchmark result directories with:

```bash
python benchmark/vision/compare_benchmark_results.py run-a run-b
```

## Verification

The focused tests cover archive download recovery, safe staging and replacement,
dataset readiness, evaluator dispatch, standardized benchmark artifacts, and
comparison validation. Hardware-backed runs require a configured Mobilint NPU;
ONNX benchmarks require the relevant optional ONNX Runtime extra.

# Vision benchmark commands

`mblt-vision-python` is a Vision-only package, so every command script lives
directly in this directory. Reusable benchmark support—argument parsing,
artifact writing, charts, and summaries—is packaged under
`mblt_vision.benchmark` for use by library code and command scripts alike.

The old per-dataset benchmark scripts and duplicated organizer wrappers were
removed. Use the unified runner and the organizer that matches your dataset.

## Organize a dataset

Public datasets can use their default download sources. ImageNet, DOTA, and
Cityscapes require the appropriate source archives or credentials.

```bash
python benchmark/organize_coco.py
python benchmark/organize_ade20k.py
python benchmark/organize_nyu_depth.py
```

For Cityscapes, provide the official archives:

```bash
python benchmark/organize_cityscapes.py \
  --image-dir path/to/leftImg8bit_trainvaltest.zip \
  --annotation-dir path/to/gtFine_trainvaltest.zip
```

## Run a benchmark

Use the unified runner for every Vision task. It chooses the evaluator from
`--task`, writes JSON/CSV/Markdown artifacts, and can create an accuracy chart.

```bash
python benchmark/benchmark_vision_models.py \
  --models ResNet50 \
  --task image_classification \
  --data-path ~/.mblt_vision/datasets/imagenet
```

Use `--framework onnx` for ONNX Runtime, `--core-mode all` to compare supported
MXQ core modes, and `--fail-fast` to stop on the first failed target. Compare
completed runs with:

```bash
python benchmark/compare_benchmark_results.py run-a run-b
```

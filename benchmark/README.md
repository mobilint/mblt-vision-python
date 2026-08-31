# Vision benchmark commands

`mblt-vision-python` is a Vision-only package, so every command script lives
directly in this directory. Reusable benchmark support—argument parsing,
artifact writing, charts, and summaries—is packaged under
`mblt_vision.benchmark` for use by library code and command scripts alike.

The old per-dataset benchmark scripts and duplicated organizer wrappers were
removed. Use the unified runner and the organizer that matches your dataset.

## Organize a dataset

Public datasets can use their default download sources. ImageNet, DOTA,
Cityscapes, and SA-V require the appropriate source archives or credentials.

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

### SA-V (mask generation)

Meta distributes SA-V through a
[form-gated download portal](https://ai.meta.com/datasets/segment-anything-video-downloads/)
and this package does not mirror it, so `--dataset-path` is required. Download
`sav_val.tar` (about 15 GB), then:

```bash
python benchmark/organize_sav.py --dataset-path path/to/sav_val.tar
```

An already-extracted `sav_val` directory is accepted in place of the archive.
The expected source layout is documented in the
[SAM 2 `sav_dataset` README](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md):

```text
sav_val
|-- sav_val.txt                                       # 155 video ids
|-- JPEGImages_24fps/{video_id}/{frame:05d}.jpg       # frames at 24fps
`-- Annotations_6fps/{video_id}/{object_id:03d}/{frame:05d}.png  # masks at 6fps
```

The organizer installs only the annotated frames, since annotations exist at
6fps while frames are extracted at 24fps and evaluation can only use annotated
frames. The result is validated before it replaces any existing cache: 155
videos, 293 masklets, and 31967 annotated masks, with every mask checked for
matching geometry and single-object values.

Organizing needs roughly 20 GB of free space for the unpacked archive plus the
installed copy; the archive itself can be deleted afterwards. Raw dataset
archives are gitignored — never commit them.

Once organized, validation reuses the cache and needs no further flags:

```bash
mblt-vision val --model sam2-hiera-large
```

## Run a benchmark

Use the unified runner for every Vision task. It chooses the evaluator from
`--task`, writes JSON/CSV/Markdown artifacts, and can create an accuracy chart.

```bash
python benchmark/benchmark_vision_models.py \
  --models ResNet50 \
  --task image_classification \
  --target-device aries-rb \
  --data-path ~/.mblt_model_zoo/datasets/imagenet
```

Use `--framework onnx` for ONNX Runtime, `--core-mode all` to compare supported
MXQ core modes, `--target-device regulus-ra` or `regulus-rb` for the corresponding
Regulus board artifact, and `--fail-fast` to stop on the first failed target. Compare
completed runs with:

```bash
python benchmark/compare_benchmark_results.py run-a run-b
```

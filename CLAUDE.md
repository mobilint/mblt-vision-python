# Claude Code Guide

@AGENTS.md

`AGENTS.md` is the canonical guide for this repository. Follow it for the Python-first
implementation, PyPI-wheel, lifecycle, and mblt-model-zoo Vision compatibility requirements.
This includes the timm pre_cfg contract — a classifier's preprocessing comes from the
timm id's default pretrained configuration, with the pre-crop resize floored as
floor(input_size / crop_pct) the way timm's own transforms_factory floors it —
the WiderFace Hard-primary and Medium/Easy-secondary metric contract,
the single-class `face_detection` postprocessing contract and its
`mblt-model-ops`-sourced input geometry (640x640, except `YOLOv8m-face` and
`YOLOv8l-face` at 960x960), plus NYU Depth delta1-primary with abs_rel/RMSE
secondary metrics.

For focused model, preprocessing, postprocessing, and model-registry work, also read
.claude/skills/mblt-vision/SKILL.md.

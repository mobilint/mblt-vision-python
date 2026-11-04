"""CPU processing and discovery contracts for additional published classifiers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import mblt_vision
from mblt_vision.utils.preprocess import build_preprocess
from mblt_vision.wrapper import resolve_model_config

# Model-specific values from the recorded ops predict_float inputs and distribution
# contracts at 6fdbf471a3bd6ec22035a42c8730fe88a68881e4. Keep the surprising TF
# normalization of ViT_Large_Patch14_CLIP_224: names do not define transforms.
SPECS = {
    "EfficientViT_B3": (235, 224, "torch"),
    "EfficientViT_L1": (224, 224, "torch"),
    "EfficientViT_L2": (224, 224, "torch"),
    "EfficientViT_L3": (224, 224, "torch"),
    "LeViT_128": (248, 224, "torch"),
    "LeViT_128S": (248, 224, "torch"),
    "LeViT_192": (248, 224, "torch"),
    "LeViT_256": (248, 224, "torch"),
    "LeViT_384": (248, 224, "torch"),
    "ViT_RelPos_Medium_Patch16_RPN_224": (248, 224, "tf"),
    "ViT_RelPos_Base_Patch32_Plus_RPN_256": (284, 256, "tf"),
    "ViT_Base_Patch16_RPN_224": (248, 224, "tf"),
    "ViT_Base_Patch16_CLIP_224": (235, 224, "openai"),
    "ViT_Base_Patch32_CLIP_448": (448, 448, "openai"),
    "ViT_Medium_Patch16_GAP_256": (269, 256, "tf"),
    "ViT_Large_Patch14_CLIP_224": (224, 224, "tf"),
    "ViT_Base_Patch32_CLIP_224": (248, 224, "openai"),
    "ViT_RelPos_Small_Patch16_224": (248, 224, "tf"),
    "ViT_Base_Patch32_CLIP_384": (384, 384, "openai"),
    "ViT_SRelPos_Medium_Patch16_224": (248, 224, "tf"),
    "ViT_Huge_Patch14_CLIP_224": (224, 224, "openai"),
    "ViT_RelPos_Base_Patch16_224": (248, 224, "tf"),
    "ViT_RelPos_Medium_Patch16_224": (248, 224, "tf"),
    "ViT_SRelPos_Small_Patch16_224": (248, 224, "tf"),
}
STATS = {
    "torch": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tf": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "openai": (
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ),
}


@pytest.mark.parametrize("model", SPECS)
def test_configs_match_recorded_ops_inputs_and_discover(model: str) -> None:
    resize, crop, style = SPECS[model]
    config = resolve_model_config(model)
    assert config["pre_cfg"] == {
        "Reader": {"style": "pil"},
        "Resize": {"size": resize, "interpolation": "bicubic"},
        "CenterCrop": {"size": [crop, crop]},
        "SetOrder": {"shape": "HWC"},
        "Normalize": {"style": style},
    }
    assert list(config["pre_cfg"]) == [
        "Reader",
        "Resize",
        "CenterCrop",
        "SetOrder",
        "Normalize",
    ]
    assert config["post_cfg"] == {"task": "image_classification", "dataset": "imagenet"}
    assert config["file_cfg"]["repo_id"] == f"mobilint/{model}"
    assert config["file_cfg"]["revision"] == "main"
    assert config["file_cfg"]["onnx_local_artifact_only"] is True
    assert (
        model in mblt_vision.list_models("image_classification")["image_classification"]
    )
    assert getattr(mblt_vision, model) is getattr(
        mblt_vision.image_classification, model
    )


@pytest.mark.parametrize("model", SPECS)
@pytest.mark.parametrize("shape", [(277, 391), (399, 271), (257, 257)])
def test_cpu_numeric_parity_with_ops_adapter_math(
    model: str, shape: tuple[int, int], tmp_path
) -> None:
    resize, crop, style = SPECS[model]
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    )
    source = tmp_path / "input.png"
    image.save(source)
    # Independent equivalent of ops customer_preprocessing_imagenet_torch.py;
    # TF/OpenAI adapters use the same resize/crop with their own recorded stats.
    width, height = image.size
    target = (
        (resize, int(resize * height / width))
        if width <= height
        else (int(resize * width / height), resize)
    )
    image = image.resize(target, Image.Resampling.BICUBIC)
    left, top = round((image.width - crop) / 2), round((image.height - crop) / 2)
    reference = (
        np.asarray(image.crop((left, top, left + crop, top + crop)), dtype=np.float32)
        / 255
    )
    mean, std = STATS[style]
    reference = (reference - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    actual = build_preprocess(resolve_model_config(model)["pre_cfg"])(str(source))
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, reference, rtol=0, atol=5e-7)

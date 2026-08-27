"""Host-side SAM2 glue: feature-map bookkeeping, prompt encoding, and mask resizing.

Only the two heavy backbone/mask-decoder networks run on the NPU (via
``MobilintNPUBackend``, see ``sam2.py``). Everything here runs on the host with
plain ``torch`` and the tiny prompt-encoder weights in ``_sam2_prompt.py`` --
no dependency on the ``sam2`` package, no ``torchvision``, no manually cloned
repository, and no ~900MB full checkpoint download. Ported from the validated
``sam2-mxq-pipeline`` reference, restructured to take plain tensors (feature
maps, weights) instead of mutating a live ``SAM2ImagePredictor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from . import _sam2_prompt as prompt

_FPN_CHANNELS = (32, 64, 256)

_NORMALIZE_MEAN = torch.tensor(prompt.NORMALIZE_MEAN, dtype=torch.float32).view(
    -1, 1, 1
)
_NORMALIZE_STD = torch.tensor(prompt.NORMALIZE_STD, dtype=torch.float32).view(-1, 1, 1)


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB ``uint8`` array."""

    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def preprocess_encoder_input(image: np.ndarray) -> np.ndarray:
    """Resize/normalize an RGB image into the encoder MXQ's NHWC input.

    Ported from ``SAM2Transforms.__call__`` (``ToTensor`` then
    ``Resize((1024, 1024))`` + ImageNet normalize) in plain torch, verified
    bit-for-bit against the torchvision pipeline it replaces: ``ToTensor``
    scales only ``uint8`` inputs by 1/255, and tensor ``Resize`` is bilinear
    with ``antialias=True``.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {array.shape}.")
    # HWC -> CHW as a fresh contiguous copy; PIL-backed arrays are read-only
    # and torch.from_numpy warns on non-writable memory.
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    if not chw.flags.writeable:
        chw = chw.copy()
    tensor = torch.from_numpy(chw)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    else:
        tensor = tensor.float()
    tensor = functional.interpolate(
        tensor[None],
        size=prompt.INPUT_IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    tensor = (tensor - _NORMALIZE_MEAN) / _NORMALIZE_STD
    return np.ascontiguousarray(
        tensor.permute(0, 2, 3, 1).float().cpu().numpy(), dtype=np.float32
    )


def fpn_from_runtime(
    outputs: Sequence[np.ndarray], device: torch.device
) -> list[torch.Tensor]:
    """Convert the three raw encoder-MXQ outputs into ordered NCHW FPN levels.

    Returns ``[feat_32ch, feat_64ch, feat_256ch]`` -- the two high-resolution
    skip-connection levels followed by the deepest ``image_embed`` level --
    identified by channel count rather than position, since qbruntime does
    not guarantee the runtime output order.
    """

    features: dict[int, torch.Tensor] = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            continue
        if array.shape[-1] in _FPN_CHANNELS:
            channel = int(array.shape[-1])
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[
                None
            ]
        elif array.shape[0] in _FPN_CHANNELS:
            channel = int(array.shape[0])
            tensor = torch.from_numpy(np.ascontiguousarray(array))[None]
        else:
            continue
        if channel in features:
            raise ValueError(f"Duplicate encoder output with {channel} channels.")
        features[channel] = tensor.to(device)

    missing = [channel for channel in _FPN_CHANNELS if channel not in features]
    if missing:
        raise ValueError(
            f"Encoder outputs are missing FPN channel count(s) {missing}; "
            f"got shapes {[np.asarray(o).shape for o in outputs]}."
        )
    return [features[32], features[64], features[256]]


def build_backbone_features(
    weights: dict[str, torch.Tensor], feature_maps: Sequence[torch.Tensor]
) -> dict[str, list[torch.Tensor]]:
    """Ported from SAM2's ``_prepare_backbone_features`` for a single image
    (no video memory: ``directly_add_no_mem_embed`` is always applied)."""

    vision_features = [value.flatten(2).permute(2, 0, 1) for value in feature_maps]
    if prompt.DIRECTLY_ADD_NO_MEM_EMBED:
        vision_features[-1] = vision_features[-1] + weights["no_mem_embed"]
    features = [
        feature.permute(1, 2, 0).view(1, -1, *feature_size)
        for feature, feature_size in zip(
            vision_features[::-1], prompt.BB_FEAT_SIZES[::-1]
        )
    ][::-1]
    return {"image_embed": features[-1], "high_res_feats": features[:-1]}


def prepare_decoder_tensors(
    weights: dict[str, torch.Tensor],
    features: dict[str, list[torch.Tensor]],
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> dict[str, np.ndarray]:
    """Run only the host prompt encoder and build the six compiled decoder inputs."""

    points_tensor = torch.as_tensor(points, dtype=torch.float32)[None, ...]
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)[None, ...]
    unnorm_coords = prompt.transform_points(points_tensor, tuple(original_hw))

    sparse = prompt.embed_points(weights, unnorm_coords, labels_tensor)
    dense = prompt.dense_embeddings_for_no_mask(weights, batch_size=sparse.size(0))

    image_embeddings = features["image_embed"][-1].unsqueeze(0)
    high_res = [value[-1].unsqueeze(0) for value in features["high_res_feats"]]
    tokens, src, pos_src = prompt.decoder_token_prep(
        weights,
        image_embeddings=image_embeddings.float(),
        dense_prompt_embeddings=dense.float(),
        sparse_prompt_embeddings=sparse.float(),
    )

    def sequence(value: torch.Tensor) -> torch.Tensor:
        return (
            value.flatten(2)
            .transpose(1, 2)
            .reshape(1, 1, -1, value.shape[1])
            .contiguous()
        )

    src_sequence = sequence(src)
    pos_sequence = sequence(pos_src)
    tensors = {
        "hrf0_nhwc": high_res[0].permute(0, 2, 3, 1).contiguous(),
        "hrf1_nhwc": high_res[1].permute(0, 2, 3, 1).contiguous(),
        "src": src_sequence,
        "tokens": tokens.reshape(1, 1, -1, prompt.EMBED_DIM).contiguous(),
        "pos_src": pos_sequence,
        "src_plus_pos_src": src_sequence + pos_sequence,
    }
    return {
        name: np.ascontiguousarray(
            value.detach().float().cpu().numpy(), dtype=np.float32
        )
        for name, value in tensors.items()
    }


def postprocess_masks(
    low_resolution_masks: np.ndarray, original_hw: Sequence[int]
) -> np.ndarray:
    """Resize low-res decoder mask logits back to the original image size.

    Ported from ``SAM2Transforms.postprocess_masks`` (``max_hole_area`` /
    ``max_sprinkle_area`` are 0 by default upstream, so that branch is a
    no-op and is not ported).
    """

    tensor = torch.from_numpy(
        np.ascontiguousarray(low_resolution_masks, dtype=np.float32)
    )[None]
    masks = functional.interpolate(
        tensor, tuple(original_hw), mode="bilinear", align_corners=False
    )[0]
    return masks.detach().float().cpu().numpy()

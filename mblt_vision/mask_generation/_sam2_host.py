"""Host-side SAM2 glue: feature-map bookkeeping, prompt encoding, and mask resizing.

Only the two heavy backbone/mask-decoder networks run on a backend -- the NPU
via ``MobilintNPUBackend`` or ONNX Runtime via ``ONNXBackend`` (see
``sam2.py``). Everything here runs on the host with
plain ``torch`` and the tiny prompt-encoder weights in ``_sam2_prompt.py`` --
no dependency on the ``sam2`` package, no ``torchvision``, no manually cloned
repository, and no ~900MB full checkpoint download. Ported from the validated
``sam2-mxq-pipeline`` reference, restructured to take plain tensors (feature
maps, weights) instead of mutating a live ``SAM2ImagePredictor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as functional

from . import _sam2_prompt as prompt


class BackboneFeatures(TypedDict):
    """``build_backbone_features``'s two differently-shaped outputs.

    ``image_embed`` is a single deepest-level feature tensor; ``high_res_feats``
    is the list of shallower skip-connection tensors -- a plain
    ``dict[str, list[Tensor]]`` cannot express that the two keys hold different
    types.
    """

    image_embed: torch.Tensor
    high_res_feats: list[torch.Tensor]


_FPN_CHANNELS = (32, 64, 256)
# Each FPN level's full (channels, height, width). `build_backbone_features`
# `view()`s these into `prompt.BB_FEAT_SIZES`, so a channel-correct level with
# the wrong spatial geometry but the same element count -- (1, 32, 128, 512)
# for (1, 32, 256, 256) -- would be silently rearranged into a plausible but
# corrupted FPN, after which the decoder feed still has the expected shape and
# passes its own validation. Pin the complete shape, not just the channel count.
_FPN_LEVEL_SHAPES: dict[int, tuple[int, int]] = dict(
    zip(_FPN_CHANNELS, prompt.BB_FEAT_SIZES)
)

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
    # Interpolation and normalization preserve NaN/infinity, so a non-finite
    # source image would reach the backend as an invalid encoder tensor and
    # fail in a backend-specific way -- by the time the decoder-output
    # finiteness check fires, the source is no longer identifiable.
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
        or array.dtype == np.bool_
    ):
        raise ValueError(f"Expected a numeric RGB image, got dtype {array.dtype}.")
    if np.issubdtype(array.dtype, np.floating) and not bool(np.isfinite(array).all()):
        raise ValueError("Source image must be finite; got NaN or infinity.")
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
        expected_spatial = _FPN_LEVEL_SHAPES[channel]
        if tuple(tensor.shape[2:]) != expected_spatial:
            raise ValueError(
                f"Encoder output with {channel} channels must be "
                f"{expected_spatial} spatially, got {tuple(tensor.shape[2:])}. "
                "A same-element-count geometry would be silently rearranged into "
                "a corrupted FPN."
            )
        features[channel] = tensor.to(device)

    missing = [channel for channel in _FPN_CHANNELS if channel not in features]
    if missing:
        raise ValueError(
            f"Encoder outputs are missing FPN channel count(s) {missing}; "
            f"got shapes {[np.asarray(o).shape for o in outputs]}."
        )
    return [features[32], features[64], features[256]]


def fpn_from_onnx(
    outputs: Sequence[np.ndarray], device: torch.device
) -> list[torch.Tensor]:
    """Convert the three batched-NCHW encoder-ONNX outputs into ordered FPN levels.

    Returns ``[feat_32ch, feat_64ch, feat_256ch]`` like :func:`fpn_from_runtime`.
    The exported graph declares batched NCHW outputs, so the channel axis is
    fixed at axis 1 -- unlike the MXQ runtime path, a square spatial size can
    never be mistaken for a channel count, and any other layout is rejected.
    """

    features: dict[int, torch.Tensor] = {}
    shapes = [tuple(np.asarray(output).shape) for output in outputs]
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if (
            array.ndim != 4
            or array.shape[0] != 1
            or array.shape[1] not in _FPN_CHANNELS
        ):
            raise ValueError(
                f"Unexpected encoder ONNX output shape {array.shape}; expected "
                f"(1, C, H, W) with C in {_FPN_CHANNELS}. Got shapes {shapes}."
            )
        channel = int(array.shape[1])
        expected_spatial = _FPN_LEVEL_SHAPES[channel]
        if tuple(array.shape[2:]) != expected_spatial:
            raise ValueError(
                f"Encoder ONNX output with {channel} channels must be "
                f"{expected_spatial} spatially, got {tuple(array.shape[2:])}. "
                "A same-element-count geometry would be silently rearranged into "
                "a corrupted FPN."
            )
        if channel in features:
            raise ValueError(f"Duplicate encoder output with {channel} channels.")
        features[channel] = torch.from_numpy(np.ascontiguousarray(array)).to(device)

    missing = [channel for channel in _FPN_CHANNELS if channel not in features]
    if missing:
        raise ValueError(
            f"Encoder ONNX outputs are missing FPN channel count(s) {missing}; "
            f"got shapes {shapes}."
        )
    return [features[32], features[64], features[256]]


def build_backbone_features(
    weights: dict[str, torch.Tensor], feature_maps: Sequence[torch.Tensor]
) -> BackboneFeatures:
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


def _as_float32_arrays(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert named torch tensors into contiguous float32 numpy arrays."""

    return {
        name: np.ascontiguousarray(
            value.detach().float().cpu().numpy(), dtype=np.float32
        )
        for name, value in tensors.items()
    }


def _decoder_prompt_tensors(
    weights: dict[str, torch.Tensor],
    features: BackboneFeatures,
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run the host prompt encoder shared by the MXQ and ONNX decoder feeds.

    Returns ``(tokens, src, pos_src, high_res)`` in the pre-flattening shapes
    the mask decoder was traced with: ``tokens (1, N+7, 256)``,
    ``src``/``pos_src`` NCHW ``(1, 256, 64, 64)``, and the two NCHW
    high-resolution feature maps.
    """

    sparse, dense, image_embeddings, high_res = _prompt_encoder_outputs(
        weights, features, points, labels, original_hw
    )
    tokens, src, pos_src = prompt.decoder_token_prep(
        weights,
        image_embeddings=image_embeddings.float(),
        dense_prompt_embeddings=dense.float(),
        sparse_prompt_embeddings=sparse.float(),
    )
    return tokens, src, pos_src, high_res


def _prompt_encoder_outputs(
    weights: dict[str, torch.Tensor],
    features: BackboneFeatures,
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run the host prompt encoder and return its raw outputs, pre-assembly.

    ``(sparse, dense, image_embeddings, high_res)`` -- the stage *before*
    ``decoder_token_prep``, which is what the bridged decoder contract consumes:
    its MBLT carries a host-bridge subgraph that does the token concatenation
    and the ``image_embeddings + dense`` sum inside the graph.
    """

    # Prompts arrive as host numpy arrays, so place them on the same device the
    # weights live on: the prompt encoder combines them with the Gaussian
    # position-encoding matrix and the learned embeddings, which torch requires
    # to share a device once the engine was constructed with (or moved to) CUDA.
    device = weights["positional_encoding_gaussian_matrix"].device
    points_tensor = torch.as_tensor(points, dtype=torch.float32, device=device)[
        None, ...
    ]
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64, device=device)[None, ...]
    height, width = original_hw
    unnorm_coords = prompt.transform_points(points_tensor, (int(height), int(width)))

    sparse = prompt.embed_points(weights, unnorm_coords, labels_tensor)
    dense = prompt.dense_embeddings_for_no_mask(weights, batch_size=sparse.size(0))
    image_embeddings = features["image_embed"][-1].unsqueeze(0)
    high_res = [value[-1].unsqueeze(0) for value in features["high_res_feats"]]
    return sparse, dense, image_embeddings, high_res


def prepare_decoder_tensors_bridged(
    weights: dict[str, torch.Tensor],
    features: BackboneFeatures,
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> dict[str, np.ndarray]:
    """Build the six bridged-contract decoder inputs from the prompt encoder.

    Keyed by semantic role because three of the six share the shape
    ``(1, 256, 64, 64)`` -- ``image_embeddings``, ``dense_prompt_embeddings``,
    and ``image_pe`` -- so a positional guess would silently swap them.
    """

    sparse, dense, image_embeddings, high_res = _prompt_encoder_outputs(
        weights, features, points, labels, original_hw
    )
    tensors = {
        "image_embeddings": image_embeddings.float(),
        "dense_prompt_embeddings": dense.float(),
        "image_pe": prompt.get_dense_pe(weights).float(),
        # (1, N+1, 256) -> (1, 1, N+1, 256); axis 2 is the dynamic prompt axis.
        "sparse_prompt_embeddings_0": sparse.float().unsqueeze(1).contiguous(),
        "high_res_features0_0": high_res[0].permute(0, 2, 3, 1).contiguous(),
        "high_res_features1_0": high_res[1].permute(0, 2, 3, 1).contiguous(),
    }
    return _as_float32_arrays(tensors)


def prepare_decoder_tensors(
    weights: dict[str, torch.Tensor],
    features: BackboneFeatures,
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> dict[str, np.ndarray]:
    """Run only the host prompt encoder and build the six compiled decoder inputs."""

    tokens, src, pos_src, high_res = _decoder_prompt_tensors(
        weights, features, points, labels, original_hw
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
    return _as_float32_arrays(tensors)


def prepare_decoder_tensors_onnx(
    weights: dict[str, torch.Tensor],
    features: BackboneFeatures,
    points: np.ndarray,
    labels: np.ndarray,
    original_hw: Sequence[int],
) -> dict[str, np.ndarray]:
    """Run the host prompt encoder and build the five named ONNX decoder inputs.

    The exported decoder consumes the pre-flattening NCHW tensors the graph was
    traced with; ``src + pos_src`` stays inside the graph, unlike the compiled
    MXQ artifact's flattened six-input runtime signature.
    """

    tokens, src, pos_src, high_res = _decoder_prompt_tensors(
        weights, features, points, labels, original_hw
    )
    return _as_float32_arrays(
        {
            "tokens": tokens.contiguous(),
            "src": src.contiguous(),
            "pos_src": pos_src.contiguous(),
            "high_res_features_0": high_res[0].contiguous(),
            "high_res_features_1": high_res[1].contiguous(),
        }
    )


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

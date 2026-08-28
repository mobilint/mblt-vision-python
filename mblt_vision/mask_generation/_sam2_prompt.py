"""Self-contained SAM2 Hiera-Large host-side prompt encoding.

Ported line-for-line from the official ``facebookresearch/sam2`` modules
(``modeling/position_encoding.py::PositionEmbeddingRandom``,
``modeling/sam/prompt_encoder.py::PromptEncoder``,
``modeling/sam/mask_decoder.py``'s token embeddings, and
``sam2_image_predictor.py``'s ``_prep_prompts``/``_bb_feat_sizes``), restricted
to the point-only prompt path this package supports (no box prompts, no mask
prompts). Depends on nothing from the ``sam2`` package -- the only weights
needed for this path are the ~3k floats extracted from the official
``facebook/sam2-hiera-large`` checkpoint into ``sam2_hiera_large_prompt_weights.pt``
(point/"not-a-point"/"no-mask" embeddings, the random Fourier position-encoding
matrix, the decoder's IoU/mask/object-score tokens, and the no-memory-embedding
parameter) -- not the ~900MB full checkpoint, which also carries the Hiera
backbone and video-memory modules this package never runs on the host (the
backbone runs on the NPU; there is no video memory in single-image inference).

That small bundle lives at ``mobilint/sam2-hiera-large``'s Hub repo root as
``sam2_hiera_large_prompt_weights.pt``, downloaded the same way as the
encoder/decoder MXQ artifacts (see ``sam2.py``) -- not shipped as package
data, consistent with every other model's artifacts living on the Hub rather
than in the wheel.

Numerically verified against the real ``facebookresearch/sam2`` predictor's
``sam_prompt_encoder``/``sam_mask_decoder`` on the same weights.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

# Fixed SAM2 Hiera-Large configuration (facebookresearch/sam2, sam2_hiera_l.yaml
# and modeling/sam2_base.py::_build_sam_heads); none of these are learned.
EMBED_DIM = 256
IMAGE_EMBEDDING_SIZE = (64, 64)  # image_size // backbone_stride == 1024 // 16
INPUT_IMAGE_SIZE = (1024, 1024)
BB_FEAT_SIZES = ((256, 256), (128, 128), (64, 64))
NUM_MASK_TOKENS = 4  # num_multimask_outputs (3) + 1
USE_MULTIMASK_TOKEN_FOR_OBJ_PTR = True
PRED_OBJ_SCORES = True
DIRECTLY_ADD_NO_MEM_EMBED = True
MASK_THRESHOLD = 0.0

# SAM2's ImageNet-style input normalization (sam2/utils/transforms.py).
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def load_prompt_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Load the point/label/token embeddings extracted from the official
    ``facebook/sam2-hiera-large`` checkpoint."""

    return torch.load(path, map_location="cpu", weights_only=True)


def positional_encoding_for_grid(
    gaussian_matrix: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Dense positional encoding for an ``size`` grid.

    Ported from ``PositionEmbeddingRandom.forward``. Used once per model
    (``get_dense_pe()``); ``size`` is always ``IMAGE_EMBEDDING_SIZE`` here.
    """

    h, w = size
    device = gaussian_matrix.device
    grid = torch.ones((h, w), device=device, dtype=torch.float32)
    y_embed = (grid.cumsum(dim=0) - 0.5) / h
    x_embed = (grid.cumsum(dim=1) - 0.5) / w
    coords = torch.stack([x_embed, y_embed], dim=-1)
    pe = _pe_encoding(coords, gaussian_matrix)
    return pe.permute(2, 0, 1)  # C x H x W


def _pe_encoding(coords: torch.Tensor, gaussian_matrix: torch.Tensor) -> torch.Tensor:
    """Ported from ``PositionEmbeddingRandom._pe_encoding``. ``coords`` in [0, 1]."""

    coords = 2 * coords - 1
    coords = coords @ gaussian_matrix
    coords = 2 * math.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


def _forward_with_coords(
    coords_input: torch.Tensor,
    image_size: tuple[int, int],
    gaussian_matrix: torch.Tensor,
) -> torch.Tensor:
    """Ported from ``PositionEmbeddingRandom.forward_with_coords``."""

    coords = coords_input.clone()
    coords[..., 0] = coords[..., 0] / image_size[1]
    coords[..., 1] = coords[..., 1] / image_size[0]
    return _pe_encoding(coords.to(torch.float32), gaussian_matrix)


def embed_points(
    weights: dict[str, torch.Tensor], points: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Sparse point-prompt embeddings.

    Ported from ``PromptEncoder._embed_points`` and ``PromptEncoder.forward``'s
    points branch, restricted to the point-only path (``pad=True``, no boxes):
    always pads with one "not a point" token, matching this package's prompts
    (there is never a box prompt to pad against instead).

    Args:
        points: ``(B, N, 2)`` pixel coordinates in the 1024x1024 encoder input
            space (already produced by :func:`transform_points`).
        labels: ``(B, N)`` point labels (``1`` positive, ``0`` negative).

    Returns:
        ``(B, N + 1, embed_dim)`` sparse embeddings (the ``+1`` is the padding
        "not a point" token SAM2 always appends when there is no box prompt).
    """

    points = points + 0.5  # shift to center of pixel
    # Built on the incoming tensors' own device (not the implicit CPU default),
    # so the concatenation below still works when the engine was constructed
    # with device="cuda". Numerically identical to the upstream port.
    padding_point = torch.zeros(
        (points.shape[0], 1, 2), dtype=points.dtype, device=points.device
    )
    padding_label = -torch.ones(
        (labels.shape[0], 1), dtype=labels.dtype, device=labels.device
    )
    points = torch.cat([points, padding_point], dim=1)
    labels = torch.cat([labels, padding_label], dim=1)

    point_embedding = _forward_with_coords(
        points, INPUT_IMAGE_SIZE, weights["positional_encoding_gaussian_matrix"]
    )
    is_padding = (labels == -1).unsqueeze(-1)
    point_embedding = torch.where(
        is_padding,
        torch.zeros_like(point_embedding) + weights["not_a_point_embed_weight"],
        point_embedding,
    )
    is_negative = (labels == 0).unsqueeze(-1)
    point_embedding = torch.where(
        is_negative,
        point_embedding + weights["point_embedding_negative"],
        point_embedding,
    )
    is_positive = (labels == 1).unsqueeze(-1)
    point_embedding = torch.where(
        is_positive,
        point_embedding + weights["point_embedding_positive"],
        point_embedding,
    )
    return point_embedding


def dense_embeddings_for_no_mask(
    weights: dict[str, torch.Tensor], batch_size: int
) -> torch.Tensor:
    """Dense embeddings when no mask prompt is given.

    Ported from ``PromptEncoder.forward``'s ``masks is None`` branch.
    """

    no_mask_embed = weights["no_mask_embed_weight"]
    return no_mask_embed.reshape(1, -1, 1, 1).expand(
        batch_size, -1, IMAGE_EMBEDDING_SIZE[0], IMAGE_EMBEDDING_SIZE[1]
    )


def get_dense_pe(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Ported from ``PromptEncoder.get_dense_pe``."""

    return positional_encoding_for_grid(
        weights["positional_encoding_gaussian_matrix"], IMAGE_EMBEDDING_SIZE
    ).unsqueeze(0)


def transform_points(
    points: torch.Tensor, original_hw: tuple[int, int]
) -> torch.Tensor:
    """Normalize original-image-pixel point coordinates into the encoder's
    1024x1024 input space.

    Ported from ``SAM2Transforms.transform_coords`` with ``normalize=True``.
    """

    height, width = original_hw
    coords = points.clone()
    coords[..., 0] = coords[..., 0] / width
    coords[..., 1] = coords[..., 1] / height
    return coords * INPUT_IMAGE_SIZE[0]


def decoder_token_prep(
    weights: dict[str, torch.Tensor],
    image_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host-side token concatenation and image/dense embedding sum used by the
    compiled decoder MXQ.

    Ported from the reference ``sam2-mxq-pipeline``'s ``sam2_decoder_prep``,
    itself ported from the mask decoder's ``forward``/``predict_masks`` token
    setup, restricted to this package's fixed, validated decoder contract
    (``pred_obj_scores=True``, ``use_multimask_token_for_obj_ptr=True``).
    """

    output_tokens = torch.cat(
        [
            weights["obj_score_token_weight"],
            weights["iou_token_weight"],
            weights["mask_tokens_weight"],
        ],
        dim=0,
    )
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0), -1, -1
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
    image_pe = get_dense_pe(weights)
    return tokens, image_embeddings + dense_prompt_embeddings, image_pe

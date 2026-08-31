"""Differential test: our from-scratch prompt encoding vs. the real
``facebookresearch/sam2`` predictor, on the same extracted weights.

Opt-in and network-dependent (downloads the official checkpoint from Hugging
Face and needs a real ``sam2`` install available on ``sys.path`` -- the
official package, never the unofficial PyPI mirror; see
``mblt_vision/mask_generation/_sam2_host.py``). Skips cleanly when ``sam2``
is not importable, since this is a fidelity check for maintainers changing
``_sam2_prompt.py``/``_sam2_host.py``, not a normal unit test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

sam2_image_predictor = pytest.importorskip("sam2.sam2_image_predictor")

from mblt_vision.mask_generation import _sam2_host as host  # noqa: E402
from mblt_vision.mask_generation import _sam2_prompt as prompt  # noqa: E402

pytestmark = pytest.mark.requires_network


@pytest.fixture(scope="module")
def real_predictor():
    predictor = sam2_image_predictor.SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large"
    )
    predictor.model.to(torch.device("cpu")).eval()
    return predictor


@pytest.fixture(scope="module")
def our_weights(real_predictor):
    encoder = real_predictor.model.sam_prompt_encoder
    decoder = real_predictor.model.sam_mask_decoder
    return {
        "no_mem_embed": real_predictor.model.no_mem_embed.detach().clone(),
        "iou_token_weight": decoder.iou_token.weight.detach().clone(),
        "mask_tokens_weight": decoder.mask_tokens.weight.detach().clone(),
        "obj_score_token_weight": decoder.obj_score_token.weight.detach().clone(),
        "positional_encoding_gaussian_matrix": encoder.pe_layer.positional_encoding_gaussian_matrix.detach().clone(),
        "point_embedding_negative": encoder.point_embeddings[0].weight.detach().clone(),
        "point_embedding_positive": encoder.point_embeddings[1].weight.detach().clone(),
        "not_a_point_embed_weight": encoder.not_a_point_embed.weight.detach().clone(),
        "no_mask_embed_weight": encoder.no_mask_embed.weight.detach().clone(),
    }


def test_config_constants_match_the_real_model(real_predictor) -> None:
    encoder = real_predictor.model.sam_prompt_encoder
    decoder = real_predictor.model.sam_mask_decoder
    assert encoder.embed_dim == prompt.EMBED_DIM
    assert tuple(encoder.image_embedding_size) == prompt.IMAGE_EMBEDDING_SIZE
    assert tuple(encoder.input_image_size) == prompt.INPUT_IMAGE_SIZE
    assert (
        tuple(tuple(size) for size in real_predictor._bb_feat_sizes)
        == prompt.BB_FEAT_SIZES
    )
    assert decoder.num_mask_tokens == prompt.NUM_MASK_TOKENS
    assert (
        decoder.use_multimask_token_for_obj_ptr
        == prompt.USE_MULTIMASK_TOKEN_FOR_OBJ_PTR
    )
    assert decoder.pred_obj_scores == prompt.PRED_OBJ_SCORES
    assert (
        real_predictor.model.directly_add_no_mem_embed
        == prompt.DIRECTLY_ADD_NO_MEM_EMBED
    )
    assert real_predictor.mask_threshold == prompt.MASK_THRESHOLD


@pytest.mark.parametrize("num_points", [1, 2, 3])
def test_sparse_and_dense_embeddings_match_bit_for_bit(
    real_predictor, our_weights, num_points: int
) -> None:
    rng = np.random.default_rng(num_points)
    original_hw = (480, 640)
    points_np = np.stack(
        [rng.uniform(0, 640, size=num_points), rng.uniform(0, 480, size=num_points)],
        axis=-1,
    ).astype(np.float32)
    labels_np = rng.integers(0, 2, size=num_points).astype(np.int64)

    real_encoder = real_predictor.model.sam_prompt_encoder
    point_coords = torch.as_tensor(points_np, dtype=torch.float32)[None, ...]
    real_unnorm = real_predictor._transforms.transform_coords(
        point_coords, normalize=True, orig_hw=original_hw
    )
    real_labels = torch.as_tensor(labels_np, dtype=torch.int32)[None, ...]
    real_sparse, real_dense = real_encoder(
        points=(real_unnorm, real_labels), boxes=None, masks=None
    )
    real_dense_pe = real_encoder.get_dense_pe()

    our_unnorm = prompt.transform_points(point_coords.clone(), original_hw)
    our_labels = torch.as_tensor(labels_np, dtype=torch.int64)[None, ...]
    our_sparse = prompt.embed_points(our_weights, our_unnorm, our_labels)
    our_dense = prompt.dense_embeddings_for_no_mask(our_weights, batch_size=1)
    our_dense_pe = prompt.get_dense_pe(our_weights)

    assert torch.equal(real_unnorm, our_unnorm)
    assert torch.equal(real_sparse, our_sparse)
    assert torch.equal(real_dense, our_dense)
    assert torch.equal(real_dense_pe, our_dense_pe)


def test_decoder_token_prep_matches_bit_for_bit(real_predictor, our_weights) -> None:
    decoder = real_predictor.model.sam_mask_decoder
    sparse = torch.zeros(1, 2, prompt.EMBED_DIM)
    dense = torch.zeros(1, prompt.EMBED_DIM, 64, 64)
    image_embeddings = torch.randn(1, prompt.EMBED_DIM, 64, 64)

    real_output_tokens = (
        torch.cat(
            [
                decoder.obj_score_token.weight,
                decoder.iou_token.weight,
                decoder.mask_tokens.weight,
            ],
            dim=0,
        )
        .unsqueeze(0)
        .expand(sparse.size(0), -1, -1)
    )
    real_tokens = torch.cat((real_output_tokens, sparse), dim=1)
    real_src = image_embeddings + dense

    our_tokens, our_src, our_pe = prompt.decoder_token_prep(
        our_weights,
        image_embeddings=image_embeddings,
        dense_prompt_embeddings=dense,
        sparse_prompt_embeddings=sparse,
    )

    assert torch.equal(real_tokens, our_tokens)
    assert torch.equal(real_src, our_src)
    assert torch.equal(real_predictor.model.sam_prompt_encoder.get_dense_pe(), our_pe)


def test_preprocess_and_postprocess_match_bit_for_bit(real_predictor) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    real_pre = real_predictor._transforms(np.ascontiguousarray(image))[None, ...]
    real_pre = real_pre.permute(0, 2, 3, 1).float().cpu().numpy()
    our_pre = host.preprocess_encoder_input(image)
    assert np.array_equal(real_pre, our_pre)

    low_res = rng.standard_normal((3, 256, 256)).astype(np.float32)
    real_post = (
        real_predictor._transforms.postprocess_masks(
            torch.from_numpy(low_res)[None], (480, 640)
        )[0]
        .detach()
        .numpy()
    )
    our_post = host.postprocess_masks(low_res, (480, 640))
    assert np.array_equal(real_post, our_post)

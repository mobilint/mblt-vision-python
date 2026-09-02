"""Fixed contracts between the SAM2 encoder/decoder artifacts and their host glue.

MXQ side: two decoder generations exist, distinguished by where SAM2's token
assembly runs, and both are supported. The loaded artifact's declared input
shapes identify which one it is (:func:`detect_decoder_contract`):

* **assembled** -- the host concatenates the output tokens and sums
  ``image_embeddings + dense_prompt_embeddings``, feeding six flattened
  tensors. Ported from the validated ``sam2-mxq-pipeline`` reference (real
  Aries2 SA-V-200 accuracy: FP32 mIoU 0.7750 vs MXQ mIoU 0.7757, mask
  agreement 0.983). Emits four outputs.
* **bridged** -- the decoder MBLT carries a host-bridge subgraph that does the
  token concat and the embedding sum itself, so the artifact takes the prompt
  encoder's raw outputs. Produced by the SDK tutorial's
  ``sam2_decoder_to_mblt.py`` (legacy-parser route). Emits two outputs
  (masks and IoU; the parse's ``output_meta`` drops the SAM tokens and
  object score).

Compile and calibration are out of scope for this phase, so each contract is
a fixed validated signature rather than a configurable MBLT-input-name
binding map.

ONNX side: the graph interface written by the SDK tutorial's
``sam2_export_onnx.py`` (``Sam2ImageEncoderWrapper``/``Sam2MaskDecoderWrapper``
traces, verified numerically against the official ``facebookresearch/sam2``
predictor). Unlike the compiled MXQ artifacts, the ONNX graphs are NCHW, keep
the pre-flattening decoder tensor shapes, and take five named decoder inputs --
``src_plus_pos_src`` stays inside the graph instead of being a sixth input.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Decoder runtime input order as fed to ``MobilintNPUBackend`` / ``qbruntime.Model.infer``.
# This is the positional signature of the assembled-contract SAM2 Hiera-Large
# decoder MXQ, which differs from that MBLT graph's declared input names.
DECODER_RUNTIME_ORDER: tuple[str, ...] = (
    "hrf0_nhwc",
    "src_plus_pos_src",
    "hrf1_nhwc",
    "src",
    "pos_src",
    "tokens",
)

# Bridged-contract runtime order. Unlike the assembled artifact, this one
# matches the MBLT input-name order, because the graph's own bridge subgraph
# consumes the prompt encoder's raw outputs directly.
DECODER_RUNTIME_ORDER_BRIDGED: tuple[str, ...] = (
    "image_embeddings",
    "dense_prompt_embeddings",
    "image_pe",
    "sparse_prompt_embeddings_0",
    "high_res_features0_0",
    "high_res_features1_0",
)

# Declared input signatures (batch stripped, as qbruntime reports them).
# ``-1`` is the prompt-dependent dynamic axis. These are what
# :func:`detect_decoder_contract` matches against.
DECODER_MXQ_INPUT_SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "assembled": (
        (256, 256, 32),
        (1, 4096, 256),
        (128, 128, 64),
        (1, 4096, 256),
        (1, 4096, 256),
        (1, -1, 256),
    ),
    "bridged": (
        (256, 64, 64),
        (256, 64, 64),
        (256, 64, 64),
        (1, -1, 256),
        (256, 256, 32),
        (128, 128, 64),
    ),
}


def detect_decoder_contract(shapes: Sequence[Sequence[int]]) -> str:
    """Identify which decoder generation a loaded artifact is, from its shapes.

    The two signatures share no prefix -- input 0 is ``(256, 256, 32)`` versus
    ``(256, 64, 64)`` -- so declared shapes are sufficient. A ``-1`` in the
    signature accepts any value there (the artifact may declare the axis
    dynamic or, if compiled for a single prompt length, fixed). Anything
    matching neither raises rather than guessing, since a wrong contract feeds
    tensors whose *roles* are wrong even where shapes coincide.
    """

    got = [tuple(int(dim) for dim in shape) for shape in shapes]
    for name, signature in DECODER_MXQ_INPUT_SHAPES.items():
        if len(got) != len(signature):
            continue
        if all(
            len(have) == len(want)
            and all(w == -1 or h == w for h, w in zip(have, want))
            for have, want in zip(got, signature)
        ):
            return name
    raise ValueError(
        f"Decoder artifact input shapes {got} match neither known contract: "
        f"assembled {list(DECODER_MXQ_INPUT_SHAPES['assembled'])} nor "
        f"bridged {list(DECODER_MXQ_INPUT_SHAPES['bridged'])}."
    )


# Exported ONNX graph interface. ``-1`` marks the prompt-count-dependent
# dynamic token axis (``6 output tokens + N points + 1 pad``).
ENCODER_ONNX_INPUT_NAME = "input_image_0"
ENCODER_ONNX_INPUT_SHAPE: tuple[int, ...] = (1, 1024, 1024, 3)
DECODER_ONNX_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "image_embeddings": (1, 256, 64, 64),
    "dense_prompt_embeddings": (1, 256, 64, 64),
    "image_pe": (1, 256, 64, 64),
    "sparse_prompt_embeddings_0": (1, 1, -1, 256),
    "high_res_features0_0": (1, 256, 256, 32),
    "high_res_features1_0": (1, 128, 128, 64),
}
DECODER_ONNX_INPUT_NAMES: tuple[str, ...] = tuple(DECODER_ONNX_INPUT_SHAPES)

MASK_SIDE = 256
MASK_AREA = MASK_SIDE * MASK_SIDE
# SAM2's multimask output (4 mask tokens minus the single-mask token), baked
# into both compiled artifacts. `eval_sav.CANDIDATES_PER_PROMPT` is the same
# fixed contract seen from the evaluation side.
MASK_CANDIDATE_COUNT = 3


def strip_runtime_batch(value: np.ndarray) -> np.ndarray:
    """Remove the outer model batch that qbruntime omits from buffer shapes."""

    array = np.asarray(value)
    if array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    return np.ascontiguousarray(array, dtype=np.float32)


def build_decoder_runtime_feed(
    tensors: Mapping[str, np.ndarray], order: Sequence[str] = DECODER_RUNTIME_ORDER
) -> list[np.ndarray]:
    """Order the named decoder tensors into the compiled artifact's positional feed."""

    missing = [role for role in order if role not in tensors]
    if missing:
        raise ValueError(f"Decoder tensors are missing role(s): {missing}.")
    return [strip_runtime_batch(tensors[role]) for role in order]


def validate_runtime_shapes(
    actual: Sequence[np.ndarray], expected: Sequence[Sequence[int]], label: str
) -> None:
    """Fail loudly at construction time if a resolved artifact's shapes drift.

    ``-1`` in ``expected`` marks a wildcard/dynamic dimension (for example the
    decoder's point-count-dependent token axis).
    """

    if len(actual) != len(expected):
        raise ValueError(
            f"{label} input count mismatch: feed={len(actual)}, runtime={len(expected)}."
        )
    for index, (array, shape) in enumerate(zip(actual, expected)):
        shape = tuple(int(dim) for dim in shape)
        got = tuple(int(dim) for dim in array.shape)
        if len(got) != len(shape) or any(
            want != -1 and have != want for have, want in zip(got, shape)
        ):
            raise ValueError(
                f"{label} input {index} shape mismatch: feed={got}, runtime={shape}."
            )


def normalize_onnx_dims(shape: Sequence[Any]) -> tuple[int, ...]:
    """Map ONNX Runtime dims to ints, with symbolic/dynamic dims as ``-1``."""

    return tuple(int(dim) if isinstance(dim, int) else -1 for dim in shape)


def validate_onnx_session_inputs(
    session_inputs: Sequence[Any], expected: Mapping[str, Sequence[int]], label: str
) -> None:
    """Fail loudly at construction time if a resolved ONNX artifact's graph drifts.

    ``session_inputs`` is ONNX Runtime input metadata (``session.get_inputs()``).
    An expected static dimension must be declared statically by the graph, and
    ``-1`` marks an axis the graph must declare *dynamic* -- not a wildcard that
    accepts anything. ONNX Runtime reports a dynamic axis as its symbolic name
    (a ``str``), so a decoder exported with a frozen token dimension such as
    ``(1, 8, 256)`` is rejected here rather than working for one-point prompts
    and failing inside ONNX Runtime for the advertised two- and three-point
    prompts.
    """

    raw_shapes = {item.name: tuple(item.shape) for item in session_inputs}
    if set(raw_shapes) != set(expected):
        raise ValueError(
            f"{label} ONNX input names mismatch: graph={sorted(raw_shapes)}, "
            f"expected={sorted(expected)}."
        )
    for name, shape in expected.items():
        want = tuple(int(dim) for dim in shape)
        raw = raw_shapes[name]
        got = normalize_onnx_dims(raw)
        if len(raw) != len(want):
            raise ValueError(
                f"{label} ONNX input '{name}' shape mismatch: graph={got}, "
                f"expected={want}."
            )
        for axis, (raw_dim, want_dim) in enumerate(zip(raw, want)):
            if want_dim == -1:
                if not isinstance(raw_dim, str):
                    raise ValueError(
                        f"{label} ONNX input '{name}' axis {axis} must be dynamic "
                        f"to accept a varying prompt-token count, but the graph "
                        f"declares it as {raw_dim!r} (graph={got})."
                    )
            elif not isinstance(raw_dim, int) or raw_dim != want_dim:
                raise ValueError(
                    f"{label} ONNX input '{name}' shape mismatch: graph={got}, "
                    f"expected={want}."
                )


def classify_decoder_outputs(outputs: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Name the Hiera decoder outputs by their unambiguous element counts.

    qbruntime does not guarantee that the runtime output order matches the
    compiled graph's declared order, so each output is identified by its
    unique flattened size instead of position: ``masks`` is the only output
    whose size is a multiple of ``256*256``; among the rest, ``iou`` has size
    ``num_masks``, ``sam_tokens`` has size ``num_masks*256``, and
    ``object_score`` has size 1.

    ``masks`` and ``iou`` are required. ``sam_tokens`` and ``object_score``
    exist only on assembled-contract artifacts; a bridged-contract decoder is
    parsed with an ``output_meta`` that keeps just masks and IoU, so those two
    keys are simply absent from its result rather than an error.
    """

    arrays = [
        np.ascontiguousarray(np.asarray(value), dtype=np.float32) for value in outputs
    ]
    # A NaN from a numerical/runtime failure would otherwise reach argmax over
    # the IoU scores (silently selecting the wrong candidate) and the `> 0`
    # mask threshold (turning non-finite logits into plausible booleans),
    # corrupting predictions and SA-V metrics instead of reporting the failure.
    for index, array in enumerate(arrays):
        if not bool(np.isfinite(array).all()):
            raise ValueError(
                f"Decoded mask generation outputs must be finite; output {index} "
                f"with shape {array.shape} contains NaN or infinity."
            )
    mask_matches = [
        array
        for array in arrays
        if array.size >= MASK_AREA and array.size % MASK_AREA == 0
    ]
    if len(mask_matches) != 1:
        raise ValueError(
            f"Expected exactly one mask output, found {len(mask_matches)} "
            f"among shapes {[array.shape for array in arrays]}."
        )
    # Reshaping blindly would silently interleave the candidates' pixels for a
    # decoder that emits NHWC `(1, 256, 256, 3)` instead of the expected layout:
    # that size is also a multiple of MASK_AREA and also yields three
    # candidates, so neither the size match above nor the count check below can
    # see it, and the result is plausible but corrupted masks. Pin the layout to
    # the two the supported artifacts actually produce -- the MXQ runtime's
    # flattened `(3, 65536)` and the ONNX graph's `(3, 256, 256)` -- ignoring
    # leading batch axes. Checked here rather than against ONNX session output
    # metadata so the MXQ path is covered by the same guard.
    mask_shape = tuple(int(dim) for dim in mask_matches[0].shape)
    unbatched_layout = (
        tuple(dim for dim in mask_shape[:-2] if dim != 1) + mask_shape[-2:]
    )
    # Structure only -- the candidate count is checked separately below, so a
    # decoder emitting the right layout with the wrong number of candidates
    # still reports that rather than a layout error.
    is_flattened = len(unbatched_layout) == 2 and unbatched_layout[-1] == MASK_AREA
    is_spatial = len(unbatched_layout) == 3 and unbatched_layout[-2:] == (
        MASK_SIDE,
        MASK_SIDE,
    )
    if not (is_flattened or is_spatial):
        raise ValueError(
            f"Decoder mask output has an unsupported layout {mask_shape}; expected "
            f"candidates as (N, {MASK_AREA}) or (N, {MASK_SIDE}, {MASK_SIDE}), "
            "optionally batched. A channels-last layout would interleave the "
            "candidates into corrupted masks."
        )
    masks = mask_matches[0].reshape(-1, MASK_SIDE, MASK_SIDE)
    num_masks = masks.shape[0]
    # The compiled artifacts bake in SAM2's multimask slice, so a stale or
    # differently exported decoder emitting 2 or 4 masks (with consistently
    # sized iou/token outputs) would otherwise be accepted here and reach the
    # caller as a Results.masks shape that violates the documented fixed
    # three-candidate contract. Mirrors eval_sav.CANDIDATES_PER_PROMPT.
    if num_masks != MASK_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected exactly {MASK_CANDIDATE_COUNT} decoder mask candidates, "
            f"got {num_masks} (mask output shape {mask_matches[0].shape})."
        )

    def unique(label: str, size: int, required: bool) -> np.ndarray | None:
        matches = [
            array
            for array in arrays
            if array is not mask_matches[0] and array.size == size
        ]
        if len(matches) > 1 or (required and not matches):
            raise ValueError(
                f"Expected exactly one '{label}' output of size {size}, found {len(matches)}."
            )
        return matches[0] if matches else None

    iou = unique("iou", num_masks, required=True)
    assert iou is not None  # narrowed by required=True
    result = {"masks": masks, "iou": iou.reshape(num_masks)}
    sam_tokens = unique("sam_tokens", num_masks * 256, required=False)
    if sam_tokens is not None:
        result["sam_tokens"] = sam_tokens.reshape(num_masks, 256)
    object_score = unique("object_score", 1, required=False)
    if object_score is not None:
        result["object_score"] = object_score.reshape(1)
    return result

"""Fixed contracts between the SAM2 encoder/decoder artifacts and their host glue.

MXQ side: ported from the validated ``sam2-mxq-pipeline`` reference (real Aries2
SA-V-200 accuracy: FP32 mIoU 0.7750 vs MXQ mIoU 0.7757, mask agreement 0.983).
Compile and calibration are out of scope for this phase, so the decoder's
compiled runtime input order is a single validated default rather than a
configurable MBLT-input-name binding map.

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
# This is the positional signature of the compiled SAM2 Hiera-Large decoder MXQ,
# which differs from the MBLT graph's declared input names.
DECODER_RUNTIME_ORDER: tuple[str, ...] = (
    "hrf0_nhwc",
    "src_plus_pos_src",
    "hrf1_nhwc",
    "src",
    "pos_src",
    "tokens",
)

# Exported ONNX graph interface. ``-1`` marks the prompt-count-dependent
# dynamic token axis (``6 output tokens + N points + 1 pad``).
ENCODER_ONNX_INPUT_NAME = "input_image"
ENCODER_ONNX_INPUT_SHAPE: tuple[int, ...] = (1, 3, 1024, 1024)
DECODER_ONNX_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "tokens": (1, -1, 256),
    "src": (1, 256, 64, 64),
    "pos_src": (1, 256, 64, 64),
    "high_res_features_0": (1, 32, 256, 256),
    "high_res_features_1": (1, 64, 128, 128),
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
    """Name the four Hiera decoder outputs by their unambiguous element counts.

    qbruntime does not guarantee that the runtime output order matches the
    compiled graph's declared order, so each output is identified by its
    unique flattened size instead of position: ``masks`` is the only output
    whose size is a multiple of ``256*256``; among the rest, ``iou`` has size
    ``num_masks``, ``sam_tokens`` has size ``num_masks*256``, and
    ``object_score`` has size 1.
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

    def unique(label: str, size: int) -> np.ndarray:
        matches = [
            array
            for array in arrays
            if array is not mask_matches[0] and array.size == size
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one '{label}' output of size {size}, found {len(matches)}."
            )
        return matches[0]

    return {
        "masks": masks,
        "iou": unique("iou", num_masks).reshape(num_masks),
        "sam_tokens": unique("sam_tokens", num_masks * 256).reshape(num_masks, 256),
        "object_score": unique("object_score", 1).reshape(1),
    }

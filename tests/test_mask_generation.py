"""Tests for the mask_generation task and the SAM2HieraLarge model.

These exercise the NPU-facing contract logic (batch stripping, decoder
output classification, two-backend wiring/cleanup) with mocked backends and
structurally-shaped (not numerically real) prompt weights, matching the
``_FakeBackend`` monkeypatch pattern used throughout ``tests/test_wrapper.py``.
The actual prompt-encoding math is verified bit-for-bit against the real
``facebookresearch/sam2`` predictor separately (not part of this repo's
default test run; see the module docstring in ``_sam2_prompt.py``), and
exercised end-to-end on real hardware only by the opt-in
``tests/test_mask_generation_hardware.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

import mblt_vision.mask_generation.sam2 as sam2_module
from mblt_vision.mask_generation._sam2_contracts import (
    DECODER_MXQ_INPUT_SHAPES,
    DECODER_RUNTIME_ORDER,
    build_decoder_runtime_feed,
    classify_decoder_outputs,
    detect_decoder_contract,
    strip_runtime_batch,
    validate_runtime_shapes,
)
from mblt_vision.mask_generation.sam2 import SAM2HieraLarge


def _decoder_output_set(num_masks: int = 3) -> list[np.ndarray]:
    """Synthetic decoder outputs shaped like the real Hiera-Large decoder."""

    return [
        np.zeros((num_masks, 256, 256), dtype=np.float32),
        np.arange(num_masks, dtype=np.float32),
        np.ones((num_masks, 256), dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    ]


def _bridged_decoder_output_set() -> list[np.ndarray]:
    """The two outputs a bridged-contract decoder MXQ declares: IoU and masks."""

    return [
        np.zeros((1, 1, 3), dtype=np.float32),
        np.zeros((1, 3, 65536), dtype=np.float32),
    ]


def test_detect_decoder_contract_identifies_both_generations() -> None:
    assert detect_decoder_contract(DECODER_MXQ_INPUT_SHAPES["assembled"]) == "assembled"
    assert detect_decoder_contract(DECODER_MXQ_INPUT_SHAPES["bridged"]) == "bridged"
    # A single-prompt-length compile declares the dynamic axis fixed; the
    # signature's -1 must accept it.
    fixed = [
        (1, 10, 256) if shape == (1, -1, 256) else shape
        for shape in DECODER_MXQ_INPUT_SHAPES["assembled"]
    ]
    assert detect_decoder_contract(fixed) == "assembled"


def test_detect_decoder_contract_rejects_unknown_signatures() -> None:
    with pytest.raises(ValueError, match="neither known contract"):
        detect_decoder_contract([(1, 2, 3)])
    # Same length as a known contract but different shapes must not pass either.
    with pytest.raises(ValueError, match="neither known contract"):
        detect_decoder_contract([(64, 64, 256)] * 6)


def test_classify_decoder_outputs_accepts_bridged_two_output_form() -> None:
    classified = classify_decoder_outputs(_bridged_decoder_output_set())
    assert sorted(classified) == ["iou", "masks"]
    assert classified["masks"].shape == (3, 256, 256)
    assert classified["iou"].shape == (3,)


def test_classify_decoder_outputs_identifies_by_shape_not_position() -> None:
    """Order-independent classification, since qbruntime output order is not guaranteed."""

    masks, iou, sam_tokens, object_score = _decoder_output_set()
    for shuffled in (
        [masks, iou, sam_tokens, object_score],
        [object_score, sam_tokens, iou, masks],
        [iou, masks, object_score, sam_tokens],
    ):
        classified = classify_decoder_outputs(shuffled)
        assert classified["masks"].shape == (3, 256, 256)
        assert classified["iou"].shape == (3,)
        assert classified["sam_tokens"].shape == (3, 256)
        assert classified["object_score"].shape == (1,)
        assert np.array_equal(classified["iou"], iou)


def test_classify_decoder_outputs_rejects_wrong_candidate_count() -> None:
    """A stale/differently exported decoder must not reach the caller.

    Two or four masks with consistently sized iou/token outputs would otherwise
    classify cleanly and return a Results.masks shape violating the documented
    fixed three-candidate contract.
    """

    for num_masks in (2, 4):
        with pytest.raises(ValueError, match="3 decoder mask candidates"):
            classify_decoder_outputs(_decoder_output_set(num_masks))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_classify_decoder_outputs_rejects_non_finite_values(bad: float) -> None:
    """A numerical/runtime failure must be reported, not silently absorbed.

    A NaN IoU would participate in argmax and select the wrong candidate, and
    non-finite mask logits become plausible booleans under the later `> 0`.
    """

    for index in range(4):
        outputs = _decoder_output_set()
        outputs[index] = outputs[index].copy()
        outputs[index].flat[0] = bad
        with pytest.raises(ValueError, match="must be finite"):
            classify_decoder_outputs(outputs)


@pytest.mark.parametrize(
    ("shape", "accepted"),
    [
        ((1, 3, 256 * 256), True),  # MXQ runtime, flattened
        ((3, 256, 256), True),  # MXQ runtime, batch already stripped
        ((1, 3, 256, 256), True),  # exported ONNX graph, NCHW
        ((1, 256, 256, 3), False),  # channels-last re-export
        ((256, 256, 3), False),
    ],
)
def test_classify_decoder_outputs_pins_the_mask_layout(
    shape: tuple[int, ...], accepted: bool
) -> None:
    """A channels-last mask output must not be reshaped into interleaved masks.

    Its size is also a multiple of 256*256 and it also yields three candidates,
    so neither the size match nor the candidate-count check can catch it.
    """

    outputs = _decoder_output_set()
    outputs[0] = np.zeros(shape, dtype=np.float32)
    if accepted:
        assert classify_decoder_outputs(outputs)["masks"].shape == (3, 256, 256)
    else:
        with pytest.raises(ValueError, match="unsupported layout"):
            classify_decoder_outputs(outputs)


def test_classify_decoder_outputs_rejects_ambiguous_mask_candidates() -> None:
    """Two same-sized mask-shaped outputs cannot be told apart -- fail loudly."""

    masks, iou, sam_tokens, object_score = _decoder_output_set()
    with pytest.raises(ValueError, match="Expected exactly one mask output"):
        classify_decoder_outputs([masks, masks, iou, sam_tokens, object_score])


def test_strip_runtime_batch_drops_leading_batch_of_one() -> None:
    """qbruntime omits the outer model batch from buffer shapes."""

    batched = np.zeros((1, 4, 4, 3), dtype=np.float64)
    stripped = strip_runtime_batch(batched)
    assert stripped.shape == (4, 4, 3)
    assert stripped.dtype == np.float32
    assert stripped.flags["C_CONTIGUOUS"]

    unbatched = np.zeros((4, 4, 3), dtype=np.float32)
    assert strip_runtime_batch(unbatched).shape == (4, 4, 3)


def test_build_decoder_runtime_feed_orders_and_strips_batch() -> None:
    """Feed tensors are reordered per the compiled artifact's positional signature."""

    tensors = {
        role: np.full((1, 2, 2, 3), fill_value=index, dtype=np.float32)
        for index, role in enumerate(DECODER_RUNTIME_ORDER)
    }
    feed = build_decoder_runtime_feed(tensors)
    assert len(feed) == len(DECODER_RUNTIME_ORDER)
    for role, array in zip(DECODER_RUNTIME_ORDER, feed):
        assert array.shape == (2, 2, 3)  # leading batch-of-1 dim stripped
        assert np.all(array == tensors[role][0])


def test_build_decoder_runtime_feed_rejects_missing_role() -> None:
    tensors = {role: np.zeros((1, 1)) for role in DECODER_RUNTIME_ORDER[:-1]}
    with pytest.raises(ValueError, match="missing role"):
        build_decoder_runtime_feed(tensors)


def test_validate_runtime_shapes_accepts_wildcard_dynamic_dim() -> None:
    """``-1`` marks the decoder's point-count-dependent token axis as dynamic."""

    actual = [np.zeros((1, 1, 9, 256), dtype=np.float32)]
    validate_runtime_shapes(actual, [(1, 1, -1, 256)], "decoder")


def test_validate_runtime_shapes_rejects_static_dim_mismatch() -> None:
    actual = [np.zeros((1, 1, 9, 256), dtype=np.float32)]
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_runtime_shapes(actual, [(1, 1, -1, 128)], "decoder")


class _FakeModelHandle:
    """Mirrors the ``qbruntime.Model`` slot-zero compatibility handle
    ``MobilintNPUBackend.mxq_model`` exposes."""

    def __init__(self, input_shape: list[tuple[int, ...]]) -> None:
        self._input_shape = input_shape

    def get_model_input_shape(self) -> list[tuple[int, ...]]:
        return self._input_shape


class _FakeBackend:
    """Mirrors MobilintNPUBackend's minimal interface (see tests/test_wrapper.py)."""

    instances: list["_FakeBackend"] = []
    input_shape: list[tuple[int, ...]] = [(1024, 1024, 3)]
    # Construction now detects the decoder contract from declared shapes, so the
    # decoder handle must present a valid signature even in wiring-only tests.
    decoder_input_shape: ClassVar[list[tuple[int, ...]]] = list(
        DECODER_MXQ_INPUT_SHAPES["assembled"]
    )

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.disposed = False
        self.launched = False
        self.is_decoder = kwargs["mxq_path"].endswith("decoder.mxq")
        shapes = self.decoder_input_shape if self.is_decoder else self.input_shape
        self.mxq_model = _FakeModelHandle(list(shapes))
        _FakeBackend.instances.append(self)

    def create(self) -> None:
        return None

    def launch(self) -> None:
        self.launched = True

    def get_dtype(self) -> str:
        return "DataType.Float32"

    def __call__(self, feed: list[np.ndarray]) -> list[np.ndarray]:
        del feed
        return _decoder_output_set()

    def dispose(self) -> None:
        self.disposed = True


class _DualContractBackend(_FakeBackend):
    """Encoder/decoder pair for driving predict end to end without hardware.

    Selects declared shapes and outputs by which artifact path it was built
    for, and records the decoder feed so a test can assert what was sent.
    """

    decoder_input_shape: ClassVar[list[tuple[int, ...]]] = list(
        DECODER_MXQ_INPUT_SHAPES["bridged"]
    )
    decoder_outputs: staticmethod = staticmethod(_bridged_decoder_output_set)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.received_feeds: list[list[np.ndarray]] = []

    def __call__(self, feed: list[np.ndarray]) -> list[np.ndarray]:
        self.received_feeds.append(feed)
        if self.is_decoder:
            return self.decoder_outputs()
        return [
            np.zeros((256, 256, 32), dtype=np.float32),
            np.zeros((128, 128, 64), dtype=np.float32),
            np.zeros((64, 64, 256), dtype=np.float32),
        ]


class _AssembledContractBackend(_DualContractBackend):
    decoder_input_shape: ClassVar[list[tuple[int, ...]]] = list(
        DECODER_MXQ_INPUT_SHAPES["assembled"]
    )
    decoder_outputs = staticmethod(_decoder_output_set)


class _FailingDecoderBackend(_FakeBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if kwargs["mxq_path"].endswith("decoder.mxq"):
            raise RuntimeError("simulated decoder load failure")


class _WrongShapeBackend(_FakeBackend):
    """Reports an encoder input shape that does not match the fed tensor --
    simulates a resolved artifact compiled from a different signature than
    expected. The decoder keeps a valid signature so construction succeeds and
    the drift is caught at the encoder feed."""

    input_shape = [(1, 2, 3)]


def _fake_prompt_weights() -> dict[str, torch.Tensor]:
    """Structurally-correct (shape-only, not numerically real) prompt weights."""

    return {
        "no_mem_embed": torch.zeros(1, 1, 256),
        "iou_token_weight": torch.zeros(1, 256),
        "mask_tokens_weight": torch.zeros(4, 256),
        "obj_score_token_weight": torch.zeros(1, 256),
        "positional_encoding_gaussian_matrix": torch.zeros(2, 128),
        "point_embedding_negative": torch.zeros(1, 256),
        "point_embedding_positive": torch.zeros(1, 256),
        "not_a_point_embed_weight": torch.zeros(1, 256),
        "no_mask_embed_weight": torch.zeros(1, 256),
    }


@pytest.fixture(autouse=True)
def _reset_fake_backend_instances() -> None:
    _FakeBackend.instances.clear()


def _make_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_cls: type[_FakeBackend] = _FakeBackend,
) -> tuple[Path, Path]:
    encoder_path = tmp_path / "encoder.mxq"
    decoder_path = tmp_path / "decoder.mxq"
    encoder_path.write_bytes(b"mxq")
    decoder_path.write_bytes(b"mxq")
    weights_path = tmp_path / "prompt_weights.pt"
    torch.save(_fake_prompt_weights(), weights_path)
    monkeypatch.setattr(sam2_module, "MobilintNPUBackend", backend_cls)
    # Every test below passes explicit encoder_mxq_path/decoder_mxq_path, so this
    # is only ever reached for the prompt-weights resolution (no explicit
    # prompt_weights_path given).
    monkeypatch.setattr(
        sam2_module, "download_hub_artifact", lambda **kwargs: str(weights_path)
    )
    return encoder_path, decoder_path


def test_sam2_hiera_large_loads_two_backends_with_single_core_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit local paths skip Hub download; both backends default to single-core."""

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)

    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        assert len(_FakeBackend.instances) == 2
        encoder_backend, decoder_backend = _FakeBackend.instances
        assert encoder_backend.kwargs["mxq_path"] == str(encoder_path)
        assert decoder_backend.kwargs["mxq_path"] == str(decoder_path)
        assert encoder_backend.kwargs["core_mode"] == "single"
        assert decoder_backend.kwargs["core_mode"] == "single"
        assert encoder_backend.launched and decoder_backend.launched
    finally:
        engine.close()

    assert all(instance.disposed for instance in _FakeBackend.instances)


def test_sam2_hiera_large_close_is_idempotent_and_context_manager_disposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)

    with SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    ) as engine:
        pass
    assert all(instance.disposed for instance in _FakeBackend.instances)
    engine.close()  # second close must not raise


def test_sam2_hiera_large_disposes_encoder_when_decoder_construction_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mirrors the reference pipeline's exception-safe encoder/decoder construction."""

    encoder_path, decoder_path = _make_engine(
        monkeypatch, tmp_path, _FailingDecoderBackend
    )

    with pytest.raises(RuntimeError, match="simulated decoder load failure"):
        SAM2HieraLarge(
            encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
        )

    # The encoder backend was fully constructed before the decoder load failed;
    # SAM2HieraLarge.__init__'s except-clause must dispose it rather than leak it.
    assert len(_FakeBackend.instances) == 2
    encoder_backend = _FakeBackend.instances[0]
    assert encoder_backend.kwargs["mxq_path"] == str(encoder_path)
    assert encoder_backend.disposed


def test_predict_preprocessed_rejects_out_of_range_point_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point-count validation happens before any backend call, per the reference's 1-3 point contract."""

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        encoder_input = np.zeros((1024, 1024, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="1 to 3 point prompts"):
            engine.predict_preprocessed(
                encoder_input, (100, 100), points=np.zeros((0, 2)), labels=[]
            )
        with pytest.raises(ValueError, match="1 to 3 point prompts"):
            engine.predict_preprocessed(
                encoder_input, (100, 100), points=np.zeros((4, 2)), labels=[1, 1, 1, 1]
            )
    finally:
        engine.close()


def test_predict_preprocessed_rejects_unsupported_point_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only 1 (positive) and 0 (negative) carry a learned embedding.

    Any other value would receive neither and silently produce a plausible but
    semantically meaningless mask, so reject it before any backend call.
    """

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        encoder_input = np.zeros((1024, 1024, 3), dtype=np.float32)
        with pytest.raises(ValueError, match=r"must be 1 \(positive\) or 0"):
            engine.predict_preprocessed(
                encoder_input, (100, 100), points=[[10.0, 10.0]], labels=[2]
            )
        with pytest.raises(ValueError, match=r"must be 1 \(positive\) or 0"):
            engine.predict_preprocessed(
                encoder_input,
                (100, 100),
                points=[[10.0, 10.0], [20.0, 20.0]],
                labels=[1, -1],
            )
        with pytest.raises(ValueError, match=r"labels shaped \(N,\)"):
            engine.predict_preprocessed(
                encoder_input, (100, 100), points=[[10.0, 10.0]], labels=[[1]]
            )
        # Casting to int64 first would truncate these into valid 0/1 labels.
        for truncating in ([0.5], [1.9], [-0.1]):
            with pytest.raises(ValueError, match="must be whole numbers"):
                engine.predict_preprocessed(
                    encoder_input, (100, 100), points=[[10.0, 10.0]], labels=truncating
                )
        with pytest.raises(ValueError, match="labels must be finite"):
            engine.predict_preprocessed(
                encoder_input,
                (100, 100),
                points=[[10.0, 10.0]],
                labels=[float("nan")],
            )
    finally:
        engine.close()


@pytest.mark.parametrize(
    "bad_hw",
    [(0, 640), (640, 0), (-1, 480), (640.5, 480), (640, 480, 3), (640,)],
)
def test_predict_preprocessed_rejects_malformed_original_hw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_hw: tuple[int, ...]
) -> None:
    """Geometry is validated before either backend runs.

    A zero dimension makes transform_points produce infinite coordinates, and a
    fractional or wrong-length size otherwise fails only in the final mask
    resize, after both backends have already executed.
    """

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        with pytest.raises(ValueError, match="original_hw"):
            engine.predict_preprocessed(
                np.zeros((1024, 1024, 3), dtype=np.float32),
                bad_hw,
                points=[[10.0, 10.0]],
                labels=[1],
            )
    finally:
        engine.close()


def test_preprocess_rejects_non_finite_and_non_numeric_images() -> None:
    """Interpolation and normalization preserve NaN, so catch it at the source.

    Otherwise the backend fails in a runtime-specific way and the decoder-output
    finiteness check fires too late to name the offending input.
    """

    from mblt_vision.mask_generation._sam2_host import preprocess_encoder_input

    non_finite = np.zeros((32, 32, 3), dtype=np.float32)
    non_finite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        preprocess_encoder_input(non_finite)

    infinite = np.zeros((32, 32, 3), dtype=np.float32)
    infinite[1, 1, 1] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        preprocess_encoder_input(infinite)

    with pytest.raises(ValueError, match="numeric RGB image"):
        preprocess_encoder_input(np.full((4, 4, 3), "x"))

    # uint8 and float sources both remain supported.
    for supported in (
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float32),
    ):
        assert preprocess_encoder_input(supported).shape == (1, 1024, 1024, 3)


@pytest.mark.parametrize("converter", ["runtime", "onnx"])
def test_fpn_converters_pin_the_complete_level_shape(converter: str) -> None:
    """A channel-correct level with wrong geometry must not be reshaped.

    (1, 32, 128, 512) has the same element count as (1, 32, 256, 256), so
    build_backbone_features would view it into a plausible but corrupted FPN
    that then passes every downstream decoder-feed check.
    """

    from mblt_vision.mask_generation._sam2_host import fpn_from_onnx, fpn_from_runtime

    if converter == "onnx":
        convert = fpn_from_onnx
        good = [
            np.zeros((1, 32, 256, 256), dtype=np.float32),
            np.zeros((1, 64, 128, 128), dtype=np.float32),
            np.zeros((1, 256, 64, 64), dtype=np.float32),
        ]
        wrong_geometry = np.zeros((1, 32, 128, 512), dtype=np.float32)
    else:
        convert = fpn_from_runtime
        good = [
            np.zeros((256, 256, 32), dtype=np.float32),
            np.zeros((128, 128, 64), dtype=np.float32),
            np.zeros((64, 64, 256), dtype=np.float32),
        ]
        wrong_geometry = np.zeros((128, 512, 32), dtype=np.float32)

    assert len(convert(good, torch.device("cpu"))) == 3
    with pytest.raises(ValueError, match="spatially"):
        convert([wrong_geometry, *good[1:]], torch.device("cpu"))


def test_predict_preprocessed_rejects_non_finite_point_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NaN/Inf coordinates would contaminate tokens, masks, and IoU scores."""

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        encoder_input = np.zeros((1024, 1024, 3), dtype=np.float32)
        for bad in ([[float("nan"), 10.0]], [[10.0, float("inf")]]):
            with pytest.raises(ValueError, match="coordinates must be finite"):
                engine.predict_preprocessed(
                    encoder_input, (100, 100), points=bad, labels=[1]
                )
    finally:
        engine.close()


def test_predict_preprocessed_rejects_encoder_artifact_shape_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A drifted artifact's declared input shape must fail loudly, not silently produce wrong masks."""

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path, _WrongShapeBackend)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        encoder_input = np.zeros((1024, 1024, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="encoder input 0 shape mismatch"):
            engine.predict_preprocessed(
                encoder_input, (100, 100), points=np.array([[1.0, 1.0]]), labels=[1]
            )
    finally:
        engine.close()


class _UnknownDecoderContractBackend(_FakeBackend):
    """Decoder declares shapes matching neither known contract."""

    decoder_input_shape: ClassVar[list[tuple[int, ...]]] = [(64, 64, 256)] * 6


def test_unknown_decoder_contract_fails_at_construction_and_disposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A drifted decoder artifact must fail while the engine is being built.

    Detection at first decode would let the engine construct successfully and
    burn an NPU encoder inference before reporting the invalid signature.
    """

    encoder_path, decoder_path = _make_engine(
        monkeypatch, tmp_path, _UnknownDecoderContractBackend
    )
    with pytest.raises(ValueError, match="neither known contract"):
        SAM2HieraLarge(
            encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
        )
    assert all(
        instance.disposed for instance in _UnknownDecoderContractBackend.instances
    )


def test_predict_preprocessed_drives_the_bridged_decoder_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end shape flow for a bridged-contract artifact (SDK tutorial MXQ).

    The engine must detect the contract from the declared shapes, feed the
    prompt encoder's raw outputs in bridged order, and classify the two-output
    result -- with ``object_score`` absent rather than an error.
    """

    encoder_path, decoder_path = _make_engine(
        monkeypatch, tmp_path, _DualContractBackend
    )
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        encoder_input = np.zeros((1024, 1024, 3), dtype=np.float32)
        result = engine.predict_preprocessed(
            encoder_input,
            (480, 640),
            points=np.array([[10.0, 20.0], [30.0, 40.0]]),
            labels=[1, 0],
        )
        assert result.masks.shape == (3, 480, 640)
        assert result.iou_predictions.shape == (3,)
        # The bridged decoder emits no object score; the key must be absent,
        # not present with a synthetic None.
        assert "object_score" not in result.output

        decoder = next(b for b in _DualContractBackend.instances if b.is_decoder)
        (feed,) = decoder.received_feeds
        # Bridged order: three (256, 64, 64) prompt-encoder tensors, the dynamic
        # prompt axis (2 points + 1 pad), then the two NHWC feature maps.
        assert [tuple(t.shape) for t in feed] == [
            (256, 64, 64),
            (256, 64, 64),
            (256, 64, 64),
            (1, 3, 256),
            (256, 256, 32),
            (128, 128, 64),
        ]
    finally:
        engine.close()


def test_predict_preprocessed_drives_the_assembled_decoder_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The validated assembled-contract path must keep working unchanged."""

    encoder_path, decoder_path = _make_engine(
        monkeypatch, tmp_path, _AssembledContractBackend
    )
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        result = engine.predict_preprocessed(
            np.zeros((1024, 1024, 3), dtype=np.float32),
            (480, 640),
            points=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
            labels=[1, 1, 0],
        )
        assert result.masks.shape == (3, 480, 640)
        assert result.output["object_score"].shape == (1,)

        decoder = next(b for b in _AssembledContractBackend.instances if b.is_decoder)
        (feed,) = decoder.received_feeds
        # Assembled order: flattened host-side tensors, tokens = 3 points + 7.
        assert [tuple(t.shape) for t in feed] == [
            (256, 256, 32),
            (1, 4096, 256),
            (128, 128, 64),
            (1, 4096, 256),
            (1, 4096, 256),
            (1, 10, 256),
        ]
    finally:
        engine.close()


def test_raw_call_and_generic_postprocess_are_not_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SAM2HieraLarge needs prompts; the generic single-image MBLT_Engine API does not apply."""

    encoder_path, decoder_path = _make_engine(monkeypatch, tmp_path)
    engine = SAM2HieraLarge(
        encoder_mxq_path=str(encoder_path), decoder_mxq_path=str(decoder_path)
    )
    try:
        with pytest.raises(NotImplementedError):
            engine(np.zeros((1, 1)))
        with pytest.raises(NotImplementedError):
            engine.postprocess(np.zeros((1, 1)))
        with pytest.raises(NotImplementedError):
            engine.set_postprocess_thresholds()
        with pytest.raises(NotImplementedError):
            engine.preprocess_with_metadata(np.zeros((1, 1)))
    finally:
        engine.close()


def test_sam2_hiera_large_is_discoverable_via_list_models() -> None:
    import mblt_vision

    assert (
        "SAM2HieraLarge"
        in mblt_vision.list_models("mask_generation")["mask_generation"]
    )
    assert mblt_vision.mask_generation.SAM2HieraLarge is SAM2HieraLarge


class _FakeOnnxInput:
    """Mirrors ONNX Runtime's ``NodeArg`` (name + shape with symbolic dims)."""

    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


_ENCODER_ONNX_INPUTS = [_FakeOnnxInput("input_image_0", [1, 1024, 1024, 3])]
_DECODER_ONNX_INPUTS = [
    _FakeOnnxInput("image_embeddings", [1, 256, 64, 64]),
    _FakeOnnxInput("dense_prompt_embeddings", [1, 256, 64, 64]),
    _FakeOnnxInput("image_pe", [1, 256, 64, 64]),
    _FakeOnnxInput("sparse_prompt_embeddings_0", [1, 1, "num_points", 256]),
    _FakeOnnxInput("high_res_features0_0", [1, 256, 256, 32]),
    _FakeOnnxInput("high_res_features1_0", [1, 128, 128, 64]),
]


class _FakeONNXBackend:
    """Mirrors ``mblt_npu.ONNXBackend``'s minimal interface (dict-fed sessions)."""

    instances: list["_FakeONNXBackend"] = []

    def __init__(
        self, model_path: str, *, providers: Any = None, ort_module: Any = None
    ) -> None:
        self.model_path = model_path
        self.providers = providers
        self.ort_module = ort_module
        self.created = False
        self.disposed = False
        self.calls: list[dict[str, tuple[int, ...]]] = []
        _FakeONNXBackend.instances.append(self)

    @property
    def _is_encoder(self) -> bool:
        return self.model_path.endswith("encoder.onnx")

    def create(self) -> None:
        self.created = True

    def get_inputs(self) -> list[_FakeOnnxInput]:
        return _ENCODER_ONNX_INPUTS if self._is_encoder else _DECODER_ONNX_INPUTS

    def __call__(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: np.asarray(x).shape for name, x in feed.items()})
        if self._is_encoder:
            # Batched NCHW FPN levels, exactly as the exported graph declares.
            return [
                np.zeros((1, 256, 256, 32), dtype=np.float32),
                np.zeros((1, 128, 128, 64), dtype=np.float32),
                np.zeros((1, 64, 64, 256), dtype=np.float32),
            ]
        return [
            np.arange(3, dtype=np.float32).reshape(1, 1, 3),
            np.zeros((1, 3, 256 * 256), dtype=np.float32),
        ]

    def dispose(self) -> None:
        self.disposed = True


class _DriftedDecoderONNXBackend(_FakeONNXBackend):
    """Reports decoder input names that do not match the exported-graph contract."""

    def get_inputs(self) -> list[_FakeOnnxInput]:
        if self._is_encoder:
            return _ENCODER_ONNX_INPUTS
        return [_FakeOnnxInput("renamed_tokens", [1, "num_tokens", 256])]


@pytest.fixture(autouse=True)
def _reset_fake_onnx_backend_instances() -> None:
    _FakeONNXBackend.instances.clear()


def _make_onnx_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_cls: type[_FakeONNXBackend] = _FakeONNXBackend,
) -> tuple[Path, Path]:
    encoder_path = tmp_path / "encoder.onnx"
    decoder_path = tmp_path / "decoder.onnx"
    encoder_path.write_bytes(b"onnx")
    decoder_path.write_bytes(b"onnx")
    weights_path = tmp_path / "prompt_weights.pt"
    torch.save(_fake_prompt_weights(), weights_path)
    monkeypatch.setattr(sam2_module, "ONNXBackend", backend_cls)
    # A structurally-complete stand-in module: _resolve_onnx_providers only
    # reads it when an explicit provider list is requested.
    monkeypatch.setattr(sam2_module, "_load_onnxruntime", lambda: object())
    monkeypatch.setattr(
        sam2_module, "download_hub_artifact", lambda **kwargs: str(weights_path)
    )
    return encoder_path, decoder_path


def test_sam2_onnx_framework_is_inferred_from_paths_and_disposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit .onnx paths select ONNX inference without an explicit framework."""

    encoder_path, decoder_path = _make_onnx_engine(monkeypatch, tmp_path)

    engine = SAM2HieraLarge(
        encoder_onnx_path=str(encoder_path), decoder_onnx_path=str(decoder_path)
    )
    try:
        assert engine.framework == "onnx"
        assert len(_FakeONNXBackend.instances) == 2
        encoder_backend, decoder_backend = _FakeONNXBackend.instances
        assert encoder_backend.model_path == str(encoder_path)
        assert decoder_backend.model_path == str(decoder_path)
        assert encoder_backend.created and decoder_backend.created
        assert encoder_backend.providers == ["CPUExecutionProvider"]
    finally:
        engine.close()

    assert all(instance.disposed for instance in _FakeONNXBackend.instances)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"framework": "mxq", "encoder_onnx_path": "encoder.onnx"},
            "conflicts with explicit encoder_onnx_path",
        ),
        (
            {"framework": "onnx", "encoder_mxq_path": "encoder.mxq"},
            "conflicts with explicit encoder_mxq_path",
        ),
        (
            {
                "encoder_mxq_path": "encoder.mxq",
                "decoder_onnx_path": "decoder.onnx",
            },
            "without an explicit framework",
        ),
        ({"framework": "tflite"}, "must be 'mxq' or 'onnx'"),
        ({"encoder_onnx_path": "encoder.mxq"}, "must end in '.onnx'"),
        ({"decoder_mxq_path": "decoder.onnx"}, "must end in '.mxq'"),
    ],
)
def test_sam2_framework_and_path_conflicts_fail_fast(
    kwargs: dict[str, Any], match: str
) -> None:
    """Suffix and framework conflicts fail before any download or backend load."""

    with pytest.raises(ValueError, match=match):
        SAM2HieraLarge(**kwargs)


@pytest.mark.parametrize("framework", ["ONNX", "Onnx", "MXQ", "Mxq"])
def test_sam2_framework_name_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, framework: str
) -> None:
    """Match `_model_paths.resolve_framework`, which lowercases explicit names."""

    assert (
        SAM2HieraLarge._resolve_framework(
            framework, mxq_path_passed=False, onnx_path_passed=False
        )
        == framework.lower()
    )

    # The conflict checks run on the normalized value too.
    with pytest.raises(ValueError, match="conflicts with explicit"):
        SAM2HieraLarge._resolve_framework(
            "ONNX", mxq_path_passed=True, onnx_path_passed=False
        )


def test_sam2_invalid_target_device_reported_before_any_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unknown board must not surface as a Hub/network failure."""

    _make_engine(monkeypatch, tmp_path)
    downloads: list[str] = []

    def _record_download(**kwargs: Any) -> str:
        downloads.append(str(kwargs.get("filename")))
        raise AssertionError("no artifact may be downloaded before target validation")

    monkeypatch.setattr(sam2_module, "download_hub_artifact", _record_download)
    with pytest.raises(ValueError, match="target_device"):
        SAM2HieraLarge(target_device="not-a-board")
    assert downloads == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder_mxq_path": "missing-encoder.mxq"},
        {"decoder_onnx_path": "missing-decoder.onnx"},
        {"prompt_weights_path": "missing-weights.pt"},
    ],
)
def test_sam2_missing_explicit_artifact_fails_before_any_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    """An invalid local path must be reported as such, not as a Hub failure.

    Offline this would otherwise surface as a network error; online it would
    cost a download before failing later in backend creation.
    """

    _make_onnx_engine(monkeypatch, tmp_path)
    downloads: list[str] = []

    def _record_download(**download_kwargs: Any) -> str:
        downloads.append(str(download_kwargs.get("filename")))
        raise AssertionError("no artifact may be downloaded before path validation")

    monkeypatch.setattr(sam2_module, "download_hub_artifact", _record_download)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SAM2HieraLarge(**kwargs)
    assert downloads == []


def test_sam2_construction_error_survives_a_failing_dispose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dispose() failure must not replace the original construction error."""

    class _UndisposableDriftedBackend(_DriftedDecoderONNXBackend):
        def dispose(self) -> None:
            raise RuntimeError("dispose exploded")

    encoder_path, decoder_path = _make_onnx_engine(
        monkeypatch, tmp_path, _UndisposableDriftedBackend
    )

    # The graph-drift ValueError is what tells the caller what is wrong; the
    # dispose RuntimeError must not surface in its place.
    with pytest.raises(ValueError, match="decoder ONNX input names mismatch"):
        SAM2HieraLarge(
            encoder_onnx_path=str(encoder_path), decoder_onnx_path=str(decoder_path)
        )


def test_sam2_onnx_rejects_a_frozen_prompt_token_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A decoder exported with a fixed token dimension must fail construction.

    It would otherwise work for one-point prompts and fail inside ONNX Runtime
    for the advertised two- and three-point prompts.
    """

    class _FrozenTokenAxisBackend(_FakeONNXBackend):
        def get_inputs(self) -> list[_FakeOnnxInput]:
            if self._is_encoder:
                return _ENCODER_ONNX_INPUTS
            return [
                *_DECODER_ONNX_INPUTS[:3],
                _FakeOnnxInput("sparse_prompt_embeddings_0", [1, 1, 2, 256]),
                *_DECODER_ONNX_INPUTS[4:],
            ]

    encoder_path, decoder_path = _make_onnx_engine(
        monkeypatch, tmp_path, _FrozenTokenAxisBackend
    )

    with pytest.raises(ValueError, match="must be dynamic"):
        SAM2HieraLarge(
            encoder_onnx_path=str(encoder_path), decoder_onnx_path=str(decoder_path)
        )


def test_sam2_onnx_predict_preprocessed_uses_the_exported_graph_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NHWC preprocess output is transposed to NCHW and fed by graph input name."""

    encoder_path, decoder_path = _make_onnx_engine(monkeypatch, tmp_path)

    with SAM2HieraLarge(
        encoder_onnx_path=str(encoder_path), decoder_onnx_path=str(decoder_path)
    ) as engine:
        result = engine.predict_preprocessed(
            np.zeros((1, 1024, 1024, 3), dtype=np.float32),
            original_hw=(480, 640),
            points=[[100.0, 200.0]],
            labels=[1],
        )
        encoder_backend, decoder_backend = _FakeONNXBackend.instances
        assert encoder_backend.calls == [{"input_image_0": (1, 1024, 1024, 3)}]
        (decoder_call,) = decoder_backend.calls
        assert decoder_call == {
            "image_embeddings": (1, 256, 64, 64),
            "dense_prompt_embeddings": (1, 256, 64, 64),
            "image_pe": (1, 256, 64, 64),
            "sparse_prompt_embeddings_0": (1, 1, 2, 256),
            "high_res_features0_0": (1, 256, 256, 32),
            "high_res_features1_0": (1, 128, 128, 64),
        }
        assert result.task == "mask_generation"
        assert result.masks is not None
        assert result.masks.shape == (3, 480, 640)
        assert result.masks.dtype == np.bool_
        # The fake decoder's iou_pred is arange(3): argmax selection is index 2.
        assert result.selected == 2


def test_sam2_onnx_rejects_session_interface_drift_and_disposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A re-exported decoder with different graph inputs must fail at construction."""

    encoder_path, decoder_path = _make_onnx_engine(
        monkeypatch, tmp_path, _DriftedDecoderONNXBackend
    )

    with pytest.raises(ValueError, match="decoder ONNX input names mismatch"):
        SAM2HieraLarge(
            encoder_onnx_path=str(encoder_path), decoder_onnx_path=str(decoder_path)
        )

    # The encoder session was fully constructed before the decoder validation
    # failed; the constructor's except-clause must dispose it rather than leak it.
    assert len(_FakeONNXBackend.instances) == 2
    assert _FakeONNXBackend.instances[0].disposed
    # The decoder session is created before validation rejects it, and is never
    # assigned to the engine -- so only _build_onnx_backend itself can dispose
    # it. Without that, the constructor's cleanup cannot reach it and it leaks.
    assert _FakeONNXBackend.instances[1].disposed


def test_sam2_onnx_missing_onnxruntime_raises_before_any_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optional-dependency error surfaces before any session or download.

    In an offline environment a Hub fetch attempted first would surface a
    network/cache failure instead of the documented package-extra install
    error -- and the download would have been useless anyway.
    """

    _make_onnx_engine(monkeypatch, tmp_path)

    def _raise() -> Any:
        raise ImportError("onnxruntime is not installed")

    downloads: list[str] = []

    def _record_download(**kwargs: Any) -> str:
        downloads.append(str(kwargs.get("filename")))
        raise AssertionError("no artifact may be downloaded before onnxruntime loads")

    monkeypatch.setattr(sam2_module, "_load_onnxruntime", _raise)
    monkeypatch.setattr(sam2_module, "download_hub_artifact", _record_download)
    with pytest.raises(ImportError, match="onnxruntime is not installed"):
        SAM2HieraLarge(framework="onnx")
    assert _FakeONNXBackend.instances == []
    assert downloads == []


def test_fpn_from_onnx_orders_levels_by_channel_and_rejects_other_layouts() -> None:
    """Batched NCHW outputs are ordered 32/64/256 regardless of runtime order."""

    from mblt_vision.mask_generation._sam2_host import fpn_from_onnx

    outputs = [
        np.zeros((1, 256, 64, 64), dtype=np.float32),
        np.zeros((1, 32, 256, 256), dtype=np.float32),
        np.zeros((1, 64, 128, 128), dtype=np.float32),
    ]
    levels = fpn_from_onnx(outputs, torch.device("cpu"))
    assert [tuple(level.shape) for level in levels] == [
        (1, 32, 256, 256),
        (1, 64, 128, 128),
        (1, 256, 64, 64),
    ]

    with pytest.raises(ValueError, match="Duplicate encoder output"):
        fpn_from_onnx([outputs[0], outputs[0], outputs[1]], torch.device("cpu"))
    # The MXQ runtime's batchless NHWC layout must be rejected, not guessed at.
    with pytest.raises(ValueError, match="Unexpected encoder ONNX output shape"):
        fpn_from_onnx([np.zeros((256, 256, 32), dtype=np.float32)], torch.device("cpu"))


@pytest.mark.parametrize(("num_points", "num_tokens"), [(1, 8), (2, 9), (3, 10)])
def test_prepare_decoder_tensors_onnx_builds_the_six_direct_inputs(
    num_points: int, num_tokens: int
) -> None:
    """The ONNX decoder feed keeps the traced pre-flattening shapes."""

    from mblt_vision.mask_generation._sam2_contracts import DECODER_ONNX_INPUT_NAMES
    from mblt_vision.mask_generation._sam2_host import (
        build_backbone_features,
        prepare_decoder_tensors_bridged,
    )

    weights = _fake_prompt_weights()
    features = build_backbone_features(
        weights,
        [
            torch.zeros(1, 32, 256, 256),
            torch.zeros(1, 64, 128, 128),
            torch.zeros(1, 256, 64, 64),
        ],
    )
    points = np.arange(num_points * 2, dtype=np.float32).reshape(num_points, 2)
    labels = np.ones(num_points, dtype=np.int64)
    tensors = prepare_decoder_tensors_bridged(
        weights, features, points, labels, (480, 640)
    )

    assert tuple(tensors) == DECODER_ONNX_INPUT_NAMES
    assert tensors["image_embeddings"].shape == (1, 256, 64, 64)
    assert tensors["dense_prompt_embeddings"].shape == (1, 256, 64, 64)
    assert tensors["image_pe"].shape == (1, 256, 64, 64)
    assert tensors["sparse_prompt_embeddings_0"].shape == (1, 1, num_points + 1, 256)
    assert tensors["high_res_features0_0"].shape == (1, 256, 256, 32)
    assert tensors["high_res_features1_0"].shape == (1, 128, 128, 64)
    assert all(array.dtype == np.float32 for array in tensors.values())

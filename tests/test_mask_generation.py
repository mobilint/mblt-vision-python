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
from typing import Any

import numpy as np
import pytest
import torch

import mblt_vision.mask_generation.sam2 as sam2_module
from mblt_vision.mask_generation._sam2_contracts import (
    DECODER_RUNTIME_ORDER,
    build_decoder_runtime_feed,
    classify_decoder_outputs,
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

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.disposed = False
        self.launched = False
        self.mxq_model = _FakeModelHandle(self.input_shape)
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


class _FailingDecoderBackend(_FakeBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if kwargs["mxq_path"].endswith("decoder.mxq"):
            raise RuntimeError("simulated decoder load failure")


class _WrongShapeBackend(_FakeBackend):
    """Reports an input shape that does not match the fed tensor -- simulates a
    resolved artifact compiled from a different signature than expected."""

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

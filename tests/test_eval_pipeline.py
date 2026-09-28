"""Checks for the evaluation loop that overlaps preparation, inference, and decoding."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from mblt_vision.utils.evaluation._pipeline import map_batched_inference
from mblt_vision.utils.preprocess import build_preprocess


def _identity_decode(batch, outputs):
    return batch, outputs


def test_map_batched_inference_decodes_off_the_consuming_thread() -> None:
    """Inference must run ahead of decoding, and results must stay in order."""

    calls = 0
    threads: set[int] = set()

    def infer(batch: int) -> int:
        nonlocal calls
        calls += 1
        return batch * 10

    def decode(batch: int, outputs: int) -> tuple[int, int]:
        threads.add(threading.get_ident())
        return batch, outputs

    stream = map_batched_inference(range(4), infer, decode, prefetch_batches=2)
    first = next(stream)
    # Two of the four batches are already through `infer` when the first result
    # comes back. A loop that decodes on the consuming thread reads 1 here,
    # because the second batch cannot start until the caller returns.
    assert calls == 2

    results = [first, *stream]
    assert results == [(batch, batch * 10) for batch in range(4)]
    assert threading.get_ident() not in threads


def test_map_batched_inference_keeps_inference_on_the_consuming_thread() -> None:
    threads: set[int] = set()

    def infer(batch: int) -> int:
        threads.add(threading.get_ident())
        return batch

    list(map_batched_inference(range(9), infer, _identity_decode))
    assert threads == {threading.get_ident()}


@pytest.mark.parametrize("count", [0, 1, 7, 15, 33])
def test_map_batched_inference_bounds_each_stage(count: int) -> None:
    pulled = inferred = consumed = 0
    prefetch = 4

    def batches():
        nonlocal pulled
        for batch in range(count):
            # Counted where the consuming thread takes a batch, so a batch whose
            # preparation is submitted but not yet started still counts as queued.
            assert pulled - inferred < prefetch
            pulled += 1
            yield batch

    def infer(batch: int) -> int:
        nonlocal inferred
        assert pulled - inferred <= prefetch
        # Include batches whose results are waiting for the consumer.
        assert inferred - consumed < prefetch
        inferred += 1
        return batch

    results = []
    for result in map_batched_inference(
        batches(), infer, _identity_decode, prefetch_batches=prefetch
    ):
        results.append(result)
        consumed += 1
    assert results == [(batch, batch) for batch in range(count)]


def test_map_batched_inference_overlaps_inference_with_blocked_decode() -> None:
    decoding = threading.Event()
    next_inference = threading.Event()

    def infer(batch: int) -> int:
        if batch == 1:
            assert decoding.wait(5), "first decode never started"
            next_inference.set()
        return batch

    def decode(batch: int, outputs: int) -> int:
        if batch == 0:
            decoding.set()
            assert next_inference.wait(5), "inference waited for decode"
        return outputs

    assert list(map_batched_inference(range(3), infer, decode, prefetch_batches=2)) == [
        0,
        1,
        2,
    ]


def test_map_batched_inference_decodes_batches_concurrently() -> None:
    second_started = threading.Event()

    def decode(batch: int, outputs: int) -> int:
        if batch == 1:
            second_started.set()
        elif batch == 0:
            assert second_started.wait(5), "decode ran one batch at a time"
        return outputs

    assert list(
        map_batched_inference(range(3), lambda batch: batch, decode, prefetch_batches=2)
    ) == [0, 1, 2]


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.size = size
        self.threads: set[int] = set()

    def __getitem__(self, index: int) -> int:
        self.threads.add(threading.get_ident())
        return index

    def __len__(self) -> int:
        return self.size


def test_map_batched_inference_prepares_dataloader_batches_on_the_pool() -> None:
    """A single-process DataLoader is fetched per batch, off the consuming thread."""

    dataset = _Dataset(11)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=list
    )

    expected = [(batch, len(batch)) for batch in loader]
    dataset.threads.clear()

    results = list(
        map_batched_inference(loader, lambda batch: len(batch), _identity_decode)
    )

    assert results == expected
    assert dataset.threads
    assert threading.get_ident() not in dataset.threads


@pytest.mark.parametrize("stage", ["prepare", "infer", "decode"])
def test_map_batched_inference_propagates_stage_errors(stage: str) -> None:
    class _FailingDataset(_Dataset):
        def __getitem__(self, index: int) -> int:
            if stage == "prepare" and index == 4:
                raise ValueError("prepare failed")
            return index

    def infer(batch: list[int]) -> list[int]:
        if stage == "infer" and 4 in batch:
            raise ValueError("infer failed")
        return batch

    def decode(batch: list[int], outputs: list[int]) -> list[int]:
        if stage == "decode" and 4 in batch:
            raise ValueError("decode failed")
        return outputs

    loader = torch.utils.data.DataLoader(
        _FailingDataset(9), batch_size=2, num_workers=0, collate_fn=list
    )
    with pytest.raises(ValueError, match=f"{stage} failed"):
        list(map_batched_inference(loader, infer, decode))


def test_map_batched_inference_stops_work_when_abandoned() -> None:
    decoded: list[int] = []

    def decode(batch: int, outputs: int) -> int:
        decoded.append(batch)
        return outputs

    stream = map_batched_inference(
        range(1000), lambda batch: batch, decode, prefetch_batches=2
    )
    assert next(stream) == 0
    stream.close()
    assert len(decoded) < 10


def test_letterbox_metadata_is_per_call_under_concurrency() -> None:
    """Concurrent preprocessing through one instance must not swap geometries."""

    preprocess = build_preprocess({"LetterBox": {"img_size": [64, 64]}})
    shapes = [(16 + 8 * (index % 7), 64 - 4 * (index % 5), 3) for index in range(200)]

    def run(shape: tuple[int, int, int]) -> tuple[object, object]:
        _, metadata = preprocess.with_metadata(np.zeros(shape, dtype=np.uint8))
        _, expected = build_preprocess(
            {"LetterBox": {"img_size": [64, 64]}}
        ).with_metadata(np.zeros(shape, dtype=np.uint8))
        return metadata["ratio_pad"], expected["ratio_pad"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for actual, expected in pool.map(run, shapes):
            assert actual == expected


def test_letterbox_call_still_records_ratio_pad() -> None:
    """The single-threaded ``ratio_pad`` attribute stays for compatibility."""

    preprocess = build_preprocess({"LetterBox": {"img_size": [64, 64]}})
    letterbox = preprocess.Ops[0]
    image = np.zeros((32, 64, 3), dtype=np.uint8)

    output = letterbox(image)
    expected_output, expected_ratio_pad = letterbox.with_ratio_pad(image)

    assert torch.equal(output, expected_output)
    assert letterbox.ratio_pad == expected_ratio_pad

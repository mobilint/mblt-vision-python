"""Overlap batch preparation, inference, and postprocessing during evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import torch

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")

PREFETCH_BATCHES = 4
WORKERS = 4


def _fetch_batch(
    dataset: Any, collate_fn: Callable[[list[Any]], Any], indices: list[int]
) -> Any:
    """Fetch and collate one batch the way a single-process DataLoader does."""

    return collate_fn([dataset[index] for index in indices])


def _batch_tasks(batches: Iterable[Any]) -> Iterator[Callable[[], Any]]:
    """Yield one callable per batch that produces the prepared batch.

    A single-process map-style ``DataLoader`` is split into per-batch fetches, so
    image decoding and preprocessing run on the preparation pool. Any other
    iterable is consumed on the calling thread as-is: its batches arrive already
    prepared, and a generic iterator is not safe to advance from several threads.
    """

    if (
        isinstance(batches, torch.utils.data.DataLoader)
        and batches.num_workers == 0
        and batches.batch_sampler is not None
        and not isinstance(batches.dataset, torch.utils.data.IterableDataset)
    ):
        dataset, collate_fn = batches.dataset, batches.collate_fn
        for indices in batches.batch_sampler:
            yield lambda indices=list(indices): _fetch_batch(
                dataset, collate_fn, indices
            )
        return
    for batch in batches:
        yield lambda batch=batch: batch


def map_batched_inference(
    batches: Iterable[BatchT],
    infer: Callable[[BatchT], OutputT],
    decode: Callable[[BatchT, OutputT], ResultT],
    prefetch_batches: int = PREFETCH_BATCHES,
    workers: int = WORKERS,
) -> Iterator[ResultT]:
    """Yield ``decode(batch, infer(batch))`` for every batch, in input order.

    Preparation and decoding use separate thread pools, so both overlap inference
    without competing for the same workers: sharing one pool would let a slow
    decode starve the prefetch that feeds the device. ``infer`` stays on the
    consuming thread, one batch at a time, so the backend is never entered
    concurrently.

    ``decode`` runs on a worker thread with several batches in it at once, so it
    must be a pure function of its two arguments; accumulate into shared state
    only from the consuming loop, which sees results in input order. Preparing a
    ``DataLoader`` batch likewise runs its dataset and collate function on a
    worker thread, so both must be thread-safe.

    Each stage holds at most ``prefetch_batches`` batches, and a slow stage or a
    slow consumer applies backpressure to the others. Peak memory is the price of
    the overlap: up to that many batches sit in each stage rather than one.

    Args:
        batches: A ``DataLoader`` or an iterable of already prepared batches.
        infer: Runs the model on one prepared batch.
        decode: Turns one batch and its raw outputs into a result.
        prefetch_batches: Batches each stage may hold.
        workers: Upper bound on the threads in each pool.

    Yields:
        One decoded result per batch, in input order.

    Raises:
        Exception: The first failure of preparation, inference, or decoding, as
            raised by that stage.
    """

    prefetch = max(prefetch_batches, 1)
    # Every task is a whole batch, so each pool's queue holds at most ``prefetch``
    # tasks; more threads than that would never have work.
    pool_workers = max(min(workers, prefetch), 1)
    pending: deque[Future[BatchT]] = deque()
    decoded: deque[Future[ResultT]] = deque()
    prepare_pool = ThreadPoolExecutor(
        max_workers=pool_workers, thread_name_prefix="mblt-vision-prepare"
    )
    decode_pool = ThreadPoolExecutor(
        max_workers=pool_workers, thread_name_prefix="mblt-vision-decode"
    )

    def infer_oldest() -> None:
        """Run inference on the oldest prepared batch and queue its decode."""

        batch = pending.popleft().result()
        outputs = infer(batch)
        decoded.append(decode_pool.submit(decode, batch, outputs))

    try:
        for task in _batch_tasks(batches):
            pending.append(prepare_pool.submit(task))
            if len(pending) >= prefetch:
                infer_oldest()
            if len(decoded) >= prefetch:
                yield decoded.popleft().result()
        while pending:
            infer_oldest()
            if len(decoded) >= prefetch:
                yield decoded.popleft().result()
        while decoded:
            yield decoded.popleft().result()
    finally:
        # On failure or an abandoned iterator, drop the batches nobody will read
        # instead of preparing and decoding them before the error surfaces.
        prepare_pool.shutdown(wait=True, cancel_futures=True)
        decode_pool.shutdown(wait=True, cancel_futures=True)

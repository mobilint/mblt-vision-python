"""Source-exact Market protocol retrieval metrics (ops ec537764)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..datasets.reid import ReIDSample
from ._pipeline import map_batched_inference


def extract_embeddings(
    batches: Iterable[Any], infer: Callable, postprocess: Callable
) -> np.ndarray:
    """Decode prepared batches in input order through the shared bounded pipeline.

    ``infer`` receives a prepared batch; ``postprocess`` receives raw outputs and
    returns a validated ``(N,D)`` tensor. No implicit normalization or flip fusion.
    """

    def decode(batch: Any, output: Any) -> np.ndarray:
        value = postprocess(output)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
            raise ValueError(
                "Decoded embeddings require a nonempty finite (N,D) matrix"
            )
        return value

    rows = list(map_batched_inference(batches, infer, decode))
    if not rows:
        raise ValueError("No embedding batches supplied")
    return np.concatenate(rows, axis=0)


def euclidean_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Source float32 squared Euclidean distance (not cosine or square root)."""
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    return (
        np.sum(query**2, axis=1, keepdims=True)
        + np.sum(gallery**2, axis=1, keepdims=True).T
        - 2.0 * (query @ gallery.T)
    )


@dataclass(frozen=True)
class RetrievalResult:
    """Unrounded mAP and CMC curve; ``as_dict`` matches ops score rounding."""

    cmc: np.ndarray
    mean_ap: float
    sample_count: int

    @property
    def primary_score(self) -> float:
        return self.mean_ap

    @property
    def secondary_score(self) -> float:
        return self.rank(1)

    def rank(self, k: int) -> float:
        """Saturate ranks beyond the gallery length, as in the source scorer."""
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("Rank must be a positive integer")
        return float(self.cmc[min(k, len(self.cmc)) - 1])

    def as_dict(self) -> dict:
        """Return source-compatible primary/secondary metrics in ratio units."""
        return {
            "primary": {"map": round(self.mean_ap, 5)},
            "secondary": {f"rank{k}": round(self.rank(k), 5) for k in (1, 5, 10)},
            "sample_count": self.sample_count,
        }


class RetrievalEvaluator:
    """Market retrieval protocol shared by three dataset-specific evaluators.

    Stable distance ties retain gallery manifest order. Exclude only same-ID,
    same-camera rows; pid 0 distractors remain. Dataset loaders own pid -1 junk
    filtering. Queries without cross-camera matches do not enter either mean.
    """

    dataset: str | None = None

    def evaluate(
        self,
        embeddings: np.ndarray,
        samples: Sequence[ReIDSample],
        *,
        max_rank: int = 50,
    ) -> RetrievalResult:
        """Score one embedding row per sample in query-then-gallery order."""
        values = np.asarray(embeddings, dtype=np.float32)
        if (
            values.ndim != 2
            or min(values.shape) == 0
            or values.shape[0] != len(samples)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Embeddings require one finite nonempty vector per sample")
        if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank <= 0:
            raise ValueError("max_rank must be a positive integer")
        if any(s.split not in {"query", "gallery"} or s.pid == -1 for s in samples):
            raise ValueError(
                "Samples require query/gallery splits with loader-filtered junk"
            )
        query = [i for i, s in enumerate(samples) if s.split == "query"]
        gallery = [i for i, s in enumerate(samples) if s.split == "gallery"]
        if not query or not gallery:
            raise ValueError("Benchmark needs both a query and a gallery split")
        cmc, mean_ap = self.evaluate_distances(
            euclidean_distance(values[query], values[gallery]),
            np.asarray([samples[i].pid for i in query]),
            np.asarray([samples[i].pid for i in gallery]),
            np.asarray([samples[i].camid for i in query]),
            np.asarray([samples[i].camid for i in gallery]),
            max_rank,
        )
        return RetrievalResult(cmc, mean_ap, len(samples))

    @staticmethod
    def evaluate_distances(
        distmat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_camids: np.ndarray,
        g_camids: np.ndarray,
        max_rank: int = 50,
    ) -> tuple[np.ndarray, float]:
        """CMC and mAP with the exact source accumulation and padding order."""
        distmat = np.asarray(distmat)
        q_pids, g_pids = np.asarray(q_pids), np.asarray(g_pids)
        q_camids, g_camids = np.asarray(q_camids), np.asarray(g_camids)
        if (
            distmat.ndim != 2
            or min(distmat.shape) == 0
            or not np.isfinite(distmat).all()
        ):
            raise ValueError("Distances require a nonempty finite matrix")
        num_q, num_g = distmat.shape
        if any(
            v.shape != (n,)
            for v, n in (
                (q_pids, num_q),
                (g_pids, num_g),
                (q_camids, num_q),
                (g_camids, num_g),
            )
        ):
            raise ValueError(
                "Identity/camera metadata does not match distance dimensions"
            )
        if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank <= 0:
            raise ValueError("max_rank must be a positive integer")
        max_rank = min(max_rank, num_g)
        indices = np.argsort(distmat, axis=1, kind="stable")
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        all_cmc = []
        all_ap = []
        for q_idx in range(num_q):
            order = indices[q_idx]
            remove = (g_pids[order] == q_pids[q_idx]) & (
                g_camids[order] == q_camids[q_idx]
            )
            orig_cmc = matches[q_idx][~remove]
            if not np.any(orig_cmc):
                continue
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(
                np.pad(
                    cmc[:max_rank], (0, max_rank - min(len(cmc), max_rank)), mode="edge"
                )
            )
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
            all_ap.append(float((tmp_cmc * orig_cmc).sum() / num_rel))
        if not all_cmc:
            raise ValueError("No query identity appears in the gallery")
        cmc_curve = np.asarray(all_cmc, dtype=np.float64).sum(0) / len(all_cmc)
        return cmc_curve, float(np.mean(all_ap))


class Market1501Evaluator(RetrievalEvaluator):
    """Market-1501 held-out retrieval protocol v1."""

    dataset = "market1501"


class DukeMTMCReIDEvaluator(RetrievalEvaluator):
    """DukeMTMC-reID held-out retrieval protocol v1."""

    dataset = "dukemtmc-reid"


class OccludedDukeEvaluator(RetrievalEvaluator):
    """Occluded-Duke held-out retrieval protocol v1."""

    dataset = "occluded-duke"

"""Verbatim functions from datasets/_shared/reid.py at
ec537764e36ca10da963fcb7afab157ea60ca022 (mblt-model-ops).
Snapshot permits ordinary offline differential tests without sibling repos.
"""
import numpy as np
MAX_RANK = 50

def euclidean_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Squared euclidean distance, the reference ``euclidean_distance`` without the sqrt.

    Ranking is monotone in the square, so CMC and mAP are unchanged.
    """
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    q2 = np.sum(query**2, axis=1, keepdims=True)
    g2 = np.sum(gallery**2, axis=1, keepdims=True).T
    return q2 + g2 - 2.0 * (query @ gallery.T)

def evaluate(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = MAX_RANK,
) -> tuple[np.ndarray, float]:
    """Market-1501 protocol CMC curve and mAP (reference ``eval_func``)."""
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1, kind="stable")
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_ap = []
    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        orig_cmc = matches[q_idx][~remove]
        if not np.any(orig_cmc):
            continue  # the query identity never appears in the gallery
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        # Same-identity/same-camera removal leaves different gallery lengths.
        # Saturate missing ranks with the final CMC value for this query.
        all_cmc.append(
            np.pad(cmc[:max_rank], (0, max_rank - min(len(cmc), max_rank)), mode="edge")
        )
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
        all_ap.append(float((tmp_cmc * orig_cmc).sum() / num_rel))
    if not all_cmc:
        raise ValueError("No query identity appears in the gallery")
    cmc_curve = np.asarray(all_cmc, dtype=np.float64).sum(0) / len(all_cmc)
    return cmc_curve, float(np.mean(all_ap))

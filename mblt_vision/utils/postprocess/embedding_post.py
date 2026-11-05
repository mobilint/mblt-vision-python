"""Validated appearance embeddings, shared by retrieval and verification tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ...datasets import get_dataset_config_for_task
from ..types import ListTensorLike, TensorLike
from .base import PostBase


class EmbeddingPost(PostBase):
    """Return floating-point ``(batch, dimension)`` embeddings without argmax.

    Accept a single matrix or singleton-spatial NCHW/NHWC output (also wrapped
    in a one-element runtime output list). L2 normalization is opt-in; zero
    vectors stay zero, matching the source embedding decoder.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict) -> None:
        super().__init__()
        self.normalize = post_cfg.get("normalize", False)
        if not isinstance(self.normalize, bool):
            raise TypeError("post_cfg.normalize must be a boolean")
        self.embedding_dim = post_cfg.get("embedding_dim")
        if self.embedding_dim is not None and (
            not isinstance(self.embedding_dim, int)
            or isinstance(self.embedding_dim, bool)
            or self.embedding_dim <= 0
        ):
            raise ValueError("post_cfg.embedding_dim must be a positive integer")

    def __call__(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Validate one dense output and optionally L2-normalize each row."""
        if isinstance(x, Sequence):
            if len(x) != 1:
                raise ValueError(
                    "Embedding postprocessing expects exactly one output tensor"
                )
            x = x[0]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("Embedding output must be a NumPy array or torch tensor")
        if not x.is_floating_point():
            raise TypeError("Embedding output must have a floating-point dtype")
        x = x.to(self.device)
        if x.ndim == 4 and (x.shape[2:] == (1, 1) or x.shape[1:3] == (1, 1)):
            x = x.reshape(x.shape[0], -1)
        if x.ndim != 2 or min(x.shape) <= 0:
            raise ValueError(
                f"Embedding output requires nonempty (N, D), got {tuple(x.shape)}"
            )
        if self.embedding_dim is not None and x.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {x.shape[1]} does not match {self.embedding_dim}"
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Embedding output must contain only finite values")
        # Float32 agrees with the ops decoder, including its 1e-12 norm floor.
        x = x.float()
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Embedding values overflow float32")
        if self.normalize and not bool(
            torch.isfinite(torch.linalg.vector_norm(x, dim=1)).all()
        ):
            raise ValueError("Embedding norms overflow float32")
        if self.normalize:
            x = F.normalize(x, p=2, dim=1, eps=1e-12)
        return x


class PersonReIDPost(EmbeddingPost):
    """Appearance embeddings tied to an explicitly configured ReID dataset."""

    def __init__(self, pre_cfg: dict, post_cfg: dict) -> None:
        super().__init__(pre_cfg, post_cfg)
        dataset = post_cfg.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Person ReID postprocessing requires post_cfg.dataset")
        self.dataset = get_dataset_config_for_task("person_reid", dataset)["name"]


class FaceRecognitionPost(EmbeddingPost):
    """One identity embedding per prepared aligned crop; no implicit flip fusion."""

    def __init__(self, pre_cfg: dict, post_cfg: dict) -> None:
        super().__init__(pre_cfg, post_cfg)
        dataset = post_cfg.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(
                "Face recognition postprocessing requires post_cfg.dataset"
            )
        self.dataset = get_dataset_config_for_task("face_recognition", dataset)["name"]

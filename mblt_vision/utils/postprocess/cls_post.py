"""
Classification postprocessing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...datasets import get_dataset_class_names, get_dataset_config
from ..types import ListTensorLike, TensorLike
from .base import PostBase


class ClsPost(PostBase):
    """Post-processing for image classification models.

    Typically applies softmax to logits and ensures correct output shape.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict) -> None:
        """Initializes the classification post-processing.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        super().__init__()
        self.softmax = post_cfg.get("softmax", False)
        self.dataset: str | None = None
        self.num_classes: int | None = None
        dataset = post_cfg.get("dataset")
        if dataset is not None:
            if not isinstance(dataset, str) or not dataset:
                raise ValueError(
                    "Classification postprocessing requires post_cfg.dataset to be a non-empty string."
                )
            dataset = dataset.lower()
            dataset_config = get_dataset_config(dataset)
            if "image_classification" not in dataset_config["tasks"]:
                raise ValueError(
                    f"Dataset '{dataset}' does not support image classification."
                )
            self.dataset = dataset
            self.num_classes = len(get_dataset_class_names(dataset))

    def __call__(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Executes classification post-processing.

        Typically applies softmax to convert logits to probabilities.

        Args:
            x (TensorLike | ListTensorLike): Raw model outputs.
                Expected to be pre-softmax logits.

        Returns:
            torch.Tensor: Softmax probabilities of shape (N, C).
        """
        if isinstance(x, Sequence):
            if len(x) != 1:
                raise ValueError(
                    f"Classification postprocessing expects one output tensor, got {len(x)}."
                )
            x = x[0]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.ndim == 3:
            if self.num_classes is None:
                raise ValueError(
                    "Classification 3D outputs require a configured class count to disambiguate layout."
                )
            if x.shape[1] == self.num_classes and x.shape[-1] == 1:
                x = x.unsqueeze(-1)
            elif x.shape[0] == self.num_classes:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unsupported 3D classification output shape {tuple(x.shape)} for {self.num_classes} classes."
                )
        if x.ndim != 4:
            raise ValueError(
                f"Classification output must be convertible to NCHW, got shape {tuple(x.shape)}."
            )
        x = x.flatten(
            1
        )  # Classification heads may retain singleton spatial dimensions.
        if self.num_classes is not None and x.shape[1] != self.num_classes:
            raise ValueError(
                f"Classification output has {x.shape[1]} classes, but dataset "
                f"'{self.dataset}' requires {self.num_classes}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("Classification output scores must all be finite.")
        if self.softmax:
            if not bool(((x >= 0) & (x <= 1)).all()):
                raise ValueError(
                    "Classification probability outputs must be in [0, 1] when post_cfg.softmax is true."
                )
            if not torch.allclose(
                x.sum(dim=-1),
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device),
                rtol=1e-4,
                atol=1e-4,
            ):
                raise ValueError(
                    "Classification probability outputs must sum to 1 per sample when post_cfg.softmax is true."
                )
            return x
        return x.softmax(dim=-1)

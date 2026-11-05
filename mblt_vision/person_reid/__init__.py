"""Person retrieval utilities; published model wrappers are added separately."""

from .._compat import create_model_class
from ..utils.datasets.reid import (
    DukeMTMCReIDDataset,
    Market1501Dataset,
    OccludedDukeDataset,
    ReIDDataset,
    ReIDSample,
)
from ..utils.evaluation.retrieval import (
    DukeMTMCReIDEvaluator,
    Market1501Evaluator,
    OccludedDukeEvaluator,
    RetrievalEvaluator,
    RetrievalResult,
    extract_embeddings,
)
from ..utils.postprocess.embedding_post import EmbeddingPost, PersonReIDPost

__all__ = [
    "DukeMTMCReIDDataset",
    "DukeMTMCReIDEvaluator",
    "EmbeddingPost",
    "Market1501Dataset",
    "Market1501Evaluator",
    "OccludedDukeDataset",
    "OccludedDukeEvaluator",
    "PersonReIDPost",
    "ReIDDataset",
    "ReIDSample",
    "RetrievalEvaluator",
    "RetrievalResult",
    "extract_embeddings",
]

# Concrete published single-input models; utilities above remain explicit exports.
CLIP_ReID_RN50_Baseline_DukeMTMC = create_model_class(
    "CLIP_ReID_RN50_Baseline_DukeMTMC", __name__
)
CLIP_ReID_RN50_Baseline_Market1501 = create_model_class(
    "CLIP_ReID_RN50_Baseline_Market1501", __name__
)
CLIP_ReID_RN50_Baseline_OccDuke = create_model_class(
    "CLIP_ReID_RN50_Baseline_OccDuke", __name__
)
CLIP_ReID_RN50_DukeMTMC = create_model_class("CLIP_ReID_RN50_DukeMTMC", __name__)
CLIP_ReID_RN50_Market1501 = create_model_class("CLIP_ReID_RN50_Market1501", __name__)
CLIP_ReID_RN50_OccDuke = create_model_class("CLIP_ReID_RN50_OccDuke", __name__)
CLIP_ReID_ViT_B_16_Baseline_DukeMTMC = create_model_class(
    "CLIP_ReID_ViT_B_16_Baseline_DukeMTMC", __name__
)
CLIP_ReID_ViT_B_16_Baseline_Market1501 = create_model_class(
    "CLIP_ReID_ViT_B_16_Baseline_Market1501", __name__
)
CLIP_ReID_ViT_B_16_Baseline_OccDuke = create_model_class(
    "CLIP_ReID_ViT_B_16_Baseline_OccDuke", __name__
)
CLIP_ReID_ViT_B_16_DukeMTMC = create_model_class(
    "CLIP_ReID_ViT_B_16_DukeMTMC", __name__
)
CLIP_ReID_ViT_B_16_Market1501 = create_model_class(
    "CLIP_ReID_ViT_B_16_Market1501", __name__
)
CLIP_ReID_ViT_B_16_OccDuke = create_model_class("CLIP_ReID_ViT_B_16_OccDuke", __name__)

MODEL_EXPORTS = [
    "CLIP_ReID_RN50_Baseline_DukeMTMC",
    "CLIP_ReID_RN50_Baseline_Market1501",
    "CLIP_ReID_RN50_Baseline_OccDuke",
    "CLIP_ReID_RN50_DukeMTMC",
    "CLIP_ReID_RN50_Market1501",
    "CLIP_ReID_RN50_OccDuke",
    "CLIP_ReID_ViT_B_16_Baseline_DukeMTMC",
    "CLIP_ReID_ViT_B_16_Baseline_Market1501",
    "CLIP_ReID_ViT_B_16_Baseline_OccDuke",
    "CLIP_ReID_ViT_B_16_DukeMTMC",
    "CLIP_ReID_ViT_B_16_Market1501",
    "CLIP_ReID_ViT_B_16_OccDuke",
]

__all__ += MODEL_EXPORTS

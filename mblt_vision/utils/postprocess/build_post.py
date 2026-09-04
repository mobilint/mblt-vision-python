"""
Postprocessing builder.
"""

from __future__ import annotations

from ..._tasks import normalize_vision_task
from .base import PostBase
from .cls_post import ClsPost
from .depth_post import DepthPost
from .semantic_seg_post import SemanticSegPost
from .yolo_anchor_post import (
    YOLOAnchorDetectionPost,
    YOLOAnchorFaceDetectionPost,
    YOLOAnchorSegPost,
)
from .yolo_anchorless_post import (
    YOLOAnchorlessDetectionPost,
    YOLOAnchorlessFaceDetectionPost,
    YOLOAnchorlessOBBPost,
    YOLOAnchorlessPosePost,
    YOLOAnchorlessSegPost,
)
from .yolo_dflfree_post import (
    YOLODFLFreeDetectionPost,
    YOLODFLFreeFaceDetectionPost,
    YOLODFLFreeOBBPost,
    YOLODFLFreePosePost,
    YOLODFLFreeSegPost,
)
from .damoyolo_post import DAMOYOLODetectionPost, DAMOYOLOFaceDetectionPost
from .yolo_nmsfree_post import YOLONMSFreeDetectionPost, YOLONMSFreeFaceDetectionPost
from .yolox_post import YOLOXDetectionPost, YOLOXFaceDetectionPost


def build_postprocess(
    pre_cfg: dict,
    post_cfg: dict,
    **kwargs: object,
) -> PostBase:
    """Builds a postprocessing object based on the model configuration.

    Args:
        pre_cfg (dict): Preprocessing configuration from the model info.
        post_cfg (dict): Postprocessing configuration from the model info.
            Must contain "task" and relevant flags for the specific task.
        **kwargs: Optional runtime overrides passed to the postprocessor.

    Returns:
        PostBase: An instance of a postprocessing class tailored for the task.

    Raises:
        NotImplementedError: If the specified task is not supported.
    """
    task = normalize_vision_task(post_cfg["task"])
    if task == "image_classification":
        return ClsPost(pre_cfg, post_cfg)
    if task == "depth_estimation":
        return DepthPost(pre_cfg, post_cfg)
    if task == "semantic_segmentation":
        return SemanticSegPost(pre_cfg, post_cfg)
    if task == "face_detection":
        if post_cfg.get("yolox", False):
            return YOLOXFaceDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("damoyolo", False):
            return DAMOYOLOFaceDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("anchors", False):
            return YOLOAnchorFaceDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("dflfree", False):  # nms free is only available for detection
            return YOLODFLFreeFaceDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("nmsfree", False):
            return YOLONMSFreeFaceDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessFaceDetectionPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task == "object_detection":
        if post_cfg.get("yolox", False):
            return YOLOXDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("damoyolo", False):
            return DAMOYOLODetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("anchors", False):
            return YOLOAnchorDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("dflfree", False):  # nms free is only available for detection
            return YOLODFLFreeDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("nmsfree", False):
            return YOLONMSFreeDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessDetectionPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task == "instance_segmentation":
        if post_cfg.get("anchors", False):
            return YOLOAnchorSegPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("dflfree", False):
            return YOLODFLFreeSegPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessSegPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task == "pose_estimation":
        if post_cfg.get("dflfree", False):
            return YOLODFLFreePosePost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessPosePost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task == "obb":
        if post_cfg.get("dflfree", False):
            return YOLODFLFreeOBBPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessOBBPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")

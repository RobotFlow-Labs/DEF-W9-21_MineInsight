"""Hungarian bipartite matcher for DETR-style detection.

.. warning::

    **ARCHIVED** — this module belongs to the legacy custom CSPDarknet +
    Hungarian + focal-loss training pipeline that was superseded by stock
    Ultralytics YOLO26s + TAL assigner in April 2026 (see ``PIVOT_PLAN.md``).
    YOLO26's TaskAlignedAssigner is a pure-torch GPU operation and does NOT
    suffer from the CPU sync penalty of the scipy call below. **Do not use
    this matcher for new training runs** — use ``scripts/train_yolo26_fusion.py``.

Performance caveat for the legacy path:
    This matcher calls ``scipy.optimize.linear_sum_assignment`` on every
    forward pass, which requires a GPU→CPU sync of the cost matrix and
    releases the Python GIL. There is no direct cupy equivalent
    (``cupyx.scipy`` does not ship Hungarian). A GPU replacement would
    require a custom ``auction-lap`` / ``lapjv`` kernel or porting
    Ultralytics' ``TaskAlignedAssigner`` — which is already what we use in
    the production pipeline.

Matches predicted anchors to ground truth targets using optimal assignment
via scipy.optimize.linear_sum_assignment. Cost matrix combines classification,
L1 bbox, and GIoU costs.

Adapted from DEF-UAVDETR (project_def_uavdetr/src/anima_def_uavdetr/matcher.py).
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from mineinsight.losses import box_cxcywh_to_xyxy


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute GIoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4) in xyxy format.
        boxes2: (M, 4) in xyxy format.

    Returns:
        (N, M) GIoU matrix.
    """
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / (union + 1e-7)

    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    return iou - (enc_area - union) / (enc_area + 1e-7)


class HungarianMatcher:
    """Bipartite matching between predictions and ground truth.

    Cost = cost_class * cls_cost + cost_bbox * l1_cost + cost_giou * giou_cost

    For dense detectors (8400 anchors), we pre-filter to top-K candidates
    per GT to keep the cost matrix tractable.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        topk_candidates: int = 20,
    ):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.topk_candidates = topk_candidates

    @torch.no_grad()
    def __call__(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        n_targets: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match predictions to targets for a single image.

        Args:
            pred_boxes: (A, 4) predicted boxes in cxcywh pixels.
            pred_logits: (A, num_classes+1) class logits (class 0 = background).
            gt_boxes: (T, 4) ground truth boxes in cxcywh pixels.
            gt_labels: (T,) ground truth class labels (1-indexed, 0=unused).
            n_targets: Number of valid targets.

        Returns:
            (pred_indices, gt_indices) — matched pairs.
        """
        if n_targets == 0:
            return (
                torch.tensor([], dtype=torch.long, device=pred_boxes.device),
                torch.tensor([], dtype=torch.long, device=pred_boxes.device),
            )

        gt_boxes = gt_boxes[:n_targets]
        gt_labels = gt_labels[:n_targets]
        n_preds = pred_boxes.shape[0]

        # Pre-filter: select top-K closest predictions per GT by center distance
        pred_centers = pred_boxes[:, :2]
        gt_centers = gt_boxes[:, :2]
        dist = torch.cdist(pred_centers, gt_centers)  # (A, T)

        k = min(self.topk_candidates, n_preds)
        _, topk_per_gt = dist.topk(k, dim=0, largest=False)  # (K, T)
        candidate_idx = topk_per_gt.unique()  # flattened unique indices

        # Subset predictions to candidates
        cand_boxes = pred_boxes[candidate_idx]  # (C, 4)
        cand_logits = pred_logits[candidate_idx]  # (C, num_classes+1)

        # Classification cost (focal)
        cand_scores = cand_logits.sigmoid()  # (C, num_classes+1)
        # +1 offset: gt_labels are 1-indexed in our system, class 0 is background
        scores_for_gt = cand_scores[:, gt_labels.long()]  # (C, T)

        neg_cost = (
            (1 - self.focal_alpha)
            * scores_for_gt**self.focal_gamma
            * (-(1 - scores_for_gt + 1e-8).log())
        )
        pos_cost = (
            self.focal_alpha
            * (1 - scores_for_gt) ** self.focal_gamma
            * (-(scores_for_gt + 1e-8).log())
        )
        cost_class = pos_cost - neg_cost  # (C, T)

        # L1 bbox cost
        cost_bbox = torch.cdist(cand_boxes.float(), gt_boxes.float(), p=1)  # (C, T)

        # GIoU cost
        cand_xyxy = box_cxcywh_to_xyxy(cand_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        cost_giou = -generalized_box_iou(cand_xyxy, gt_xyxy)  # (C, T)

        # Final cost matrix
        cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        # Hungarian assignment
        cost_np = cost.detach().cpu().numpy()
        cand_idx_matched, gt_idx_matched = linear_sum_assignment(cost_np)

        # Map candidate indices back to original prediction indices
        pred_idx_matched = candidate_idx[cand_idx_matched]

        return (
            pred_idx_matched.to(pred_boxes.device),
            torch.tensor(gt_idx_matched, dtype=torch.long, device=pred_boxes.device),
        )

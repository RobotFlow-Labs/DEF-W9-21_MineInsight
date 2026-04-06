"""Detection loss functions for MineInsight.

Components:
- CIoU loss for bounding box regression
- Focal loss for classification (handles mine/distractor imbalance)
- BCE loss for objectness confidence
- Combined DetectionLoss with configurable weights
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Complete IoU loss for bounding box regression.

    Accounts for overlap area, center distance, and aspect ratio consistency.

    Args:
        pred_boxes: (N, 4) predicted boxes in cxcywh format
        target_boxes: (N, 4) target boxes in cxcywh format

    Returns:
        Scalar CIoU loss (mean over batch).
    """
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area_pred = pred_boxes[..., 2] * pred_boxes[..., 3]
    area_tgt = target_boxes[..., 2] * target_boxes[..., 3]
    union = area_pred + area_tgt - inter

    iou = inter / (union + eps)

    # Enclosing box diagonal
    enc_x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    enc_y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    enc_x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    enc_y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    c_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    # Center distance
    d_sq = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 + (
        pred_boxes[..., 1] - target_boxes[..., 1]
    ) ** 2

    # Aspect ratio consistency
    with torch.no_grad():
        arctan_pred = torch.atan2(pred_boxes[..., 2], pred_boxes[..., 3] + eps)
        arctan_tgt = torch.atan2(target_boxes[..., 2], target_boxes[..., 3] + eps)
        v = (4 / (math.pi**2)) * (arctan_pred - arctan_tgt) ** 2

    alpha = v / (1 - iou + v + eps)

    ciou = iou - d_sq / (c_diag_sq + eps) - alpha * v
    return (1 - ciou).mean()


class FocalLoss(nn.Module):
    """Focal loss for dense classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Balancing factor (default 0.25).
        gamma: Focusing parameter (default 2.0).
        reduction: "mean" or "sum".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: (N, C) logits (pre-sigmoid).
            target: (N,) class indices or (N, C) one-hot.
        """
        if target.dim() == 1:
            # Convert to one-hot
            num_classes = pred.shape[-1]
            target_oh = F.one_hot(target.long(), num_classes).float()
        else:
            target_oh = target.float()

        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target_oh, reduction="none")

        p_t = p * target_oh + (1 - p) * (1 - target_oh)
        alpha_t = self.alpha * target_oh + (1 - self.alpha) * (1 - target_oh)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """Combined detection loss: box regression + classification + objectness.

    Args:
        num_classes: Number of object classes.
        box_weight: Weight for CIoU box loss.
        cls_weight: Weight for focal classification loss.
        obj_weight: Weight for BCE objectness loss.
        focal_alpha: Focal loss alpha.
        focal_gamma: Focal loss gamma.
    """

    def __init__(
        self,
        num_classes: int = 35,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight

        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined detection loss.

        Args:
            predictions: List of (B, N_i, 5+num_classes) per scale.
            targets: (B, max_targets, 5) with [cls_id, cx, cy, w, h] in pixels.
            target_counts: (B,) number of valid targets per sample.

        Returns:
            Dict with "loss", "box_loss", "cls_loss", "obj_loss".
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        num_pos = 0

        # Concatenate all scale predictions
        all_preds = torch.cat(predictions, dim=1)  # (B, total_anchors, 5+C)

        for b in range(batch_size):
            n_targets = target_counts[b].item()
            pred = all_preds[b]  # (total_anchors, 5+C)

            pred_box = pred[:, :4]   # (A, 4) cx, cy, w, h
            pred_obj = pred[:, 4]    # (A,) objectness logit
            pred_cls = pred[:, 5:]   # (A, C) class logits

            if n_targets == 0:
                # No targets: all predictions should have low objectness
                obj_target = torch.zeros_like(pred_obj)
                total_obj = total_obj + self.bce(pred_obj, obj_target)
                continue

            gt = targets[b, :n_targets]  # (T, 5)
            gt_cls = gt[:, 0].long()     # (T,)
            gt_box = gt[:, 1:5]          # (T, 4) cx, cy, w, h

            # Simple assignment: match each prediction to closest target by center distance
            pred_centers = pred_box[:, :2]    # (A, 2)
            gt_centers = gt_box[:, :2]        # (T, 2)
            dist = torch.cdist(pred_centers, gt_centers)  # (A, T)
            min_dist, assigned_gt = dist.min(dim=1)       # (A,)

            # Positive mask: predictions within a radius of any target center
            # Use a dynamic radius based on target size
            gt_sizes = (gt_box[:, 2] + gt_box[:, 3]) / 2  # mean of w, h
            assigned_radius = gt_sizes[assigned_gt] * 0.5
            pos_mask = min_dist < assigned_radius

            num_pos_this = pos_mask.sum().item()
            if num_pos_this == 0:
                # Fallback: take top-K closest per target
                k = min(3, pred_box.shape[0])
                _, topk_idx = dist.topk(k, dim=0, largest=False)
                pos_mask_flat = torch.zeros(pred_box.shape[0], dtype=torch.bool, device=device)
                pos_mask_flat[topk_idx.flatten()] = True
                pos_mask = pos_mask_flat
                num_pos_this = pos_mask.sum().item()

            if num_pos_this > 0:
                num_pos += num_pos_this

                # Box loss (positive predictions only)
                pos_pred_box = pred_box[pos_mask]
                pos_gt_idx = assigned_gt[pos_mask]
                pos_gt_box = gt_box[pos_gt_idx]
                total_box = total_box + ciou_loss(pos_pred_box, pos_gt_box)

                # Classification loss (positive predictions only)
                pos_pred_cls = pred_cls[pos_mask]
                pos_gt_cls = gt_cls[pos_gt_idx]
                total_cls = total_cls + self.focal(pos_pred_cls, pos_gt_cls)

            # Objectness loss (all predictions)
            obj_target = pos_mask.float()
            total_obj = total_obj + self.bce(pred_obj, obj_target)

        # Normalize
        num_pos = max(num_pos, 1)
        box_loss = total_box / batch_size
        cls_loss = total_cls / batch_size
        obj_loss = total_obj / batch_size

        total_loss = (
            self.box_weight * box_loss
            + self.cls_weight * cls_loss
            + self.obj_weight * obj_loss
        )

        return {
            "loss": total_loss,
            "box_loss": box_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "obj_loss": obj_loss.detach(),
        }

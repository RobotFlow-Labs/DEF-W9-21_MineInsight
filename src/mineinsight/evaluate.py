"""MineInsight evaluation pipeline.

Computes mAP@0.5, mAP@0.5:0.95, per-class AP, mine-specific metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mineinsight.dataset import (
    ALL_CLASSES,
    MINE_CLASS_IDS,
    MineInsightDataset,
    collate_fn,
)
from mineinsight.losses import box_cxcywh_to_xyxy, box_iou
from mineinsight.model import build_model
from mineinsight.utils import Config, get_device, load_config, set_seed


def decode_predictions(
    predictions: list[torch.Tensor],
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
) -> list[dict[str, torch.Tensor]]:
    """Decode raw model predictions into detection results.

    Uses CUDA-accelerated NMS when available (shared vectorized_nms kernel).

    Args:
        predictions: List of (B, N, 5+C) per scale.
        conf_threshold: Minimum confidence to keep.
        nms_iou_threshold: IoU threshold for NMS.

    Returns:
        List of dicts per image: {"boxes": (M, 4) xyxy, "scores": (M,), "labels": (M,)}.
    """
    from mineinsight.cuda_ops import cuda_nms_2d

    all_preds = torch.cat(predictions, dim=1)  # (B, total, 5+C)
    batch_size = all_preds.shape[0]
    results = []
    empty = lambda dev: {  # noqa: E731
        "boxes": torch.zeros((0, 4), device=dev),
        "scores": torch.zeros(0, device=dev),
        "labels": torch.zeros(0, dtype=torch.long, device=dev),
    }

    for b in range(batch_size):
        pred = all_preds[b]  # (A, 4+C+1)

        box_cxcywh = pred[:, :4]
        cls_logit = pred[:, 4:]  # (A, num_classes+1) — class 0 is background

        cls_conf = torch.sigmoid(cls_logit)

        # Skip background class (index 0) — detection score = max non-background class
        fg_conf = cls_conf[:, 1:]  # (A, num_classes)
        scores, labels = fg_conf.max(dim=-1)
        labels = labels + 1  # offset back to 1-indexed class IDs

        mask = scores > conf_threshold
        if mask.sum() == 0:
            results.append(empty(pred.device))
            continue

        boxes_xyxy = box_cxcywh_to_xyxy(box_cxcywh[mask])
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]

        # Per-class NMS using CUDA kernel
        keep_indices = []
        for cls_id in filtered_labels.unique():
            cls_mask = filtered_labels == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = filtered_scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            nms_keep = cuda_nms_2d(cls_boxes, cls_scores, nms_iou_threshold)
            keep_indices.extend(cls_indices[nms_keep].tolist())

        if keep_indices:
            keep_t = torch.tensor(keep_indices, dtype=torch.long, device=pred.device)
            results.append({
                "boxes": boxes_xyxy[keep_t],
                "scores": filtered_scores[keep_t],
                "labels": filtered_labels[keep_t],
            })
        else:
            results.append(empty(pred.device))

    return results


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using the 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_recall = mpre[mrec >= t]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    return ap / 101.0


def compute_map(
    all_detections: list[dict[str, torch.Tensor]],
    all_targets: list[torch.Tensor],
    all_target_counts: list[int],
    num_classes: int = 58,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mAP at a given IoU threshold.

    Returns dict with per-class AP and overall mAP.
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_scores = []
        all_tp = []
        n_gt = 0

        for det, gt, n_targets in zip(all_detections, all_targets, all_target_counts, strict=False):
            # Ground truth boxes for this class
            gt_valid = gt[:n_targets]
            gt_cls_mask = gt_valid[:, 0] == cls_id
            gt_boxes = gt_valid[gt_cls_mask, 1:5]  # cx, cy, w, h
            n_gt += len(gt_boxes)
            gt_matched = [False] * len(gt_boxes)

            # Detection boxes for this class
            if len(det["labels"]) == 0:
                continue
            det_cls_mask = det["labels"] == cls_id
            det_boxes = det["boxes"][det_cls_mask]  # xyxy
            det_scores = det["scores"][det_cls_mask]

            if len(det_boxes) == 0 or len(gt_boxes) == 0:
                for _ in range(len(det_boxes)):
                    all_scores.append(det_scores[_].item() if len(det_scores) > _ else 0)
                    all_tp.append(0)
                continue

            # Convert gt to xyxy for IoU
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

            # Sort detections by score
            order = det_scores.argsort(descending=True)
            det_boxes = det_boxes[order]
            det_scores = det_scores[order]

            for d_idx in range(len(det_boxes)):
                ious = box_iou(det_boxes[d_idx:d_idx + 1], gt_xyxy)[0]
                best_iou, best_gt = ious.max(dim=0) if len(ious) > 0 else (
                    torch.tensor(0.0), torch.tensor(0),
                )

                all_scores.append(det_scores[d_idx].item())
                if best_iou.item() >= iou_threshold and not gt_matched[best_gt.item()]:
                    all_tp.append(1)
                    gt_matched[best_gt.item()] = True
                else:
                    all_tp.append(0)

        if n_gt == 0:
            continue

        # Sort by score
        if len(all_scores) == 0:
            per_class_ap[cls_id] = 0.0
            continue

        indices = np.argsort(-np.array(all_scores))
        tp = np.array(all_tp)[indices]
        fp = 1 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)

        per_class_ap[cls_id] = compute_ap(recall, precision)

    # Overall mAP
    if per_class_ap:
        mean_ap = float(np.mean(list(per_class_ap.values())))
    else:
        mean_ap = 0.0

    # Mine-specific mAP
    mine_aps = [per_class_ap.get(c, 0.0) for c in MINE_CLASS_IDS if c in per_class_ap]
    mine_map = float(np.mean(mine_aps)) if mine_aps else 0.0

    return {
        "mAP": mean_ap,
        "mine_mAP": mine_map,
        "per_class_ap": {ALL_CLASSES[k]: v for k, v in per_class_ap.items()},
    }


@torch.no_grad()
def evaluate(
    cfg: Config,
    checkpoint_path: str | None = None,
    split: str = "test",
    conf_threshold: float = 0.25,
) -> dict:
    """Run full evaluation on a dataset split.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to model checkpoint.
        split: "val" or "test".
        conf_threshold: Detection confidence threshold.

    Returns:
        Evaluation results dict.
    """
    set_seed(cfg.training.seed)
    device = get_device()

    # Build model
    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
        architecture=cfg.model.architecture,
        pretrained=cfg.model.pretrained,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[EVAL] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)

    model.eval()

    # Build dataloader
    sequences = cfg.data.test_sequences if split == "test" else cfg.data.val_sequences
    dataset = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=sequences,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    print(f"[EVAL] Evaluating on {split} split ({len(dataset)} samples)")

    all_detections = []
    all_targets = []
    all_target_counts = []
    total_time = 0.0
    num_images = 0

    precision_type = cfg.training.precision
    use_amp = precision_type in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision_type == "bf16" else torch.float16

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["targets"]
        target_counts = batch["target_counts"]

        t0 = time.time()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                preds = model(images)
        else:
            preds = model(images)
        total_time += time.time() - t0
        num_images += images.shape[0]

        dets = decode_predictions(preds, conf_threshold=conf_threshold)
        all_detections.extend(dets)

        for b in range(len(targets)):
            all_targets.append(targets[b])
            all_target_counts.append(target_counts[b].item())

    # Compute mAP at multiple IoU thresholds
    results_50 = compute_map(
        all_detections, all_targets, all_target_counts,
        num_classes=cfg.model.num_classes, iou_threshold=0.5,
    )

    # mAP@0.5:0.95
    map_values = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        r = compute_map(
            all_detections, all_targets, all_target_counts,
            num_classes=cfg.model.num_classes, iou_threshold=iou_thresh,
        )
        map_values.append(r["mAP"])
    map_50_95 = float(np.mean(map_values))

    fps = num_images / max(total_time, 1e-6)

    results = {
        "split": split,
        "num_images": num_images,
        "mAP@0.5": results_50["mAP"],
        "mAP@0.5:0.95": map_50_95,
        "mine_mAP@0.5": results_50["mine_mAP"],
        "per_class_ap@0.5": results_50["per_class_ap"],
        "fps": fps,
        "conf_threshold": conf_threshold,
    }

    print(f"[RESULTS] mAP@0.5={results_50['mAP']:.4f}")
    print(f"[RESULTS] mAP@0.5:0.95={map_50_95:.4f}")
    print(f"[RESULTS] mine_mAP@0.5={results_50['mine_mAP']:.4f}")
    print(f"[RESULTS] FPS={fps:.1f}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MineInsight Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate(cfg, args.checkpoint, args.split, args.conf_threshold)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[SAVED] Results to {output_path}")


if __name__ == "__main__":
    main()

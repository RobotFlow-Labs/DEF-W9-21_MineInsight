#!/usr/bin/env python3
"""Comprehensive eval sweep for the canonical v5 fusion checkpoint.

Runs:
- Both val and test splits
- Multiple confidence thresholds (0.05, 0.10, 0.15, 0.25, 0.40, 0.60)
- mAP@0.5, mAP@0.5:0.95, mine_mAP, per-class AP
- TP/FP/FN counts, precision, recall, F1
- FPS benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mineinsight.dataset import (
    MINE_CLASS_IDS,
    MineInsightDataset,
    collate_fn,
)
from mineinsight.evaluate import compute_map, decode_predictions
from mineinsight.losses import box_cxcywh_to_xyxy, box_iou
from mineinsight.model import build_model
from mineinsight.utils import get_device, load_config, set_seed


def count_tp_fp_fn(
    all_detections: list[dict],
    all_targets: list[torch.Tensor],
    all_target_counts: list[int],
    num_classes: int = 58,
    iou_threshold: float = 0.5,
) -> dict:
    """Count TP / FP / FN per class and overall."""
    per_cls = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)}

    for det, gt, n_t in zip(all_detections, all_targets, all_target_counts, strict=False):
        gt_valid = gt[:n_t]
        gt_classes = gt_valid[:, 0].long()
        gt_boxes_cxcywh = gt_valid[:, 1:5]

        det_boxes = det["boxes"]
        det_scores = det["scores"]
        det_labels = det["labels"]

        # For each class present in either preds or gt
        classes_here = set(gt_classes.tolist()) | set(det_labels.tolist())
        for c in classes_here:
            gt_mask = gt_classes == c
            cls_gt = gt_boxes_cxcywh[gt_mask]
            cls_gt_xyxy = box_cxcywh_to_xyxy(cls_gt) if len(cls_gt) else cls_gt

            det_mask = det_labels == c
            cls_det = det_boxes[det_mask]
            cls_sc = det_scores[det_mask]

            if len(cls_det) == 0:
                per_cls[c]["fn"] += len(cls_gt)
                continue
            if len(cls_gt) == 0:
                per_cls[c]["fp"] += len(cls_det)
                continue

            order = cls_sc.argsort(descending=True)
            cls_det = cls_det[order]

            gt_matched = [False] * len(cls_gt)
            for d in cls_det:
                ious = box_iou(d.unsqueeze(0), cls_gt_xyxy)[0]
                best_iou, best_gt = ious.max(dim=0)
                if best_iou.item() >= iou_threshold and not gt_matched[best_gt.item()]:
                    per_cls[c]["tp"] += 1
                    gt_matched[best_gt.item()] = True
                else:
                    per_cls[c]["fp"] += 1
            per_cls[c]["fn"] += gt_matched.count(False)

    tp = sum(v["tp"] for v in per_cls.values())
    fp = sum(v["fp"] for v in per_cls.values())
    fn = sum(v["fn"] for v in per_cls.values())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    mine_tp = sum(per_cls[c]["tp"] for c in MINE_CLASS_IDS)
    mine_fp = sum(per_cls[c]["fp"] for c in MINE_CLASS_IDS)
    mine_fn = sum(per_cls[c]["fn"] for c in MINE_CLASS_IDS)
    mine_prec = mine_tp / max(mine_tp + mine_fp, 1)
    mine_rec = mine_tp / max(mine_tp + mine_fn, 1)
    mine_f1 = 2 * mine_prec * mine_rec / max(mine_prec + mine_rec, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mine_tp": mine_tp, "mine_fp": mine_fp, "mine_fn": mine_fn,
        "mine_precision": mine_prec, "mine_recall": mine_rec, "mine_f1": mine_f1,
        "classes_with_detections": sum(
            1 for v in per_cls.values() if v["tp"] + v["fp"] > 0
        ),
        "classes_with_tp": sum(1 for v in per_cls.values() if v["tp"] > 0),
    }


@torch.no_grad()
def run_eval(
    cfg,
    checkpoint_path: str,
    split: str,
    conf_thresholds: list[float],
    batch_size: int = 8,
) -> dict:
    set_seed(cfg.training.seed)
    device = get_device()

    print(f"\n{'='*70}")
    print(f"[EVAL] Split={split}  Checkpoint={checkpoint_path}")
    print(f"{'='*70}")

    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
        architecture=cfg.model.architecture,
        pretrained=cfg.model.pretrained,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[EVAL] Loaded checkpoint (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    sequences = cfg.data.test_sequences if split == "test" else cfg.data.val_sequences
    dataset = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=sequences,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )
    print(f"[EVAL] Dataset: {len(dataset)} samples, batch={batch_size}")

    # Forward pass once, collect RAW predictions. Decode multiple times
    # at different thresholds to save GPU time.
    all_raw_preds = []
    all_targets = []
    all_target_counts = []
    total_time = 0.0
    num_images = 0

    precision_type = cfg.training.precision
    use_amp = precision_type in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision_type == "bf16" else torch.float16

    for i, batch in enumerate(loader):
        targets = batch["targets"]
        target_counts = batch["target_counts"]

        if "images" in batch:
            model_input = {mod: t.to(device) for mod, t in batch["images"].items()}
            batch_n = next(iter(model_input.values())).shape[0]
        else:
            model_input = batch["image"].to(device)
            batch_n = model_input.shape[0]

        t0 = time.time()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                preds = model(model_input)
        else:
            preds = model(model_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        num_images += batch_n

        # Keep on CPU to save GPU memory (will move back for decode)
        all_raw_preds.append([p.cpu() for p in preds])

        for b in range(len(targets)):
            all_targets.append(targets[b])
            all_target_counts.append(target_counts[b].item())

        if (i + 1) % 20 == 0:
            print(f"  forward {i+1}/{len(loader)}")

    fps = num_images / max(total_time, 1e-6)
    print(f"[EVAL] Forward done: {num_images} imgs in {total_time:.1f}s = {fps:.1f} FPS")

    # Sweep conf thresholds
    sweep = {}
    for conf in conf_thresholds:
        print(f"\n  [CONF={conf:.2f}] decoding + NMS + mAP…")
        all_dets = []
        for raw in all_raw_preds:
            raw_gpu = [p.to(device) for p in raw]
            dets = decode_predictions(raw_gpu, conf_threshold=conf, nms_iou_threshold=0.45)
            # Move back to CPU for storage
            dets_cpu = [
                {
                    "boxes": d["boxes"].cpu(),
                    "scores": d["scores"].cpu(),
                    "labels": d["labels"].cpu(),
                }
                for d in dets
            ]
            all_dets.extend(dets_cpu)

        map50 = compute_map(
            all_dets, all_targets, all_target_counts,
            num_classes=cfg.model.num_classes, iou_threshold=0.5,
        )
        # mAP@0.5:0.95 — skip full 10-IoU sweep if conf != primary, just report [0.5, 0.75]
        map75 = compute_map(
            all_dets, all_targets, all_target_counts,
            num_classes=cfg.model.num_classes, iou_threshold=0.75,
        )

        counts = count_tp_fp_fn(
            all_dets, all_targets, all_target_counts,
            num_classes=cfg.model.num_classes, iou_threshold=0.5,
        )

        # Extract per-class AP sorted
        per_cls = map50["per_class_ap"]
        top_cls = sorted(per_cls.items(), key=lambda kv: -kv[1])[:15]

        sweep[f"{conf:.2f}"] = {
            "mAP@0.5": map50["mAP"],
            "mAP@0.75": map75["mAP"],
            "mine_mAP@0.5": map50["mine_mAP"],
            "mine_mAP@0.75": map75["mine_mAP"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": counts["precision"],
            "recall": counts["recall"],
            "f1": counts["f1"],
            "mine_tp": counts["mine_tp"],
            "mine_fp": counts["mine_fp"],
            "mine_fn": counts["mine_fn"],
            "mine_precision": counts["mine_precision"],
            "mine_recall": counts["mine_recall"],
            "mine_f1": counts["mine_f1"],
            "classes_with_tp": counts["classes_with_tp"],
            "top_15_classes": [{"class": k, "ap": v} for k, v in top_cls if v > 0],
        }

        print(
            f"    mAP@0.5={map50['mAP']:.4f}  mine_mAP={map50['mine_mAP']:.4f}  "
            f"P={counts['precision']:.4f}  R={counts['recall']:.4f}  "
            f"F1={counts['f1']:.4f}  TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}",
        )

    return {
        "split": split,
        "checkpoint": checkpoint_path,
        "num_images": num_images,
        "fps": fps,
        "modality": cfg.model.modality,
        "input_size": list(cfg.model.input_size),
        "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default=(
            "/mnt/artifacts-datai/reports/project_mineinsight/canonical_eval.json"
        ),
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--conf-thresholds",
        default="0.05,0.10,0.15,0.25,0.40,0.60",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = args.splits.split(",")
    confs = [float(x) for x in args.conf_thresholds.split(",")]

    all_results = {}
    for split in splits:
        all_results[split] = run_eval(
            cfg, args.checkpoint, split.strip(), confs, batch_size=args.batch_size,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {out}")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = (
        f"{'split':<6} {'conf':<6} {'mAP@0.5':<10} {'mine_mAP':<10} "
        f"{'prec':<8} {'recall':<8} {'f1':<8} "
        f"{'TP':<6} {'FP':<8} {'FN':<6}"
    )
    print(header)
    print("-" * 90)
    for split, r in all_results.items():
        for conf, s in r["sweep"].items():
            row = (
                f"{split:<6} {conf:<6} "
                f"{s['mAP@0.5']:<10.4f} {s['mine_mAP@0.5']:<10.4f} "
                f"{s['precision']:<8.4f} {s['recall']:<8.4f} {s['f1']:<8.4f} "
                f"{s['tp']:<6} {s['fp']:<8} {s['fn']:<6}"
            )
            print(row)


if __name__ == "__main__":
    main()

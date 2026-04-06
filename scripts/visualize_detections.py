#!/usr/bin/env python3
"""Visualize MineInsight detections on images.

Draws bounding boxes with class labels and confidence scores.
Outputs annotated images for qualitative evaluation.

Usage:
    python scripts/visualize_detections.py \
        --config configs/paper.toml \
        --checkpoint best.pth \
        --output /mnt/artifacts-datai/reports/project_mineinsight/vis/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mineinsight.dataset import ALL_CLASSES, MINE_CLASS_IDS, MineInsightDataset
from mineinsight.evaluate import decode_predictions
from mineinsight.model import build_model
from mineinsight.utils import get_device, load_config

# Colors: mines in red, distractors in blue
MINE_COLOR = (0, 0, 255)  # BGR red
DISTRACTOR_COLOR = (255, 128, 0)  # BGR orange-blue


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Draw detection boxes on image.

    Args:
        image: (H, W, 3) BGR image.
        boxes: (N, 4) xyxy format.
        scores: (N,) confidence scores.
        labels: (N,) class indices.
        conf_threshold: Minimum confidence to draw.

    Returns:
        Annotated image.
    """
    img = image.copy()
    for i in range(len(scores)):
        if scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(labels[i])
        score = scores[i]

        color = MINE_COLOR if cls_id in MINE_CLASS_IDS else DISTRACTOR_COLOR
        cls_name = ALL_CLASSES[cls_id] if cls_id < len(ALL_CLASSES) else f"cls_{cls_id}"
        label = f"{cls_name} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    return img


@torch.no_grad()
def visualize(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    split: str = "test",
    max_images: int = 50,
    conf_threshold: float = 0.25,
) -> None:
    """Run inference and save annotated images."""
    cfg = load_config(config_path)
    device = get_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
        architecture=cfg.model.architecture,
        pretrained=cfg.model.pretrained,
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)

    model.eval()

    sequences = cfg.data.test_sequences if split == "test" else cfg.data.val_sequences
    dataset = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=sequences,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=False,
    )

    count = min(max_images, len(dataset))
    print(f"[VIS] Generating {count} annotated images...")

    for idx in range(count):
        sample = dataset[idx]
        img_tensor = sample["image"].unsqueeze(0).to(device)

        preds = model(img_tensor)
        dets = decode_predictions(preds, conf_threshold=conf_threshold)
        det = dets[0]

        # Convert image tensor back to BGR for OpenCV
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()

        annotated = draw_detections(img_bgr, boxes, scores, labels, conf_threshold)

        out_path = out / f"det_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), annotated)

    print(f"[VIS] Saved {count} images to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detections")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()

    visualize(args.config, args.checkpoint, args.output,
              args.split, args.max_images, args.conf_threshold)


if __name__ == "__main__":
    main()

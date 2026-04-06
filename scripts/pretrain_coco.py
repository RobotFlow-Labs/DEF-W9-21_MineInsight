#!/usr/bin/env python3
"""Pretrain MineInsight backbone on COCO val2017 for objectness learning.

Converts COCO annotations to YOLO format on-the-fly, trains for a few epochs
to warm up the backbone + objectness head, then saves weights for fine-tuning.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/pretrain_coco.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mineinsight.losses import DetectionLoss
from mineinsight.model import build_model
from mineinsight.utils import WarmupCosineScheduler, set_seed


class COCODetectionDataset(Dataset):
    """COCO val2017 in YOLO-compatible format for pretraining."""

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        input_size: tuple[int, int] = (640, 640),
        max_targets: int = 50,
    ):
        self.img_dir = Path(img_dir)
        self.input_size = input_size
        self.max_targets = max_targets

        with open(ann_file) as f:
            coco = json.load(f)

        # Build image index
        self.images = {img["id"]: img for img in coco["images"]}

        # Group annotations by image
        self.img_anns: dict[int, list] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)

        # Only keep images that have annotations
        self.img_ids = sorted(
            [iid for iid in self.img_anns if iid in self.images]
        )
        print(f"[COCO] {len(self.img_ids)} images with annotations")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            h, w = self.input_size
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img_h, img_w = h, w
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]

        h_new, w_new = self.input_size
        img = cv2.resize(img, (w_new, h_new))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Convert COCO bbox [x,y,w,h] to YOLO [cls, cx, cy, w, h] in pixels
        anns = self.img_anns.get(img_id, [])
        targets = []
        for ann in anns[: self.max_targets]:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img_w * w_new
            cy = (y + h / 2) / img_h * h_new
            bw = w / img_w * w_new
            bh = h / img_h * h_new
            # Use category_id modulo num_classes to fit our head
            cls_id = ann["category_id"] % 58
            targets.append([cls_id, cx, cy, bw, bh])

        if targets:
            targets_t = torch.tensor(targets, dtype=torch.float32)
        else:
            targets_t = torch.zeros((0, 5), dtype=torch.float32)

        return {"image": tensor, "targets": targets_t, "image_id": str(img_id)}


def collate_coco(batch):
    images = torch.stack([b["image"] for b in batch])
    max_t = max(len(b["targets"]) for b in batch)
    if max_t == 0:
        max_t = 1
    padded = torch.zeros(len(batch), max_t, 5)
    counts = []
    for i, b in enumerate(batch):
        n = len(b["targets"])
        counts.append(n)
        if n > 0:
            padded[i, :n] = b["targets"]
    return {
        "image": images,
        "targets": padded,
        "target_counts": torch.tensor(counts, dtype=torch.long),
    }


def pretrain(epochs: int = 20, batch_size: int = 64, lr: float = 1e-3) -> str:
    """Pretrain on COCO and save backbone weights."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco_img = "/mnt/train-data/datasets/coco/val2017"
    coco_ann = "/mnt/train-data/datasets/coco/annotations/instances_val2017.json"

    if not Path(coco_img).exists():
        print("[ERROR] COCO val2017 not found")
        return ""

    dataset = COCODetectionDataset(coco_img, coco_ann, (640, 640))
    # 80/20 split
    n_train = int(len(dataset) * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_coco, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_coco,
    )

    model = build_model("rgb", num_classes=58).to(device)
    criterion = DetectionLoss(num_classes=58).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, int(total_steps * 0.05), total_steps)

    print(f"[PRETRAIN] COCO val2017: {len(train_ds)} train, {len(val_ds)} val")
    print(f"[PRETRAIN] {epochs} epochs, batch={batch_size}, lr={lr}")

    best_val = float("inf")
    out_dir = Path("/mnt/artifacts-datai/checkpoints/project_mineinsight_coco_pretrain")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            tgts = batch["targets"].to(device)
            cnts = batch["target_counts"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(imgs)
                loss_dict = criterion(preds, tgts, cnts)
                loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                tgts = batch["targets"].to(device)
                cnts = batch["target_counts"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(imgs)
                    ld = criterion(preds, tgts, cnts)
                val_loss += ld["loss"].item()
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"[Epoch {epoch+1}/{epochs}] train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            save_path = out_dir / "coco_pretrained.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)

    save_path = out_dir / "coco_pretrained.pth"
    print(f"[PRETRAIN] Best val={best_val:.4f}, saved to {save_path}")
    return str(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    pretrain(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()

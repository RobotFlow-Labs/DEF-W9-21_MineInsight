"""MineInsight dataset loader for multi-modal landmine detection.

Supports RGB, VIS-SWIR, LWIR modalities with YOLO-format annotations.
Dataset structure (expected after download):
    mineinsight/
        rgb/
            track1_seq1/
                images/   *.jpg
                labels/   *.txt  (YOLO: class cx cy w h)
            track1_seq2/ ...
        lwir/
            track1_seq1/
                images/   *.jpg
                labels/   *.txt
        swir/
            track1_seq1/
                images/   *.jpg
                labels/   *.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------
# Class names: 15 landmines + 20 distractor objects = 35 total
# Indices 0-14 are landmines, 15-34 are distractors
# --------------------------------------------------------------------------

MINE_CLASSES = [
    "ap_mine_type1", "ap_mine_type2", "ap_mine_type3",
    "ap_mine_type4", "ap_mine_type5", "ap_mine_type6",
    "ap_mine_type7", "ap_mine_type8", "ap_mine_type9",
    "at_mine_type1", "at_mine_type2", "at_mine_type3",
    "at_mine_type4", "at_mine_type5", "at_mine_type6",
]

DISTRACTOR_CLASSES = [
    "bottle", "can", "rock", "branch", "bag",
    "tire", "shoe", "cloth", "metal_scrap", "plastic_container",
    "wire", "glass_shard", "rubber_piece", "wood_block", "brick",
    "pipe", "cardboard", "foam", "rope", "shell_casing",
]

ALL_CLASSES = MINE_CLASSES + DISTRACTOR_CLASSES
NUM_CLASSES = len(ALL_CLASSES)  # 35
MINE_CLASS_IDS = list(range(len(MINE_CLASSES)))  # 0..14


def _parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """Parse a YOLO-format label file.

    Returns tensor of shape (N, 5): [class_id, x_center, y_center, width, height]
    in pixel coordinates.
    """
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            labels.append([cls_id, cx, cy, w, h])

    if not labels:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.float32)


class MineInsightDataset(Dataset):
    """Multi-modal detection dataset for MineInsight.

    Args:
        root: Path to the mineinsight dataset root.
        sequences: List of sequence names to include (e.g. ["track1_seq1"]).
        modality: Which modality to load ("rgb", "lwir", "swir", "rgb+lwir", etc).
        input_size: (H, W) to resize images to.
        augment: Whether to apply data augmentation.
        flip_lr: Probability of horizontal flip.
        hsv_h: HSV hue jitter magnitude.
        hsv_s: HSV saturation jitter magnitude.
        hsv_v: HSV value jitter magnitude.
    """

    def __init__(
        self,
        root: str | Path,
        sequences: list[str],
        modality: str = "rgb",
        input_size: tuple[int, int] = (640, 640),
        augment: bool = False,
        flip_lr: float = 0.5,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
    ):
        self.root = Path(root)
        self.modality = modality
        self.modalities = self._parse_modalities(modality)
        self.input_size = input_size
        self.augment = augment
        self.flip_lr = flip_lr
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

        # Primary modality is the first one (used for labels)
        self.primary = self.modalities[0]

        # Build sample list: each sample is (sequence, image_stem)
        self.samples: list[tuple[str, str]] = []
        self._build_index(sequences)

    def _parse_modalities(self, modality: str) -> list[str]:
        """Parse modality string into list of modality names."""
        if "+" in modality:
            return modality.split("+")
        return [modality]

    def _build_index(self, sequences: list[str]) -> None:
        """Scan filesystem and build list of (sequence, image_stem) pairs."""
        for seq in sequences:
            img_dir = self.root / self.primary / seq / "images"
            if not img_dir.exists():
                continue
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                    self.samples.append((seq, img_file.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, modality: str, seq: str, stem: str) -> np.ndarray:
        """Load an image for a given modality, sequence, and file stem."""
        img_dir = self.root / modality / seq / "images"
        # Try common extensions
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
            path = img_dir / f"{stem}{ext}"
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Return blank image if not found (graceful fallback)
        h, w = self.input_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Resize image to input_size, return (resized, scale_x, scale_y)."""
        h_orig, w_orig = img.shape[:2]
        h_new, w_new = self.input_size
        resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        sx = w_new / max(w_orig, 1)
        sy = h_new / max(h_orig, 1)
        return resized, sx, sy

    def _augment_hsv(self, img: np.ndarray) -> np.ndarray:
        """Apply HSV jitter augmentation."""
        if self.hsv_h == 0 and self.hsv_s == 0 and self.hsv_v == 0:
            return img
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq, stem = self.samples[idx]

        # Load primary image for original size (needed for label parsing)
        primary_img = self._load_image(self.primary, seq, stem)
        h_orig, w_orig = primary_img.shape[:2]

        # Load labels (from primary modality)
        label_path = self.root / self.primary / seq / "labels" / f"{stem}.txt"
        targets = _parse_yolo_label(label_path, w_orig, h_orig)

        # Load and resize all modality images
        images: dict[str, torch.Tensor] = {}
        do_flip = self.augment and np.random.random() < self.flip_lr
        h_new, w_new = self.input_size

        for mod in self.modalities:
            img = self._load_image(mod, seq, stem)
            img, sx, sy = self._resize(img)

            if self.augment:
                img = self._augment_hsv(img)
                if do_flip:
                    img = np.ascontiguousarray(img[:, ::-1])

            # HWC -> CHW, normalize to [0, 1]
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images[mod] = tensor

        # Scale targets to resized coordinates
        if len(targets) > 0:
            sx = w_new / max(w_orig, 1)
            sy = h_new / max(h_orig, 1)
            targets[:, 1] *= sx  # cx
            targets[:, 2] *= sy  # cy
            targets[:, 3] *= sx  # w
            targets[:, 4] *= sy  # h

            if do_flip and len(targets) > 0:
                targets[:, 1] = w_new - targets[:, 1]

        result: dict[str, Any] = {
            "targets": targets,
            "image_id": f"{seq}/{stem}",
        }

        # If single modality, put tensor directly under "image"
        if len(self.modalities) == 1:
            result["image"] = images[self.modalities[0]]
        else:
            result["images"] = images
            # Also provide concatenated tensor for simple fusion
            result["image"] = torch.cat(list(images.values()), dim=0)

        return result


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate for variable-length targets."""
    images = torch.stack([b["image"] for b in batch])
    image_ids = [b["image_id"] for b in batch]

    # Pad targets to same length within batch
    max_targets = max(len(b["targets"]) for b in batch)
    if max_targets == 0:
        max_targets = 1  # at least 1 slot
    padded = torch.zeros(len(batch), max_targets, 5)
    target_counts = []
    for i, b in enumerate(batch):
        n = len(b["targets"])
        target_counts.append(n)
        if n > 0:
            padded[i, :n] = b["targets"]

    result: dict[str, Any] = {
        "image": images,
        "targets": padded,
        "target_counts": torch.tensor(target_counts, dtype=torch.long),
        "image_ids": image_ids,
    }

    # Multi-modal: also collate per-modality tensors
    if "images" in batch[0]:
        modalities = list(batch[0]["images"].keys())
        result["images"] = {
            mod: torch.stack([b["images"][mod] for b in batch])
            for mod in modalities
        }

    return result

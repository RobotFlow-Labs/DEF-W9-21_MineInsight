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
# --------------------------------------------------------------------------
# MineInsight class mapping (from targets_list.yaml)
# YOLO labels use per-target instance IDs (1-57), not sequential class IDs.
# 35 physical targets: 15 landmines + 20 distractors across 3 tracks.
# --------------------------------------------------------------------------

# Mine instance IDs (from targets_list.yaml)
MINE_IDS = {
    # PFM-1 (15 instances across tracks)
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    # PMN
    21, 43,
    # M6
    39, 40,
    # TC-3.6, TMA-2, MON-90, Type 72P, TM-46, TMM-1, MON-50
    41, 42, 46, 47, 45, 48, 44,
    # C-3, PROM-1, M-35, VS-50 (Track 3)
    55, 57, 52, 56, 53,
}

# ID → human-readable name
CLASS_NAMES: dict[int, str] = {
    1: "Coke Can", 2: "Chips Bag", 3: "Tuna Can", 4: "Glass Jar",
    5: "Chips Bag", 6: "Glass Jar", 7: "Pepper Dispenser", 8: "Corn Tin",
    9: "Beer Bottle", 10: "Plastic Cup", 11: "Shampoo Bottle",
    12: "Vinegar Bottle", 13: "Plastic Bottle", 14: "Plastic Bottle",
    15: "Plastic Bottle", 16: "Soda Can", 17: "Metal Pot", 18: "Sponge",
    19: "Paper Cup", 20: "Plastic Charger",
    21: "PMN", 22: "PFM-1", 23: "PFM-1", 24: "PFM-1", 25: "PFM-1",
    26: "PFM-1", 27: "PFM-1", 28: "PFM-1", 29: "PFM-1", 30: "PFM-1",
    31: "PFM-1", 32: "PFM-1", 33: "PFM-1", 34: "PFM-1", 35: "PFM-1",
    36: "PFM-1", 39: "M6", 40: "M6", 41: "TC-3.6", 42: "TMA-2",
    43: "PMN", 44: "MON-50", 45: "TM-46", 46: "MON-90", 47: "Type 72P",
    48: "TMM-1", 52: "PROM-1", 53: "VS-50", 55: "C-3", 56: "M-35", 57: "C-3",
}

# Max instance ID + 1 = number of output classes for the model
NUM_CLASSES = 58  # IDs 0-57 (some IDs unused, that's fine)
MINE_CLASS_IDS = sorted(MINE_IDS)
ALL_CLASSES = [CLASS_NAMES.get(i, f"unused_{i}") for i in range(NUM_CLASSES)]


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

        # For multi-modal: build nearest-timestamp index for secondary modalities
        self._cross_modal_map: dict[str, dict[str, str]] = {}
        if len(self.modalities) > 1:
            self._build_cross_modal_index(sequences)

    def _parse_modalities(self, modality: str) -> list[str]:
        """Parse modality string into list of modality names."""
        if "+" in modality:
            return modality.split("+")
        return [modality]

    def _find_img_dir(self, modality: str, seq: str) -> Path | None:
        """Find the image directory for a given modality and sequence.

        Supports two layouts:
          Flat:   root/track_1_s1_rgb_images/
          Nested: root/rgb/track1_seq1/images/
        """
        # Flat layout: root/{seq}_{modality}_images/
        flat = self.root / f"{seq}_{modality}_images"
        if flat.exists():
            return flat
        # Nested layout: root/{modality}/{seq}/images/
        nested = self.root / modality / seq / "images"
        if nested.exists():
            return nested
        return None

    def _find_label_dir(self, modality: str, seq: str) -> Path | None:
        """Find the label directory for a given modality and sequence.

        Supports:
          root/{seq}_{modality}_labels_reproj/   (preferred for LWIR)
          root/{seq}_{modality}_labels/
          root/{modality}/{seq}/labels/           (nested)
        """
        reproj = self.root / f"{seq}_{modality}_labels_reproj"
        if reproj.exists():
            return reproj
        flat = self.root / f"{seq}_{modality}_labels"
        if flat.exists():
            return flat
        nested = self.root / modality / seq / "labels"
        if nested.exists():
            return nested
        return None

    def _build_index(self, sequences: list[str]) -> None:
        """Scan filesystem and build list of (sequence, image_stem) pairs."""
        for seq in sequences:
            img_dir = self._find_img_dir(self.primary, seq)
            if img_dir is None:
                continue
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                    self.samples.append((seq, img_file.stem))

    @staticmethod
    def _extract_timestamp(stem: str) -> int:
        """Extract numeric timestamp from filename stem.

        Stems look like: track_1_s1_rgb_1730290670_166206123
        We combine the last two numeric parts as a single timestamp.
        """
        parts = stem.split("_")
        # Find the last two numeric parts
        nums = [p for p in parts if p.isdigit()]
        if len(nums) >= 2:
            return int(nums[-2]) * 1_000_000_000 + int(nums[-1])
        if len(nums) >= 1:
            return int(nums[-1])
        return 0

    def _build_cross_modal_index(self, sequences: list[str]) -> None:
        """Build nearest-timestamp mapping from primary to secondary modalities.

        RGB runs at ~10Hz, LWIR at ~30Hz — timestamps don't match exactly.
        For each primary frame, find the closest secondary frame by timestamp.
        """
        import bisect

        for mod in self.modalities:
            if mod == self.primary:
                continue
            self._cross_modal_map[mod] = {}
            for seq in sequences:
                img_dir = self._find_img_dir(mod, seq)
                if img_dir is None:
                    continue
                # Build sorted timestamp → stem index for this modality
                mod_stems = []
                for f in sorted(img_dir.iterdir()):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                        ts = self._extract_timestamp(f.stem)
                        mod_stems.append((ts, f.stem))
                if not mod_stems:
                    continue
                mod_timestamps = [t[0] for t in mod_stems]

                # For each primary sample in this sequence, find nearest
                for prim_seq, prim_stem in self.samples:
                    if prim_seq != seq:
                        continue
                    prim_ts = self._extract_timestamp(prim_stem)
                    idx = bisect.bisect_left(mod_timestamps, prim_ts)
                    # Check idx and idx-1 for closest
                    best_stem = None
                    best_dist = float("inf")
                    for ci in (idx - 1, idx):
                        if 0 <= ci < len(mod_timestamps):
                            dist = abs(mod_timestamps[ci] - prim_ts)
                            if dist < best_dist:
                                best_dist = dist
                                best_stem = mod_stems[ci][1]
                    if best_stem is not None:
                        key = f"{seq}/{prim_stem}"
                        self._cross_modal_map[mod][key] = best_stem

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, modality: str, seq: str, stem: str) -> np.ndarray:
        """Load an image for a given modality, sequence, and file stem."""
        img_dir = self._find_img_dir(modality, seq)
        if img_dir is not None:
            # For cross-modal: use nearest-timestamp matched stem
            cross_key = f"{seq}/{stem}"
            if modality in self._cross_modal_map and cross_key in self._cross_modal_map[modality]:
                matched_stem = self._cross_modal_map[modality][cross_key]
                candidates = [matched_stem]
            else:
                mod_stem = stem.replace(f"_{self.primary}_", f"_{modality}_")
                candidates = [stem, mod_stem]
            for candidate_stem in candidates:
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    path = img_dir / f"{candidate_stem}{ext}"
                    if path.exists():
                        # IMREAD_UNCHANGED preserves 16-bit TIFFs (FLIR Boson
                        # LWIR output). Plain cv2.imread silently returns None
                        # for 16-bit, which was feeding the model blank frames.
                        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        # uint16 → uint8 (normalize via shift)
                        if img.dtype == np.uint16:
                            img = (img >> 8).astype(np.uint8)
                        # Mono → 3-channel
                        if img.ndim == 2:
                            img = np.stack([img] * 3, axis=-1)
                        elif img.shape[-1] == 4:
                            img = img[..., :3]
                        elif img.shape[-1] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return img
        import warnings

        warnings.warn(
            f"Image not found: {modality}/{seq}/{stem}.*, using blank", stacklevel=2,
        )
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

        # Load labels (from primary modality) — search flat + nested layouts
        label_dir = self._find_label_dir(self.primary, seq)
        if label_dir is not None:
            label_path = label_dir / f"{stem}.txt"
        else:
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
                # Skip HSV jitter for LWIR and SWIR: hue-shifting thermal /
                # near-IR pseudocolor has no physical meaning and only adds
                # noise to the fusion branch.
                if mod == "rgb":
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

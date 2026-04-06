#!/usr/bin/env python3
"""Split MineInsight dataset into train/val/test within sequences.

Since we may only have 2 sequences, this script creates frame-level splits
within sequences to ensure proper train/val/test separation.

Usage:
    python scripts/split_data.py --root /mnt/train-data/datasets/mineinsight --modality rgb
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_sequence(
    root: Path,
    modality: str,
    sequence: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split frames within a sequence into train/val/test.

    Args:
        root: Dataset root path.
        modality: "rgb", "lwir", or "swir".
        sequence: Sequence name (e.g., "track_1_s1_rgb_images").
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with "train", "val", "test" lists of frame stems.
    """
    img_dir = root / modality / sequence / "images"
    if not img_dir.exists():
        # Try flat structure: root/sequence/images/
        img_dir = root / sequence / "images"
    if not img_dir.exists():
        # Try: root/sequence/*.jpg
        img_dir = root / sequence
    if not img_dir.exists():
        print(f"[WARN] No images found at {img_dir}")
        return {"train": [], "val": [], "test": []}

    # Collect all image stems
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    stems = sorted(
        f.stem for f in img_dir.iterdir()
        if f.suffix.lower() in exts
    )

    if not stems:
        print(f"[WARN] No images in {img_dir}")
        return {"train": [], "val": [], "test": []}

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": sorted(stems[:n_train]),
        "val": sorted(stems[n_train:n_train + n_val]),
        "test": sorted(stems[n_train + n_val:]),
    }

    print(f"[SPLIT] {sequence}: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MineInsight data")
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--modality", type=str, default="rgb")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}")
        return

    # Auto-discover sequences
    sequences = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            sequences.append(p.name)

    if not sequences:
        print(f"[ERROR] No sequences found in {root}")
        return

    print(f"[INFO] Found sequences: {sequences}")

    all_splits = {}
    for seq in sequences:
        splits = split_sequence(
            root, args.modality, seq,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        all_splits[seq] = splits

    # Save split file
    output = Path(args.output) if args.output else root / "data_split.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_splits, f, indent=2)
    print(f"[SAVED] Split indices to {output}")


if __name__ == "__main__":
    main()

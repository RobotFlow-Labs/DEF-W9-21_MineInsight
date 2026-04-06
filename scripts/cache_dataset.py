#!/usr/bin/env python3
"""Pre-process MineInsight images to tensor cache for fast training.

Reads raw JPEGs, resizes to 640x640, converts to float32 CHW tensors,
and saves as .pt files for instant loading during training.

Output: /mnt/forge-data/shared_infra/datasets/mineinsight_{modality}_cache/

Usage:
    python scripts/cache_dataset.py --modality lwir --sequences track_1_s1 track_2_s1
    python scripts/cache_dataset.py --modality rgb --sequences track_1_s1 track_2_s1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mineinsight.dataset import MineInsightDataset


def cache_dataset(
    root: str,
    modality: str,
    sequences: list[str],
    output_dir: str,
    input_size: tuple[int, int] = (640, 640),
) -> None:
    """Pre-process and cache dataset as tensors."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = MineInsightDataset(
        root=root,
        sequences=sequences,
        modality=modality,
        input_size=input_size,
        augment=False,
    )

    print(f"[CACHE] {len(ds)} samples, modality={modality}")
    print(f"[CACHE] Output: {out}")

    all_images = []
    all_targets = []
    all_target_counts = []
    all_ids = []

    for idx in tqdm(range(len(ds)), desc="Caching"):
        sample = ds[idx]
        all_images.append(sample["image"])  # (3, H, W) float32
        t = sample["targets"]
        all_targets.append(t)
        all_target_counts.append(len(t))
        all_ids.append(sample["image_id"])

    # Save as single files for mmap loading
    images_t = torch.stack(all_images)
    size_gb = images_t.element_size() * images_t.nelement() / 1e9
    print(f"[CACHE] Images tensor: {images_t.shape}, {size_gb:.1f}GB")

    torch.save(images_t, out / "images.pt")
    torch.save(all_targets, out / "targets.pt")
    torch.save(all_target_counts, out / "target_counts.pt")
    torch.save(all_ids, out / "image_ids.pt")

    # Save metadata
    meta = {
        "modality": modality,
        "sequences": sequences,
        "num_samples": len(ds),
        "input_size": list(input_size),
        "root": root,
    }
    torch.save(meta, out / "meta.pt")

    total_size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"[CACHE] Total cache size: {total_size:.1f}GB")
    print(f"[CACHE] Done! Use with: --cache-dir {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache MineInsight dataset")
    parser.add_argument("--root", default="/mnt/train-data/datasets/mineinsight")
    parser.add_argument("--modality", default="lwir")
    parser.add_argument("--sequences", nargs="+", default=["track_1_s1", "track_2_s1"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/mnt/forge-data/shared_infra/datasets/mineinsight_{args.modality}_cache"

    cache_dataset(args.root, args.modality, args.sequences, args.output_dir)


if __name__ == "__main__":
    main()

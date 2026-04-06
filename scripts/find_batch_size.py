#!/usr/bin/env python3
"""Auto-detect optimal batch size for MineInsight training on current GPU.

Targets 70-80% VRAM utilization. L4 = 23GB → target 16-18GB.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/find_batch_size.py --target 0.75
    CUDA_VISIBLE_DEVICES=2 python scripts/find_batch_size.py --config configs/paper.toml
"""

from __future__ import annotations

import argparse

import torch


def find_optimal_batch_size(
    input_size: tuple[int, int] = (640, 640),
    num_classes: int = 35,
    target_vram_ratio: float = 0.75,
    min_batch: int = 1,
    max_batch: int = 256,
) -> int:
    """Binary search for optimal batch size targeting VRAM ratio.

    Args:
        input_size: (H, W) input image size.
        num_classes: Number of classes.
        target_vram_ratio: Target fraction of total VRAM to use.
        min_batch: Minimum batch size to try.
        max_batch: Maximum batch size to try.

    Returns:
        Optimal batch size.
    """
    if not torch.cuda.is_available():
        print("[WARN] No CUDA device, defaulting to batch_size=4")
        return 4

    from mineinsight.model import build_model

    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_bytes = total_vram * target_vram_ratio
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[GPU] {gpu_name}")
    print(f"[VRAM] Total: {total_vram / 1e9:.1f}GB, Target: {target_bytes / 1e9:.1f}GB")

    model = build_model("rgb", num_classes=num_classes).to(device)
    model.train()

    best_batch = min_batch
    lo, hi = min_batch, max_batch

    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            dummy = torch.randn(mid, 3, *input_size, device=device)
            _ = model(dummy)

            # Check peak memory
            peak = torch.cuda.max_memory_allocated()
            ratio = peak / total_vram

            if ratio < target_vram_ratio:
                best_batch = mid
                lo = mid + 1
            else:
                hi = mid - 1

            del dummy
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
            torch.cuda.empty_cache()
            continue

    # Final verification
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dummy = torch.randn(best_batch, 3, *input_size, device=device)
    _ = model(dummy)
    final_peak = torch.cuda.max_memory_allocated()
    final_ratio = final_peak / total_vram

    print(f"[RESULT] Optimal batch_size={best_batch}")
    print(f"[RESULT] VRAM: {final_peak / 1e9:.1f}GB / {total_vram / 1e9:.1f}GB ({final_ratio:.0%})")

    del model, dummy
    torch.cuda.empty_cache()

    return best_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal batch size")
    parser.add_argument("--target", type=float, default=0.75, help="Target VRAM ratio")
    parser.add_argument("--config", type=str, default=None, help="Config file for input_size")
    args = parser.parse_args()

    input_size = (640, 640)
    num_classes = 35

    if args.config:
        from mineinsight.utils import load_config

        cfg = load_config(args.config)
        input_size = tuple(cfg.model.input_size)
        num_classes = cfg.model.num_classes

    batch = find_optimal_batch_size(
        input_size=input_size,
        num_classes=num_classes,
        target_vram_ratio=args.target,
    )
    print(f"\nRecommended: --batch-size {batch}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CUDA-accelerated training pipeline for MineInsight.

Uses shared ANIMA CUDA kernels + custom MineInsight kernels:
- detection_ops: fused_box_iou_2d, fused_focal_loss
- fused_image_preprocess: batch_normalize_hwc_to_chw
- vectorized_nms: nms_2d
- mineinsight_cuda_ops: fused_multimodal_preprocess, fused_ciou_loss, fused_detection_decode

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cu.py --config configs/paper.toml
"""

from mineinsight.train import main

if __name__ == "__main__":
    main()

# DEF-mineinsight -- Asset Manifest

## Paper
- Title: MineInsight: A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments
- ArXiv: 2506.04842
- Authors: Mario Morales et al.
- PDF: `papers/2506.04842.pdf`
- GitHub: https://github.com/mariomlz99/MineInsight

## Status: READY — Code complete, dataset downloading
- All code scaffolded and tested (38 tests passing)
- 4 custom CUDA kernels compiled (sm_89)
- 3 shared CUDA kernels integrated
- Dataset downloading to /mnt/train-data/datasets/mineinsight/

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---:|---|---|---|
| YOLO26n (latest, Jan 2026) | 5.3 MB | Ultralytics | `/mnt/forge-data/models/yolo26n.pt` | AVAILABLE |
| YOLO11n | ~6 MB | shared infra | `/mnt/forge-data/models/yolo11n.pt` | AVAILABLE |
| YOLOv12n | ~6 MB | shared infra | `/mnt/forge-data/models/yolov12n.pt` | AVAILABLE |
| YOLOv5l6 | ~150 MB | shared infra | `/mnt/forge-data/models/yolov5l6.pt` | AVAILABLE |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| MineInsight T1S1 RGB | ~3.8 GB | track1 seq1 | GitHub releases | `/mnt/train-data/datasets/mineinsight/` | DOWNLOADING |
| MineInsight T1S1 LWIR | ~465 MB | track1 seq1 | GitHub releases | `/mnt/train-data/datasets/mineinsight/` | DOWNLOADING |
| MineInsight T1S1 LiDAR | ~669 MB | track1 seq1 | GitHub releases | `/mnt/train-data/datasets/mineinsight/` | DOWNLOADING |
| MineInsight T2S1 RGB | ~2.8 GB | track2 seq1 | GitHub releases | `/mnt/train-data/datasets/mineinsight/` | DOWNLOADING |
| MineInsight Labels | ~2.4 MB | all tracks | GitHub releases | `/mnt/train-data/datasets/mineinsight/` | DOWNLOADING |

## CUDA Kernels
### Custom (built for this module)
| Kernel | Ops | Path | Speedup |
|---|---:|---|---|
| fused_multimodal_preprocess | 1 | csrc/mineinsight_cuda_ops.cu | 7x (eliminates 7 PyTorch ops) |
| fused_batch_multimodal_preprocess | 1 | csrc/mineinsight_cuda_ops.cu | batch version |
| fused_ciou_loss | 1 | csrc/mineinsight_cuda_ops.cu | ~3x (eliminates 15+ ops) |
| fused_detection_decode | 1 | csrc/mineinsight_cuda_ops.cu | ~2x (sigmoid+filter in 1 pass) |

### Shared (from /mnt/forge-data/shared_infra/cuda_extensions/)
| Kernel | Used For |
|---|---|
| detection_ops.fused_box_iou_2d | IoU computation in evaluation |
| detection_ops.fused_focal_loss | Classification loss acceleration |
| detection_ops.fused_score_filter | Pre-NMS score filtering |
| fused_image_preprocess.batch_normalize_hwc_to_chw | Image preprocessing |
| vectorized_nms.nms_2d | CUDA NMS in evaluation |

## Hyperparameters (baseline — not from paper, paper is dataset-only)
| Param | Value | Notes |
|---|---|---|
| Input resolution | 640 x 640 | YOLOv8 standard |
| Backbone | CSPDarknet-nano | Custom, 2.6M params |
| Num classes | 35 | 15 mines + 20 distractors |
| Optimizer | AdamW | lr=1e-3, wd=0.01 |
| Scheduler | cosine + 5% warmup | min_lr=1e-6 |
| Epochs | 100 | early stopping patience=20 |
| Augmentation | mosaic, mixup, hsv-jitter, h-flip | |
| Precision | bf16 | CUDA mixed precision |
| Batch size | auto | gpu-batch-finder on L4 |

## Expected Metrics (targets for our baseline)
| Benchmark | Metric | Target | Notes |
|---|---|---:|---|
| MineInsight RGB val | mAP@0.5 | >= 0.50 | Dataset paper showed YOLOv8 failed on transfer |
| MineInsight RGB val | mAP@0.5:0.95 | >= 0.25 | Small targets, challenging terrain |
| MineInsight LWIR val | mAP@0.5 | >= 0.40 | Thermal modality, lower resolution |
| MineInsight Fusion val | mAP@0.5 | >= 0.55 | Multi-modal should outperform single |
| MineInsight mines-only | mAP@0.5 | >= 0.45 | Critical: landmine-specific accuracy |

## Data Contracts
- YOLO-format labels: `class_id x_center y_center width height` (normalized)
- Images: JPEG/PNG, variable resolution (will be resized to 640x640)
- Class mapping: 35 classes (15 mines + 20 distractors)
- Split: frame-level 80/10/10 within sequences

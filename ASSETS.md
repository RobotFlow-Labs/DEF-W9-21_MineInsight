# DEF-mineinsight -- Asset Manifest

## Paper
- Title: MineInsight: A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments
- ArXiv: 2506.04842
- Authors: Mario Morales et al.
- PDF: `papers/2506.04842.pdf`
- GitHub: https://github.com/mariomlz99/MineInsight

## Status: BLOCKED -- Dataset not downloaded
- Paper parsed from arXiv HTML.
- No local dataset yet. MineInsight ROS2 bags and extracted images need to be downloaded.
- Implementation scaffold is complete and runnable once data is provisioned.

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---:|---|---|---|
| YOLOv8n (COCO pretrained) | ~6 MB | Ultralytics | `/mnt/forge-data/models/yolov8n.pt` | NEEDS CHECK |
| YOLO11n (fallback) | ~6 MB | shared infra | `/mnt/forge-data/models/yolo11n.pt` | AVAILABLE |
| YOLOv5l6 (fallback) | ~150 MB | shared infra | `/mnt/forge-data/models/yolov5l6.pt` | AVAILABLE |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| MineInsight RGB images | ~2 GB est. | 6 sequences | GitHub releases | `/mnt/forge-data/datasets/mineinsight/rgb/` | MISSING |
| MineInsight LWIR images | ~4 GB est. | 6 sequences | GitHub releases | `/mnt/forge-data/datasets/mineinsight/lwir/` | MISSING |
| MineInsight VIS-SWIR images | ~3 GB est. | 6 sequences | GitHub releases | `/mnt/forge-data/datasets/mineinsight/swir/` | MISSING |
| MineInsight YOLO labels | ~10 MB est. | per modality | GitHub releases | `/mnt/forge-data/datasets/mineinsight/labels/` | MISSING |
| MineInsight SAM2 masks | ~1 GB est. | auto-generated | GitHub releases | `/mnt/forge-data/datasets/mineinsight/masks/` | MISSING |
| MineInsight LiDAR bags | ~100 GB est. | 6 sequences | GitHub releases | `/mnt/forge-data/datasets/mineinsight/lidar/` | MISSING |
| MineInsight calibration | ~1 MB | all sensors | GitHub repo | `/mnt/forge-data/datasets/mineinsight/calibration/` | MISSING |

## Download Commands (when ready)
```bash
# Clone the dataset repository
cd /mnt/forge-data/datasets/
git clone https://github.com/mariomlz99/MineInsight mineinsight_repo

# Extract images and labels from the repo structure
# NOTE: Large files (ROS2 bags) are hosted on external links -- check repo README
# Full ROS2 bags are very large (19-77 GB per sequence). Start with extracted images only.

# Minimal start: download only RGB images + labels for initial training
# Exact download links TBD -- check GitHub releases page
```

## Hyperparameters (baseline -- not from paper, paper is dataset-only)
| Param | Value | Notes |
|---|---|---|
| Input resolution | 640 x 640 | YOLOv8 standard |
| Backbone | CSPDarknet-nano | YOLOv8n |
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
- Class mapping: defined in `targets_list.yaml` from dataset repo (35 classes)
- Split: 4 sequences train, 1 val, 1 test (track-based split to avoid data leakage)

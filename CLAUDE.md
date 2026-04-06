# 21_MineInsight

## Module Identity
- Module: `DEF-mineinsight`
- Scope: Multi-sensor landmine detection for humanitarian demining robotics.
- Primary source: `papers/2506.04842.pdf` (MineInsight).

## Mission
Build an ANIMA-ready multi-modal object detection pipeline that fuses RGB, VIS-SWIR,
LWIR (thermal), and LiDAR data for landmine and surface-object detection in off-road
environments. The paper is a **dataset paper** (not a model paper), so this module
implements a strong detection baseline on the MineInsight dataset using YOLOv8 and a
custom multi-modal fusion detector.

## Paper Summary
- **Title**: MineInsight: A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments
- **ArXiv**: 2506.04842
- **Venue**: Submitted to IEEE (cs.RO, cs.CV)
- **Key contribution**: First publicly available multi-sensor, multi-spectral dataset
  integrating dual-view scans (UGV body + robotic arm) for landmine detection.
- **Dataset**: 35 targets (15 inert landmines + 20 distractor objects) on 3 tracks,
  recorded in daylight and nighttime, ~1 hour total.

## Sensor Suite
| Sensor | Platform | Spectral Range | Frame Rate | Frames |
|--------|----------|---------------|------------|--------|
| Alvium 1800 U-240 (RGB) | Arm | 300-1100 nm | ~10 Hz | ~38,000 |
| Alvium 1800 U-130 (VIS-SWIR) | Arm | 400-1700 nm | ~15 Hz | ~53,000 |
| FLIR Boson 640 (LWIR) | Arm | 8-13.5 um | ~30 Hz | ~108,000 |
| Livox AVIA LiDAR | Arm | 905 nm | 10 Hz | continuous |
| Livox Mid-360 LiDAR | UGV body | 905 nm | 10 Hz | continuous |
| Sevensense Core (stereo) | UGV body | visible | ~20 Hz | continuous |

## Dataset Structure
- 3 tracks (~15 m x 2.5 m each)
- 6 sequences (2 per track: daylight + nighttime)
- Annotations: YOLO-format 2D bounding boxes (class cx cy w h)
- Additional: SAM2-generated segmentation masks (auto, not human-verified)
- Calibration: intrinsic + extrinsic YAML files for all sensors

## Architecture
Since this is a dataset paper (no novel model), we implement:
1. **Single-modal baselines**: YOLOv8 on each modality (RGB, SWIR, LWIR)
2. **Multi-modal fusion detector**: Early/late fusion of RGB + LWIR + optional SWIR
   using a shared backbone + modality-specific heads + fusion neck
3. **LiDAR-camera fusion**: Project LiDAR points onto image plane for depth-aware detection

### Fusion Detector Architecture
```
RGB image  --> ResNet/CSPDarknet backbone --> FPN neck --\
LWIR image --> ResNet/CSPDarknet backbone --> FPN neck --> Fusion module --> Detection head
SWIR image --> ResNet/CSPDarknet backbone --> FPN neck --/   (optional)
```

## Hyperparameters (baseline training)
| Param | Value | Notes |
|-------|-------|-------|
| Input resolution | 640 x 640 | Standard YOLO input |
| Backbone | YOLOv8n / CSPDarknet-S | Fits L4 23GB |
| Optimizer | AdamW | weight_decay=0.01 |
| Learning rate | 1e-3 | with cosine schedule |
| Warmup | 5% of total steps | linear warmup |
| Batch size | auto | gpu-batch-finder |
| Epochs | 100 | with early stopping patience=20 |
| Precision | bf16 | mixed precision on CUDA |
| Augmentation | mosaic, mixup, hsv-jitter, flip | standard YOLO augmentation |
| Num classes | 35 | 15 mines + 20 distractors |
| Grad clip | max_norm=1.0 | |

## Evaluation Metrics
| Metric | Description |
|--------|-------------|
| mAP@0.5 | Primary detection metric |
| mAP@0.5:0.95 | COCO-style mAP |
| Precision / Recall | Per-class and overall |
| F1 score | Harmonic mean |
| Mine-specific mAP | mAP on the 15 landmine classes only |
| FPS | Inference throughput |

## Datasets Required
| Dataset | Source | Path |
|---------|--------|------|
| MineInsight (RGB images + labels) | GitHub mariomlz99/MineInsight | /mnt/forge-data/datasets/mineinsight/ |
| MineInsight (LWIR images + labels) | GitHub mariomlz99/MineInsight | /mnt/forge-data/datasets/mineinsight/ |
| MineInsight (VIS-SWIR images + labels) | GitHub mariomlz99/MineInsight | /mnt/forge-data/datasets/mineinsight/ |
| YOLOv8n pretrained | Ultralytics | /mnt/forge-data/models/yolov8n.pt |

## Local Conventions
- Python package root: `src/mineinsight/`
- Configs: `configs/*.toml`
- CLIs: `scripts/train.py`, `scripts/evaluate.py`
- Tests: `tests/`

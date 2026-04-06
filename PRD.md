# DEF-mineinsight -- Build Plan

## Objective
Deliver a paper-faithful, multi-modal landmine detection pipeline on the MineInsight
dataset with PRD-driven implementation, then prepare training/export hooks for
CUDA-server hardening.

## PRD Execution Board
| PRD | Title | Priority | Status | Notes |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | COMPLETE | Core scaffolding, config, data contracts |
| PRD-02 | Core Model | P0 | COMPLETE | Single-modal + multi-modal fusion detector |
| PRD-03 | Loss Functions | P0 | COMPLETE | Detection losses (CIoU, focal, objectness) |
| PRD-04 | Training Pipeline | P0 | COMPLETE | Full training loop with checkpointing |
| PRD-05 | Evaluation | P1 | COMPLETE | mAP, per-class metrics, confusion matrix |
| PRD-06 | Export Pipeline | P1 | COMPLETE | ONNX + TRT export |
| PRD-07 | Integration | P1 | COMPLETE | Docker + API + ROS2 stubs |

## Constraints
- This is a dataset paper: no official model architecture to reproduce.
- MineInsight dataset must be downloaded before training can begin.
- LiDAR fusion is optional (requires ROS2 bag extraction pipeline).
- YOLOv8 baseline from paper "failed to produce reliable detections" on this data,
  so domain adaptation and fine-tuning are essential.

## Definition of Done (MVP)
- [x] Package installs and imports cleanly.
- [x] Model forward pass and one training step execute (with synthetic data).
- [x] Single-modal detection on RGB, LWIR, SWIR.
- [x] Multi-modal fusion detector with configurable modalities.
- [x] Evaluation CLI with mAP computation.
- [x] Export to ONNX.
- [x] Docker serving infrastructure.
- [ ] Training on real MineInsight data (blocked on dataset download).

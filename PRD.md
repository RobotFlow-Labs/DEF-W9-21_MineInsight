# DEF-mineinsight -- Build Plan

> **PIVOTED 2026-04-10** — the custom CSPDarknet + Hungarian matcher training
> pipeline (v3/v4/v5) plateaued at mAP@0.5=0.010 with 3/58 classes learning.
> Root cause: impossible 58-class schema (34 unique objects with instance-level
> duplication), 93% of data unused, hard train/val leak, Hungarian gradient
> starvation. Active strategy: stock Ultralytics YOLO26s-p2 with native
> 6-channel RGB+LWIR fusion, partial COCO pretrain, and proper 49K-frame
> dataset. **See `PIVOT_PLAN.md` for full rationale.** The legacy PRD-01..PRD-07
> scaffolding below is kept for reference and backward compatibility with old
> checkpoints, but is NOT the live training path.

## Objective
Deliver a paper-faithful, multi-modal landmine detection pipeline on the MineInsight
dataset with PRD-driven implementation, then prepare training/export hooks for
CUDA-server hardening.

## PRD Execution Board (legacy — superseded by PIVOT_PLAN.md for training)
| PRD | Title | Priority | Status | Notes |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | COMPLETE | Core scaffolding, config, data contracts |
| PRD-02 | Core Model | P0 | COMPLETE (legacy) | Custom fusion detector; **superseded by stock YOLO26s-p2** |
| PRD-03 | Loss Functions | P0 | COMPLETE (legacy) | Hungarian + focal + CIoU; **superseded by YOLO26 ProgLoss + TAL** |
| PRD-04 | Training Pipeline | P0 | COMPLETE (legacy) | Custom trainer; **new path: `scripts/train_yolo26_fusion.py`** |
| PRD-05 | Evaluation | P1 | COMPLETE (legacy) | Custom mAP; **new path: `scripts/eval_yolo26_fusion.py`** |
| PRD-06 | Export Pipeline | P1 | COMPLETE | ONNX + TRT export (still applies post-training) |
| PRD-07 | Integration | P1 | COMPLETE | Docker + API + ROS2 stubs |
| **PIVOT** | YOLO26s-p2 + 6ch fusion | P0 | **IN PROGRESS** | PIVOT_PLAN.md — Phase 1 building dataset, Phase 2 ready |

## Constraints
- This is a dataset paper: no official model architecture to reproduce.
- MineInsight dataset now fully extracted: 6 sequences × up to 3 modalities = 181K+ frames on disk at `/mnt/train-data/datasets/mineinsight/`.
- LiDAR fusion is optional (requires ROS2 bag extraction pipeline).
- **True class count is 34** (unique object types), not 58 — see `src/mineinsight/label_remap.py`. The 51 raw IDs include 15+ PFM-1 instance-level duplicates that are physically identical and unlearnable as separate classes.
- Legacy custom model reached mAP=0.010 and plateaued; pivot to stock Ultralytics is the active strategy.

## Definition of Done (MVP)
- [x] Package installs and imports cleanly.
- [x] Model forward pass and one training step execute (with synthetic data).
- [x] Single-modal detection on RGB, LWIR, SWIR.
- [x] Multi-modal fusion detector with configurable modalities.
- [x] Evaluation CLI with mAP computation.
- [x] Export to ONNX.
- [x] Docker serving infrastructure.
- [x] MineInsight dataset downloaded + fully extracted (181K frames).
- [x] PIVOT_PLAN.md written + reviewed + adversarially validated.
- [x] Phase 0 preflight gates (7/7 passed) — YOLO26 + ch=6 + nc=34 verified on GPU 1.
- [x] Pivot code shipped: `label_remap.py`, `build_fusion_dataset.py`, `train_yolo26_fusion.py`, `eval_yolo26_fusion.py`, `test_label_remap.py` (8/8 passing).
- [ ] Phase 1b — fusion dataset build v2 completes (49,156 frames target, currently in progress).
- [ ] Phase 2 — YOLO26s-p2 150-epoch training on `data_mixed.yaml` (blocked on dataset build).
- [ ] Phase 3 — eval gates: mAP@0.5 > 0.50 headline, mine_mAP > 0.30, ≥10 of 14 mine classes learned.
- [ ] Cross-track generalization run on `data_crosstrack.yaml` (train=track_1_*, val=track_2_s1).
- [ ] Export new YOLO26 checkpoint to safetensors / ONNX / TRT FP16 / TRT FP32.
- [ ] Push to HuggingFace: `ilessio-aiflowlab/project_mineinsight`.

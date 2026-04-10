# NEXT_STEPS.md
> Last updated: 2026-04-10
> MVP Readiness: 72%
> Status: **PIVOTED** — custom model failed at mAP=0.010 after 3 iterations. Moving to stock Ultralytics YOLO26s with 6-channel multi-page TIFF fusion. See `PIVOT_PLAN.md` for the full plan and rationale.

## Current phase
**Phase 1 ready to run** — all pivot code written, reviewed (incl. cupy-vs-scipy concern addressed: YOLO26's TAL assigner is GPU-native), and GPU-smoke-tested. Blocked only on you saying "go" for the dataset build.

## Done

### Original scaffolding (preserved, legacy)
- [x] Paper analyzed (arXiv 2506.04842 — dataset paper for humanitarian demining)
- [x] Full scaffolding created (CLAUDE.md, ASSETS.md, PRD.md, prds/, configs/)
- [x] Python package (`src/mineinsight/`) + TOML config system
- [x] Dataset loader (YOLO format, multi-modal RGB/LWIR/SWIR)
- [x] Custom model architecture (YOLOv8-style CSPDarknet + multi-modal fusion)
- [x] Loss functions (CIoU, focal, DETR-style with Hungarian matcher)
- [x] Training pipeline with checkpointing, early stopping, LR scheduling
- [x] Evaluation pipeline (mAP, per-class metrics, CUDA NMS)
- [x] Export pipeline (pth, safetensors, ONNX)
- [x] Docker serving files (CUDA/MLX dual backend)
- [x] 4 custom CUDA kernels compiled (sm_89)
- [x] Ruff lint clean, initial unit tests passing

### v3/v4/v5 failed training runs (archived)
- [x] Trained custom model through v3 (broken loss), v4 (Hungarian fix), v5 (hard-neg mining)
- [x] Identified 5 root causes: impossible class schema (58 nominal IDs / 34 real classes), data leak, wrong assigner, no COCO pretrain, 93% of data unused
- [x] Best checkpoint preserved: `/mnt/artifacts-datai/models/project_mineinsight_CANONICAL/v5_fusion_rgb_lwir_attn_hires_ep54_val1.5422.pth` (mAP@0.5=0.010, val_loss=1.5422)
- [x] Legacy experimental configs (v3_*, v4_*, v5_1..5_4_*, paper_v2, lwir_v2) moved to `configs/archive/`

### Pivot to stock Ultralytics YOLO26s (2026-04-10)
- [x] PIVOT_PLAN.md (570 lines, adversarially reviewed)
- [x] Phase 0 preflight: 7/7 gates passed including 1-epoch GPU smoke train on multi-page TIFFs
- [x] `src/mineinsight/label_remap.py` — 51 raw IDs → 34 real classes (PFM-1 instances collapsed)
- [x] `tests/test_label_remap.py` — **8/8 passing**
- [x] `scripts/build_fusion_dataset.py` — offline 6-ch TIFF builder with LWIR calibration + BUILD_MANIFEST
- [x] `scripts/train_yolo26_fusion.py` — Ultralytics wrapper with verified partial COCO pretrain loader
- [x] `scripts/eval_yolo26_fusion.py` — thin `model.val()` wrapper, per-class AP by name
- [x] GPU 1 smoke test: yolo26s-p2 ch=6 nc=34 = 9.68M params, 172 img/s bf16, 1-epoch train passed
- [x] Two rounds of /code-review: 3 critical bugs found + fixed on pivot code, 18 legacy bugs found + fixed on existing code

### Hygiene + fixes (2026-04-10)
- [x] Deleted 12.2 GB of stray dataset zips + corrupted track_2_s2_rgb.zip from project root
- [x] Updated `.gitignore` with *.zip, *.pt, *.tiff, track_*/, runs/ patterns
- [x] Added `ultralytics>=8.4` and `tifffile>=2024.2` to `pyproject.toml`
- [x] Fixed `src/mineinsight/dataset.py` 16-bit TIFF load (IMREAD_UNCHANGED + uint16→uint8)
- [x] Skipped HSV jitter on LWIR (physically meaningless on thermal pseudocolor)
- [x] Cached `is_multimodal` in train.py instead of calling `dataset[0]` per epoch
- [x] Fixed `evaluate.py` hardcoded `batch_size=8` to use config
- [x] Fixed `serve.py` `/info` endpoint hardcoded `num_classes=35` to read from model
- [x] Renamed misleading `YOLO26Wrapper` to `CSPDarknetWideWrapper` (it never loaded yolo26*.pt)
- [x] Fixed `tests/test_cuda_ops.py` relative sys.path + unconditional import → `pytest.importorskip`
- [x] Deleted duplicate `scripts/train_cu.py` (identical passthrough to train.py)
- [x] Fixed v5_fusion_hires.toml train/val leak (track_2_s1 was in both)

## In Progress
- [ ] Run unit tests + lint to confirm no regressions from the batch fixes

## TODO (pivot pipeline, sequential)
- [ ] Phase 1b: run `scripts/build_fusion_dataset.py` to generate `/mnt/forge-data/shared_infra/datasets/mineinsight_fusion/` (CPU-bound, ~1-2h, ~8 GB output)
- [ ] Phase 2: launch `scripts/train_yolo26_fusion.py --data data_mixed.yaml` on GPU 1 (150 epochs, ~6h)
- [ ] Phase 3: `scripts/eval_yolo26_fusion.py` + verify Phase 3 gates (mAP>0.50 headline, mine_mAP>0.30)
- [ ] Phase 2b: launch cross-track training `--data data_crosstrack.yaml` for generalization stress test
- [ ] Export best model to safetensors, ONNX, TensorRT FP16, TensorRT FP32
- [ ] Push to HuggingFace: `ilessio-aiflowlab/project_mineinsight`
- [ ] Update ROS2 node + Docker serving to use YOLO26 checkpoint
- [ ] Push commits to GitHub develop branch

## Blocking
- Waiting on user go-ahead to run the dataset builder. GPU 1 is free (0 MiB, 0% util).

## Key Info
- **Dataset root**: `/mnt/train-data/datasets/mineinsight/`
  - track_1_s1: 3,781 RGB / 7,635 LWIR ✓
  - track_1_s2: **17,960 RGB** / 36,282 LWIR (previously unused, biggest data find)
  - track_2_s1: 3,331 RGB / 6,732 LWIR
  - track_2_s2: **RGB missing** (labels only); LWIR 26,838 — unusable for fusion without RGB
- **Target class count: 34** (not 58) — verified: `targets_list.yaml` has 51 IDs for 34 unique object names (all PFM-1 placements collapse to one class)
- **Target model**: YOLO26s-p2 (9.68M params, stride-4 P2 head for small mines) with `ch=6` (RGB+LWIR early fusion)
- **COCO pretrain source**: `/mnt/train-data/models/yolo26/yolo26s.pt` (partial load, skip stem)
- **GPU**: 8× L4, GPU 1 allocated to MineInsight
- **Canonical legacy baseline**: v5 checkpoint mAP@0.5=0.010 — the bar to beat

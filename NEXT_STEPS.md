# NEXT_STEPS.md
> Last updated: 2026-04-06
> MVP Readiness: 55%

## Done
- [x] Paper analyzed (arXiv 2506.04842 -- dataset paper for humanitarian demining)
- [x] Full scaffolding created (CLAUDE.md, ASSETS.md, PRD.md, prds/, configs/)
- [x] Python package structure (src/mineinsight/ with all core modules)
- [x] Config system (TOML-based, dataclass)
- [x] Dataset loader (YOLO-format, multi-modal: RGB/LWIR/SWIR)
- [x] Model architecture (single-modal YOLOv8-style + multi-modal fusion detector)
- [x] YOLO26n wrapper for baseline comparison (latest Ultralytics, Jan 2026)
- [x] Loss functions (CIoU, focal, objectness, combined detection loss)
- [x] Training pipeline with checkpointing, early stopping, LR scheduling
- [x] Multi-modal training support (dict-based modality input)
- [x] Evaluation pipeline (mAP, per-class metrics, CUDA NMS)
- [x] Export pipeline (pth, safetensors, ONNX, TRT FP16, TRT FP32)
- [x] Docker serving files + CUDA/MLX containers
- [x] 38 tests passing (model, dataset, CUDA ops)
- [x] Ruff lint clean (0 errors)
- [x] venv setup (torch cu128, all deps installed)
- [x] 4 custom CUDA kernels compiled (sm_89)
- [x] 3 shared CUDA kernels integrated
- [x] Code review: all CRITICAL + HIGH issues fixed
- [x] GPU batch size finder
- [x] Data split utility
- [x] Visualization utility
- [x] Download scripts (minimal 44GB, full 56GB)
- [x] PIPELINE_MAP.md

## In Progress
- [ ] Dataset download (user managing, waiting for confirmation)

## TODO (when data arrives)
- [ ] Extract archives and verify directory structure
- [ ] Adapt dataset loader paths to match actual data layout
- [ ] Download targets_list.yaml for class mapping
- [ ] Run data split (80/10/10)
- [ ] Smoke test training (2 epochs, debug config)
- [ ] GPU batch size finder on GPU 2
- [ ] Full RGB training (100 epochs, GPU 2, nohup+disown)
- [ ] Full LWIR training
- [ ] RGB+LWIR fusion training
- [ ] YOLO26n baseline comparison
- [ ] Evaluation on test split
- [ ] Export all 5 formats
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_mineinsight
- [ ] Copy custom CUDA kernel to shared_infra

## Blocking
- Dataset download not complete yet
- User confirmed: GPU 2 is free for training

## Key Info
- Dataset: ~44GB (RGB+LWIR), download to /mnt/train-data/datasets/mineinsight/
- Download script: bash download_mineinsight_minimal.sh
- Track 3 has NO RGB (camera failure) — LWIR only
- 35 classes: 15 mines + 20 distractors
- YOLO26n weights at /mnt/forge-data/models/yolo26n.pt (2.6M params)

# NEXT_STEPS.md
> Last updated: 2026-04-06
> MVP Readiness: 50%

## Done
- [x] Paper analyzed (arXiv 2506.04842 -- dataset paper for humanitarian demining)
- [x] Full scaffolding created (CLAUDE.md, ASSETS.md, PRD.md, prds/, configs/)
- [x] Python package structure (src/mineinsight/ with all core modules)
- [x] Config system (TOML-based, Pydantic-compatible)
- [x] Dataset loader (YOLO-format, multi-modal: RGB/LWIR/SWIR)
- [x] Model architecture (single-modal YOLOv8-style + multi-modal fusion detector)
- [x] Loss functions (CIoU, focal, objectness, combined detection loss)
- [x] Training pipeline with checkpointing, early stopping, LR scheduling
- [x] Evaluation pipeline (mAP, per-class metrics, CUDA NMS)
- [x] Export utilities (ONNX, safetensors)
- [x] Docker serving files (Dockerfile.serve, docker-compose.serve.yml)
- [x] Docker CUDA/MLX build files (docker/Dockerfile.cuda, docker/Dockerfile.mlx)
- [x] Unit tests (38 tests passing: model, dataset, CUDA ops)
- [x] PRD-01 through PRD-07 all scaffolded
- [x] Ruff lint clean (0 errors)
- [x] venv setup (torch cu128, all deps installed)
- [x] CUDA kernels compiled (4 custom kernels for sm_89):
  - fused_multimodal_preprocess (RGB+LWIR → 6ch CHW in 1 pass)
  - fused_batch_multimodal_preprocess (batch version)
  - fused_ciou_loss (CIoU loss on GPU, eliminates 15+ ops)
  - fused_detection_decode (sigmoid+score+filter in 1 pass)
- [x] Shared CUDA kernels integrated (3 from shared_infra):
  - detection_ops: fused_box_iou_2d, fused_focal_loss, fused_score_filter
  - fused_image_preprocess: batch_normalize_hwc_to_chw
  - vectorized_nms: nms_2d (CUDA bitmask NMS)
- [x] Evaluation uses CUDA NMS (vectorized_nms.nms_2d)

## In Progress
- [ ] Dataset downloading (Track 1 Seq 1: RGB 3.8GB + Thermal 465MB + LiDAR 669MB + Labels 1.2MB; Track 2 Seq 1: RGB 2.8GB)

## TODO
- [ ] Verify dataset loader with real data when download completes
- [ ] Run smoke test training (2 epochs, debug config)
- [ ] Full training on RGB modality (ask for GPU)
- [ ] Full training on LWIR modality
- [ ] Multi-modal fusion training (RGB + LWIR)
- [ ] Evaluation on test split
- [ ] ONNX + TRT FP16 + TRT FP32 export of best model
- [ ] Push checkpoint to HuggingFace
- [ ] Copy custom CUDA kernel to shared_infra

## Blocking
- MineInsight dataset downloading to /mnt/train-data/datasets/mineinsight/
- Will need to inspect actual directory structure when download completes to ensure dataset loader paths match

## Downloads In Progress
- Track 1 Seq 1: RGB (3.8GB) + Thermal (465MB) + LiDAR images (669MB) + Labels (1.2MB)
- Track 2 Seq 1: RGB (2.8GB) + Labels (1.2MB)
- Target: /mnt/train-data/datasets/mineinsight/

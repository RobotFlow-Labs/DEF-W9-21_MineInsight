# PIPELINE_MAP.md — MineInsight Execution Pipeline

## Paper: MineInsight (arXiv 2506.04842)
## Type: Dataset paper — no novel model, we implement detection baselines

---

## Step 1: Data Preparation
- [x] DONE — Download MineInsight dataset (RGB + LWIR + labels)
- [x] DONE — Dataset loader (multi-modal YOLO format)
- [ ] TODO — Extract and verify downloaded archives
- [ ] TODO — Split frames into train/val/test (80/10/10)
- [ ] TODO — Verify label format matches YOLO spec
> Paper ref: Section 3 (Dataset Description)

## Step 2: Single-Modal RGB Baseline
- [x] DONE — CSPDarknet-nano backbone + FPN + detection head
- [x] DONE — CIoU + focal + objectness loss functions
- [x] DONE — Training pipeline with AMP, checkpointing, early stopping
- [ ] TODO — Train on RGB data (100 epochs, early stop)
- [ ] TODO — Evaluate mAP@0.5 on test split
> Paper ref: Section 4.1 (YOLOv8 baseline failed — we train from scratch)

## Step 3: LWIR (Thermal) Baseline
- [ ] TODO — Train on LWIR data with same architecture
- [ ] TODO — Compare day vs night performance
> Paper ref: Section 3.2 (FLIR Boson 640, 8-13.5 um)

## Step 4: Multi-Modal Fusion (RGB + LWIR)
- [x] DONE — Attention fusion module
- [x] DONE — Multi-modal detector architecture
- [ ] TODO — Train RGB+LWIR fusion model
- [ ] TODO — Compare against single-modal baselines
> Paper ref: Section 3.4 (multi-sensor integration)

## Step 5: YOLO26 Comparison
- [x] DONE — YOLO26n wrapper integrated
- [ ] TODO — Fine-tune YOLO26n on RGB data
- [ ] TODO — Compare custom CSPDarknet vs YOLO26n
> Bonus: not in paper, modern baseline comparison

## Step 6: Export + Deploy
- [x] DONE — Export pipeline (pth, safetensors, ONNX, TRT)
- [ ] TODO — Export best model all 5 formats
- [ ] TODO — Push to HuggingFace
- [ ] TODO — Docker container test
> Required by ANIMA infrastructure

## Step 7: Final Evaluation
- [ ] TODO — Generate TRAINING_REPORT.md
- [ ] TODO — Mine-specific mAP analysis
- [ ] TODO — FPS benchmarking
- [ ] TODO — Copy custom CUDA kernel to shared_infra

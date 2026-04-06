# Task Index -- DEF-mineinsight

## PRD-01: Foundation & Config
- [x] T01.1: Create pyproject.toml with hatchling backend
- [x] T01.2: Create src/mineinsight/__init__.py package
- [x] T01.3: Implement config loading from TOML (utils.py)
- [x] T01.4: Create paper.toml, debug.toml, fusion.toml configs
- [x] T01.5: Implement YOLO-format dataset loader (dataset.py)
- [x] T01.6: Add multi-modal dataset support (RGB + LWIR + SWIR)
- [x] T01.7: Write dataset unit tests
- [x] T01.8: Create anima_module.yaml manifest

## PRD-02: Core Model
- [x] T02.1: Implement CSPDarknet backbone (nano scale)
- [x] T02.2: Implement FPN neck with lateral connections
- [x] T02.3: Implement detection head (box + obj + cls)
- [x] T02.4: Build SingleModalDetector wrapper
- [x] T02.5: Implement attention-gated fusion module
- [x] T02.6: Build MultiModalDetector wrapper
- [x] T02.7: Write model forward-pass shape tests

## PRD-03: Loss Functions
- [x] T03.1: Implement CIoU loss
- [x] T03.2: Implement focal loss with configurable alpha/gamma
- [x] T03.3: Implement objectness BCE loss
- [x] T03.4: Build combined DetectionLoss with configurable weights

## PRD-04: Training Pipeline
- [x] T04.1: Implement training loop with epoch/step tracking
- [x] T04.2: Add AdamW optimizer with cosine + warmup schedule
- [x] T04.3: Add bf16 mixed precision support
- [x] T04.4: Implement checkpoint manager (save top-K, auto-delete)
- [x] T04.5: Add early stopping
- [x] T04.6: Add NaN detection
- [x] T04.7: Add TensorBoard logging
- [x] T04.8: Add --resume from checkpoint
- [x] T04.9: Create scripts/train.py CLI entry point

## PRD-05: Evaluation
- [x] T05.1: Implement IoU computation
- [x] T05.2: Implement per-class AP calculation
- [x] T05.3: Compute mAP@0.5 and mAP@0.5:0.95
- [x] T05.4: Add mine-specific metric subset
- [x] T05.5: Save evaluation report as JSON
- [x] T05.6: Create scripts/evaluate.py CLI entry point

## PRD-06: Export Pipeline
- [x] T06.1: ONNX export with dynamic batch
- [x] T06.2: Safetensors weight export
- [x] T06.3: Document TRT conversion path

## PRD-07: Integration
- [x] T07.1: Create Dockerfile.serve
- [x] T07.2: Create docker-compose.serve.yml
- [x] T07.3: Define FastAPI endpoints in serve module
- [x] T07.4: Configure ROS2 topics in anima_module.yaml

## Post-Build Tasks (blocked on dataset)
- [ ] T08.1: Download MineInsight dataset
- [ ] T08.2: Verify dataset loader with real images
- [ ] T08.3: Run smoke test training (debug config, 2 epochs)
- [ ] T08.4: Full RGB training (paper config, 100 epochs)
- [ ] T08.5: Full LWIR training
- [ ] T08.6: Multi-modal fusion training (RGB + LWIR)
- [ ] T08.7: Evaluate on test split
- [ ] T08.8: Export best model to ONNX + TRT
- [ ] T08.9: Push checkpoint to HuggingFace

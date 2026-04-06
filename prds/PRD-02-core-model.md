# PRD-02: Core Model

> Module: DEF-mineinsight | Priority: P0
> Depends on: PRD-01
> Status: COMPLETE

## Objective
Implement single-modal and multi-modal detection models for landmine detection on the
MineInsight dataset.

## Context
The paper provides no novel architecture -- it tests YOLOv8 (trained on SULAND) which
fails due to domain gap. We implement:
1. A YOLOv8-nano-style single-modal detector (CSPDarknet backbone + FPN + detection head)
2. A multi-modal fusion module that merges features from RGB + LWIR (+ optional SWIR)

## Architecture
### Single-Modal Detector
- CSPDarknet-Nano backbone (3 scale outputs: P3, P4, P5)
- FPN neck with upsampling and lateral connections
- Detection head: per-scale conv layers predicting (x, y, w, h, obj, cls*35)

### Multi-Modal Fusion Detector
- Separate backbones per modality (shared architecture, independent weights)
- Fusion module options: concatenation, element-wise addition, attention-gated fusion
- Shared FPN neck after fusion
- Same detection head as single-modal

## Acceptance Criteria
- [x] Single-modal forward pass: input (B, 3, 640, 640) -> list of (B, N, 40) predictions.
- [x] Multi-modal forward pass: dict of images -> list of predictions.
- [x] Configurable modality selection via TOML.
- [x] Attention fusion module implemented.
- [x] Parameter count matches YOLOv8-nano scale (~3M params single, ~6M fusion).

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `src/mineinsight/model.py` | CSPDarknet + FPN + DetectionHead + FusionDetector | ~450 |
| `tests/test_model.py` | Forward pass shape tests | ~100 |

## Test Plan
```bash
uv run pytest tests/test_model.py -v
```

## References
- YOLOv8 architecture (Ultralytics)
- Paper Section IV (baseline evaluation)
- Feeds into: PRD-03

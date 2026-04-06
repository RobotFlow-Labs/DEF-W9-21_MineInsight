# PRD-03: Loss Functions

> Module: DEF-mineinsight | Priority: P0
> Depends on: PRD-02
> Status: COMPLETE

## Objective
Implement detection loss functions for training the MineInsight detector.

## Loss Components
1. **Box regression loss (CIoU)**: Complete Intersection over Union for bounding box
   regression. Accounts for overlap, center distance, and aspect ratio.
2. **Classification loss (Focal)**: Focal loss to handle class imbalance between
   15 mine classes and 20 distractor classes. Alpha=0.25, gamma=2.0.
3. **Objectness loss (BCE)**: Binary cross-entropy for object confidence scoring.
4. **Combined detection loss**: Weighted sum of box + cls + obj losses.

## Acceptance Criteria
- [x] CIoU loss computes correctly for predicted vs target boxes.
- [x] Focal loss handles class imbalance with configurable alpha/gamma.
- [x] BCE objectness loss implemented.
- [x] Combined loss with configurable weights.
- [x] All losses are differentiable and support bf16.

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `src/mineinsight/losses.py` | CIoU, FocalLoss, DetectionLoss | ~200 |

## Test Plan
```bash
uv run pytest tests/test_model.py -v -k loss
```

## References
- CIoU: Zheng et al., "Distance-IoU Loss", AAAI 2020
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Feeds into: PRD-04

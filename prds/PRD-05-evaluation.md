# PRD-05: Evaluation

> Module: DEF-mineinsight | Priority: P1
> Depends on: PRD-02, PRD-04
> Status: COMPLETE

## Objective
Implement evaluation pipeline computing detection metrics on the MineInsight dataset.

## Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: COCO-style mAP across multiple IoU thresholds
- **Per-class AP**: AP for each of the 35 target classes
- **Mine-specific mAP**: mAP computed only on the 15 landmine classes
- **Precision / Recall**: At optimal confidence threshold
- **F1 score**: Harmonic mean of precision and recall
- **FPS**: Inference throughput on L4 GPU

## Acceptance Criteria
- [x] mAP computation with IoU matching.
- [x] Per-class AP breakdown.
- [x] Mine-specific vs distractor-specific metrics.
- [x] Results saved to JSON report.
- [x] CLI entry point.

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `src/mineinsight/evaluate.py` | Evaluation logic + CLI | ~300 |
| `scripts/evaluate.py` | CLI entry point | ~30 |

## Test Plan
```bash
uv run python scripts/evaluate.py --config configs/debug.toml --checkpoint best.pth
```

## References
- COCO evaluation protocol
- Feeds into: PRD-06

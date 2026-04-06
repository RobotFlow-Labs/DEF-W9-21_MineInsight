# PRD-04: Training Pipeline

> Module: DEF-mineinsight | Priority: P0
> Depends on: PRD-01, PRD-02, PRD-03
> Status: COMPLETE

## Objective
Implement the full training loop with config-driven hyperparameters, checkpointing,
early stopping, LR scheduling, and crash protection.

## Features
- TOML config loading (no hardcoded hyperparameters)
- AdamW optimizer with cosine annealing + linear warmup
- bf16 mixed precision on CUDA
- Checkpoint manager: save top-K by val metric, auto-delete old
- Early stopping with configurable patience
- NaN detection and graceful stop
- Resume from checkpoint
- TensorBoard logging
- GPU memory check at startup
- nohup/disown compatible

## Acceptance Criteria
- [x] Training loop runs with synthetic data (no real dataset needed).
- [x] Config drives all hyperparameters.
- [x] Checkpoint save/load cycle works.
- [x] Early stopping triggers correctly.
- [x] LR schedule: warmup then cosine decay.
- [x] Logging to TensorBoard and console.
- [x] `--resume` flag works.

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `src/mineinsight/train.py` | Full training loop | ~350 |
| `scripts/train.py` | CLI entry point | ~30 |

## Test Plan
```bash
# Smoke test with synthetic data
uv run python scripts/train.py --config configs/debug.toml --max-steps 5
```

## References
- ANIMA Training Standards (mandatory rules)
- Feeds into: PRD-05

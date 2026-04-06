# PRD-01: Foundation & Config

> Module: DEF-mineinsight | Priority: P0
> Depends on: None
> Status: COMPLETE

## Objective
Create a reproducible foundation (package, config, dataset I/O, sanity tests) for
MineInsight multi-modal landmine detection.

## Context (from paper)
MineInsight is a dataset paper providing YOLO-format annotations across three spectral
modalities (RGB, VIS-SWIR, LWIR). The dataset uses 35 object classes (15 landmines +
20 distractors) across 3 tracks with daylight and nighttime sequences.

## Acceptance Criteria
- [x] Package installs with `uv sync`.
- [x] Configs load (`paper`, `debug`, `fusion`).
- [x] YOLO-format dataset loader returns image + targets tensors.
- [x] Multi-modal dataset loader handles RGB + LWIR + SWIR.
- [x] Unit tests pass for config and dataset parsing.

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| `pyproject.toml` | Build and dependencies | ~60 |
| `src/mineinsight/__init__.py` | Package init | ~15 |
| `src/mineinsight/utils.py` | Config loading, seed, device utils | ~150 |
| `src/mineinsight/dataset.py` | YOLO detection dataset (multi-modal) | ~300 |
| `configs/paper.toml` | Baseline training config | ~80 |
| `configs/debug.toml` | Quick smoke test config | ~60 |
| `configs/fusion.toml` | Multi-modal fusion config | ~80 |
| `tests/test_dataset.py` | Dataset loader tests | ~100 |

## Test Plan
```bash
uv run pytest tests/test_dataset.py -v
```

## References
- Paper: arXiv 2506.04842 (dataset description, sensor specs, annotation format)
- Feeds into: PRD-02

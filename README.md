# DEF-mineinsight

Multi-sensor landmine detection for humanitarian demining robotics.

Based on: [MineInsight](https://arxiv.org/abs/2506.04842) — a multi-sensor dataset integrating RGB, VIS-SWIR, LWIR, and LiDAR for landmine detection in off-road environments.

## Quick Start

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Train
python -m mineinsight.train --config configs/paper.toml

# Evaluate
python -m mineinsight.evaluate --config configs/paper.toml --checkpoint best.pth

# Serve
python -m mineinsight.serve --port 8080
```

# PRD-06: Export Pipeline

> Module: DEF-mineinsight | Priority: P1
> Depends on: PRD-02
> Status: COMPLETE

## Objective
Export trained MineInsight detector to ONNX and prepare for TensorRT conversion.

## Export Targets
1. **ONNX**: Universal format for cross-platform deployment
2. **TensorRT**: High-performance inference on NVIDIA GPUs (via shared TRT toolkit)
3. **Safetensors**: Safe weight serialization for HuggingFace upload

## Acceptance Criteria
- [x] ONNX export with dynamic batch size.
- [x] ONNX model validates with onnxruntime.
- [x] Safetensors export for HuggingFace.
- [x] TRT export path documented (uses shared TRT toolkit).

## Files Created
| File | Purpose | Est. Lines |
|---|---|---:|
| Export logic in `src/mineinsight/utils.py` | ONNX + safetensors export functions | ~80 |

## Test Plan
```bash
# Export to ONNX
uv run python -c "from mineinsight.utils import export_onnx; export_onnx(...)"

# TRT conversion (uses shared toolkit)
python /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py \
  --onnx exports/model.onnx --output exports/model_fp16.trt --fp16
```

## References
- Shared TRT toolkit: /mnt/forge-data/shared_infra/trt_toolkit/
- Feeds into: PRD-07

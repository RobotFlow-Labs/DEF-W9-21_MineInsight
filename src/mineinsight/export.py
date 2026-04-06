"""MineInsight model export pipeline.

Exports trained models to multiple formats:
  pth → safetensors → ONNX → TensorRT FP16 → TensorRT FP32

Usage:
    python -m mineinsight.export --config configs/paper.toml --checkpoint best.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from mineinsight.model import build_model
from mineinsight.utils import Config, export_onnx, export_safetensors, get_device, load_config


def export_all(
    cfg: Config,
    checkpoint_path: str,
    output_dir: str | None = None,
) -> dict[str, Path]:
    """Export model to all required formats.

    Args:
        cfg: Model configuration.
        checkpoint_path: Path to trained checkpoint.
        output_dir: Output directory (default: artifacts exports dir).

    Returns:
        Dict mapping format name to output path.
    """
    if output_dir is None:
        output_dir = "/mnt/artifacts-datai/exports/project_mineinsight"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Build model
    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
        architecture=cfg.model.architecture,
        pretrained=cfg.model.pretrained,
    ).to(device)

    # Load checkpoint
    print(f"[EXPORT] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    results: dict[str, Path] = {}

    # 1. PyTorch (.pth)
    pth_path = out / "model.pth"
    torch.save({"model": model.state_dict()}, pth_path)
    results["pth"] = pth_path
    print(f"[EXPORT] pth: {pth_path} ({pth_path.stat().st_size / 1e6:.1f}MB)")

    # 2. SafeTensors
    safe_path = out / "model.safetensors"
    export_safetensors(model, safe_path)
    results["safetensors"] = safe_path
    print(f"[EXPORT] safetensors: {safe_path} ({safe_path.stat().st_size / 1e6:.1f}MB)")

    # 3. ONNX
    input_size = tuple(cfg.model.input_size)
    onnx_path = out / "model.onnx"
    export_onnx(model, onnx_path, input_size=input_size, opset=17)
    results["onnx"] = onnx_path
    print(f"[EXPORT] onnx: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f}MB)")

    # 4. TensorRT FP16
    trt_fp16_path = _export_trt(onnx_path, out / "model_fp16.engine", precision="fp16")
    if trt_fp16_path:
        results["trt_fp16"] = trt_fp16_path
        print(f"[EXPORT] trt_fp16: {trt_fp16_path}")

    # 5. TensorRT FP32
    trt_fp32_path = _export_trt(onnx_path, out / "model_fp32.engine", precision="fp32")
    if trt_fp32_path:
        results["trt_fp32"] = trt_fp32_path
        print(f"[EXPORT] trt_fp32: {trt_fp32_path}")

    # Copy config alongside exports
    shutil.copy2(checkpoint_path, out / "checkpoint.pth")

    print(f"[EXPORT] All exports saved to {out}/")
    return results


def _export_trt(
    onnx_path: Path,
    output_path: Path,
    precision: str = "fp16",
) -> Path | None:
    """Export ONNX to TensorRT using shared toolkit.

    Uses /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py
    """
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if not trt_script.exists():
        print(f"[WARN] TRT toolkit not found at {trt_script}, skipping TRT export")
        return None

    import subprocess

    cmd = [
        "python", str(trt_script),
        "--onnx", str(onnx_path),
        "--output", str(output_path),
        "--precision", precision,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return output_path
        print(f"[WARN] TRT {precision} export failed: {result.stderr[:200]}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] TRT {precision} export error: {e}")
    return None


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MineInsight Model Export")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_all(cfg, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

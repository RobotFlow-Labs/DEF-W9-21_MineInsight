"""MineInsight FastAPI serving endpoint.

Endpoints:
    GET  /health  -- Module health status
    GET  /ready   -- Readiness probe
    GET  /info    -- Module metadata
    POST /predict -- Run inference on uploaded image
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]

from mineinsight.evaluate import decode_predictions
from mineinsight.model import build_model
from mineinsight.utils import get_device

MODULE_NAME = "mineinsight"
MODULE_VERSION = "0.1.0"

_model = None
_device = None
_start_time = time.time()


def _get_app() -> FastAPI:
    """Create FastAPI application."""
    if FastAPI is None:
        raise ImportError("fastapi not installed")

    app = FastAPI(title=f"ANIMA {MODULE_NAME}", version=MODULE_VERSION)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "module": MODULE_NAME,
            "uptime_s": round(time.time() - _start_time, 1),
            "gpu_available": torch.cuda.is_available(),
        }

    @app.get("/ready")
    async def ready():
        is_ready = _model is not None
        status_code = 200 if is_ready else 503
        return JSONResponse(
            content={
                "ready": is_ready,
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "weights_loaded": is_ready,
            },
            status_code=status_code,
        )

    @app.get("/info")
    async def info():
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "task": "multi_modal_landmine_detection",
            "num_classes": 35,
            "modalities": ["rgb", "lwir", "swir"],
            "input_size": [640, 640],
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if _model is None:
            return JSONResponse(
                content={"error": "Model not loaded"},
                status_code=503,
            )

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        # Preprocess
        import cv2

        resized = cv2.resize(img_np, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(_device)

        with torch.no_grad():
            preds = _model(tensor)
        dets = decode_predictions(preds, conf_threshold=0.25)

        det = dets[0]
        results = []
        for i in range(len(det["scores"])):
            results.append({
                "box": det["boxes"][i].cpu().tolist(),
                "score": det["scores"][i].cpu().item(),
                "label": int(det["labels"][i].cpu().item()),
            })

        return {"detections": results, "num_detections": len(results)}

    return app


def main() -> None:
    """Start serving."""
    global _model, _device
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANIMA_SERVE_PORT", 8080)))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ros2", action="store_true")
    args = parser.parse_args()

    _device = get_device()
    _model = build_model("rgb", num_classes=35).to(_device)
    _model.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=_device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        _model.load_state_dict(state)
        print(f"[SERVE] Loaded checkpoint: {args.checkpoint}")

    print(f"[SERVE] Model loaded on {_device}")
    print(f"[SERVE] Starting on port {args.port}")

    app = _get_app()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

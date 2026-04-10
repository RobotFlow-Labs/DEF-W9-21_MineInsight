#!/usr/bin/env python3
"""Train stock Ultralytics YOLO26s-p2 on the 6-channel MineInsight fusion dataset.

This script is the Phase 2 training entrypoint from ``PIVOT_PLAN.md``. It:

1. Loads the ``yolo26s-p2.yaml`` architecture with ``ch=6``, ``nc=34``.
2. Attempts partial COCO pretrain transfer from ``yolo26s.pt`` (skipping the
   first conv, which has a channel mismatch).
3. Launches a stock Ultralytics ``model.train(...)`` call with conservative
   hyperparameters (no copy_paste, no mixup, default cls weight).
4. Writes every decision and parameter to a dedicated log file under
   ``/mnt/artifacts-datai/logs/project_mineinsight_{EXP_NAME}/``.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python scripts/train_yolo26_fusion.py \
        --data /mnt/forge-data/shared_infra/datasets/mineinsight_fusion/data_mixed.yaml \
        --exp  yolo26s_p2_mixed_v1 \
        --epochs 150 \
        --imgsz 640 \
        --batch -1

All output (checkpoints, tfevents, results.csv, log) lands in
``/mnt/artifacts-datai/checkpoints/project_mineinsight_{EXP_NAME}/``.

Recovery note: if training crashes mid-run, you can resume with
``--resume /mnt/artifacts-datai/checkpoints/.../last.pt``. Ultralytics handles
LR scheduler and optimizer state restoration transparently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch

# Ensure our src/ is importable for any ad-hoc checks
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-5s] %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(logging.INFO)
    stdout_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(stdout_h)

    file_h = logging.FileHandler(log_file, mode="w")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_h)

    log = logging.getLogger("train_yolo26")
    log.info(f"[LOG] {log_file}")
    return log


# ---------------------------------------------------------------------------
# Pretrain transfer
# ---------------------------------------------------------------------------

def load_partial_pretrain(
    model,  # ultralytics.YOLO
    pretrained_pt: Path,
    log: logging.Logger,
) -> tuple[int, int, int]:
    """Load matching weights from a pretrained checkpoint into the custom model.

    Skips any key whose shape differs (typically ``model.0`` = first conv, because
    COCO YOLO26s has ``Conv(3→32)`` but our 6-ch model has ``Conv(6→32)``).

    Uses ``strict=False`` on ``load_state_dict`` so missing/unexpected keys are
    reported but do not crash the load. This is the canonical partial-load
    pattern and is robust across PyTorch versions (unlike the earlier
    ``dst_state.update(matched)`` in-place mutation which depended on
    ``state_dict()`` returning live parameter references).

    Returns
    -------
    (n_loaded, n_shape_skipped, n_missing) : tuple[int, int, int]
    """
    from ultralytics import YOLO

    if not pretrained_pt.exists():
        log.warning(f"[PRETRAIN] not found: {pretrained_pt}")
        return 0, 0, 0

    log.info(f"[PRETRAIN] loading from {pretrained_pt}")
    try:
        src = YOLO(str(pretrained_pt))
    except Exception as e:
        log.exception(f"[PRETRAIN] failed to load source: {e}")
        return 0, 0, 0

    src_state = src.model.state_dict()
    dst_state = model.model.state_dict()

    # Build the filtered dict of weights we can actually apply
    matched: dict = {}
    shape_skipped: list[str] = []
    for k, v in src_state.items():
        if k in dst_state:
            if v.shape == dst_state[k].shape:
                matched[k] = v
            else:
                shape_skipped.append(
                    f"{k}: src{tuple(v.shape)} vs dst{tuple(dst_state[k].shape)}"
                )

    # Apply via strict=False. This is the canonical partial-load pattern —
    # missing_keys = params NOT in matched (they keep their random init),
    # unexpected_keys = keys in the checkpoint that aren't in the model.
    try:
        missing, unexpected = model.model.load_state_dict(matched, strict=False)
    except Exception as e:
        log.exception(f"[PRETRAIN] load_state_dict failed: {e}")
        return 0, len(shape_skipped), 0

    # **Verification** (defense in depth): spot-check that at least one tensor
    # was actually copied. Compare one matched key's first element to the
    # original source value. If they differ, the load silently no-op'd.
    if matched:
        first_key = next(iter(matched))
        src_val = src_state[first_key].flatten()[0].item()
        dst_val = model.model.state_dict()[first_key].flatten()[0].item()
        if abs(src_val - dst_val) > 1e-8:
            log.error(
                f"[PRETRAIN] VERIFICATION FAILED: key={first_key} "
                f"src={src_val} dst={dst_val} — weights NOT applied!",
            )
        else:
            log.info(f"[PRETRAIN] verification passed (key={first_key} src==dst)")

    log.info(
        f"[PRETRAIN] loaded={len(matched)} "
        f"/ src={len(src_state)} / dst={len(dst_state)} "
        f"shape_skipped={len(shape_skipped)} "
        f"missing_after_load={len(missing)} "
        f"unexpected={len(unexpected)}",
    )
    if shape_skipped:
        log.info("[PRETRAIN] shape-skipped keys:")
        for s in shape_skipped[:5]:
            log.info(f"  {s}")
        if len(shape_skipped) > 5:
            log.info(f"  ... and {len(shape_skipped) - 5} more")

    return len(matched), len(shape_skipped), len(missing)


# ---------------------------------------------------------------------------
# GPU memory watchdog (prints first sample for early sanity check)
# ---------------------------------------------------------------------------

def report_gpu_memory(log: logging.Logger, device: int | str) -> None:
    if not torch.cuda.is_available():
        log.info("[GPU] CUDA not available")
        return
    try:
        dev = int(device) if isinstance(device, (int, str)) and str(device).isdigit() else 0
        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        log.info(
            f"[GPU] device={dev} used={used / 1024**3:.2f}GB / "
            f"total={total / 1024**3:.2f}GB ({used / total * 100:.0f}%)",
        )
    except Exception as e:
        log.warning(f"[GPU] query failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="MineInsight YOLO26 fusion training")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml path")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--arch", type=str, default="yolo26s-p2.yaml",
        help="Ultralytics model yaml (e.g. yolo26s-p2.yaml)",
    )
    parser.add_argument(
        "--pretrained", type=Path,
        default=Path("/mnt/train-data/models/yolo26/yolo26s.pt"),
        help="Source for partial COCO pretrain transfer",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0, help="0 avoids fork-with-CUDA hangs")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="Optional last.pt to resume")
    parser.add_argument(
        "--project", type=Path,
        default=Path("/mnt/artifacts-datai/checkpoints"),
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=None,
        help="Override log directory. Default: /mnt/artifacts-datai/logs/project_mineinsight_{exp}",
    )
    args = parser.parse_args()

    project_name = f"project_mineinsight_{args.exp}"
    if args.log_dir is None:
        args.log_dir = Path("/mnt/artifacts-datai/logs") / project_name

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f"train_{ts}.log"
    log = setup_logging(log_file)

    log.info(f"[START] args={vars(args)}")
    log.info(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log.info(f"[ENV] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"[ENV] cuda.device_count={torch.cuda.device_count()}")

    if not args.data.exists():
        log.error(f"[FATAL] data yaml not found: {args.data}")
        return 2

    # Mark data yaml contents for the log (helps debugging remote runs)
    try:
        log.info(f"[DATA] yaml contents:\n{args.data.read_text()}")
    except Exception as e:
        log.warning(f"[DATA] failed to read yaml: {e}")

    # Import YOLO lazily so that the log file is created even on import failure
    from ultralytics import YOLO

    # Build model
    if args.resume is not None:
        if not args.resume.exists():
            log.error(f"[FATAL] resume checkpoint not found: {args.resume}")
            return 2
        log.info(f"[RESUME] loading {args.resume}")
        model = YOLO(str(args.resume))
    else:
        log.info(f"[MODEL] building {args.arch}")
        model = YOLO(str(args.arch))
        load_partial_pretrain(model, args.pretrained, log)

    report_gpu_memory(log, args.device)

    # Graceful shutdown: log SIGTERM/SIGINT before Ultralytics handles it
    def _handle_signal(sig, frame):  # noqa: ARG001
        log.warning(f"[SIGNAL] received signal {sig}, shutting down gracefully")
        sys.exit(130)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Launch training
    log.info("[TRAIN] starting model.train(...)")
    try:
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project),
            name=project_name,
            exist_ok=True,
            patience=args.patience,
            save=True,
            save_period=-1,       # only best + last
            cache=False,
            workers=args.workers,
            amp=True,
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            mosaic=args.mosaic,
            mixup=args.mixup,
            close_mosaic=args.close_mosaic,
            copy_paste=args.copy_paste,
            cls=args.cls,
            seed=args.seed,
            verbose=True,
            plots=True,
        )
    except KeyboardInterrupt:
        log.warning("[TRAIN] interrupted by user")
        return 130
    except Exception as e:
        log.exception(f"[FATAL] training crashed: {e}")
        return 3

    report_gpu_memory(log, args.device)

    # Dump a machine-readable summary
    summary_path = args.log_dir / f"summary_{ts}.json"
    try:
        summary = {
            "exp": args.exp,
            "best_fitness": float(getattr(results, "fitness", float("nan"))),
            "save_dir": str(getattr(results, "save_dir", "")),
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info(f"[SUMMARY] wrote {summary_path}")
    except Exception as e:
        log.warning(f"[SUMMARY] failed to write: {e}")

    log.info("[DONE] training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

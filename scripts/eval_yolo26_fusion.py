#!/usr/bin/env python3
"""Thin Ultralytics eval wrapper for the YOLO26 fusion checkpoints.

This is a separate script from ``scripts/eval_comprehensive.py`` on purpose:
``eval_comprehensive.py`` is hard-wired to the custom v5 model (``build_model``,
``num_classes=58``, custom ``compute_map``) and keeping it untouched preserves
the historical v5 eval path.

This script calls ``model.val(...)`` on each split and writes a JSON report in
the same shape as the comprehensive eval (``split → sweep[conf] → metrics``)
plus a **per-class AP dict keyed by NAME**, which is what we need to compare
across the v5 (58-class) and v6 (34-class) schemas.

The per-class mAP breakdown is pulled from Ultralytics' ``DetMetrics`` object,
which exposes ``maps`` (per-class mAP50-95) and ``results_dict`` (overall).

Usage
-----
    python scripts/eval_yolo26_fusion.py \\
        --checkpoint \\
          /mnt/artifacts-datai/checkpoints/.../weights/best.pt \\
        --data \\
          /mnt/forge-data/shared_infra/datasets/mineinsight_fusion/data_mixed.yaml \\
        --output \\
          /mnt/artifacts-datai/reports/project_mineinsight/yolo26s.json \\
        --splits val,test

``--mine-report`` additionally prints a table of mine-class AP alongside the
overall numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mineinsight.label_remap import LabelRemap  # noqa: E402

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
    stdout_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(stdout_h)

    file_h = logging.FileHandler(log_file, mode="w")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_h)

    log = logging.getLogger("eval_yolo26")
    log.info(f"[LOG] {log_file}")
    return log


# ---------------------------------------------------------------------------
# Mine classification (by name, for cross-schema comparability)
# ---------------------------------------------------------------------------

_MINE_KEYWORDS = (
    "PFM", "PMN", "PROM", "MON", "TC-", "TM-", "TMA", "TMM", "M6", "M-35",
    "Type 72", "VS-", "C-3",
)


def is_mine_name(name: str) -> bool:
    return any(kw in name for kw in _MINE_KEYWORDS)


# ---------------------------------------------------------------------------
# Run one eval pass at multiple conf thresholds
# ---------------------------------------------------------------------------

def eval_one_split(
    model,  # ultralytics.YOLO
    data_yaml: Path,
    split: str,
    conf_thresholds: list[float],
    iou: float,
    imgsz: int,
    device: str,
    log: logging.Logger,
) -> dict:
    """Run ``model.val(...)`` once per conf threshold and collect metrics.

    Returns a dict matching the ``eval_comprehensive.py`` JSON schema.
    """
    log.info(f"[EVAL split={split}] starting")
    out = {
        "split": split,
        "data_yaml": str(data_yaml),
        "iou_threshold": iou,
        "imgsz": imgsz,
        "sweep": {},
    }

    for conf in conf_thresholds:
        log.info(f"  [conf={conf:.2f}] running model.val()")
        t0 = time.time()
        try:
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                plots=False,
                save_json=False,
                workers=0,
            )
        except Exception as e:
            log.exception(f"  [conf={conf:.2f}] model.val crashed: {e}")
            continue
        elapsed = time.time() - t0

        # Ultralytics returns a DetMetrics object:
        # metrics.box.map   = mAP@0.5:0.95 overall
        # metrics.box.map50 = mAP@0.5 overall
        # metrics.box.maps  = per-class mAP@0.5:0.95 array of length nc
        # metrics.box.p, metrics.box.r = per-class P / R
        # metrics.names = {id: name}

        try:
            names = metrics.names if hasattr(metrics, "names") else {}
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            per_class_map = metrics.box.maps.tolist() \
                if hasattr(metrics.box.maps, "tolist") else list(metrics.box.maps)
            # maps array is indexed by CLASS ID (0..nc-1)
            per_class_by_name = {
                names.get(i, f"class_{i}"): float(per_class_map[i])
                for i in range(len(per_class_map))
            }
        except Exception as e:
            log.warning(f"  [conf={conf:.2f}] metrics parsing error: {e}")
            per_class_by_name = {}
            map50, map5095 = 0.0, 0.0

        mine_aps = [
            ap for name, ap in per_class_by_name.items()
            if is_mine_name(name) and ap > 0
        ]
        mine_map = sum(mine_aps) / len(mine_aps) if mine_aps else 0.0

        out["sweep"][f"{conf:.2f}"] = {
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map5095,
            "mine_mAP@0.5": mine_map,
            "per_class_by_name": per_class_by_name,
            "num_mine_classes_with_ap": len(mine_aps),
            "elapsed_sec": elapsed,
        }

        log.info(
            f"  [conf={conf:.2f}] mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
            f"mine_mAP={mine_map:.4f}  mines_with_ap={len(mine_aps)}",
        )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Ultralytics-based YOLO26 eval wrapper")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path")
    parser.add_argument("--splits", type=str, default="val", help="Comma-separated: val,test")
    parser.add_argument("--conf-thresholds", type=str, default="0.25,0.10")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--label-remap",
        type=Path,
        default=None,
        help="label_remap.json (used only to classify mines by name)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path("/mnt/artifacts-datai/logs/project_mineinsight_eval"),
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log = setup_logging(args.log_dir / f"eval_{args.checkpoint.stem}_{ts}.log")
    log.info(f"[ARGS] {vars(args)}")

    if not args.checkpoint.exists():
        log.error(f"[FATAL] checkpoint not found: {args.checkpoint}")
        return 2
    if not args.data.exists():
        log.error(f"[FATAL] data yaml not found: {args.data}")
        return 2

    # Load the remap solely for a sanity check that mine-name keywords agree
    if args.label_remap and args.label_remap.exists():
        try:
            remap = LabelRemap.load(args.label_remap)
            log.info(
                f"[REMAP] loaded, {remap.num_classes()} classes, "
                f"{len(remap.mine_new_ids)} mines",
            )
        except Exception as e:
            log.warning(f"[REMAP] failed to load: {e}")

    from ultralytics import YOLO

    log.info(f"[LOAD] {args.checkpoint}")
    model = YOLO(str(args.checkpoint))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    confs = [float(c) for c in args.conf_thresholds.split(",")]

    report = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
    }

    for split in splits:
        report["results"][split] = eval_one_split(
            model, args.data, split, confs,
            iou=args.iou, imgsz=args.imgsz, device=args.device, log=log,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    log.info(f"[SAVED] {args.output}")

    # Print a small summary table
    log.info("=" * 78)
    log.info("SUMMARY")
    log.info("=" * 78)
    log.info(f"{'split':<8}{'conf':<8}{'mAP50':<10}{'mAP50-95':<12}{'mine_mAP':<10}")
    log.info("-" * 78)
    for split, r in report["results"].items():
        for conf, s in r.get("sweep", {}).items():
            log.info(
                f"{split:<8}{conf:<8}"
                f"{s['mAP@0.5']:<10.4f}"
                f"{s['mAP@0.5:0.95']:<12.4f}"
                f"{s['mine_mAP@0.5']:<10.4f}",
            )
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

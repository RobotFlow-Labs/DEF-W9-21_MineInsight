#!/usr/bin/env python3
"""Offline builder: MineInsight → 6-channel RGB+LWIR multi-page TIFF dataset.

This script pre-processes the raw MineInsight dataset into an
Ultralytics-compatible layout with **multi-page TIFF images** where each TIFF
contains 6 grayscale pages: 3 from the RGB frame and 3 from the nearest-in-time
LWIR frame. Ultralytics 8.4.33's ``imread`` patch uses ``cv2.imdecodemulti``
which correctly decodes this into a ``(H, W, 6)`` array when combined with
``channels: 6`` in the dataset YAML.

For each primary (RGB) frame, the nearest-timestamp LWIR frame is located,
spatially resized to RGB resolution, and stacked. Class IDs in the label files
are remapped from the raw MineInsight instance IDs [1..57] to the contiguous
object-type IDs [0..33] via :class:`mineinsight.label_remap.LabelRemap`.

Outputs (under ``--out``):

    images/
      train/ track_1_*.tiff
      val/   track_2_s1_*.tiff
      test/  10% random slice of train
    labels/
      train/ *.txt (class IDs remapped)
      val/
      test/
    data_mixed.yaml       — 90/5/5 random split over {track_1_*, track_2_s1}
    data_crosstrack.yaml  — train=track_1_*, val=track_2_s1 (generalization)
    label_remap.json      — remap table (for eval + archival)
    BUILD_MANIFEST.json   — git SHA, counts, md5 of a 1% sample
    build.log             — full log of this run

Usage
-----
    python scripts/build_fusion_dataset.py \
        --raw /mnt/train-data/datasets/mineinsight \
        --out /mnt/forge-data/shared_infra/datasets/mineinsight_fusion \
        --tolerance-ms 100

Design rationale — see ``PIVOT_PLAN.md`` §Phase 1.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add src/ to path so we can import the module without a venv install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mineinsight.label_remap import LabelRemap, remap_label_file  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure root logger with stdout + file handlers."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-5s] %(message)s"
    datefmt = "%H:%M:%S"
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any pre-existing handlers (e.g., from pytest)
    for h in list(root.handlers):
        root.removeHandler(h)

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(level)
    stdout_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(stdout_h)

    file_h = logging.FileHandler(log_file, mode="w")
    file_h.setLevel(logging.DEBUG)  # always verbose in file
    file_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_h)

    log = logging.getLogger("build_fusion")
    log.info(f"[LOG] writing to {log_file}")
    return log


# ---------------------------------------------------------------------------
# Timestamp parsing (ported from src/mineinsight/dataset.py:202-216)
# ---------------------------------------------------------------------------

def extract_timestamp(stem: str) -> int:
    """Extract numeric timestamp (nanoseconds) from a MineInsight filename stem.

    Stems look like:  ``track_1_s1_rgb_1730290670_166206123``

    Returns a single 64-bit int in nanoseconds: ``sec * 1e9 + ns``.
    """
    parts = stem.split("_")
    nums = [p for p in parts if p.isdigit()]
    if len(nums) >= 2:
        return int(nums[-2]) * 1_000_000_000 + int(nums[-1])
    if len(nums) >= 1:
        return int(nums[-1])
    return 0


# ---------------------------------------------------------------------------
# Disk layout discovery
# ---------------------------------------------------------------------------

@dataclass
class SequenceLayout:
    """Resolved file locations for one (sequence, modality) combination."""
    name: str                           # e.g. "track_1_s1"
    rgb_img_dir: Path
    rgb_label_dir: Path
    lwir_img_dir: Path
    lwir_label_dir: Path | None         # may be None if reproj-only
    rgb_stems: list[str] = field(default_factory=list)
    lwir_stems_sorted: list[tuple[int, str]] = field(default_factory=list)


def find_sequence_dirs(raw_root: Path, seq: str) -> SequenceLayout | None:
    """Resolve the four image/label directories for a sequence.

    Returns None if required RGB images or labels are missing.
    """
    rgb_img = raw_root / f"{seq}_rgb_images"
    rgb_lbl = raw_root / f"{seq}_rgb_labels"
    lwir_img = raw_root / f"{seq}_lwir_images"
    lwir_lbl_reproj = raw_root / f"{seq}_lwir_labels_reproj"
    lwir_lbl = raw_root / f"{seq}_lwir_labels"

    if not rgb_img.is_dir() or not rgb_lbl.is_dir():
        logging.warning(f"[SEQ {seq}] RGB images or labels missing, skipping")
        return None
    if not lwir_img.is_dir():
        logging.warning(f"[SEQ {seq}] LWIR images missing, skipping")
        return None

    lwir_lbl_dir = lwir_lbl_reproj if lwir_lbl_reproj.is_dir() else (
        lwir_lbl if lwir_lbl.is_dir() else None
    )

    return SequenceLayout(
        name=seq,
        rgb_img_dir=rgb_img,
        rgb_label_dir=rgb_lbl,
        lwir_img_dir=lwir_img,
        lwir_label_dir=lwir_lbl_dir,
    )


def populate_layout(layout: SequenceLayout, log: logging.Logger) -> None:
    """Fill in rgb_stems and lwir_stems_sorted."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    rgb_stems = sorted(
        p.stem for p in layout.rgb_img_dir.iterdir()
        if p.suffix.lower() in img_exts
    )
    # Only keep stems that also have a matching label file
    lbl_stems = {
        p.stem for p in layout.rgb_label_dir.iterdir() if p.suffix == ".txt"
    }
    layout.rgb_stems = [s for s in rgb_stems if s in lbl_stems]

    lwir_files = sorted(
        p for p in layout.lwir_img_dir.iterdir()
        if p.suffix.lower() in img_exts
    )
    layout.lwir_stems_sorted = sorted(
        (extract_timestamp(p.stem), p.stem) for p in lwir_files
    )

    log.info(
        f"[SEQ {layout.name}] rgb={len(layout.rgb_stems)} frames "
        f"(labels matched), lwir={len(layout.lwir_stems_sorted)} frames",
    )


def nearest_lwir(
    layout: SequenceLayout, rgb_stem: str, tolerance_ns: int,
) -> str | None:
    """Find the LWIR stem closest in time to ``rgb_stem``.

    Returns None if no LWIR frame is within ``tolerance_ns`` nanoseconds.
    """
    if not layout.lwir_stems_sorted:
        return None

    target = extract_timestamp(rgb_stem)
    timestamps = [t for t, _ in layout.lwir_stems_sorted]
    idx = bisect.bisect_left(timestamps, target)

    best_stem: str | None = None
    best_dist = tolerance_ns + 1
    for ci in (idx - 1, idx):
        if 0 <= ci < len(layout.lwir_stems_sorted):
            ts, stem = layout.lwir_stems_sorted[ci]
            dist = abs(ts - target)
            if dist < best_dist:
                best_dist = dist
                best_stem = stem

    if best_stem is None or best_dist > tolerance_ns:
        return None
    return best_stem


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _find_image_file(dirpath: Path, stem: str) -> Path | None:
    """Locate a file with the given stem, trying common extensions first.

    Falls back to ``dirpath.glob(stem + '.*')`` if the known extensions miss.
    Returns None if no matching file exists.
    """
    for ext in _IMAGE_EXTS:
        cand = dirpath / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Defensive glob fallback (slower but catches weird extensions)
    matches = list(dirpath.glob(f"{stem}.*"))
    return matches[0] if matches else None


def load_rgb(img_path: Path) -> np.ndarray | None:
    """Load an RGB image as (H, W, 3) BGR uint8 (cv2 convention)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    return img


def load_lwir(
    img_path: Path,
    lwir_lo: float | None = None,
    lwir_hi: float | None = None,
) -> np.ndarray | None:
    """Load an LWIR image as (H, W, 3) uint8.

    Handles three source formats seen in MineInsight:
    1. 16-bit grayscale  → clip to [lwir_lo, lwir_hi] then scale to uint8
    2. 8-bit  grayscale  → stack 3×
    3. 8-bit  3-channel  → return as-is

    **Thermal calibration**: when ``lwir_lo`` / ``lwir_hi`` are provided, the
    16-bit thermal signal is clipped and scaled against those FIXED dataset-wide
    bounds. This preserves the absolute thermal value across frames — a warm
    frame and a cold frame end up with different pixel intensities, so the
    model can learn actual thermal signatures rather than relative contrast.

    If ``lwir_lo/lwir_hi`` are None we fall back to per-frame NORM_MINMAX, which
    is WRONG for detection training (erases absolute thermal info) — this
    fallback is only here for quick smoke tests.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.dtype != np.uint8:
        if lwir_lo is not None and lwir_hi is not None:
            span = max(lwir_hi - lwir_lo, 1.0)
            img = np.clip((img.astype(np.float32) - lwir_lo) / span, 0, 1)
            img = (img * 255.0).astype(np.uint8)
        else:
            # WARNING: per-frame normalization loses absolute thermal signal.
            # Only used when bounds are not provided (smoke test).
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def compute_lwir_percentiles(
    layouts: list[SequenceLayout],
    n_sample: int = 200,
    log: logging.Logger | None = None,
) -> tuple[float, float]:
    """Scan a random sample of LWIR frames and return (p1, p99) for clipping.

    This is a single pre-pass that determines the dataset-wide thermal value
    distribution so every frame can be normalized against fixed bounds. Uses
    the 1st and 99th percentiles to trim extreme outliers (broken pixels,
    thermal calibration spikes).

    If no 16-bit frames are found (all are already uint8), returns (0, 255).
    """
    rng = random.Random(0)
    samples: list[np.ndarray] = []
    all_lwir_paths: list[Path] = []
    img_exts = {".tif", ".tiff", ".png", ".bmp", ".jpg"}
    for lyt in layouts:
        for p in lyt.lwir_img_dir.iterdir():
            if p.suffix.lower() in img_exts:
                all_lwir_paths.append(p)
    rng.shuffle(all_lwir_paths)

    n_uint16 = 0
    for p in all_lwir_paths[: n_sample * 3]:  # oversample in case of load fails
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.dtype == np.uint8:
            continue
        n_uint16 += 1
        # Flatten + subsample to keep memory bounded
        flat = img.ravel()
        if flat.size > 50_000:
            flat = flat[::max(1, flat.size // 50_000)]
        samples.append(flat)
        if n_uint16 >= n_sample:
            break

    if not samples:
        if log:
            log.info("[LWIR_CALIB] no 16-bit LWIR frames found, using (0, 255)")
        return 0.0, 255.0

    pooled = np.concatenate(samples)
    lo = float(np.percentile(pooled, 1))
    hi = float(np.percentile(pooled, 99))
    if log:
        log.info(
            f"[LWIR_CALIB] sampled {n_uint16} 16-bit frames "
            f"({pooled.size} values). "
            f"dtype={samples[0].dtype}, "
            f"p1={lo:.1f}, p99={hi:.1f}, "
            f"min={pooled.min()}, max={pooled.max()}",
        )
    return lo, hi


def write_6ch_tiff(
    out_path: Path, rgb_bgr: np.ndarray, lwir_bgr: np.ndarray,
) -> bool:
    """Write a 6-page TIFF: 3 RGB pages + 3 LWIR pages (all grayscale).

    Both inputs are resized to LWIR native resolution BEFORE writing. This
    matches the PIVOT_PLAN decision to downsample RGB (from 2448×2048 to
    ~640×512) rather than upsample LWIR.

    Returns True on success, False on failure.
    """
    # Target size = LWIR native (H_lwir, W_lwir)
    h_t, w_t = lwir_bgr.shape[:2]

    if rgb_bgr.shape[:2] != (h_t, w_t):
        rgb_resized = cv2.resize(
            rgb_bgr, (w_t, h_t), interpolation=cv2.INTER_AREA,
        )
    else:
        rgb_resized = rgb_bgr

    # Split channels → 6 grayscale pages (BGR order from cv2)
    pages = [
        rgb_resized[..., 0],   # B
        rgb_resized[..., 1],   # G
        rgb_resized[..., 2],   # R
        lwir_bgr[..., 0],
        lwir_bgr[..., 1],
        lwir_bgr[..., 2],
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencodemulti(".tiff", pages)
    if not ok:
        return False
    out_path.write_bytes(buf.tobytes())
    return True


# ---------------------------------------------------------------------------
# Build plan (sequence → split assignment)
# ---------------------------------------------------------------------------

@dataclass
class BuildSplits:
    """Map each (seq, stem) to a split for a given YAML config."""
    mixed_split: dict[tuple[str, str], str] = field(default_factory=dict)
    crosstrack_split: dict[tuple[str, str], str] = field(default_factory=dict)


def plan_splits(
    layouts: list[SequenceLayout],
    mixed_train_pct: float = 0.90,
    mixed_val_pct: float = 0.05,
    seed: int = 42,
) -> BuildSplits:
    """Build two independent split dicts for the same underlying frames.

    Mixed split: 90/5/5 random across all sequences.
    Cross-track split: train=track_1_*, val=track_2_s1, test=none.
    """
    out = BuildSplits()

    # Mixed: shuffle and slice
    all_keys: list[tuple[str, str]] = []
    for lyt in layouts:
        all_keys.extend((lyt.name, stem) for stem in lyt.rgb_stems)
    rng = random.Random(seed)
    rng.shuffle(all_keys)
    n = len(all_keys)
    n_train = int(n * mixed_train_pct)
    n_val = int(n * mixed_val_pct)
    for k in all_keys[:n_train]:
        out.mixed_split[k] = "train"
    for k in all_keys[n_train:n_train + n_val]:
        out.mixed_split[k] = "val"
    for k in all_keys[n_train + n_val:]:
        out.mixed_split[k] = "test"

    # Cross-track
    for lyt in layouts:
        split = "train" if lyt.name.startswith("track_1") else "val"
        for stem in lyt.rgb_stems:
            out.crosstrack_split[(lyt.name, stem)] = split

    return out


# ---------------------------------------------------------------------------
# Main build pass
# ---------------------------------------------------------------------------

@dataclass
class BuildStats:
    total_frames: int = 0
    written: int = 0
    skipped_no_lwir: int = 0
    skipped_load_fail: int = 0
    skipped_encode_fail: int = 0
    n_labels_written: int = 0
    n_labels_dropped: int = 0
    per_seq_counts: dict[str, int] = field(default_factory=dict)
    per_split_counts_mixed: dict[str, int] = field(default_factory=dict)
    per_split_counts_crosstrack: dict[str, int] = field(default_factory=dict)
    lwir_lo: float = 0.0
    lwir_hi: float = 255.0


def _md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(
    raw_root: Path,
    out_root: Path,
    sequences: list[str],
    tolerance_ms: int,
    limit: int | None,
    log: logging.Logger,
) -> BuildStats:
    """Walk every sequence, stack RGB+LWIR, remap labels, write outputs."""
    stats = BuildStats()
    tolerance_ns = tolerance_ms * 1_000_000

    log.info(f"[CONFIG] raw_root   = {raw_root}")
    log.info(f"[CONFIG] out_root   = {out_root}")
    log.info(f"[CONFIG] sequences  = {sequences}")
    log.info(f"[CONFIG] tolerance  = {tolerance_ms} ms ({tolerance_ns} ns)")
    log.info(f"[CONFIG] limit      = {limit or 'ALL'}")

    # ---- Build remap and save it
    remap = LabelRemap.from_targets_yaml(raw_root / "targets_list.yaml")
    remap.save(out_root / "label_remap.json")
    num_classes = remap.num_classes()
    log.info(f"[REMAP] {num_classes} classes, {len(remap.mine_new_ids)} mines")

    # ---- Resolve + populate layouts
    layouts: list[SequenceLayout] = []
    for seq in sequences:
        lyt = find_sequence_dirs(raw_root, seq)
        if lyt is None:
            continue
        populate_layout(lyt, log)
        if not lyt.rgb_stems:
            log.warning(f"[SEQ {seq}] no usable RGB frames, skipping")
            continue
        layouts.append(lyt)
    if not layouts:
        raise RuntimeError("no usable sequences found")

    # ---- Calibrate LWIR thermal range (fixes CRITICAL bug from code review:
    # per-frame NORM_MINMAX destroys absolute thermal signal).
    lwir_lo, lwir_hi = compute_lwir_percentiles(layouts, n_sample=200, log=log)
    log.info(f"[LWIR_CALIB] fixed bounds for build: lo={lwir_lo:.1f}, hi={lwir_hi:.1f}")
    stats.lwir_lo = lwir_lo
    stats.lwir_hi = lwir_hi

    # ---- Plan splits (mixed + cross-track)
    plan = plan_splits(layouts)
    log.info(
        f"[SPLIT mixed] "
        f"train={sum(1 for v in plan.mixed_split.values() if v == 'train')} "
        f"val={sum(1 for v in plan.mixed_split.values() if v == 'val')} "
        f"test={sum(1 for v in plan.mixed_split.values() if v == 'test')}",
    )
    log.info(
        f"[SPLIT crosstrack] "
        f"train={sum(1 for v in plan.crosstrack_split.values() if v == 'train')} "
        f"val={sum(1 for v in plan.crosstrack_split.values() if v == 'val')}",
    )

    # ---- Create output dirs
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- Main loop
    t0 = time.time()
    last_report = t0
    frame_no = 0
    for lyt in layouts:
        stats.per_seq_counts[lyt.name] = 0
        for rgb_stem in lyt.rgb_stems:
            frame_no += 1
            stats.total_frames += 1

            if limit is not None and frame_no > limit:
                log.info(f"[LIMIT] stopping at {limit} frames")
                break

            # 1. Find LWIR match
            lwir_stem = nearest_lwir(lyt, rgb_stem, tolerance_ns)
            if lwir_stem is None:
                stats.skipped_no_lwir += 1
                continue

            # 2. Locate image files (RGB and LWIR probes are independent)
            rgb_path = _find_image_file(lyt.rgb_img_dir, rgb_stem)
            lwir_path = _find_image_file(lyt.lwir_img_dir, lwir_stem)
            if rgb_path is None or lwir_path is None:
                stats.skipped_load_fail += 1
                log.debug(
                    f"[LOCATE FAIL] rgb_stem={rgb_stem} "
                    f"lwir_stem={lwir_stem} rgb_path={rgb_path} "
                    f"lwir_path={lwir_path}",
                )
                continue

            rgb = load_rgb(rgb_path)
            lwir = load_lwir(lwir_path, lwir_lo=lwir_lo, lwir_hi=lwir_hi)
            if rgb is None or lwir is None:
                stats.skipped_load_fail += 1
                log.debug(f"[LOAD FAIL] rgb={rgb_path.name} lwir={lwir_path.name}")
                continue

            # 3. Decide split for mixed build (cross-track handled by the yaml only)
            split_mixed = plan.mixed_split.get((lyt.name, rgb_stem))
            split_cross = plan.crosstrack_split.get((lyt.name, rgb_stem))
            if split_mixed is None:
                continue  # should not happen

            # 4. Encode 6-ch TIFF once, used for both splits
            out_name = f"{lyt.name}__{rgb_stem}.tiff"
            tiff_path = out_root / "images" / split_mixed / out_name
            if not write_6ch_tiff(tiff_path, rgb, lwir):
                stats.skipped_encode_fail += 1
                continue

            # 5. Remap label file → mixed split path
            src_lbl = lyt.rgb_label_dir / f"{rgb_stem}.txt"
            dst_lbl = out_root / "labels" / split_mixed / f"{lyt.name}__{rgb_stem}.txt"
            n_w, n_d = remap_label_file(src_lbl, dst_lbl, remap)
            stats.n_labels_written += n_w
            stats.n_labels_dropped += n_d

            stats.written += 1
            stats.per_seq_counts[lyt.name] += 1
            stats.per_split_counts_mixed[split_mixed] = (
                stats.per_split_counts_mixed.get(split_mixed, 0) + 1
            )
            if split_cross:
                stats.per_split_counts_crosstrack[split_cross] = (
                    stats.per_split_counts_crosstrack.get(split_cross, 0) + 1
                )

            # 6. Progress report every 2 seconds
            now = time.time()
            if now - last_report > 2.0:
                rate = stats.written / max(now - t0, 1e-6)
                log.info(
                    f"[BUILD] seq={lyt.name} "
                    f"{stats.written}/{stats.total_frames} written, "
                    f"skipped(nolwir={stats.skipped_no_lwir}, "
                    f"load={stats.skipped_load_fail}, "
                    f"enc={stats.skipped_encode_fail}), "
                    f"{rate:.1f} f/s",
                )
                last_report = now

        if limit is not None and frame_no > limit:
            break

    elapsed = time.time() - t0
    log.info(f"[DONE] wrote {stats.written} frames in {elapsed:.1f}s "
             f"({stats.written / max(elapsed, 1e-6):.1f} f/s)")
    log.info(f"[STATS] {asdict(stats)}")
    return stats


# ---------------------------------------------------------------------------
# Write Ultralytics YAMLs + build manifest
# ---------------------------------------------------------------------------

def write_data_yaml(
    path: Path, out_root: Path, remap: LabelRemap,
    train_sub: str, val_sub: str, test_sub: str | None,
) -> None:
    """Write an Ultralytics-compatible data.yaml with ``channels: 6``."""
    names = {new_id: remap.new_to_name[new_id] for new_id in sorted(remap.new_to_name)}
    payload = {
        "path": str(out_root),
        "train": f"images/{train_sub}",
        "val": f"images/{val_sub}",
        "channels": 6,
        "nc": remap.num_classes(),
        "names": names,
    }
    if test_sub:
        payload["test"] = f"images/{test_sub}"
    path.write_text(yaml.dump(payload, sort_keys=False))
    logging.info(f"[YAML] wrote {path}")


def write_crosstrack_yamls(
    out_root: Path, remap: LabelRemap, log: logging.Logger,
) -> None:
    """Build the crosstrack YAML via a symlink tree.

    Because the cross-track split shares the same TIFFs as the mixed split, we
    re-use all files but via different train/val directories. We construct a
    sibling directory `images_crosstrack/` with symlinks and point the YAML
    there.
    """
    ct_root = out_root / "crosstrack"
    ct_images_train = ct_root / "images" / "train"
    ct_images_val = ct_root / "images" / "val"
    ct_labels_train = ct_root / "labels" / "train"
    ct_labels_val = ct_root / "labels" / "val"
    for d in (ct_images_train, ct_images_val, ct_labels_train, ct_labels_val):
        d.mkdir(parents=True, exist_ok=True)

    # Walk mixed splits; for each frame, decide new split by track prefix
    for split in ("train", "val", "test"):
        img_dir = out_root / "images" / split
        if not img_dir.exists():
            continue
        for tiff in img_dir.glob("*.tiff"):
            name = tiff.name   # "track_1_s1__xxx.tiff"
            new_split = "train" if name.startswith("track_1") else "val"
            link_img = ct_root / "images" / new_split / name
            link_lbl = ct_root / "labels" / new_split / f"{tiff.stem}.txt"
            src_lbl = out_root / "labels" / split / f"{tiff.stem}.txt"
            if not link_img.exists():
                link_img.symlink_to(tiff.resolve())
            if src_lbl.exists() and not link_lbl.exists():
                link_lbl.symlink_to(src_lbl.resolve())

    ct_yaml = out_root / "data_crosstrack.yaml"
    names = {new_id: remap.new_to_name[new_id] for new_id in sorted(remap.new_to_name)}
    payload = {
        "path": str(ct_root),
        "train": "images/train",
        "val": "images/val",
        "channels": 6,
        "nc": remap.num_classes(),
        "names": names,
    }
    ct_yaml.write_text(yaml.dump(payload, sort_keys=False))
    log.info(f"[YAML] wrote {ct_yaml}")


def write_manifest(
    out_root: Path, stats: BuildStats, sequences: list[str],
    tolerance_ms: int, lwir_lo: float, lwir_hi: float,
    log: logging.Logger,
) -> None:
    """Write BUILD_MANIFEST.json with git SHA + frame counts + 1%-sample md5."""
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"

    # 1% md5 sample: pick every 100th TIFF
    all_tiffs = sorted((out_root / "images").rglob("*.tiff"))
    sample = all_tiffs[::100][:50]
    sample_hashes = {str(p.relative_to(out_root)): _md5_of_file(p) for p in sample}

    manifest = {
        "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequences": sequences,
        "tolerance_ms": tolerance_ms,
        "lwir_calibration": {
            "lo": lwir_lo,
            "hi": lwir_hi,
            "method": "fixed_p1_p99_over_dataset_sample",
        },
        "stats": asdict(stats),
        "sample_md5": sample_hashes,
    }
    manifest_path = out_root / "BUILD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info(f"[MANIFEST] wrote {manifest_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build MineInsight fusion dataset")
    parser.add_argument(
        "--raw", type=Path,
        default=Path("/mnt/train-data/datasets/mineinsight"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/mnt/forge-data/shared_infra/datasets/mineinsight_fusion"),
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["track_1_s1", "track_1_s2", "track_2_s1"],
        help="Which sequences to include",
    )
    parser.add_argument(
        "--tolerance-ms", type=int, default=100,
        help="Max time delta (ms) between RGB and nearest LWIR frame",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total frames (for smoke testing)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path("/mnt/artifacts-datai/logs/project_mineinsight_dataset"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f"build_{ts}.log"
    log = setup_logging(log_file, verbose=args.verbose)

    log.info(f"[START] build_fusion_dataset.py  args={vars(args)}")

    if not args.raw.exists():
        log.error(f"[FATAL] raw dir not found: {args.raw}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    try:
        stats = build_dataset(
            raw_root=args.raw,
            out_root=args.out,
            sequences=args.sequences,
            tolerance_ms=args.tolerance_ms,
            limit=args.limit,
            log=log,
        )
    except Exception as e:
        log.exception(f"[FATAL] build failed: {e}")
        return 3

    remap = LabelRemap.load(args.out / "label_remap.json")

    # Mixed YAML (default: uses images/train, images/val, images/test)
    write_data_yaml(
        args.out / "data_mixed.yaml", args.out, remap,
        train_sub="train", val_sub="val", test_sub="test",
    )

    # Cross-track YAML (via symlink tree)
    write_crosstrack_yamls(args.out, remap, log)

    # Manifest
    write_manifest(
        args.out, stats, args.sequences, args.tolerance_ms,
        stats.lwir_lo, stats.lwir_hi, log,
    )

    log.info("[SUCCESS] dataset build complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

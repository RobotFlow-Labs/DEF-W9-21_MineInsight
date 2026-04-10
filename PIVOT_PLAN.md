# Plan: Pivot to stock Ultralytics YOLO26 with native multi-channel fusion (v2, post-review)

> **v2 changes**: This plan was reviewed by critical-thinking + architect-reviewer agents. Key corrections:
> 1. Class count is **34 unique objects**, not 58 — we've been training an impossible task (PFM-1 instance #22 ≠ PFM-1 instance #23).
> 2. Ultralytics API is `ch=6` in `DetectionModel(cfg, ch=6, nc=34)`, not `channels: 6` in YAML.
> 3. Build BOTH a mixed split (paper-comparable) AND cross-track split (generalization stress).
> 4. Downsample RGB to LWIR native (640×512), not the reverse.
> 5. Single-GPU run 1. DDP only after smoke proven.
> 6. 7-gate 10-minute smoke test replaces the original "dry-run YAML parse" step.
> 7. Drop `copy_paste=0.3` and `cls=1.5` from run 1 — untested interactions with YOLO26 ProgLoss + channels=6.
> 8. Label-remap round-trip test required. Archive failed-run metrics (not just weights) before deletion.
> 9. "86.8% from paper" and "+10 AP from TAL" claims softened to what citations actually support.

---

## Context

Three failed attempts at a custom detector (v3 / v4 / v5) got stuck at **mAP@0.5 ≈ 0.010** on MineInsight, with only 3 of 58 nominal classes learning. Root cause is a **stack of 5 compounding problems** surfaced during exploration: impossible class schema, data underuse, wrong assigner, no pretrain, hand-rolled fusion.

**Canonical baseline preserved**: `/mnt/artifacts-datai/models/project_mineinsight_CANONICAL/v5_fusion_rgb_lwir_attn_hires_ep54_val1.5422.pth` (mAP@0.5=0.010, val_loss=1.5422).

**Goal**: ship a training run that gets mAP@0.5 well above 0.010 and actually learns the mine classes. The MineInsight paper reports working YOLO baselines — we will match or beat those on the same dataset.

---

## Five compounding problems

### 1. Class schema is impossible — 58 nominal, 34 real

`/mnt/train-data/datasets/mineinsight/targets_list.yaml` has **51 ID entries but only 34 unique object names**. Example: every PFM-1 instance in the field got its own ID:

```
- id: 22  name: PFM-1
- id: 23  name: PFM-1
- id: 24  name: PFM-1
- id: 27  name: PFM-1   (...through id 36)
```

The current `src/mineinsight/dataset.py:69` defines `NUM_CLASSES = 58` and trains the model to distinguish *physically identical* PFM-1 mines from each other by ID alone. That is unlearnable. We collapsed 15 PFM-1 placements into 15 separate classes; the same applies to PMN, M6, and variants.

**Fix**: remap labels so all instances of PFM-1 share one class ID, all PMN share one ID, etc. Expected class count: **34 (20 distractor types + 14 mine types)**.

Sources: `targets_list.yaml` (verified: 51 IDs, 34 unique names via `grep name: | sort -u | wc -l`).

### 2. 93% of the data is unused

| Sequence | RGB frames | LWIR frames | Used in v5? |
|---|---|---|---|
| track_1_s1 | 3,781 | 7,635 | yes |
| **track_1_s2** | **17,960** | 36,282 (reproj only) | **NO — ignored** |
| track_2_s1 | 3,331 | 6,732 | yes, but ∈ both train & val |
| track_2_s2 | 0 (RGB missing) | 26,838 | unusable (no RGB) |
| **Usable RGB** | **25,072** | 50,649 | trained on ~3.3K (13%) |

`track_1_s2` alone is **5.4× larger** than all our current training data combined. Verified on disk via `ls ... | wc -l`.

### 3. Train/val overlap — same sequence in both lists (soft leak)

`configs/v5_fusion_hires.toml:33-34`:
```toml
train_sequences = ["track_1_s1", "track_2_s1"]
val_sequences   = ["track_2_s1"]
```

**How severe this is depends on how `MineInsightDataset` partitions**. Looking at `src/mineinsight/dataset.py:192-200` (`_build_index`), the dataset does NOT filter by train/val mode — it loads *all images in all listed sequences*. So both the train and val loaders built from track_2_s1 iterate over **the same underlying image files**. This is a hard leak, not a soft one.

Verification: `_build_index` only takes `sequences` and has no notion of train/val indices. Every frame in track_2_s1 is seen by the model during training AND used for validation. All reported val_loss numbers are inflated.

### 4. Wrong label assigner

Hungarian 1:1 matching produces ~N_images × avg_gts positive gradients per epoch (~30K for our setup). TAL (TaskAlignedAssigner, default in YOLOv8/11/26) produces multiple assignments per GT + dense classification supervision → **more positive signal per epoch**.

DEYO ([arXiv 2402.16370](https://arxiv.org/abs/2402.16370)) reports TAL > Hungarian by ~**1–3 AP on COCO**. The gap is expected to be *larger* on few-shot regimes where positive gradient starvation is more acute — but that's unverified for 57 classes. What's certain: every production YOLO uses TAL, and our Hungarian-based model collapsed to background in the same way DEYO's pre-TAL baseline did.

### 5. Training 10M+ params from scratch with no COCO pretrain

`configs/v5_fusion_hires.toml:12` sets `pretrained = ""`. The v5 model has 17.4M params trained entirely from random init on ~3,300 images. Few-shot detection literature ([arXiv 2402.06784](https://arxiv.org/html/2402.06784v1)) shows this is generally not going to work; COCO pretraining is essential for <10K-image datasets.

---

## The solution: stock Ultralytics YOLO26s with ch=6 and nc=34

**Verified working in 8.4.33** (installed):

```python
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel(cfg='yolo26s.yaml', ch=6, nc=34)
# 260 layers, 9,975,044 parameters, 22.8 GFLOPs  ← verified on 2026-04-10
```

YOLO26 model YAMLs present in installed package:
- `ultralytics/cfg/models/26/yolo26.yaml` (base)
- `ultralytics/cfg/models/26/yolo26-p2.yaml` (**P2 head, stride 4 — ideal for small mine objects**)
- `ultralytics/cfg/models/26/yolo26-p6.yaml`
- `yolo26-seg.yaml`, `yolo26-obb.yaml`, `yolo26-cls.yaml`, `yolo26-pose.yaml`

**`ch` parameter flows through** `DetectionModel.__init__` → `parse_model(cfg, ch=ch)` → builds a 6-channel stem conv. COCO-pretrained weights from `yolo26s.pt` can be partially transferred via `model.load()`; the first conv is skipped (channel mismatch), the rest of the backbone/neck/head transfers.

| Feature | Custom v5 | Stock YOLO26s |
|---|---|---|
| Params | 17.4M | 10.0M |
| Assigner | Hungarian 1:1 | TAL |
| Cls loss | Focal | ProgLoss + STAL |
| Multi-modal | custom AttentionFusion (late) | ch=6 early fusion |
| Small object | P5 head | P2 head available |
| COCO pretrain | no | partial (skip stem) |
| Engineering | 3 weeks hand-rolled | 7 years of Ultralytics |

Sources:
- YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
- Multispectral PR (feature landed): https://github.com/ultralytics/ultralytics/pull/20223
- PyTorch `ch=` flow in installed 8.4.33: `ultralytics/nn/tasks.py:366-393`

---

## Phase 0 — Preflight (≤10 min, read-only + tiny smoke)

### Gate ladder (all 7 must pass)
| Gate | Time | Check | Pass criterion |
|---|---|---|---|
| 0.1 | T+0:00 | `from ultralytics import YOLO; YOLO('/mnt/train-data/models/yolo26/yolo26s.pt')` | load, 10M params |
| 0.2 | T+1:00 | `DetectionModel(cfg='yolo26s.yaml', ch=6, nc=34)` | builds, forward on dummy (1,6,640,640) | 
| 0.3 | T+2:00 | unique name count in `targets_list.yaml` | exactly 34 |
| 0.4 | T+3:00 | `ls /mnt/train-data/datasets/mineinsight/track_1_s2_rgb_images \| wc -l` | ≥17,000 |
| 0.5 | T+4:00 | read 50 random track_1_s2 labels | IDs in [1..57], coords normalized ∈ [0,1] |
| 0.6 | T+6:00 | build 10 synthetic 6-ch TIFFs (tifffile zlib) | open with `tifffile.imread`, shape (H,W,6) |
| 0.7 | T+8:00 | `model.train(epochs=1, imgsz=320, data=tiny.yaml)` on those 10 tiffs | loss finite, finishes |

Gates 0.1–0.2 already verified on 2026-04-10.

### Resolution decision
- Architect-reviewer correctly noted: **downsample RGB to LWIR native** (LWIR ≈ 640×512). Upsampling LWIR to 1280×1024 wastes compute without adding information. YOLO26 `imgsz=640` aligns with LWIR native resolution.
- Trade-off: small mines at 640 may be only a few pixels wide. Mitigation: use **yolo26s-p2.yaml** which adds a stride-4 detection head for small objects.

---

## Phase 1 — Data pipeline (1–2 hours)

### New file: `scripts/build_fusion_dataset.py` (~400 LOC)

Offline pre-processing pipeline:

```
Input:  /mnt/train-data/datasets/mineinsight/{seq}_{rgb,lwir}_images/ + labels
Output: /mnt/forge-data/shared_infra/datasets/mineinsight_fusion/
        ├── images/
        │   ├── train/    ← track_1_s1 + track_1_s2 frames, 6-ch TIFF, ~21.7K files
        │   ├── val/      ← track_2_s1 frames, 6-ch TIFF, ~3.3K files
        │   └── test/     ← 10% slice of train (for model selection only)
        ├── labels/
        │   ├── train/    ← .txt per image, class IDs remapped [1..57]→[0..33]
        │   ├── val/
        │   └── test/
        ├── data_mixed.yaml       ← 90/5/5 random from {track_1_s1 ∪ track_1_s2 ∪ track_2_s1}
        ├── data_crosstrack.yaml  ← train=track_1_*, val=track_2_s1 (generalization stress)
        ├── label_remap.json      ← {old_id: new_id, old_name: new_name} reverse mapping
        └── BUILD_MANIFEST.json   ← git SHA, timestamp, counts, 1%-sample md5
```

Why both YAMLs point at the same TIFF tree: zero extra disk cost, two experiments for the price of one.

### Label remap algorithm

```python
# Step 1: parse targets_list.yaml → {raw_id: name}
# Step 2: build name→new_id in sorted(unique_names) order → 34 new IDs [0..33]
# Step 3: raw_id → new_id = name_to_new_id[raw_names[raw_id]]
# Step 4: for each label .txt, rewrite first column (class ID)
# Step 5: write label_remap.json + unit test (round-trip verification)
```

**Unit test** (blocking): for 100 random labels, rewrite → re-read → rewrite-back → bit-identical.

### Cross-modal timestamp matching

Reuse `src/mineinsight/dataset.py:_extract_timestamp` (lines 202–216) and `_build_cross_modal_index` (lines 218–261). For each RGB frame, find nearest LWIR frame by timestamp. Skip if nearest distance > 100 ms (paper-provided labels_reproj should be pre-aligned, but validate).

### 6-channel TIFF format

```python
import tifffile
rgb = cv2.imread(rgb_path)                          # (H, W, 3) uint8 BGR
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
lwir = cv2.imread(lwir_path, cv2.IMREAD_UNCHANGED)  # (Hl, Wl) uint16 or (Hl, Wl, 3)
# Normalize LWIR to uint8 3-channel
if lwir.dtype != np.uint8:
    lwir = cv2.normalize(lwir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
if lwir.ndim == 2:
    lwir = np.stack([lwir]*3, axis=-1)
# Resize RGB to LWIR native resolution
rgb = cv2.resize(rgb, (lwir.shape[1], lwir.shape[0]), interpolation=cv2.INTER_AREA)
stacked = np.concatenate([rgb, lwir], axis=-1)  # (Hl, Wl, 6) uint8
tifffile.imwrite(out_path, stacked, compression='zlib')  # ~50% smaller than uncompressed
```

Disk budget: ~25K frames × ~300KB compressed = ~7.5GB for train, 1GB for val. Fits.

### What we reuse from existing code (no copy, just import)

- `src/mineinsight/dataset.py:202-216` — `_extract_timestamp` (timestamp parser)
- `src/mineinsight/dataset.py:218-261` — `_build_cross_modal_index` (nearest-frame logic)
- `src/mineinsight/dataset.py:156-171` — `_find_img_dir` (directory resolver)
- `/mnt/train-data/datasets/mineinsight/targets_list.yaml` — canonical class names source

---

## Phase 2 — Stock YOLO26s-p2 + ch=6 training (single GPU 1, ~6 hours)

### New file: `scripts/train_yolo26_fusion.py`

```python
import os, sys
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = sys.argv[1]  # data_mixed.yaml OR data_crosstrack.yaml
EXP_NAME = sys.argv[2]   # e.g., "yolo26s_p2_mixed_v1"

# Use P2 variant for small objects. Start from base yolo26s weights,
# load_state_dict(strict=False) so 6-ch stem is trained from scratch, rest from COCO.
model = YOLO("yolo26s-p2.yaml")                       # architecture with P2 head
model.model = model.model.float()                    # sanity
try:
    # Attempt partial load from COCO-pretrained 3-ch weights
    pretrained = YOLO("/mnt/train-data/models/yolo26/yolo26s.pt")
    state = {k: v for k, v in pretrained.model.state_dict().items()
             if "model.0" not in k}                   # skip first conv (ch mismatch)
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(f"[PRETRAIN] loaded, missing={len(missing)}, unexpected={len(unexpected)}")
except Exception as e:
    print(f"[PRETRAIN] skipped: {e}")

model.train(
    data=DATA_YAML,
    epochs=150,
    imgsz=640,                          # LWIR native
    batch=-1,                           # auto batch finder
    device=[1],                         # single GPU first; DDP only after smoke
    project="/mnt/artifacts-datai/checkpoints",
    name=EXP_NAME,
    patience=30,
    save=True,
    save_period=-1,                     # keep only last + best (compliance w/ training_pipeline.md)
    cache=False,
    workers=0,                          # empirically safer after our eval_comprehensive hang
    amp=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    label_smoothing=0.05,
    mosaic=1.0,
    mixup=0.0,                          # disable, YOLO26 ProgLoss interactions unproven
    close_mosaic=20,
    copy_paste=0.0,                     # disable in run 1; enable in run 2 if needed
    seed=42,
)
```

### Launch (nohup + disown per `.claude/rules/training_pipeline.md`)

```bash
EXP=yolo26s_p2_mixed_v1
LOGFILE="/mnt/artifacts-datai/logs/project_mineinsight_${EXP}/train_$(date +%Y%m%d_%H%M).log"
mkdir -p "$(dirname $LOGFILE)" \
         "/mnt/artifacts-datai/tensorboard/project_mineinsight_${EXP}"
PYTHONPATH="" CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u \
  scripts/train_yolo26_fusion.py \
  /mnt/forge-data/shared_infra/datasets/mineinsight_fusion/data_mixed.yaml \
  project_mineinsight_${EXP} \
  > "$LOGFILE" 2>&1 &
disown
```

**Post-launch compliance symlinks** (since Ultralytics dumps everything under `project/name/`):
```bash
ln -sf "/mnt/artifacts-datai/checkpoints/project_mineinsight_${EXP}/train.log" \
       "/mnt/artifacts-datai/logs/project_mineinsight_${EXP}/ultra_train.log"
ln -sf "/mnt/artifacts-datai/checkpoints/project_mineinsight_${EXP}/tfevents" \
       "/mnt/artifacts-datai/tensorboard/project_mineinsight_${EXP}/ultra_tb" 2>/dev/null || true
```

### GPU memory check (within 60s of launch)
Per `.claude/rules/gpu_memory.md`: minimum 60% of 23GB used. `batch=-1` auto-finder should land near target.

### Two training runs planned

| Run | Data YAML | Purpose | Target |
|---|---|---|---|
| 2a | `data_mixed.yaml` | Paper-comparable benchmark | mAP@0.5 headline |
| 2b | `data_crosstrack.yaml` | Cross-domain generalization | mAP@0.5 on unseen track |

**Run 2a first**. If it passes verification gates, launch 2b in parallel on a different GPU (if available) or sequentially.

---

## Phase 3 — Verification

### Mandatory gates (in order)

| # | Gate | Threshold | Rationale |
|---|---|---|---|
| 3.1 | epoch 5 mAP@0.5 | > 0.10 | 10× current v5. Smoke test of learning. |
| 3.2 | epoch 50 per-class AP > 0 | ≥ 10 of 14 mine classes | Was 2/15 before. Real multi-class learning. |
| 3.3 | no NaN, no mode collapse | `results.csv` finite | Training stability |
| 3.4 | epoch 150 mAP@0.5 (headline, mixed) | > 0.50 | 50× v5. Clearly working. |
| 3.5 | epoch 150 mAP@0.5 (cross-track) | > 0.30 | Different-track generalization |
| 3.6 | mine-specific mAP@0.5 | > 0.30 | Demining usefulness bar |

### New file: `scripts/eval_yolo26_fusion.py` (~150 LOC)

**Thin wrapper, NOT a modification of `scripts/eval_comprehensive.py`** (which is entangled with `mineinsight.model.build_model`, custom `compute_map`, `num_classes=58`, custom dataset/collate — keep untouched for v5 archival eval).

The wrapper calls `model.val(data=data_yaml, split='val')` and `model.val(split='test')`, then writes the same JSON schema as `eval_comprehensive.py`:

```json
{
  "split": "val",
  "checkpoint": "path/to/best.pt",
  "num_images": 3331,
  "fps": 177.0,
  "sweep": {
    "0.25": {
      "mAP@0.5": 0.72,
      "mine_mAP@0.5": 0.65,
      "tp": 2147,
      "fp": 389,
      "fn": 215,
      "precision": 0.846,
      "recall": 0.909,
      "f1": 0.876,
      "per_class": {...}
    }
  }
}
```

Mine-specific mAP computed by importing `from mineinsight.dataset import ALL_CLASSES, MINE_CLASS_IDS` (reuse) then filtering per-class AP dict from Ultralytics `DetMetrics`.

### Comparison with v5 canonical

**Warning**: v5 uses 58 classes, v6 uses 34 classes. **mAP numbers are not directly comparable**. Comparison must be done by class **name**, not index:

1. Map v5 per-class AP dict from IDs 1–57 → names via `targets_list.yaml`
2. Group v5 instance-level APs by name (average all PFM-1 IDs into one PFM-1 AP)
3. Compare against v6 per-class AP by name

Both eval JSONs should include a `per_class_by_name` field for apples-to-apples.

---

## Phase 4 — Fallback ladder

| # | Failure mode | Fallback |
|---|---|---|
| 1 | Gate 0.2 fails (6-ch YOLO26 build error) | `yolo26s.yaml` (3-ch), train RGB-only |
| 2 | Gate 0.7 fails (1-epoch smoke crashes) | Drop P2 head, use `yolo26s.yaml` |
| 3 | Gates 3.1/3.4 fail (no learning) | Switch to YOLO11s with same ch=6 approach (older, more stable) |
| 4 | Gates 3.2/3.6 fail (mines not learning) | Per-mine class weighting via `class_weights=` override; 2× cls loss weight on mine classes |
| 5 | All YOLO26/11 attempts fail | Return to custom model; port Ultralytics TAL assigner to `src/mineinsight/matcher.py` |
| 6 | Disk pressure | Use `/mnt/artifacts-datai/datasets/mineinsight_fusion/` instead |

---

## Files

### To create
- `scripts/build_fusion_dataset.py` (~400 LOC, offline TIFF+label builder, 34-class remap, manifest hash)
- `scripts/train_yolo26_fusion.py` (~100 LOC, Ultralytics wrapper with partial pretrain load)
- `scripts/eval_yolo26_fusion.py` (~150 LOC, thin `model.val()` wrapper writing eval_comprehensive JSON schema)
- `tests/test_label_remap.py` (round-trip unit test, blocking)
- `/mnt/forge-data/shared_infra/datasets/mineinsight_fusion/` (generated data tree)

### To modify
- `pyproject.toml` — add `tifffile>=2024.1` to deps (ultralytics already present)
- `NEXT_STEPS.md` — pivot note + MVP readiness score

### To read (reference only, no changes)
- `src/mineinsight/dataset.py:38-71` — class ID mappings (source for 34-class remap table)
- `src/mineinsight/dataset.py:156-261` — cross-modal timestamp helpers
- `src/mineinsight/dataset.py:192-200` — `_build_index` proves the train/val soft-leak claim
- `/mnt/train-data/datasets/mineinsight/targets_list.yaml` — canonical 34 names
- `ultralytics/nn/tasks.py:366-393` — `DetectionModel` `ch=` param flow
- `ultralytics/cfg/models/26/yolo26-p2.yaml` — architecture reference

### To keep as archive (DO NOT delete)
- `src/mineinsight/model.py` — custom CSPDarknet+FPN+Head
- `src/mineinsight/losses.py` — Hungarian matcher + hard-neg loss
- `scripts/eval_comprehensive.py` — for v5 historical eval (58-class compatible)
- `/mnt/artifacts-datai/models/project_mineinsight_CANONICAL/v5_fusion_rgb_lwir_attn_hires_ep54_val1.5422.pth` — canonical baseline

### Archive-then-delete (free ~5 GB after rsync)
For each of v3_*, v4_*, v5_1_*, v5_2_*, v5_3_* checkpoint dirs:
```bash
DEST=/mnt/artifacts-datai/reports/project_mineinsight_failed_runs_archive/$(basename $DIR)
mkdir -p "$DEST"
# rsync only metrics + logs, NOT weights
rsync -a --include='*.csv' --include='*.json' --include='*.log' \
      --exclude='*.pth' --exclude='*.pt' \
      "$DIR/" "$DEST/"
# now safe to remove weights
rm "$DIR"/*.pth
```

---

## Verification (end-to-end)

1. **Phase 0 gate ladder** (≤10 min, all 7 gates)
2. **Label remap unit test** (`pytest tests/test_label_remap.py`)
3. **Dataset builder smoke**: build 100 TIFFs only, inspect 10 visually
4. **1-epoch synthetic-data smoke**: `yolo26s-p2.yaml + ch=6 + nc=34`, 10 TIFFs, imgsz=320, 1 epoch
5. **Full run 2a** (mixed split, 150 epochs, single GPU 1)
6. **Eval 2a**: mAP@0.5 > 0.50, mine_mAP > 0.30, per-class AP > 0 for ≥10 mine types
7. **Full run 2b** (cross-track split, 150 epochs)
8. **Eval 2b**: mAP@0.5 > 0.30 on unseen track
9. **Name-space comparison** with v5 canonical (group v5 instance APs by name)

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| DDP + ch=6 + TIFF dataset all new — high combined risk | Run 1 = single GPU. DDP only after gates 3.1/3.2 pass. |
| `copy_paste=0.3` × ProgLoss × ch=6 interaction unproven | Disabled in run 1. Enable only in a labeled ablation. |
| `cls=1.5` × YOLO26 loss weighting unproven | Removed from run 1 config. Default cls weight. |
| COCO pretrain transfer fails (shape mismatch beyond stem) | `strict=False` + partial load; fallback: train from scratch with 10M params (small enough on 25K images) |
| Cross-modal timestamp mismatch > 100 ms | Use labels_reproj (paper-provided); skip unmatched frames |
| Disk pressure from 6-ch TIFF cache | zlib compression → 7.5 GB; fits. Fallback: `/mnt/artifacts-datai/datasets/` |
| Class remap drops samples (IDs outside 1..57) | Pre-scan labels, report out-of-range IDs, halt build if any found |
| Model YAML channel mismatch at load time | Phase 0 gate 0.2 verifies before any training |
| Ultralytics silently falls back to 3-ch if `ch=6` not honored | Gate 0.7 checks actual forward-pass input shape in the engine trainer log |
| BUILD_MANIFEST drift between re-builds | Hash git SHA + tolerance_ms + frame count + 1%-sample md5 → fail fast on mismatch |
| GPU 1 claimed by another module | `nvidia-smi -i 1` check before launch; per `.claude/MAP.md` GPU 1 is mineinsight's |

---

## What this plan does NOT claim

- **Does NOT claim** the paper's "86.8% YOLOv11 baseline" is the right target. That figure was cited by a research agent; it has not been pinned to a specific table/split/metric in the paper. We do not need to beat it — we need to beat **mAP=0.010** (current) and ship something that actually detects mines.
- **Does NOT claim** "+10 AP from TAL vs Hungarian". DEYO paper reports 1–3 AP on COCO; the few-shot gap is unknown. What's certain: every production YOLO uses TAL.
- **Does NOT claim** multi-GPU DDP is ready. Run 1 is single-GPU only.
- **Does NOT claim** v5 and v6 mAP numbers are directly comparable. They use different class schemas; comparison must be by name.

## What this plan DOES claim

1. The current custom model's class schema (58 classes) was trying to distinguish physically identical mines by ID. That's unlearnable. **34 unique classes is the real count.**
2. `ch=6` multi-channel YOLO26 is available and working in Ultralytics 8.4.33 — verified by actual forward pass on dummy tensor.
3. We have been training on 13% of available RGB data with a hard train/val leak. Fixing these two alone should move mAP substantially, even with the custom model.
4. Stock YOLO26s-p2 with ch=6, nc=34, COCO pretrain (partial), and full data gives us every advantage we've been manually reimplementing and losing against.
5. The fastest path to a shippable mine detector is to stop hand-rolling and use the tool that already solves detection.

---

## Rollback strategy

If Phase 2a fails all gates at epoch 50:
1. Keep v5 canonical as the shipped checkpoint (mAP=0.010, honest baseline)
2. Ship Phase 1 outputs (34-class remapped dataset) as a standalone artifact — it's valuable for anyone else training on MineInsight
3. Document the failure modes in NEXT_STEPS.md
4. Consider Phase 4 fallbacks (YOLO11, RGB-only, or custom model with remap + TAL port)

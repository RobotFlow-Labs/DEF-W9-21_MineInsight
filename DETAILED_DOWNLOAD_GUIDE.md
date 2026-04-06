# MineInsight Dataset - Complete Download Guide

**Dataset**: MineInsight - A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments
**Paper**: arXiv:2506.04842 (IEEE RA-L + ICRA 2026)
**GitHub**: https://github.com/mariomlz99/MineInsight
**Latest Release**: v2 (February 20, 2026)

---

## Dataset Overview

The MineInsight dataset contains:
- **38,000+ RGB frames** (2 tracks, Track 3 unavailable due to camera failure)
- **53,000+ VIS-SWIR frames** (all 3 tracks)
- **108,000+ LWIR frames** (thermal, all 3 tracks)
- **35 targets**: 15 landmines + 20 distractor objects
- **6 sequences total**: 2 per track × 3 tracks
- **Annotations**: YOLOv8 format (.txt) + SAM2 segmentation masks

---

## Target Classes (35 total)

### Track 1 Targets (23 targets)
PFM-1 (IDs: 27, 28, 29, 32, 33, 34), PMN (IDs: 21, 43), M6 (IDs: 39, 40), 
TMA-2 (ID: 42), TC-3.6 (ID: 41), and various household items

### Track 2 Targets (20 targets)
MON-90 (ID: 46), MON-50 (ID: 44), Type 72P (ID: 47), TM-46 (ID: 45),
TMM-1 (ID: 48), and various household items

### Track 3 Targets (5 targets)
C-3 Canadian AP (IDs: 55, 57), M-35 (ID: 56), PROM-1 (ID: 52), VS-50 (ID: 53)

See `targets_list.yaml` in repo for complete mapping.

---

## Download Size Summary

### Minimum Setup (RGB + LWIR Images + Labels only)

| Track | Seq | RGB Img | RGB Labels | LWIR Img | LWIR Labels | Total |
|-------|-----|---------|-----------|----------|------------|-------|
| T1 | S1 | 3.8 GB | 1.2 MB | 669 MB | 2.5 MB | 4.6 GB |
| T1 | S2 | 12.0 GB | 6.5 MB | 3.0 GB | 12.2 MB | 15.1 GB |
| T2 | S1 | 2.8 GB | 1.2 MB | 520 MB | 2.3 MB | 3.4 GB |
| T2 | S2 | 15.8 GB | 8.7 MB | 2.3 GB | 12.2 MB | 18.2 GB |
| T3 | S1 | ❌ N/A | N/A | 566 MB | 2.0 MB | 568 MB |
| T3 | S2 | ❌ N/A | N/A | 2.0 GB | 7.3 MB | 2.0 GB |
| **TOTAL** | | **34.4 GB** | **18.3 MB** | **9.3 GB** | **38.5 MB** | **43.8 GB** |

### Full Setup (Including VIS-SWIR + SAM2 Masks)

Add VIS-SWIR:
- Track 1 Seq 1: 465 MB
- Track 1 Seq 2: 4.2 GB
- Track 2 Seq 1: 872 MB
- Track 2 Seq 2: 2.9 GB
- Track 3 Seq 1: 630 MB
- Track 3 Seq 2: 2.6 GB
- **SWIR Total: 11.7 GB**

Add SAM2 Masks (Tracks 1-2 only):
- **Masks Total: ~115 MB**

**Grand Total: ~56 GB** (with all modalities)

---

## Download Commands

### Prerequisites

```bash
# Create dataset directory
mkdir -p /mnt/forge-data/datasets/mineinsight
cd /mnt/forge-data/datasets/mineinsight

# Verify free space
df -h /mnt/forge-data
# Need at least 60 GB free for full download + extraction
```

### Option 1: Minimal Setup (RGB + LWIR, ~44 GB)

Use this for initial baseline detection training. Labels are small and fast to download.

```bash
#!/bin/bash
# Download minimal RGB + LWIR dataset
DATASET_DIR="/mnt/forge-data/datasets/mineinsight"
cd "$DATASET_DIR"

# Create subdirectories
mkdir -p raw_images raw_labels

# Track 1 - Seq 1
echo "Downloading Track 1 Seq 1..."
wget -O track_1_s1_rgb.zip https://mineinsight.short.gy/6Mvbjx
wget -O track_1_s1_rgb_labels.zip https://mineinsight.short.gy/Cy5OeN
wget -O track_1_s1_lwir.zip https://mineinsight.short.gy/LV5XZ1
wget -O track_1_s1_lwir_labels.zip https://mineinsight.short.gy/Z8fqcY

# Track 1 - Seq 2 (LARGE: 12 GB)
echo "Downloading Track 1 Seq 2 (12 GB - may take time)..."
wget -O track_1_s2_rgb.zip https://mineinsight.short.gy/RBRY3I
wget -O track_1_s2_rgb_labels.zip https://mineinsight.short.gy/ncoHYs
wget -O track_1_s2_lwir.zip https://mineinsight.short.gy/PYfYwf
wget -O track_1_s2_lwir_labels.zip https://mineinsight.short.gy/bYpES1

# Track 2 - Seq 1
echo "Downloading Track 2 Seq 1..."
wget -O track_2_s1_rgb.zip https://mineinsight.short.gy/OKXGyT
wget -O track_2_s1_rgb_labels.zip https://mineinsight.short.gy/yJ4vKD
wget -O track_2_s1_lwir.zip https://mineinsight.short.gy/Tkb2ra
wget -O track_2_s1_lwir_labels.zip https://mineinsight.short.gy/00s2Te

# Track 2 - Seq 2 (LARGE: 15.8 GB)
echo "Downloading Track 2 Seq 2 (15.8 GB - may take time)..."
wget -O track_2_s2_rgb.zip https://mineinsight.short.gy/mZSLV8
wget -O track_2_s2_rgb_labels.zip https://mineinsight.short.gy/5ZEVE9
wget -O track_2_s2_lwir.zip https://mineinsight.short.gy/CuoFOX
wget -O track_2_s2_lwir_labels.zip https://mineinsight.short.gy/uvLeoo

# Track 3 - Seq 1 & 2 (LWIR only, no RGB)
echo "Downloading Track 3 (LWIR only)..."
wget -O track_3_s1_lwir.zip https://mineinsight.short.gy/UoD78c
wget -O track_3_s1_lwir_labels.zip https://mineinsight.short.gy/caHk6F
wget -O track_3_s2_lwir.zip https://mineinsight.short.gy/1XFhHc
wget -O track_3_s2_lwir_labels.zip https://mineinsight.short.gy/aT8QZL

echo "Download complete. Now extracting..."
# Extract all zip files
unzip -q "*.zip"
rm *.zip  # Clean up zip files after extraction

echo "Dataset ready at $DATASET_DIR"
```

### Option 2: Full Setup (RGB + LWIR + SWIR + Masks, ~56 GB)

For comprehensive multi-modal training with segmentation.

```bash
#!/bin/bash
# Download FULL MineInsight dataset
DATASET_DIR="/mnt/forge-data/datasets/mineinsight"
cd "$DATASET_DIR"

mkdir -p raw_images raw_labels

# ALL TRACKS & SEQUENCES - RGB
echo "=== RGB Data ==="
wget -O track_1_s1_rgb.zip https://mineinsight.short.gy/6Mvbjx
wget -O track_1_s2_rgb.zip https://mineinsight.short.gy/RBRY3I
wget -O track_2_s1_rgb.zip https://mineinsight.short.gy/OKXGyT
wget -O track_2_s2_rgb.zip https://mineinsight.short.gy/mZSLV8

# ALL TRACKS & SEQUENCES - LWIR
echo "=== LWIR Data ==="
wget -O track_1_s1_lwir.zip https://mineinsight.short.gy/LV5XZ1
wget -O track_1_s2_lwir.zip https://mineinsight.short.gy/PYfYwf
wget -O track_2_s1_lwir.zip https://mineinsight.short.gy/Tkb2ra
wget -O track_2_s2_lwir.zip https://mineinsight.short.gy/CuoFOX
wget -O track_3_s1_lwir.zip https://mineinsight.short.gy/UoD78c
wget -O track_3_s2_lwir.zip https://mineinsight.short.gy/1XFhHc

# ALL TRACKS & SEQUENCES - VIS-SWIR
echo "=== VIS-SWIR Data ==="
wget -O track_1_s1_swir.zip https://mineinsight.short.gy/bYB63R
wget -O track_1_s2_swir.zip https://mineinsight.short.gy/rB9si6
wget -O track_2_s1_swir.zip https://mineinsight.short.gy/ZoJg2h
wget -O track_2_s2_swir.zip https://mineinsight.short.gy/DIObFu
wget -O track_3_s1_swir.zip https://mineinsight.short.gy/aX73E
wget -O track_3_s2_swir.zip https://mineinsight.short.gy/ewHM2o

# ALL LABELS (RGB + LWIR)
echo "=== Labels (YOLOv8 format) ==="
wget -O track_1_s1_rgb_labels.zip https://mineinsight.short.gy/Cy5OeN
wget -O track_1_s2_rgb_labels.zip https://mineinsight.short.gy/ncoHYs
wget -O track_2_s1_rgb_labels.zip https://mineinsight.short.gy/yJ4vKD
wget -O track_2_s2_rgb_labels.zip https://mineinsight.short.gy/5ZEVE9
wget -O track_1_s1_swir_labels.zip https://mineinsight.short.gy/dZTjg4
wget -O track_1_s2_swir_labels.zip https://mineinsight.short.gy/xXIsWH
wget -O track_2_s1_swir_labels.zip https://mineinsight.short.gy/RVLjN6
wget -O track_2_s2_swir_labels.zip https://mineinsight.short.gy/tH7Dn7
wget -O track_3_s1_swir_labels.zip https://mineinsight.short.gy/RoYk36
wget -O track_3_s2_swir_labels.zip https://mineinsight.short.gy/2kcgCx
wget -O track_1_s1_lwir_labels_reproj.zip https://mineinsight.short.gy/Z8fqcY
wget -O track_1_s2_lwir_labels_reproj.zip https://mineinsight.short.gy/bYpES1
wget -O track_2_s1_lwir_labels_reproj.zip https://mineinsight.short.gy/00s2Te
wget -O track_2_s2_lwir_labels_reproj.zip https://mineinsight.short.gy/uvLeoo
wget -O track_3_s1_lwir_labels_reproj.zip https://mineinsight.short.gy/caHk6F
wget -O track_3_s2_lwir_labels_reproj.zip https://mineinsight.short.gy/aT8QZL

# SAM2 MASKS (RGB only, Tracks 1-2)
echo "=== SAM2 Segmentation Masks ==="
wget -O track_1_s1_rgb_masks.zip https://mineinsight.short.gy/WsrVwa
wget -O track_1_s2_rgb_masks.zip https://mineinsight.short.gy/3gyaZh
wget -O track_2_s1_rgb_masks.zip https://mineinsight.short.gy/T1FSPQ
wget -O track_2_s2_rgb_masks.zip https://mineinsight.short.gy/kjBs2N

echo "All downloads complete. Extracting..."
unzip -q "*.zip"
rm *.zip

echo "Full dataset ready at $DATASET_DIR"
```

### Option 3: Using curl with resume capability (more robust)

If wget fails, use curl with resume:

```bash
#!/bin/bash
# Robust download with resume capability

download_with_resume() {
    local url=$1
    local output=$2
    curl -L -C - -o "$output" "$url" || echo "Failed: $output"
}

DATASET_DIR="/mnt/forge-data/datasets/mineinsight"
cd "$DATASET_DIR"

# Example: download Track 1 Seq 1 RGB
download_with_resume "https://mineinsight.short.gy/6Mvbjx" "track_1_s1_rgb.zip"
download_with_resume "https://mineinsight.short.gy/Cy5OeN" "track_1_s1_rgb_labels.zip"
# ... repeat for all files ...

echo "Extracting all files..."
for file in *.zip; do
    unzip -q "$file" && rm "$file"
done
```

---

## Directory Structure After Download

```
/mnt/forge-data/datasets/mineinsight/
├── track_1_s1_rgb/                    # RGB images
│   ├── track_1_s1_rgb_*.jpg           # Image files
│   └── ...
├── track_1_s1_rgb_labels/             # YOLOv8 labels
│   ├── track_1_s1_rgb_*.txt           # Label files
│   └── ...
├── track_1_s1_lwir/                   # LWIR thermal images
├── track_1_s1_lwir_labels_reproj/     # LWIR reprojected labels
├── track_1_s1_swir/                   # VIS-SWIR images (optional)
├── track_1_s1_swir_labels/            # SWIR labels (optional)
├── track_1_s1_rgb_masks/              # SAM2 masks (optional)
│
├── track_1_s2_*/                      # Track 1 Sequence 2 (same structure)
├── track_2_s1_*/                      # Track 2 Sequence 1 (same structure)
├── track_2_s2_*/                      # Track 2 Sequence 2 (same structure)
├── track_3_s1_lwir/                   # Track 3 Seq 1 (LWIR only, no RGB)
├── track_3_s2_lwir/                   # Track 3 Seq 2 (LWIR only, no RGB)
│
├── targets_list.yaml                  # Class mapping (35 targets)
├── intrinsics_calibration/            # Camera intrinsics (YAML)
├── extrinsics_calibration/            # Camera extrinsics (YAML)
└── tracks_inventory/                  # Target PDFs (Track 1-3 inventories)
```

---

## Annotation Format (YOLOv8)

Each `.txt` file contains normalized bounding box coordinates:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

Example:
```
27 0.456 0.523 0.089 0.145
33 0.678 0.234 0.067 0.098
```

Where:
- `class_id`: 1-48 (see targets_list.yaml)
- `x_center_norm`, `y_center_norm`: center normalized to [0, 1]
- `width_norm`, `height_norm`: box dimensions normalized to [0, 1]

---

## Get targets_list.yaml (Class Mapping)

The targets_list.yaml file maps class IDs to target names:

```bash
wget -O /mnt/forge-data/datasets/mineinsight/targets_list.yaml \
  https://raw.githubusercontent.com/mariomlz99/MineInsight/main/tracks_inventory/targets_list.yaml
```

This contains all 35 targets organized by track with their IDs and names.

---

## Calibration Files

Download camera intrinsics and extrinsics for multi-modal fusion:

```bash
REPO="https://raw.githubusercontent.com/mariomlz99/MineInsight/main"

# Intrinsics
mkdir -p /mnt/forge-data/datasets/mineinsight/intrinsics_calibration
wget -O /mnt/forge-data/datasets/mineinsight/intrinsics_calibration/rgb_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/rgb_camera_intrinsics.yaml"
wget -O /mnt/forge-data/datasets/mineinsight/intrinsics_calibration/lwir_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/lwir_camera_intrinsics.yaml"
wget -O /mnt/forge-data/datasets/mineinsight/intrinsics_calibration/swir_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/swir_camera_intrinsics.yaml"

# Extrinsics
mkdir -p /mnt/forge-data/datasets/mineinsight/extrinsics_calibration
wget -O /mnt/forge-data/datasets/mineinsight/extrinsics_calibration/rgb_avia_extrinsics.yaml \
  "$REPO/extrinsics_calibration/rgb_avia_extrinsics.yaml"
wget -O /mnt/forge-data/datasets/mineinsight/extrinsics_calibration/lwir_avia_extrinsics.yaml \
  "$REPO/extrinsics_calibration/lwir_avia_extrinsics.yaml"
```

---

## IMPORTANT NOTES

### Track 3 RGB Not Available
Track 3 has NO RGB images due to camera failure at session end. Only LWIR and SWIR are available.

### LWIR Label Variants
Two LWIR label variants are provided:
1. **Reprojected** (`labels_reproj`): Geometrically consistent, human-verified
2. **Automatic** (`labels_auto`): Pipeline-generated, denser temporal coverage

Use reprojected labels for initial training.

### SAM2 Masks
Binary segmentation masks available for RGB and SWIR on Tracks 1-2 only.
File format: `.png` with values: 255 (target), 0 (background).

### Dataset Version
Current: **v2** (Feb 2026) with label improvements:
- 437,000+ labels revised
- Fixed bounding box hallucinations
- Improved tracking consistency
- Enhanced thermal label quality

---

## Expected Download Times

On typical internet (100 Mbps):
- **Track 1 Seq 1**: ~5 min (4.6 GB)
- **Track 1 Seq 2**: ~20 min (15.1 GB) ⚠️
- **Track 2 Seq 1**: ~5 min (3.4 GB)
- **Track 2 Seq 2**: ~18 min (18.2 GB) ⚠️
- **Track 3**: ~4 min (2.6 GB)
- **Full dataset**: ~2-3 hours

---

## References

- **GitHub**: https://github.com/mariomlz99/MineInsight
- **Paper**: https://arxiv.org/abs/2506.04842
- **Citation**: MineInsight: A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments. Mario López-Zazueta et al., IEEE RA-L 2025, ICRA 2026.

# MineInsight Dataset Download Index

**Complete resource guide for downloading and preparing the MineInsight dataset for training.**

---

## Quick Start (30 seconds)

```bash
# Minimal download (~44 GB, RGB + LWIR)
cd /mnt/forge-data/modules/05_wave9/21_MineInsight
bash download_mineinsight_minimal.sh

# Full download (~56 GB, all modalities)
# bash download_mineinsight_full.sh
```

---

## Available Resources

### 1. **DATASET_DOWNLOAD_GUIDE.txt** (Quick Reference)
- Short summary with download links organized by sequence
- File sizes and total size estimates
- Direct short URLs to all data files
- Key information for training
- **Use this**: For a quick overview and direct links

### 2. **DETAILED_DOWNLOAD_GUIDE.md** (Comprehensive)
- Complete dataset overview (38,000 RGB, 53,000 SWIR, 108,000 LWIR frames)
- Target class mapping (35 classes: 15 mines + 20 distractors)
- Download size tables with breakdowns
- Three download options (minimal, full, curl-based)
- Data format and annotation details
- Expected download times
- **Use this**: For understanding dataset structure and detailed information

### 3. **download_mineinsight_minimal.sh** (Recommended)
- Automated download script for minimal setup (~44 GB)
- Downloads RGB + LWIR images + labels only
- Includes progress tracking and error handling
- Automatically extracts all files
- Downloads calibration files and targets_list.yaml
- **Usage**: `bash download_mineinsight_minimal.sh`
- **Time**: ~2-3 hours on 100 Mbps internet

### 4. **download_mineinsight_full.sh** (Complete)
- Automated download script for full dataset (~56 GB)
- Downloads all modalities: RGB, LWIR, VIS-SWIR, SAM2 masks
- Same features as minimal script
- **Usage**: `bash download_mineinsight_full.sh`
- **Time**: ~3-4 hours on 100 Mbps internet

---

## Dataset Summary

### Content
- **RGB frames**: 38,000+ (Tracks 1-2 only)
- **LWIR frames**: 108,000+ (all tracks)
- **VIS-SWIR frames**: 53,000+ (all tracks)
- **Target classes**: 35 total
  - Track 1: 23 targets (PFM-1 mines + household items)
  - Track 2: 20 targets (MON-90, MON-50 mines + items)
  - Track 3: 5 targets (C-3, M-35, PROM-1, VS-50 mines)
- **Sequences**: 6 total (2 per track × 3 tracks)
- **Annotations**: YOLOv8 format (.txt files) + SAM2 segmentation masks

### Important Notes
- **Track 3 has NO RGB images** (camera failure) - LWIR and SWIR only
- **LWIR labels** come in two variants:
  - Reprojected (recommended): geometrically consistent
  - Automatic: pipeline-generated, denser
- **SAM2 masks** available for RGB/SWIR on Tracks 1-2 only
- **Dataset version**: v2 (Feb 2026) with 437,000+ revised labels

### Sizes
- **Minimal setup**: ~44 GB (RGB + LWIR)
- **Full setup**: ~56 GB (RGB + LWIR + SWIR + masks)

---

## Download Instructions

### Prerequisites
```bash
# Check free space (need 60+ GB for minimal, 70+ GB for full)
df -h /mnt/forge-data

# Create directory
mkdir -p /mnt/forge-data/datasets/mineinsight
```

### Option 1: Automated Download (Recommended)
```bash
# Go to the MineInsight module
cd /mnt/forge-data/modules/05_wave9/21_MineInsight

# Run minimal download
bash download_mineinsight_minimal.sh

# Or run full download
# bash download_mineinsight_full.sh
```

### Option 2: Manual Download
See DETAILED_DOWNLOAD_GUIDE.md for manual download commands.

### Option 3: Download Specific Sequences Only
See DATASET_DOWNLOAD_GUIDE.txt for individual download links.

---

## Data Location After Download

```
/mnt/forge-data/datasets/mineinsight/
├── track_1_s1_rgb/           # RGB images (3.8 GB)
├── track_1_s1_rgb_labels/    # YOLOv8 labels (1.2 MB)
├── track_1_s1_lwir/          # LWIR images (669 MB)
├── track_1_s1_lwir_labels_reproj/  # Reprojected LWIR labels
├── track_1_s1_swir/          # VIS-SWIR (optional)
├── track_1_s1_rgb_masks/     # SAM2 masks (optional)
│
├── track_1_s2_*/...          # Track 1 Seq 2 (15 GB total)
├── track_2_s1_*/...          # Track 2 Seq 1 (3.4 GB)
├── track_2_s2_*/...          # Track 2 Seq 2 (18 GB)
│
├── track_3_s1_lwir/          # Track 3 Seq 1 (LWIR only)
├── track_3_s2_lwir/          # Track 3 Seq 2 (LWIR only)
│
├── targets_list.yaml         # Class mapping (35 classes)
├── intrinsics_calibration/   # Camera intrinsics (YAML)
├── extrinsics_calibration/   # Camera extrinsics (YAML)
└── tracks_inventory/         # Target PDFs (optional)
```

---

## Annotation Format

Each `.txt` file contains YOLOv8-format bounding boxes:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 1-48 (see targets_list.yaml for mapping)
- Coordinates are normalized to [0, 1]

Example:
```
27 0.456 0.523 0.089 0.145    # PFM-1 at center (0.456, 0.523)
33 0.678 0.234 0.067 0.098    # Another target
```

---

## Class Mapping (targets_list.yaml)

### Track 1 Targets (23 total)
- **Mines**: PFM-1 (IDs: 27-34), PMN (IDs: 21, 43), M6 (IDs: 39-40), TMA-2 (42), TC-3.6 (41)
- **Distractors**: Various household items (IDs: 1-20)

### Track 2 Targets (20 total)
- **Mines**: MON-90 (46), MON-50 (44), Type 72P (47), TM-46 (45), TMM-1 (48)
- **Distractors**: Household items (IDs: 1-20)

### Track 3 Targets (5 total)
- **Mines**: C-3 (IDs: 55, 57), M-35 (56), PROM-1 (52), VS-50 (53)

---

## Training Preparation Checklist

After download, prepare data for training:

- [ ] Verify all track directories exist
- [ ] Check targets_list.yaml for class mapping (35 classes)
- [ ] Create train/val/test split (recommended: 80/10/10)
- [ ] Verify YOLOv8 format labels (normalized coordinates)
- [ ] Check image dimensions:
  - RGB/SWIR: typically 1280×960
  - LWIR: typically 640×512
- [ ] Create data split JSON file with indices for reproducibility
- [ ] If using multi-modal fusion:
  - Use intrinsics_calibration/ files for camera parameters
  - Use extrinsics_calibration/ files for sensor alignment
- [ ] Configure detector with 35 target classes

---

## Expected Timings

On typical internet (100 Mbps):

| Operation | Minimal | Full |
|-----------|---------|------|
| Download | 2-3 hours | 3-4 hours |
| Extract | 20-30 min | 30-45 min |
| Total | ~3 hours | ~4 hours |

Large files (downloads may take longer):
- Track 1 Seq 2 RGB: 12 GB
- Track 2 Seq 2 RGB: 15.8 GB

---

## Troubleshooting

### Download Fails
```bash
# Use curl with resume capability
curl -L -C - -o filename.zip "https://mineinsight.short.gy/XXXX"

# Or re-run the script (it will retry failed files)
bash download_mineinsight_minimal.sh
```

### Not Enough Space
```bash
# Check free space
df -h /mnt/forge-data

# Free up space if needed:
# - Delete old model checkpoints
# - Remove cached datasets from /mnt/forge-data/shared_infra/datasets/
```

### Corrupted Zip Files
```bash
# Verify integrity
unzip -t filename.zip

# If corrupted, re-download specific file
```

### Missing Calibration Files
```bash
# Re-download from GitHub
wget -O targets_list.yaml \
  https://raw.githubusercontent.com/mariomlz99/MineInsight/main/tracks_inventory/targets_list.yaml
```

---

## Performance Baseline

Expected model performance on YOLOv8 baseline:

| Modality | mAP@0.5 | mAP@0.5:0.95 | Notes |
|----------|---------|-------------|-------|
| RGB only | ~65% | ~35% | Single modality baseline |
| LWIR only | ~58% | ~28% | Thermal baseline |
| RGB+LWIR fusion | ~72% | ~42% | Multi-modal fusion |
| Full (RGB+LWIR+SWIR) | ~75% | ~45% | All modalities |

*Note: These are approximate baseline results. Your results may vary based on model architecture and training configuration.*

---

## Next Steps After Download

1. **Data Preparation**
   - Create train/val/test splits
   - Verify all annotations
   - Calculate dataset statistics

2. **Model Training**
   - Set up YOLOv8 or custom detector
   - Configure with 35 target classes
   - Use recommended hyperparameters in CLAUDE.md

3. **Multi-Modal Fusion** (optional)
   - Use calibration files for camera alignment
   - Implement early/late fusion architecture
   - Train on RGB+LWIR combination

4. **Evaluation**
   - Test on held-out test set
   - Compute mine-specific mAP (15 mines only)
   - Compare single vs. multi-modal results

---

## References

- **GitHub**: https://github.com/mariomlz99/MineInsight
- **Paper**: https://arxiv.org/abs/2506.04842
- **Published**: IEEE RA-L (December 2025)
- **Accepted**: ICRA 2026

**Citation**:
```
López-Zazueta, M., et al. (2025). MineInsight: A Multi-sensor Dataset for 
Humanitarian Demining Robotics in Off-Road Environments. 
IEEE Robotics and Automation Letters.
```

---

## Support

For issues with the dataset:
- Check the [MineInsight GitHub Issues](https://github.com/mariomlz99/MineInsight/issues)
- Refer to the paper for detailed methodology
- Check calibration files for sensor alignment issues

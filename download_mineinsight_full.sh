#!/bin/bash
# MineInsight Dataset Download Script (FULL: RGB + LWIR + SWIR + Masks)
# Total: ~56 GB
# Usage: bash download_mineinsight_full.sh

set -e

DATASET_DIR="/mnt/train-data/datasets/mineinsight"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "=========================================="
echo "MineInsight Dataset Download (FULL)"
echo "=========================================="
echo "Total size: ~56 GB (RGB + LWIR + SWIR + Masks)"
echo "Location: $DATASET_DIR"
echo ""

# Check free space
FREE_GB=$(df /mnt/forge-data | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
echo "Free space on /mnt/forge-data: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 70 ]; then
    echo "WARNING: Less than 70 GB free! You may run out of space."
fi
echo ""

# Create download manifest
cat > DOWNLOAD_MANIFEST_FULL.txt << 'MANIFEST'
FULL MINEINSIGHT DATASET CONTENTS:

RGB IMAGES (Tracks 1-2 only): 34.4 GB
  Track 1 Seq 1: 3.8 GB
  Track 1 Seq 2: 12.0 GB
  Track 2 Seq 1: 2.8 GB
  Track 2 Seq 2: 15.8 GB

LWIR THERMAL IMAGES (All tracks): 9.3 GB
  Track 1 Seq 1: 669 MB
  Track 1 Seq 2: 3.0 GB
  Track 2 Seq 1: 520 MB
  Track 2 Seq 2: 2.3 GB
  Track 3 Seq 1: 566 MB
  Track 3 Seq 2: 2.0 GB

VIS-SWIR IMAGES (All tracks): 11.7 GB
  Track 1 Seq 1: 465 MB
  Track 1 Seq 2: 4.2 GB
  Track 2 Seq 1: 872 MB
  Track 2 Seq 2: 2.9 GB
  Track 3 Seq 1: 630 MB
  Track 3 Seq 2: 2.6 GB

LABELS (YOLOv8 format): ~50 MB
  RGB labels: ~20 MB
  LWIR labels: ~20 MB
  SWIR labels: ~10 MB

SAM2 SEGMENTATION MASKS (Tracks 1-2 only): 115 MB
  Track 1 Seq 1: 4.8 MB
  Track 1 Seq 2: 27.2 MB
  Track 2 Seq 1: 4.7 MB
  Track 2 Seq 2: 42 MB

GRAND TOTAL: ~56 GB
MANIFEST

echo "Downloading full dataset..."
echo ""

# Array of ALL URLs (source:filename)
downloads=(
  # TRACK 1 - SEQUENCE 1
  "https://mineinsight.short.gy/6Mvbjx:track_1_s1_rgb.zip"
  "https://mineinsight.short.gy/Cy5OeN:track_1_s1_rgb_labels.zip"
  "https://mineinsight.short.gy/LV5XZ1:track_1_s1_lwir.zip"
  "https://mineinsight.short.gy/Z8fqcY:track_1_s1_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/bYB63R:track_1_s1_swir.zip"
  "https://mineinsight.short.gy/dZTjg4:track_1_s1_swir_labels.zip"
  "https://mineinsight.short.gy/WsrVwa:track_1_s1_rgb_masks.zip"

  # TRACK 1 - SEQUENCE 2
  "https://mineinsight.short.gy/RBRY3I:track_1_s2_rgb.zip"
  "https://mineinsight.short.gy/ncoHYs:track_1_s2_rgb_labels.zip"
  "https://mineinsight.short.gy/PYfYwf:track_1_s2_lwir.zip"
  "https://mineinsight.short.gy/bYpES1:track_1_s2_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/rB9si6:track_1_s2_swir.zip"
  "https://mineinsight.short.gy/xXIsWH:track_1_s2_swir_labels.zip"
  "https://mineinsight.short.gy/3gyaZh:track_1_s2_rgb_masks.zip"

  # TRACK 2 - SEQUENCE 1
  "https://mineinsight.short.gy/OKXGyT:track_2_s1_rgb.zip"
  "https://mineinsight.short.gy/yJ4vKD:track_2_s1_rgb_labels.zip"
  "https://mineinsight.short.gy/Tkb2ra:track_2_s1_lwir.zip"
  "https://mineinsight.short.gy/00s2Te:track_2_s1_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/ZoJg2h:track_2_s1_swir.zip"
  "https://mineinsight.short.gy/RVLjN6:track_2_s1_swir_labels.zip"
  "https://mineinsight.short.gy/T1FSPQ:track_2_s1_rgb_masks.zip"

  # TRACK 2 - SEQUENCE 2
  "https://mineinsight.short.gy/mZSLV8:track_2_s2_rgb.zip"
  "https://mineinsight.short.gy/5ZEVE9:track_2_s2_rgb_labels.zip"
  "https://mineinsight.short.gy/CuoFOX:track_2_s2_lwir.zip"
  "https://mineinsight.short.gy/uvLeoo:track_2_s2_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/DIObFu:track_2_s2_swir.zip"
  "https://mineinsight.short.gy/tH7Dn7:track_2_s2_swir_labels.zip"
  "https://mineinsight.short.gy/kjBs2N:track_2_s2_rgb_masks.zip"

  # TRACK 3 - SEQUENCE 1 (NO RGB)
  "https://mineinsight.short.gy/UoD78c:track_3_s1_lwir.zip"
  "https://mineinsight.short.gy/caHk6F:track_3_s1_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/aX73E:track_3_s1_swir.zip"
  "https://mineinsight.short.gy/RoYk36:track_3_s1_swir_labels.zip"

  # TRACK 3 - SEQUENCE 2 (NO RGB)
  "https://mineinsight.short.gy/1XFhHc:track_3_s2_lwir.zip"
  "https://mineinsight.short.gy/aT8QZL:track_3_s2_lwir_labels_reproj.zip"
  "https://mineinsight.short.gy/ewHM2o:track_3_s2_swir.zip"
  "https://mineinsight.short.gy/2kcgCx:track_3_s2_swir_labels.zip"
)

count=0
total=${#downloads[@]}
failed_downloads=()

for entry in "${downloads[@]}"; do
  count=$((count + 1))
  IFS=':' read -r link filename <<< "$entry"
  printf "[%2d/%d] Downloading %-40s " "$count" "$total" "$filename"

  if wget -q --show-progress -O "$filename" "$link" 2>&1 | grep -q "saved"; then
    echo "✓"
  else
    echo "✗ RETRY"
    if wget -q --show-progress -O "$filename" "$link" 2>&1 | grep -q "saved"; then
      echo "  (retry succeeded)"
    else
      echo "  (failed)"
      failed_downloads+=("$filename")
    fi
  fi
done

echo ""
echo "=========================================="
echo "Download phase complete."
if [ ${#failed_downloads[@]} -gt 0 ]; then
  echo "Failed files (${#failed_downloads[@]}):"
  for f in "${failed_downloads[@]}"; do
    echo "  - $f"
  done
  echo ""
fi
echo "Extracting files..."
echo "=========================================="
echo ""

# Extract all zips
failed_extracts=0
extracted_count=0
for file in *.zip; do
  printf "Extracting %-40s " "$file"
  if unzip -q "$file" 2>/dev/null; then
    echo "✓"
    extracted_count=$((extracted_count + 1))
  else
    echo "✗"
    failed_extracts=$((failed_extracts + 1))
  fi
done

# Cleanup
echo ""
echo "Cleaning up zip files..."
rm -f *.zip

# Download metadata
echo ""
echo "=========================================="
echo "Downloading metadata and calibration..."
echo "=========================================="
echo ""

REPO="https://raw.githubusercontent.com/mariomlz99/MineInsight/main"

wget -q -O targets_list.yaml "$REPO/tracks_inventory/targets_list.yaml" 2>/dev/null && \
  echo "✓ targets_list.yaml"

mkdir -p intrinsics_calibration extrinsics_calibration

wget -q -O intrinsics_calibration/rgb_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/rgb_camera_intrinsics.yaml" 2>/dev/null && \
  echo "✓ rgb_camera_intrinsics.yaml"

wget -q -O intrinsics_calibration/lwir_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/lwir_camera_intrinsics.yaml" 2>/dev/null && \
  echo "✓ lwir_camera_intrinsics.yaml"

wget -q -O intrinsics_calibration/swir_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/swir_camera_intrinsics.yaml" 2>/dev/null && \
  echo "✓ swir_camera_intrinsics.yaml"

wget -q -O extrinsics_calibration/rgb_avia_extrinsics.yaml \
  "$REPO/extrinsics_calibration/rgb_avia_extrinsics.yaml" 2>/dev/null && \
  echo "✓ rgb_avia_extrinsics.yaml"

wget -q -O extrinsics_calibration/lwir_avia_extrinsics.yaml \
  "$REPO/extrinsics_calibration/lwir_avia_extrinsics.yaml" 2>/dev/null && \
  echo "✓ lwir_avia_extrinsics.yaml"

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
echo ""
echo "Dataset location: $DATASET_DIR"
echo ""
echo "Summary:"
echo "  - Downloaded: $count files"
echo "  - Failed downloads: ${#failed_downloads[@]}"
echo "  - Extracted: $extracted_count files"
echo "  - Failed extracts: $failed_extracts"
echo ""
echo "Directory size:"
du -sh "$DATASET_DIR" 2>/dev/null || echo "  (calculating...)"
echo ""
echo "Next steps:"
echo "  1. Check DATASET_DOWNLOAD_GUIDE.txt in /mnt/forge-data/modules/05_wave9/21_MineInsight/"
echo "  2. Verify all track_*_*_* directories exist"
echo "  3. Create data splits for training (80% train, 10% val, 10% test)"
echo "  4. Configure YOLOv8 or custom detector with 35 target classes"

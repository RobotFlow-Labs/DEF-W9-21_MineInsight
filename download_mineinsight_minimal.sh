#!/bin/bash
# MineInsight Dataset Download Script (Minimal: RGB + LWIR)
# Total: ~44 GB
# Usage: bash download_mineinsight_minimal.sh

set -e

DATASET_DIR="/mnt/train-data/datasets/mineinsight"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "=========================================="
echo "MineInsight Dataset Download (Minimal)"
echo "=========================================="
echo "Total size: ~44 GB (RGB + LWIR + Labels)"
echo "Location: $DATASET_DIR"
echo ""

# Check free space
FREE_GB=$(df /mnt/forge-data | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
echo "Free space on /mnt/forge-data: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 60 ]; then
    echo "WARNING: Less than 60 GB free! You may run out of space."
fi
echo ""

# Create download manifest
cat > DOWNLOAD_MANIFEST.txt << 'MANIFEST'
Track 1 - Sequence 1
  RGB Images: 3.8 GB
  RGB Labels: 1.2 MB
  LWIR Images: 669 MB
  LWIR Labels: 2.5 MB
  Subtotal: 4.6 GB

Track 1 - Sequence 2 [LARGE]
  RGB Images: 12.0 GB
  RGB Labels: 6.5 MB
  LWIR Images: 3.0 GB
  LWIR Labels: 12.2 MB
  Subtotal: 15.1 GB

Track 2 - Sequence 1
  RGB Images: 2.8 GB
  RGB Labels: 1.2 MB
  LWIR Images: 520 MB
  LWIR Labels: 2.3 MB
  Subtotal: 3.4 GB

Track 2 - Sequence 2 [LARGE]
  RGB Images: 15.8 GB
  RGB Labels: 8.7 MB
  LWIR Images: 2.3 GB
  LWIR Labels: 12.2 MB
  Subtotal: 18.2 GB

Track 3 - Sequences 1 & 2 (LWIR ONLY, no RGB)
  T3 S1 LWIR: 566 MB
  T3 S1 Labels: 2.0 MB
  T3 S2 LWIR: 2.0 GB
  T3 S2 Labels: 7.3 MB
  Subtotal: 2.6 GB

GRAND TOTAL: ~44 GB
MANIFEST

echo "Downloading dataset..."
echo ""

# Array of URLs (source:filename)
downloads=(
  "https://mineinsight.short.gy/6Mvbjx:track_1_s1_rgb.zip"
  "https://mineinsight.short.gy/Cy5OeN:track_1_s1_rgb_labels.zip"
  "https://mineinsight.short.gy/LV5XZ1:track_1_s1_lwir.zip"
  "https://mineinsight.short.gy/Z8fqcY:track_1_s1_lwir_labels.zip"
  "https://mineinsight.short.gy/RBRY3I:track_1_s2_rgb.zip"
  "https://mineinsight.short.gy/ncoHYs:track_1_s2_rgb_labels.zip"
  "https://mineinsight.short.gy/PYfYwf:track_1_s2_lwir.zip"
  "https://mineinsight.short.gy/bYpES1:track_1_s2_lwir_labels.zip"
  "https://mineinsight.short.gy/OKXGyT:track_2_s1_rgb.zip"
  "https://mineinsight.short.gy/yJ4vKD:track_2_s1_rgb_labels.zip"
  "https://mineinsight.short.gy/Tkb2ra:track_2_s1_lwir.zip"
  "https://mineinsight.short.gy/00s2Te:track_2_s1_lwir_labels.zip"
  "https://mineinsight.short.gy/mZSLV8:track_2_s2_rgb.zip"
  "https://mineinsight.short.gy/5ZEVE9:track_2_s2_rgb_labels.zip"
  "https://mineinsight.short.gy/CuoFOX:track_2_s2_lwir.zip"
  "https://mineinsight.short.gy/uvLeoo:track_2_s2_lwir_labels.zip"
  "https://mineinsight.short.gy/UoD78c:track_3_s1_lwir.zip"
  "https://mineinsight.short.gy/caHk6F:track_3_s1_lwir_labels.zip"
  "https://mineinsight.short.gy/1XFhHc:track_3_s2_lwir.zip"
  "https://mineinsight.short.gy/aT8QZL:track_3_s2_lwir_labels.zip"
)

count=0
total=${#downloads[@]}

for entry in "${downloads[@]}"; do
  count=$((count + 1))
  IFS=':' read -r link filename <<< "$entry"
  echo "[$count/$total] Downloading $filename..."
  wget -q --show-progress -O "$filename" "$link" || echo "FAILED: $filename"
done

echo ""
echo "=========================================="
echo "Download complete. Extracting files..."
echo "=========================================="
echo ""

# Extract all zips
failed=0
for file in *.zip; do
  echo "Extracting $file..."
  unzip -q "$file" || { echo "FAILED: $file"; failed=$((failed+1)); }
done

# Cleanup
echo "Cleaning up zip files..."
rm -f *.zip

# Download metadata
echo ""
echo "Downloading metadata..."
REPO="https://raw.githubusercontent.com/mariomlz99/MineInsight/main"

wget -q -O targets_list.yaml "$REPO/tracks_inventory/targets_list.yaml" && \
  echo "✓ targets_list.yaml"

mkdir -p intrinsics_calibration extrinsics_calibration

wget -q -O intrinsics_calibration/rgb_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/rgb_camera_intrinsics.yaml" && \
  echo "✓ rgb_camera_intrinsics.yaml"

wget -q -O intrinsics_calibration/lwir_camera_intrinsics.yaml \
  "$REPO/intrinsics_calibration/lwir_camera_intrinsics.yaml" && \
  echo "✓ lwir_camera_intrinsics.yaml"

echo ""
echo "=========================================="
if [ $failed -eq 0 ]; then
  echo "SUCCESS! Dataset ready."
else
  echo "WARNING: $failed file(s) failed to extract"
fi
echo "=========================================="
echo ""
echo "Dataset location: $DATASET_DIR"
echo ""
echo "Contents:"
du -sh "$DATASET_DIR"/* 2>/dev/null | sort -h || true
echo ""
echo "Total size:"
du -sh "$DATASET_DIR" 2>/dev/null || true

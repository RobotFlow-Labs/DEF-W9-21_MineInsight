#!/bin/bash
# Auto-launch fusion models when LWIR training completes
# Watches LWIR log for [DONE] or [EARLY STOP], then launches both fusion variants

set -e
cd /mnt/forge-data/modules/05_wave9/21_MineInsight
source .venv/bin/activate

LWIR_LOG=$(ls -t /mnt/artifacts-datai/logs/project_mineinsight_v3_lwir/train_r3_*.log 2>/dev/null | head -1)
RGB_CKPT="/mnt/artifacts-datai/checkpoints/project_mineinsight_v3_rgb/best.pth"
LWIR_CKPT="/mnt/artifacts-datai/checkpoints/project_mineinsight_v3_lwir/best.pth"

echo "[WATCHER] Watching LWIR training for completion..."
echo "[WATCHER] Log: $LWIR_LOG"

# Wait for LWIR to finish
while true; do
    if grep -q "\[DONE\]\|EARLY STOP" "$LWIR_LOG" 2>/dev/null; then
        echo "[WATCHER] LWIR training complete! Launching fusion models..."
        break
    fi
    sleep 30
done

# Create directories
mkdir -p /mnt/artifacts-datai/checkpoints/project_mineinsight_v3_fusion_attn
mkdir -p /mnt/artifacts-datai/logs/project_mineinsight_v3_fusion_attn
mkdir -p /mnt/artifacts-datai/tensorboard/project_mineinsight_v3_fusion_attn
mkdir -p /mnt/artifacts-datai/checkpoints/project_mineinsight_v3_fusion_concat
mkdir -p /mnt/artifacts-datai/logs/project_mineinsight_v3_fusion_concat
mkdir -p /mnt/artifacts-datai/tensorboard/project_mineinsight_v3_fusion_concat

# Launch ATTENTION fusion on GPU 1
LOGFILE1="/mnt/artifacts-datai/logs/project_mineinsight_v3_fusion_attn/train_$(date +%Y%m%d_%H%M).log"
PYTHONPATH="" CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/train_fusion.py \
    --config configs/v3_fusion_attention.toml \
    --rgb-ckpt "$RGB_CKPT" \
    --lwir-ckpt "$LWIR_CKPT" \
    > "$LOGFILE1" 2>&1 &
disown
echo "[LAUNCHED] Fusion ATTENTION on GPU 1, PID=$!, Log=$LOGFILE1"

# Launch CONCAT fusion on GPU 6
LOGFILE6="/mnt/artifacts-datai/logs/project_mineinsight_v3_fusion_concat/train_$(date +%Y%m%d_%H%M).log"
PYTHONPATH="" CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python scripts/train_fusion.py \
    --config configs/v3_fusion_concat.toml \
    --rgb-ckpt "$RGB_CKPT" \
    --lwir-ckpt "$LWIR_CKPT" \
    > "$LOGFILE6" 2>&1 &
disown
echo "[LAUNCHED] Fusion CONCAT on GPU 6, PID=$!, Log=$LOGFILE6"

echo "[WATCHER] Both fusion models launched. Monitoring will continue via cron."

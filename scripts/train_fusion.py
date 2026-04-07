#!/usr/bin/env python3
"""Train RGB+LWIR fusion model initialized from best single-modal checkpoints.

Loads RGB backbone from best RGB checkpoint and LWIR backbone from best LWIR checkpoint,
then trains the fusion model end-to-end.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_fusion.py \
        --config configs/v3_fusion_attention.toml \
        --rgb-ckpt /mnt/artifacts-datai/checkpoints/project_mineinsight_v3_rgb/best.pth \
        --lwir-ckpt /mnt/artifacts-datai/checkpoints/project_mineinsight_v3_lwir/best.pth
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from mineinsight.dataset import MineInsightDataset, collate_fn
from mineinsight.losses import DetectionLoss
from mineinsight.model import MultiModalDetector, build_model
from mineinsight.utils import (
    CheckpointManager,
    Config,
    EarlyStopping,
    WarmupCosineScheduler,
    get_device,
    load_config,
    set_seed,
)


def load_backbone_weights(
    fusion_model: MultiModalDetector,
    rgb_ckpt: str,
    lwir_ckpt: str,
) -> int:
    """Load single-modal backbone weights into fusion model's per-modality backbones."""
    loaded = 0

    if rgb_ckpt and os.path.exists(rgb_ckpt):
        ckpt = torch.load(rgb_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        # Extract backbone weights from single-modal model
        rgb_backbone = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                rgb_backbone[k.replace("backbone.", "")] = v

        if "rgb" in fusion_model.backbones:
            matched = fusion_model.backbones["rgb"].load_state_dict(rgb_backbone, strict=False)
            n = len(rgb_backbone) - len(matched.missing_keys)
            print(f"[FUSION] RGB backbone: loaded {n} params from {rgb_ckpt}")
            loaded += n

    if lwir_ckpt and os.path.exists(lwir_ckpt):
        ckpt = torch.load(lwir_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        lwir_backbone = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                lwir_backbone[k.replace("backbone.", "")] = v

        if "lwir" in fusion_model.backbones:
            matched = fusion_model.backbones["lwir"].load_state_dict(lwir_backbone, strict=False)
            n = len(lwir_backbone) - len(matched.missing_keys)
            print(f"[FUSION] LWIR backbone: loaded {n} params from {lwir_ckpt}")
            loaded += n

    # Also load neck/head from RGB checkpoint (better than random init)
    if rgb_ckpt and os.path.exists(rgb_ckpt):
        ckpt = torch.load(rgb_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        neck_head = {}
        for k, v in state.items():
            if k.startswith("neck.") or k.startswith("head."):
                neck_head[k] = v
        if neck_head:
            matched = fusion_model.load_state_dict(neck_head, strict=False)
            n = len(neck_head) - len(matched.missing_keys)
            print(f"[FUSION] Neck+Head: loaded {n} params from RGB checkpoint")
            loaded += n

    return loaded


def train_fusion(
    cfg: Config,
    rgb_ckpt: str,
    lwir_ckpt: str,
    resume: str | None = None,
) -> None:
    """Train fusion model with pre-initialized backbones."""
    set_seed(cfg.training.seed)
    device = get_device()

    print(f"[CONFIG] modality={cfg.model.modality}")
    print(f"[CONFIG] fusion_method={cfg.model.fusion_method}")
    print(f"[CONFIG] num_classes={cfg.model.num_classes}")
    print(f"[CONFIG] epochs={cfg.training.epochs}, lr={cfg.training.learning_rate}")
    print(f"[DEVICE] {device}")

    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
    )

    if isinstance(model, MultiModalDetector) and not resume:
        loaded = load_backbone_weights(model, rgb_ckpt, lwir_ckpt)
        print(f"[FUSION] Total pre-loaded: {loaded} parameters")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {num_params / 1e6:.1f}M parameters")

    # Build dataloaders
    train_seqs = cfg.data.train_sequences
    val_seqs = cfg.data.val_sequences

    train_ds = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=train_seqs,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=True,
        flip_lr=cfg.data.flip_lr,
        hsv_h=cfg.data.hsv_h,
        hsv_s=cfg.data.hsv_s,
        hsv_v=cfg.data.hsv_v,
    )
    val_ds = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=val_seqs,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=False,
    )

    batch_size = cfg.training.batch_size if isinstance(cfg.training.batch_size, int) else 48

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} samples")

    criterion = DetectionLoss(
        num_classes=cfg.model.num_classes,
        box_weight=cfg.loss.box_weight,
        cls_weight=cfg.loss.cls_weight,
        obj_weight=cfg.loss.obj_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
    )

    total_steps = cfg.training.epochs * max(len(train_loader), 1)
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    os.makedirs(cfg.checkpoint.output_dir, exist_ok=True)
    ckpt_manager = CheckpointManager(
        cfg.checkpoint.output_dir, keep_top_k=cfg.checkpoint.keep_top_k,
        metric=cfg.checkpoint.metric, mode=cfg.checkpoint.mode,
    )
    early_stop = EarlyStopping(
        patience=cfg.early_stopping.patience,
        min_delta=cfg.early_stopping.min_delta, mode=cfg.checkpoint.mode,
    ) if cfg.early_stopping.enabled else None

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(cfg.logging.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(cfg.logging.tensorboard_dir)
    except ImportError:
        pass

    start_epoch = 0
    global_step = 0
    if resume and os.path.exists(resume):
        print(f"[RESUME] Loading from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)

    use_amp = cfg.training.precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16

    print(f"[TRAIN] Starting {cfg.training.epochs} epochs, warmup={warmup_steps} steps")

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            targets = batch["targets"].to(device)
            target_counts = batch["target_counts"].to(device)

            # Multi-modal: pass dict of modality tensors
            if "images" in batch:
                model_input = {mod: t.to(device) for mod, t in batch["images"].items()}
            else:
                model_input = batch["image"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    preds = model(model_input)
                    loss_dict = criterion(preds, targets, target_counts)
                    loss = loss_dict["loss"]
            else:
                preds = model(model_input)
                loss_dict = criterion(preds, targets, target_counts)
                loss = loss_dict["loss"]

            if torch.isnan(loss):
                print("[FATAL] NaN loss — stopping")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Step {global_step}] loss={loss.item():.4f} "
                    f"box={loss_dict['box_loss'].item():.4f} "
                    f"cls={loss_dict['cls_loss'].item():.4f} "
                    f"obj={loss_dict['obj_loss'].item():.4f} lr={lr:.6f}",
                )

            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)

        avg_train = total_loss / max(num_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                targets = batch["targets"].to(device)
                target_counts = batch["target_counts"].to(device)
                if "images" in batch:
                    model_input = {mod: t.to(device) for mod, t in batch["images"].items()}
                else:
                    model_input = batch["image"].to(device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        preds = model(model_input)
                        ld = criterion(preds, targets, target_counts)
                else:
                    preds = model(model_input)
                    ld = criterion(preds, targets, target_counts)
                val_loss += ld["loss"].item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch + 1}/{cfg.training.epochs}] "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} time={elapsed:.1f}s",
        )

        if writer:
            writer.add_scalar("val/loss", avg_val, global_step)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": avg_val,
        }
        ckpt_manager.save(state, avg_val, global_step)

        if early_stop and early_stop.step(avg_val):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs.")
            break

    if writer:
        writer.close()
    print("[DONE] Fusion training complete.")


def main():
    parser = argparse.ArgumentParser(description="MineInsight Fusion Training")
    parser.add_argument("--config", type=str, required=True)
    rgb_default = "/mnt/artifacts-datai/checkpoints/project_mineinsight_v3_rgb/best.pth"
    lwir_default = "/mnt/artifacts-datai/checkpoints/project_mineinsight_v3_lwir/best.pth"
    parser.add_argument("--rgb-ckpt", type=str, default=rgb_default)
    parser.add_argument("--lwir-ckpt", type=str, default=lwir_default)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_fusion(cfg, args.rgb_ckpt, args.lwir_ckpt, args.resume)


if __name__ == "__main__":
    main()

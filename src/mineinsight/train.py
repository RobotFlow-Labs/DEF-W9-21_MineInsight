"""MineInsight training pipeline.

Config-driven, supports single-modal and multi-modal training.
Includes checkpointing, early stopping, LR scheduling, bf16 mixed precision.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from mineinsight.dataset import MineInsightDataset, collate_fn
from mineinsight.losses import DetectionLoss
from mineinsight.model import build_model
from mineinsight.utils import (
    CheckpointManager,
    Config,
    EarlyStopping,
    WarmupCosineScheduler,
    get_device,
    load_config,
    set_seed,
)


def build_dataloader(cfg: Config, split: str = "train") -> DataLoader:
    """Build a DataLoader for the given split."""
    if split == "train":
        sequences = cfg.data.train_sequences
        augment = True
    elif split == "val":
        sequences = cfg.data.val_sequences
        augment = False
    else:
        sequences = cfg.data.test_sequences
        augment = False

    dataset = MineInsightDataset(
        root=cfg.data.dataset_root,
        sequences=sequences,
        modality=cfg.model.modality,
        input_size=tuple(cfg.model.input_size),
        augment=augment,
        flip_lr=cfg.data.flip_lr if augment else 0.0,
        hsv_h=cfg.data.hsv_h if augment else 0.0,
        hsv_s=cfg.data.hsv_s if augment else 0.0,
        hsv_v=cfg.data.hsv_v if augment else 0.0,
    )

    batch_size = cfg.training.batch_size if isinstance(cfg.training.batch_size, int) else 8

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: Config,
    epoch: int,
    global_step: int,
    ckpt_manager: CheckpointManager | None = None,
    writer: object | None = None,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, updated_global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    precision = cfg.training.precision
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    is_multimodal = "images" in (loader.dataset[0] if len(loader.dataset) > 0 else {})

    for batch in loader:
        targets = batch["targets"].to(device)
        target_counts = batch["target_counts"].to(device)

        # Multi-modal: pass dict of modality tensors; single: pass concatenated tensor
        if is_multimodal and "images" in batch:
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

        # NaN detection
        if torch.isnan(loss):
            print("[FATAL] Loss is NaN -- stopping training")
            print("[FIX] Reduce lr by 10x, check data for corrupt samples")
            return float("nan"), global_step

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()

        scheduler.step()
        global_step += 1
        total_loss += loss.item()
        num_batches += 1

        # Logging
        if global_step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Step {global_step}] loss={loss.item():.4f} "
                f"box={loss_dict['box_loss'].item():.4f} "
                f"cls={loss_dict['cls_loss'].item():.4f} "
                f"obj={loss_dict['obj_loss'].item():.4f} "
                f"lr={lr:.6f}",
            )

        # TensorBoard
        if writer is not None:
            try:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/box_loss", loss_dict["box_loss"].item(), global_step)
                writer.add_scalar("train/cls_loss", loss_dict["cls_loss"].item(), global_step)
                writer.add_scalar("train/obj_loss", loss_dict["obj_loss"].item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            except Exception:
                pass

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def detection_health_check(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 50,
) -> float:
    """Quick check: are foreground scores high enough to detect objects?

    Returns max foreground score across sampled images.
    If < 0.10, the model is not detecting anything useful.
    """
    model.eval()
    max_fg = 0.0
    step = max(1, len(dataset) // n_samples)
    for i in range(0, min(len(dataset), n_samples * step), step):
        sample = dataset[i]
        if "images" in sample:
            inp = {mod: t.unsqueeze(0).to(device) for mod, t in sample["images"].items()}
        else:
            inp = sample["image"].unsqueeze(0).to(device)
        preds = model(inp)
        all_p = torch.cat(preds, dim=1)[0]
        cls_logits = all_p[:, 4:]
        fg = torch.sigmoid(cls_logits[:, 1:]).max().item()
        if fg > max_fg:
            max_fg = fg
    return max_fg


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    cfg: Config,
) -> dict[str, float]:
    """Run validation and return metrics dict."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    precision = cfg.training.precision
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    is_multimodal = "images" in (loader.dataset[0] if len(loader.dataset) > 0 else {})

    for batch in loader:
        targets = batch["targets"].to(device)
        target_counts = batch["target_counts"].to(device)

        if is_multimodal and "images" in batch:
            model_input = {mod: t.to(device) for mod, t in batch["images"].items()}
        else:
            model_input = batch["image"].to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                preds = model(model_input)
                loss_dict = criterion(preds, targets, target_counts)
        else:
            preds = model(model_input)
            loss_dict = criterion(preds, targets, target_counts)

        total_loss += loss_dict["loss"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    # Placeholder mAP -- real mAP computed in evaluate.py
    return {"val_loss": avg_loss, "val_mAP50": 0.0}


def train(cfg: Config, resume: str | None = None, max_steps: int | None = None) -> None:
    """Main training loop."""
    set_seed(cfg.training.seed)
    device = get_device()

    # Print training config
    print(f"[CONFIG] modality={cfg.model.modality}")
    print(f"[CONFIG] num_classes={cfg.model.num_classes}")
    print(f"[CONFIG] input_size={cfg.model.input_size}")
    print(f"[CONFIG] epochs={cfg.training.epochs}, lr={cfg.training.learning_rate}")
    print(f"[CONFIG] precision={cfg.training.precision}")
    print(f"[DEVICE] {device}")

    # Build model
    model = build_model(
        modality=cfg.model.modality,
        num_classes=cfg.model.num_classes,
        fusion_method=cfg.model.fusion_method,
        architecture=cfg.model.architecture,
        pretrained=cfg.model.pretrained,
    )

    # Load pretrained weights if available (skip for YOLO26 — handled internally)
    if (
        cfg.model.architecture != "yolo26"
        and cfg.model.pretrained
        and os.path.exists(cfg.model.pretrained)
    ):
        print(f"[WEIGHTS] Loading pretrained from {cfg.model.pretrained}")
        try:
            state = torch.load(cfg.model.pretrained, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            # Partial load (ignore mismatched keys)
            model_state = model.state_dict()
            matched = {
                k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model_state.update(matched)
            model.load_state_dict(model_state)
            print(f"[WEIGHTS] Loaded {len(matched)}/{len(model_state)} parameters")
        except Exception as e:
            print(f"[WEIGHTS] Failed to load pretrained: {e}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {num_params / 1e6:.1f}M parameters")

    # Build dataloaders
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")
    print(f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)} samples")

    # Loss
    criterion = DetectionLoss(
        num_classes=cfg.model.num_classes,
        box_weight=cfg.loss.box_weight,
        cls_weight=cfg.loss.cls_weight,
        obj_weight=cfg.loss.obj_weight,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    total_steps = cfg.training.epochs * max(len(train_loader), 1)
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # AMP scaler
    scaler = None
    if cfg.training.precision == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # Checkpoint manager
    os.makedirs(cfg.checkpoint.output_dir, exist_ok=True)
    ckpt_manager = CheckpointManager(
        cfg.checkpoint.output_dir,
        keep_top_k=cfg.checkpoint.keep_top_k,
        metric=cfg.checkpoint.metric,
        mode=cfg.checkpoint.mode,
    )

    # Early stopping
    early_stop = None
    if cfg.early_stopping.enabled:
        early_stop = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode=cfg.checkpoint.mode,
        )

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(cfg.logging.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(cfg.logging.tensorboard_dir)
    except ImportError:
        pass

    # Resume
    start_epoch = 0
    global_step = 0
    if resume and os.path.exists(resume):
        print(f"[RESUME] Loading from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[RESUME] Starting from epoch {start_epoch}, step {global_step}")

    # Training loop
    print(f"[TRAIN] Starting {cfg.training.epochs} epochs, warmup={warmup_steps} steps")
    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        avg_train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg, epoch, global_step, ckpt_manager, writer,
        )

        if avg_train_loss != avg_train_loss:  # NaN check
            break

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, cfg)
        val_loss = val_metrics["val_loss"]
        val_map = val_metrics.get("val_mAP50", 0.0)

        # Detection health check every 5 epochs
        max_fg = 0.0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            max_fg = detection_health_check(model, val_loader.dataset, device, 50)

        elapsed = time.time() - t0
        fg_str = f" max_fg={max_fg:.4f}" if max_fg > 0 else ""
        print(
            f"[Epoch {epoch + 1}/{cfg.training.epochs}] "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}"
            f"{fg_str} time={elapsed:.1f}s",
        )
        if max_fg > 0 and max_fg < 0.10:
            print(
                f"  [WARN] Model not detecting — max_fg={max_fg:.4f}. "
                f"Check loss balance.",
            )

        if writer is not None:
            writer.add_scalar("val/loss", val_loss, global_step)
            if max_fg > 0:
                writer.add_scalar("val/max_fg", max_fg, global_step)

        # Save checkpoint
        metric_val = val_map if cfg.checkpoint.mode == "max" else val_loss
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "val_mAP50": val_map,
        }
        ckpt_manager.save(state, metric_val, global_step)

        # Early stopping
        if early_stop is not None and early_stop.step(metric_val):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

        # Max steps override (for smoke testing)
        if max_steps is not None and global_step >= max_steps:
            print(f"[MAX STEPS] Reached {max_steps} steps. Stopping.")
            break

    if writer is not None:
        writer.close()

    print("[DONE] Training complete.")
    print(f"[CKPT] Best model saved to {cfg.checkpoint.output_dir}/best.pth")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MineInsight Training")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

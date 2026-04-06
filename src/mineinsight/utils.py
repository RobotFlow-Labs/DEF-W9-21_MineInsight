"""Utilities: config loading, seed, device detection, export helpers."""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    architecture: str = "yolov8"
    backbone: str = "cspdarknet_nano"
    num_classes: int = 58
    input_size: list[int] = field(default_factory=lambda: [640, 640])
    modality: str = "rgb"
    pretrained: str = ""
    fusion_enabled: bool = False
    fusion_method: str = "concat"
    fusion_channels: int = 256


@dataclass
class TrainingConfig:
    batch_size: int | str = "auto"
    learning_rate: float = 1e-3
    epochs: int = 100
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    precision: str = "bf16"
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    dataset_root: str = "/mnt/forge-data/datasets/mineinsight"
    train_sequences: list[str] = field(default_factory=list)
    val_sequences: list[str] = field(default_factory=list)
    test_sequences: list[str] = field(default_factory=list)
    num_workers: int = 4
    pin_memory: bool = True
    mosaic: bool = True
    mixup: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flip_lr: float = 0.5
    flip_ud: float = 0.0
    scale: list[float] = field(default_factory=lambda: [0.5, 1.5])


@dataclass
class LossConfig:
    box_weight: float = 7.5
    cls_weight: float = 0.5
    obj_weight: float = 1.0
    box_loss: str = "ciou"
    cls_loss: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class CheckpointConfig:
    output_dir: str = "/mnt/artifacts-datai/checkpoints/project_mineinsight"
    save_every_n_steps: int = 500
    keep_top_k: int = 2
    metric: str = "val_mAP50"
    mode: str = "max"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 20
    min_delta: float = 0.001


@dataclass
class LoggingConfig:
    log_dir: str = "/mnt/artifacts-datai/logs/project_mineinsight"
    tensorboard_dir: str = "/mnt/artifacts-datai/tensorboard/project_mineinsight"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    """Load a TOML config and return a Config dataclass."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config()

    # Model
    m = raw.get("model", {})
    cfg.model.architecture = m.get("architecture", cfg.model.architecture)
    cfg.model.backbone = m.get("backbone", cfg.model.backbone)
    cfg.model.num_classes = m.get("num_classes", cfg.model.num_classes)
    cfg.model.input_size = m.get("input_size", cfg.model.input_size)
    cfg.model.modality = m.get("modality", cfg.model.modality)
    cfg.model.pretrained = m.get("pretrained", cfg.model.pretrained)
    fusion = m.get("fusion", {})
    cfg.model.fusion_enabled = fusion.get("enabled", cfg.model.fusion_enabled)
    cfg.model.fusion_method = fusion.get("method", cfg.model.fusion_method)
    cfg.model.fusion_channels = fusion.get("fusion_channels", cfg.model.fusion_channels)

    # Training
    t = raw.get("training", {})
    for k in (
        "batch_size", "learning_rate", "epochs", "optimizer", "weight_decay",
        "scheduler", "warmup_ratio", "precision", "gradient_accumulation",
        "max_grad_norm", "seed",
    ):
        if k in t:
            setattr(cfg.training, k, t[k])

    # Data
    d = raw.get("data", {})
    for k in ("dataset_root", "train_sequences", "val_sequences", "test_sequences",
              "num_workers", "pin_memory"):
        if k in d:
            setattr(cfg.data, k, d[k])
    aug = d.get("augmentation", {})
    for k in ("mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v", "flip_lr", "flip_ud", "scale"):
        if k in aug:
            setattr(cfg.data, k, aug[k])

    # Loss
    lo = raw.get("loss", {})
    for k in ("box_weight", "cls_weight", "obj_weight", "box_loss", "cls_loss",
              "focal_alpha", "focal_gamma"):
        if k in lo:
            setattr(cfg.loss, k, lo[k])

    # Checkpoint
    cp = raw.get("checkpoint", {})
    for k in ("output_dir", "save_every_n_steps", "keep_top_k", "metric", "mode"):
        if k in cp:
            setattr(cfg.checkpoint, k, cp[k])

    # Early stopping
    es = raw.get("early_stopping", {})
    for k in ("enabled", "patience", "min_delta"):
        if k in es:
            setattr(cfg.early_stopping, k, es[k])

    # Logging
    lg = raw.get("logging", {})
    for k in ("log_dir", "tensorboard_dir"):
        if k in lg:
            setattr(cfg.logging, k, lg[k])

    return cfg


# ---------------------------------------------------------------------------
# Seed & device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set reproducibility seeds for torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save top-K checkpoints ranked by a metric; auto-delete old ones."""

    def __init__(
        self,
        save_dir: str | Path,
        keep_top_k: int = 2,
        metric: str = "val_mAP50",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict[str, Any],
        metric_value: float,
        step: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(
            key=lambda x: x[0], reverse=(self.mode == "max"),
        )
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        # Always keep best as best.pth
        best_val, best_path = self.history[0]
        best_dst = self.save_dir / "best.pth"
        if best_path != best_dst:
            shutil.copy2(best_path, best_dst)
        return path


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when metric stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps,
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self) -> dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: tuple[int, int] = (640, 640),
    batch_size: int = 1,
    opset: int = 17,
) -> Path:
    """Export model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, 3, *input_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={
            "images": {0: "batch"},
            "detections": {0: "batch"},
        },
    )
    return output_path


def export_safetensors(
    model: torch.nn.Module,
    output_path: str | Path,
) -> Path:
    """Export model weights in safetensors format."""
    from safetensors.torch import save_file

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(output_path))
    return output_path

"""CUDA-accelerated operations for MineInsight.

Uses shared ANIMA CUDA kernels from /mnt/forge-data/shared_infra/cuda_extensions/:
- detection_ops: fused_box_iou_2d, fused_focal_loss, fused_score_filter
- fused_image_preprocess: batch_normalize_hwc_to_chw, fused_resize
- vectorized_nms: nms_2d (CUDA bitmask NMS)

Falls back to pure PyTorch when CUDA kernels are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Shared CUDA kernel paths
# ---------------------------------------------------------------------------

_CUDA_EXT_ROOT = Path("/mnt/forge-data/shared_infra/cuda_extensions")
_LOADED = False

_detection_ops = None
_fused_preprocess = None
_nms_2d_fn = None


def _ensure_loaded() -> None:
    """Lazy-load shared CUDA extensions once."""
    global _LOADED, _detection_ops, _fused_preprocess, _nms_2d_fn
    if _LOADED:
        return
    _LOADED = True

    if not torch.cuda.is_available():
        return

    # detection_ops
    det_path = str(_CUDA_EXT_ROOT / "detection_ops")
    if det_path not in sys.path:
        sys.path.insert(0, det_path)
    try:
        import detection_ops as _det
        _detection_ops = _det
    except ImportError:
        pass

    # fused_image_preprocess
    prep_path = str(_CUDA_EXT_ROOT / "fused_image_preprocess")
    if prep_path not in sys.path:
        sys.path.insert(0, prep_path)
    try:
        import fused_image_preprocess as _prep
        _fused_preprocess = _prep
    except ImportError:
        pass

    # vectorized_nms (parent dir must be on path for package import)
    nms_parent = str(_CUDA_EXT_ROOT)
    if nms_parent not in sys.path:
        sys.path.insert(0, nms_parent)
    try:
        from vectorized_nms import nms_2d
        _nms_2d_fn = nms_2d
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CUDA-accelerated 2D Box IoU
# ---------------------------------------------------------------------------

def cuda_box_iou_2d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Fused 2D box IoU on CUDA. Falls back to PyTorch if kernel unavailable.

    Args:
        boxes1: (N, 4) in xyxy format, CUDA float32.
        boxes2: (M, 4) in xyxy format, CUDA float32.

    Returns:
        (N, M) IoU matrix.
    """
    _ensure_loaded()

    if _detection_ops is not None and boxes1.is_cuda:
        return _detection_ops.fused_box_iou_2d(
            boxes1.contiguous().float(),
            boxes2.contiguous().float(),
        )

    # PyTorch fallback
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# CUDA-accelerated Focal Loss
# ---------------------------------------------------------------------------

def cuda_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Fused focal loss on CUDA. Falls back to PyTorch if kernel unavailable.

    Args:
        logits: (N, C) pre-sigmoid logits.
        targets: (N,) class indices.
        alpha: Balancing factor.
        gamma: Focusing parameter.

    Returns:
        Scalar loss.
    """
    _ensure_loaded()

    if _detection_ops is not None and logits.is_cuda:
        per_sample = _detection_ops.fused_focal_loss(
            logits.contiguous().float(),
            targets.contiguous().int(),
            alpha,
            gamma,
        )
        return per_sample.mean()

    # PyTorch fallback
    import torch.nn.functional as F

    num_classes = logits.shape[-1]
    target_oh = F.one_hot(targets.long(), num_classes).float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target_oh, reduction="none")
    p_t = p * target_oh + (1 - p) * (1 - target_oh)
    alpha_t = alpha * target_oh + (1 - alpha) * (1 - target_oh)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# CUDA-accelerated NMS
# ---------------------------------------------------------------------------

def cuda_nms_2d(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """Vectorized 2D NMS on CUDA. Falls back to PyTorch if kernel unavailable.

    Args:
        boxes: (N, 4) in xyxy format.
        scores: (N,) confidence scores.
        iou_threshold: NMS IoU threshold.

    Returns:
        Indices of kept boxes.
    """
    _ensure_loaded()

    if _nms_2d_fn is not None and boxes.is_cuda:
        return _nms_2d_fn(
            boxes.contiguous().float(),
            scores.contiguous().float(),
            iou_threshold,
        )

    # PyTorch fallback
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        b1 = boxes[i].unsqueeze(0)
        b2 = boxes[rest]
        inter_min = torch.max(b1[:, :2], b2[:, :2])
        inter_max = torch.min(b1[:, 2:], b2[:, 2:])
        inter = (inter_max - inter_min).clamp(min=0).prod(dim=-1)
        area1 = (b1[:, 2:] - b1[:, :2]).prod(dim=-1)
        area2 = (b2[:, 2:] - b2[:, :2]).prod(dim=-1)
        iou = inter / (area1 + area2 - inter).clamp(min=1e-8)
        order = rest[iou.squeeze() <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ---------------------------------------------------------------------------
# CUDA-accelerated Image Preprocessing
# ---------------------------------------------------------------------------

def cuda_normalize_hwc_to_chw(
    image: torch.Tensor,
    mean: tuple[float, ...] = (0.0, 0.0, 0.0),
    std: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Fused HWC uint8 → CHW float32 with normalization.

    Replaces: .permute(2,0,1).float().div(255).sub(mean).div(std)
    With a single CUDA kernel pass.

    Args:
        image: (H, W, C) uint8 tensor on CUDA.
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        (C, H, W) float32 normalized tensor.
    """
    _ensure_loaded()

    if _fused_preprocess is not None and image.is_cuda:
        mean_t = torch.tensor(mean, dtype=torch.float32, device=image.device)
        std_t = torch.tensor(std, dtype=torch.float32, device=image.device)
        return _fused_preprocess.fused_normalize_hwc_to_chw(
            image.contiguous(),
            mean_t,
            std_t,
        )

    # PyTorch fallback
    img = image.float() / 255.0
    img = img.permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=image.device).view(3, 1, 1)
    return (img - mean_t) / std_t


def cuda_batch_normalize(
    images: torch.Tensor,
    mean: tuple[float, ...] = (0.0, 0.0, 0.0),
    std: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Batch fused HWC uint8 → CHW float32 with normalization.

    Args:
        images: (B, H, W, C) uint8 tensor on CUDA.
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        (B, C, H, W) float32 normalized tensor.
    """
    _ensure_loaded()

    if _fused_preprocess is not None and images.is_cuda:
        mean_t = torch.tensor(mean, dtype=torch.float32, device=images.device)
        std_t = torch.tensor(std, dtype=torch.float32, device=images.device)
        return _fused_preprocess.batch_normalize_hwc_to_chw(
            images.contiguous(),
            mean_t,
            std_t,
        )

    # PyTorch fallback
    imgs = images.float() / 255.0
    imgs = imgs.permute(0, 3, 1, 2)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    return (imgs - mean_t) / std_t


# ---------------------------------------------------------------------------
# Module availability check
# ---------------------------------------------------------------------------

def cuda_kernels_available() -> dict[str, bool]:
    """Check which CUDA kernels are loaded."""
    _ensure_loaded()
    return {
        "detection_ops": _detection_ops is not None,
        "fused_image_preprocess": _fused_preprocess is not None,
        "vectorized_nms": _nms_2d_fn is not None,
        "cuda": torch.cuda.is_available(),
    }

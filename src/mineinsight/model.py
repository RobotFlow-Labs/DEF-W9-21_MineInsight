"""MineInsight detection model: CSPDarknet backbone + FPN neck + detection head.

Supports single-modal (RGB, LWIR, SWIR) and multi-modal fusion detection.
Architecture follows YOLOv8-nano scale (~3M params single, ~6M fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: 1x1 reduce -> 3x3 conv -> residual add."""

    def __init__(self, ch: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(ch * expansion)
        self.cv1 = ConvBnSiLU(ch, hidden, 1, 1, 0)
        self.cv2 = ConvBnSiLU(hidden, ch, 3, 1, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return out + x if self.shortcut else out


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.hidden = out_ch // 2
        self.cv1 = ConvBnSiLU(in_ch, 2 * self.hidden, 1, 1, 0)
        self.cv2 = ConvBnSiLU((2 + n) * self.hidden, out_ch, 1, 1, 0)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.hidden, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        chunks = list(y.chunk(2, dim=1))
        for bn in self.bottlenecks:
            chunks.append(bn(chunks[-1]))
        return self.cv2(torch.cat(chunks, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBnSiLU(in_ch, hidden, 1, 1, 0)
        self.cv2 = ConvBnSiLU(hidden * 4, out_ch, 1, 1, 0)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ---------------------------------------------------------------------------
# CSPDarknet Backbone (nano scale)
# ---------------------------------------------------------------------------

class CSPDarknet(nn.Module):
    """CSPDarknet backbone outputting features at 3 scales (P3, P4, P5).

    Nano scale: base width = 16, depth multiplier = 0.33.
    """

    def __init__(self, in_ch: int = 3, base_width: int = 16):
        super().__init__()
        w = base_width  # 16 for nano

        # Stem: 640 -> 320
        self.stem = ConvBnSiLU(in_ch, w, 3, 2, 1)

        # Stage 1: 320 -> 160
        self.stage1 = nn.Sequential(
            ConvBnSiLU(w, w * 2, 3, 2, 1),
            C2f(w * 2, w * 2, n=1),
        )

        # Stage 2: 160 -> 80 (P3)
        self.stage2 = nn.Sequential(
            ConvBnSiLU(w * 2, w * 4, 3, 2, 1),
            C2f(w * 4, w * 4, n=2),
        )

        # Stage 3: 80 -> 40 (P4)
        self.stage3 = nn.Sequential(
            ConvBnSiLU(w * 4, w * 8, 3, 2, 1),
            C2f(w * 8, w * 8, n=2),
        )

        # Stage 4: 40 -> 20 (P5)
        self.stage4 = nn.Sequential(
            ConvBnSiLU(w * 8, w * 16, 3, 2, 1),
            C2f(w * 16, w * 16, n=1),
            SPPF(w * 16, w * 16),
        )

        self.out_channels = [w * 4, w * 8, w * 16]  # P3, P4, P5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


# ---------------------------------------------------------------------------
# FPN Neck
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """Feature Pyramid Network neck with top-down + bottom-up paths."""

    def __init__(self, in_channels: list[int]):
        super().__init__()
        c3, c4, c5 = in_channels

        # Top-down path
        self.up5_to_4 = ConvBnSiLU(c5, c4, 1, 1, 0)
        self.fuse_p4 = C2f(c4 * 2, c4, n=1, shortcut=False)
        self.up4_to_3 = ConvBnSiLU(c4, c3, 1, 1, 0)
        self.fuse_p3 = C2f(c3 * 2, c3, n=1, shortcut=False)

        # Bottom-up path
        self.down3_to_4 = ConvBnSiLU(c3, c3, 3, 2, 1)
        self.fuse_n4 = C2f(c3 + c4, c4, n=1, shortcut=False)
        self.down4_to_5 = ConvBnSiLU(c4, c4, 3, 2, 1)
        self.fuse_n5 = C2f(c4 + c5, c5, n=1, shortcut=False)

        self.out_channels = in_channels

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features

        # Top-down
        up5 = F.interpolate(self.up5_to_4(p5), size=p4.shape[2:], mode="nearest")
        p4 = self.fuse_p4(torch.cat([up5, p4], dim=1))

        up4 = F.interpolate(self.up4_to_3(p4), size=p3.shape[2:], mode="nearest")
        p3 = self.fuse_p3(torch.cat([up4, p3], dim=1))

        # Bottom-up
        dn3 = self.down3_to_4(p3)
        p4 = self.fuse_n4(torch.cat([dn3, p4], dim=1))

        dn4 = self.down4_to_5(p4)
        p5 = self.fuse_n5(torch.cat([dn4, p5], dim=1))

        return p3, p4, p5


# ---------------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """Per-scale detection head predicting (x, y, w, h, cls*(num_classes+1)).

    No separate objectness head — class 0 is background (DETR-style).
    Output: (B, num_anchors, 4 + num_classes + 1) per scale.
    """

    def __init__(self, in_channels: list[int], num_classes: int = 58):
        super().__init__()
        self.num_classes = num_classes
        # box(4) + cls(num_classes+1) where class 0 = background
        out_ch = 4 + num_classes + 1

        self.heads = nn.ModuleList()
        for ch in in_channels:
            self.heads.append(
                nn.Sequential(
                    ConvBnSiLU(ch, ch, 3, 1, 1),
                    ConvBnSiLU(ch, ch, 3, 1, 1),
                    nn.Conv2d(ch, out_ch, 1),
                ),
            )

    def forward(
        self, features: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Returns list of (B, num_anchors, 4+num_classes+1) per scale."""
        outputs = []
        for feat, head in zip(features, self.heads, strict=True):
            pred = head(feat)  # (B, out_ch, H, W)
            b, c, h, w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(b, h * w, c)
            outputs.append(pred)
        return outputs


# ---------------------------------------------------------------------------
# Fusion Module
# ---------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """Attention-gated fusion of multi-modal features at each scale."""

    def __init__(self, channels: int, num_modalities: int = 2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * num_modalities, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.proj = ConvBnSiLU(channels * num_modalities, channels, 1, 1, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple modalities.

        Args:
            features: list of (B, C, H, W) tensors, one per modality.
        """
        cat = torch.cat(features, dim=1)
        gate = self.gate(cat)
        proj = self.proj(cat)
        return proj * gate


class ConcatFusion(nn.Module):
    """Simple concatenation + projection fusion."""

    def __init__(self, channels: int, num_modalities: int = 2):
        super().__init__()
        self.proj = ConvBnSiLU(channels * num_modalities, channels, 1, 1, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(features, dim=1))


class AddFusion(nn.Module):
    """Element-wise addition fusion (all modalities must have same channels)."""

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(features).sum(0)


# ---------------------------------------------------------------------------
# Single-Modal Detector
# ---------------------------------------------------------------------------

class SingleModalDetector(nn.Module):
    """Complete single-modality detection model.

    Backbone -> FPN Neck -> Detection Head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 58,
        base_width: int = 16,
    ):
        super().__init__()
        self.backbone = CSPDarknet(in_channels, base_width)
        self.neck = FPNNeck(self.backbone.out_channels)
        self.head = DetectionHead(self.neck.out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, C, H, W) input image tensor.

        Returns:
            List of prediction tensors per scale, each (B, N, 5+num_classes).
        """
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)


# ---------------------------------------------------------------------------
# Multi-Modal Fusion Detector
# ---------------------------------------------------------------------------

class MultiModalDetector(nn.Module):
    """Multi-modal detection with per-modality backbones and fusion.

    Separate backbone per modality -> Fuse at each scale -> Shared FPN + Head.
    """

    def __init__(
        self,
        modalities: list[str],
        num_classes: int = 58,
        base_width: int = 16,
        fusion_method: str = "attention",
    ):
        super().__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        num_mod = len(modalities)

        # One backbone per modality (each takes 3-channel input)
        self.backbones = nn.ModuleDict({
            mod: CSPDarknet(3, base_width) for mod in modalities
        })

        ch = self.backbones[modalities[0]].out_channels  # [c3, c4, c5]

        # Fusion at each scale
        fusion_cls = {
            "attention": AttentionFusion,
            "concat": ConcatFusion,
            "add": AddFusion,
        }
        fuse_fn = fusion_cls.get(fusion_method, AttentionFusion)
        self.fusions = nn.ModuleList([
            fuse_fn(c, num_mod) if fusion_method != "add" else fuse_fn()
            for c in ch
        ])

        # Shared neck and head after fusion
        self.neck = FPNNeck(ch)
        self.head = DetectionHead(self.neck.out_channels, num_classes)

    def forward(
        self,
        images: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Forward pass with dictionary of modality images.

        Args:
            images: dict mapping modality name -> (B, 3, H, W) tensor.

        Returns:
            List of prediction tensors per scale.
        """
        # Extract features per modality
        all_features: list[tuple[torch.Tensor, ...]] = []
        for mod in self.modalities:
            feats = self.backbones[mod](images[mod])
            all_features.append(feats)

        # Fuse at each scale
        fused = []
        for scale_idx, fusion in enumerate(self.fusions):
            scale_feats = [all_features[m][scale_idx] for m in range(len(self.modalities))]
            fused.append(fusion(scale_feats))

        fused_tuple = tuple(fused)
        features = self.neck(fused_tuple)
        return self.head(features)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CSPDarknetWideWrapper(nn.Module):
    """Wider CSPDarknet variant (base_width=32) for the legacy custom model.

    **This is NOT Ultralytics YOLO26.** It was previously misnamed
    ``YOLO26Wrapper`` and accepted a ``model_path`` argument that was silently
    ignored — the name promised Ultralytics YOLO26 weights but the code only
    built a wider CSPDarknet with random init. Any historical run labelled
    "yolo26 baseline" was actually this wide CSPDarknet.

    For the **real** stock Ultralytics YOLO26s path, use
    ``scripts/train_yolo26_fusion.py`` which loads
    ``/mnt/train-data/models/yolo26/yolo26s.pt`` directly via the Ultralytics
    library. This class is kept purely for backward-compatible deserialization
    of older checkpoints.
    """

    def __init__(self, num_classes: int = 58, *_, **__):
        super().__init__()
        self.detector = SingleModalDetector(
            in_channels=3, num_classes=num_classes, base_width=32,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.detector(x)


# Backward-compatibility alias so any existing checkpoints / imports don't
# break at load time. New code should use CSPDarknetWideWrapper directly.
YOLO26Wrapper = CSPDarknetWideWrapper


def build_model(
    modality: str = "rgb",
    num_classes: int = 58,
    base_width: int = 16,
    fusion_method: str = "attention",
    architecture: str = "yolov8",
    pretrained: str = "",  # noqa: ARG001 — kept for backward compat
) -> nn.Module:
    """Build a detector based on modality configuration.

    Args:
        modality: "rgb", "lwir", "swir" for single; "rgb+lwir" etc for fusion.
        num_classes: Number of object classes.
        base_width: Base channel width (16 = nano).
        fusion_method: "attention", "concat", or "add".
        architecture:
            "yolov8" — custom CSPDarknet (default, legacy).
            "cspdarknet_wide" / "yolo26" — wider CSPDarknet (legacy alias).
        pretrained: DEPRECATED — silently ignored by legacy wrappers. Use
            ``scripts/train_yolo26_fusion.py`` if you need real Ultralytics
            weight loading.

    Returns:
        Detection model.
    """
    if architecture in ("yolo26", "cspdarknet_wide"):
        return CSPDarknetWideWrapper(num_classes=num_classes)
    if "+" in modality:
        mods = modality.split("+")
        return MultiModalDetector(mods, num_classes, base_width, fusion_method)
    return SingleModalDetector(3, num_classes, base_width)

"""Tests for MineInsight model architecture."""

import torch

from mineinsight.losses import (
    DetectionLoss,
    FocalLoss,
    box_cxcywh_to_xyxy,
    box_iou,
    ciou_loss,
)
from mineinsight.model import (
    CSPDarknet,
    DetectionHead,
    FPNNeck,
    MultiModalDetector,
    SingleModalDetector,
    build_model,
)


class TestCSPDarknet:
    def test_output_shapes(self):
        backbone = CSPDarknet(in_ch=3, base_width=16)
        x = torch.randn(2, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape == (2, 64, 80, 80), f"P3 shape: {p3.shape}"
        assert p4.shape == (2, 128, 40, 40), f"P4 shape: {p4.shape}"
        assert p5.shape == (2, 256, 20, 20), f"P5 shape: {p5.shape}"

    def test_out_channels(self):
        backbone = CSPDarknet(in_ch=3, base_width=16)
        assert backbone.out_channels == [64, 128, 256]


class TestFPNNeck:
    def test_preserves_shapes(self):
        neck = FPNNeck([64, 128, 256])
        p3 = torch.randn(2, 64, 80, 80)
        p4 = torch.randn(2, 128, 40, 40)
        p5 = torch.randn(2, 256, 20, 20)
        o3, o4, o5 = neck((p3, p4, p5))
        assert o3.shape == p3.shape
        assert o4.shape == p4.shape
        assert o5.shape == p5.shape


class TestDetectionHead:
    def test_output_channels(self):
        head = DetectionHead([64, 128, 256], num_classes=35)
        feats = (
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        )
        outputs = head(feats)
        assert len(outputs) == 3
        # P3: 80*80 = 6400 anchors, P4: 40*40=1600, P5: 20*20=400
        assert outputs[0].shape == (2, 6400, 40)  # 5 + 35
        assert outputs[1].shape == (2, 1600, 40)
        assert outputs[2].shape == (2, 400, 40)


class TestSingleModalDetector:
    def test_forward(self):
        model = SingleModalDetector(in_channels=3, num_classes=35, base_width=16)
        x = torch.randn(2, 3, 640, 640)
        outputs = model(x)
        assert len(outputs) == 3
        total_preds = sum(o.shape[1] for o in outputs)
        assert total_preds == 6400 + 1600 + 400  # 8400

    def test_param_count_reasonable(self):
        model = SingleModalDetector(in_channels=3, num_classes=35, base_width=16)
        n = sum(p.numel() for p in model.parameters())
        # Nano should be roughly 1-5M params
        assert 500_000 < n < 10_000_000, f"Unexpected param count: {n}"


class TestMultiModalDetector:
    def test_forward(self):
        model = MultiModalDetector(
            modalities=["rgb", "lwir"],
            num_classes=35,
            base_width=16,
            fusion_method="attention",
        )
        images = {
            "rgb": torch.randn(2, 3, 640, 640),
            "lwir": torch.randn(2, 3, 640, 640),
        }
        outputs = model(images)
        assert len(outputs) == 3
        total_preds = sum(o.shape[1] for o in outputs)
        assert total_preds == 8400

    def test_concat_fusion(self):
        model = MultiModalDetector(
            modalities=["rgb", "lwir"],
            num_classes=35,
            base_width=16,
            fusion_method="concat",
        )
        images = {
            "rgb": torch.randn(1, 3, 640, 640),
            "lwir": torch.randn(1, 3, 640, 640),
        }
        outputs = model(images)
        assert len(outputs) == 3


class TestBuildModel:
    def test_single_modal(self):
        model = build_model("rgb", num_classes=35)
        assert isinstance(model, SingleModalDetector)

    def test_multi_modal(self):
        model = build_model("rgb+lwir", num_classes=35)
        assert isinstance(model, MultiModalDetector)


class TestCIoULoss:
    def test_identical_boxes(self):
        boxes = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        loss = ciou_loss(boxes, boxes)
        assert loss.item() < 0.01

    def test_nonoverlap_high_loss(self):
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        tgt = torch.tensor([[500.0, 500.0, 10.0, 10.0]])
        loss = ciou_loss(pred, tgt)
        assert loss.item() > 0.5


class TestFocalLoss:
    def test_correct_prediction_low_loss(self):
        fl = FocalLoss(alpha=0.25, gamma=2.0)
        # High logit for correct class
        pred = torch.tensor([[10.0, -10.0, -10.0]])
        target = torch.tensor([0])
        loss = fl(pred, target)
        assert loss.item() < 0.1

    def test_wrong_prediction_high_loss(self):
        fl = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.tensor([[-10.0, 10.0, -10.0]])
        target = torch.tensor([0])
        loss = fl(pred, target)
        assert loss.item() > 0.01


class TestDetectionLoss:
    def test_forward(self):
        criterion = DetectionLoss(num_classes=35)
        predictions = [
            torch.randn(2, 100, 40, requires_grad=True),
            torch.randn(2, 25, 40, requires_grad=True),
        ]
        targets = torch.zeros(2, 3, 5)
        targets[0, 0] = torch.tensor([0, 320, 320, 50, 50])
        targets[0, 1] = torch.tensor([1, 100, 100, 30, 30])
        targets[1, 0] = torch.tensor([5, 200, 200, 40, 40])
        target_counts = torch.tensor([2, 1])

        loss_dict = criterion(predictions, targets, target_counts)
        assert "loss" in loss_dict
        assert "box_loss" in loss_dict
        assert "cls_loss" in loss_dict
        assert "obj_loss" in loss_dict
        assert loss_dict["loss"].requires_grad


class TestBoxUtils:
    def test_cxcywh_to_xyxy_roundtrip(self):
        from mineinsight.losses import box_xyxy_to_cxcywh
        boxes = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        xyxy = box_cxcywh_to_xyxy(boxes)
        back = box_xyxy_to_cxcywh(xyxy)
        assert torch.allclose(boxes, back, atol=1e-5)

    def test_box_iou_self(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        iou = box_iou(boxes, boxes)
        assert torch.allclose(iou.diag(), torch.ones(2))

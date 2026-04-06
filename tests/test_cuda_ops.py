"""Tests for MineInsight CUDA-accelerated operations."""

import sys

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestCudaOps:
    def test_kernels_available(self):
        from mineinsight.cuda_ops import cuda_kernels_available

        avail = cuda_kernels_available()
        assert avail["cuda"] is True
        assert avail["detection_ops"] is True
        assert avail["fused_image_preprocess"] is True

    def test_cuda_box_iou_2d(self):
        from mineinsight.cuda_ops import cuda_box_iou_2d

        boxes1 = torch.tensor(
            [[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32, device="cuda"
        )
        boxes2 = torch.tensor(
            [[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32, device="cuda"
        )
        iou = cuda_box_iou_2d(boxes1, boxes2)
        assert iou.shape == (2, 2)
        assert abs(iou[0, 0].item() - 1.0) < 0.01  # self-IoU = 1
        assert iou[1, 1].item() < 0.01  # non-overlapping

    def test_cuda_focal_loss(self):
        from mineinsight.cuda_ops import cuda_focal_loss

        logits = torch.randn(10, 35, device="cuda")
        targets = torch.randint(0, 35, (10,), device="cuda")
        loss = cuda_focal_loss(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_cuda_nms_2d(self):
        from mineinsight.cuda_ops import cuda_nms_2d

        boxes = torch.tensor(
            [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]],
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
        keep = cuda_nms_2d(boxes, scores, 0.5)
        assert len(keep) == 2  # 2 non-overlapping kept
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_cuda_normalize_hwc_to_chw(self):
        from mineinsight.cuda_ops import cuda_normalize_hwc_to_chw

        img = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8, device="cuda")
        out = cuda_normalize_hwc_to_chw(img)
        assert out.shape == (3, 480, 640)
        assert out.dtype == torch.float32
        assert out.min() >= -0.01  # with mean=0, std=1, values in [0,1]
        assert out.max() <= 1.01

    def test_cuda_batch_normalize(self):
        from mineinsight.cuda_ops import cuda_batch_normalize

        imgs = torch.randint(0, 255, (4, 480, 640, 3), dtype=torch.uint8, device="cuda")
        out = cuda_batch_normalize(imgs)
        assert out.shape == (4, 3, 480, 640)
        assert out.dtype == torch.float32


class TestCustomCudaKernels:
    def test_fused_multimodal_preprocess(self):
        sys.path.insert(0, "src")
        import mineinsight_cuda_ops as ops

        rgb = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8, device="cuda")
        thermal = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8, device="cuda")
        out = ops.fused_multimodal_preprocess(rgb, thermal)
        assert out.shape == (6, 640, 640)
        assert out.dtype == torch.float32
        # First 3 channels from RGB, next 3 from thermal
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_fused_batch_multimodal_preprocess(self):
        sys.path.insert(0, "src")
        import mineinsight_cuda_ops as ops

        rgb = torch.randint(0, 255, (4, 640, 640, 3), dtype=torch.uint8, device="cuda")
        thermal = torch.randint(0, 255, (4, 640, 640, 3), dtype=torch.uint8, device="cuda")
        out = ops.fused_batch_multimodal_preprocess(rgb, thermal)
        assert out.shape == (4, 6, 640, 640)

    def test_fused_ciou_loss(self):
        sys.path.insert(0, "src")
        import mineinsight_cuda_ops as ops

        # Identical boxes → loss near 0
        boxes = torch.tensor(
            [[100, 100, 50, 50]] * 10, dtype=torch.float32, device="cuda"
        )
        loss = ops.fused_ciou_loss(boxes, boxes)
        assert loss.item() < 0.01

        # Different boxes → loss > 0
        pred = torch.tensor(
            [[100, 100, 50, 50]] * 10, dtype=torch.float32, device="cuda"
        )
        tgt = torch.tensor(
            [[200, 200, 50, 50]] * 10, dtype=torch.float32, device="cuda"
        )
        loss = ops.fused_ciou_loss(pred, tgt)
        assert loss.item() > 0.5

    def test_fused_detection_decode(self):
        sys.path.insert(0, "src")
        import mineinsight_cuda_ops as ops

        preds = torch.randn(1000, 40, device="cuda")  # 5 + 35 classes
        boxes, scores, labels = ops.fused_detection_decode(preds, 0.25)
        assert boxes.shape == (1000, 4)
        assert scores.shape == (1000,)
        assert labels.shape == (1000,)
        assert labels.dtype == torch.int32

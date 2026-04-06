"""Tests for MineInsight dataset loader and config utilities."""


import numpy as np
import pytest

from mineinsight.dataset import (
    ALL_CLASSES,
    MINE_IDS,
    NUM_CLASSES,
    MineInsightDataset,
    _parse_yolo_label,
    collate_fn,
)
from mineinsight.utils import load_config


class TestConstants:
    def test_class_counts(self):
        assert NUM_CLASSES == 58  # IDs 0-57
        assert len(ALL_CLASSES) == 58

    def test_mine_ids(self):
        assert len(MINE_IDS) > 0
        assert all(mid < NUM_CLASSES for mid in MINE_IDS)
        # Known mine IDs from targets_list.yaml
        assert 21 in MINE_IDS  # PMN
        assert 42 in MINE_IDS  # TMA-2
        assert 33 in MINE_IDS  # PFM-1


class TestParseYoloLabel:
    def test_valid_label(self, tmp_path):
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n3 0.2 0.8 0.05 0.05\n")
        targets = _parse_yolo_label(label, 640, 640)
        assert targets.shape == (2, 5)
        assert targets[0, 0] == 0  # class
        assert abs(targets[0, 1].item() - 320.0) < 1.0  # cx = 0.5 * 640

    def test_missing_label(self, tmp_path):
        targets = _parse_yolo_label(tmp_path / "nonexistent.txt", 640, 640)
        assert targets.shape == (0, 5)

    def test_empty_label(self, tmp_path):
        label = tmp_path / "empty.txt"
        label.write_text("")
        targets = _parse_yolo_label(label, 640, 640)
        assert targets.shape == (0, 5)


class TestMineInsightDataset:
    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a minimal dataset directory structure."""
        # Create RGB modality with one sequence
        seq_dir = tmp_path / "rgb" / "track1_seq1"
        img_dir = seq_dir / "images"
        lbl_dir = seq_dir / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 5 dummy images and labels
        for i in range(5):
            # Create a simple image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_dir / f"frame_{i:04d}.jpg"), img)

            # Create YOLO label
            label_text = f"0 0.5 0.5 0.1 0.1\n{i % 20 + 15} 0.3 0.7 0.05 0.05\n"
            (lbl_dir / f"frame_{i:04d}.txt").write_text(label_text)

        # Create LWIR modality
        lwir_dir = tmp_path / "lwir" / "track1_seq1"
        lwir_img_dir = lwir_dir / "images"
        lwir_lbl_dir = lwir_dir / "labels"
        lwir_img_dir.mkdir(parents=True)
        lwir_lbl_dir.mkdir(parents=True)

        for i in range(5):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(lwir_img_dir / f"frame_{i:04d}.jpg"), img)
            label_text = "0 0.5 0.5 0.1 0.1\n"
            (lwir_lbl_dir / f"frame_{i:04d}.txt").write_text(label_text)

        return tmp_path

    def test_single_modal_rgb(self, dataset_dir):
        ds = MineInsightDataset(
            root=dataset_dir,
            sequences=["track1_seq1"],
            modality="rgb",
            input_size=(640, 640),
            augment=False,
        )
        assert len(ds) == 5

        sample = ds[0]
        assert "image" in sample
        assert sample["image"].shape == (3, 640, 640)
        assert "targets" in sample
        assert sample["targets"].shape[1] == 5

    def test_multi_modal(self, dataset_dir):
        ds = MineInsightDataset(
            root=dataset_dir,
            sequences=["track1_seq1"],
            modality="rgb+lwir",
            input_size=(640, 640),
            augment=False,
        )
        assert len(ds) == 5

        sample = ds[0]
        assert "images" in sample
        assert "rgb" in sample["images"]
        assert "lwir" in sample["images"]
        # Concatenated image should have 6 channels
        assert sample["image"].shape == (6, 640, 640)

    def test_augmentation(self, dataset_dir):
        ds = MineInsightDataset(
            root=dataset_dir,
            sequences=["track1_seq1"],
            modality="rgb",
            input_size=(640, 640),
            augment=True,
            flip_lr=1.0,  # always flip for deterministic test
        )
        sample = ds[0]
        assert sample["image"].shape == (3, 640, 640)

    def test_collate(self, dataset_dir):
        ds = MineInsightDataset(
            root=dataset_dir,
            sequences=["track1_seq1"],
            modality="rgb",
            input_size=(640, 640),
            augment=False,
        )
        batch = collate_fn([ds[0], ds[1], ds[2]])
        assert batch["image"].shape[0] == 3
        assert batch["targets"].shape[0] == 3
        assert batch["target_counts"].shape[0] == 3

    def test_empty_sequence(self, dataset_dir):
        ds = MineInsightDataset(
            root=dataset_dir,
            sequences=["nonexistent_seq"],
            modality="rgb",
            input_size=(640, 640),
        )
        assert len(ds) == 0


class TestConfig:
    def test_load_config(self, tmp_path):
        config_content = """
[model]
architecture = "yolov8"
num_classes = 35
input_size = [640, 640]
modality = "rgb"

[model.fusion]
enabled = false

[training]
batch_size = 8
learning_rate = 0.001
epochs = 10

[data]
dataset_root = "/tmp/test"
train_sequences = ["track1_seq1"]
val_sequences = ["track1_seq2"]
test_sequences = ["track2_seq1"]

[data.augmentation]
mosaic = false
mixup = false

[loss]
box_weight = 7.5
cls_weight = 0.5

[checkpoint]
output_dir = "/tmp/ckpt"

[early_stopping]
enabled = true
patience = 10

[logging]
log_dir = "/tmp/logs"
tensorboard_dir = "/tmp/tb"
"""
        config_path = tmp_path / "test.toml"
        config_path.write_text(config_content)

        cfg = load_config(config_path)
        assert cfg.model.num_classes == 35
        assert cfg.training.epochs == 10
        assert cfg.training.learning_rate == 0.001
        assert cfg.data.train_sequences == ["track1_seq1"]
        assert cfg.loss.box_weight == 7.5
        assert cfg.early_stopping.patience == 10

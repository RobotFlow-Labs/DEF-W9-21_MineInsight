"""MineInsight: Multi-modal landmine detection for humanitarian demining robotics.

Paper: arXiv 2506.04842
Dataset: Multi-sensor (RGB, VIS-SWIR, LWIR, LiDAR) with 35 target classes.
"""

__version__ = "0.1.0"
__all__ = [
    "MineInsightDataset",
    "SingleModalDetector",
    "MultiModalDetector",
    "DetectionLoss",
]

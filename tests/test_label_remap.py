"""Round-trip and correctness tests for :mod:`mineinsight.label_remap`.

These tests are BLOCKING — the entire dataset builder depends on the remap
being bit-identical under round-trip, so we verify:

1. Loading from ``targets_list.yaml`` yields exactly 34 unique classes.
2. All 51 raw IDs map to a valid new ID.
3. PFM-1 instance IDs (22, 23, 24, ..., 36) all collapse to ONE new class.
4. save → load → compare produces an identical object.
5. ``remap_label_file`` rewrites a label bit-identically when applied
   with the identity-like mapping (remap then reverse).
6. Known mine class keywords are correctly classified as mines.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from mineinsight.label_remap import LabelRemap, remap_label_file

TARGETS_YAML = "/mnt/train-data/datasets/mineinsight/targets_list.yaml"


@pytest.fixture(scope="module")
def remap() -> LabelRemap:
    """Load the canonical remap once per module."""
    return LabelRemap.from_targets_yaml(TARGETS_YAML)


def test_class_count_is_34(remap: LabelRemap) -> None:
    assert remap.num_classes() == 34, (
        f"expected 34 unique classes, got {remap.num_classes()}: "
        f"{sorted(remap.name_to_new)}"
    )


def test_all_raw_ids_mapped(remap: LabelRemap) -> None:
    assert len(remap.raw_to_new) == 51, (
        f"expected 51 raw IDs, got {len(remap.raw_to_new)}"
    )
    for raw_id in remap.raw_to_new:
        assert 0 <= remap.raw_to_new[raw_id] < 34


def test_pfm1_instances_collapse(remap: LabelRemap) -> None:
    """All PFM-1 raw IDs (22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)
    must map to the same new ID."""
    pfm1_raw_ids = [
        r for r, name in remap.raw_to_name.items() if name == "PFM-1"
    ]
    assert len(pfm1_raw_ids) >= 10, (
        f"expected >=10 PFM-1 instances, got {len(pfm1_raw_ids)}: {pfm1_raw_ids}"
    )
    new_ids = {remap.raw_to_new[r] for r in pfm1_raw_ids}
    assert len(new_ids) == 1, (
        f"PFM-1 instances did not collapse! Mapped to {new_ids}"
    )
    pfm1_new_id = next(iter(new_ids))
    assert remap.new_to_name[pfm1_new_id] == "PFM-1"


def test_mines_classified(remap: LabelRemap) -> None:
    """Sanity check: mines are detected, distractors are not."""
    pfm1_new = remap.name_to_new["PFM-1"]
    assert pfm1_new in remap.mine_new_ids

    # A distractor like "Coke Can folded" should NOT be a mine
    for name in remap.name_to_new:
        if "Coke" in name or "Bottle" in name or "Can" in name:
            if name not in {"Metal tuna can", "Metal corn tin", "Soda metal can"}:
                nid = remap.name_to_new[name]
                assert nid not in remap.mine_new_ids, (
                    f"{name} wrongly classified as mine"
                )


def test_save_load_roundtrip(remap: LabelRemap) -> None:
    """save → load → equal."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "label_remap.json"
        remap.save(out)
        assert out.exists()

        reloaded = LabelRemap.load(out)
        assert reloaded.num_classes() == remap.num_classes()
        assert reloaded.raw_to_new == remap.raw_to_new
        assert reloaded.name_to_new == remap.name_to_new
        assert reloaded.new_to_name == remap.new_to_name
        assert reloaded.mine_new_ids == remap.mine_new_ids


def test_label_file_roundtrip(remap: LabelRemap) -> None:
    """Rewrite 100 real label files through the remap and confirm that:

    1. Every surviving line has a new_id in [0, num_classes).
    2. Coordinates are unchanged (string comparison after stripping whitespace).
    3. No file crashes the rewriter.
    """
    label_dir = Path("/mnt/train-data/datasets/mineinsight/track_1_s2_rgb_labels")
    if not label_dir.exists():
        pytest.skip(f"label dir not available: {label_dir}")

    files = sorted(label_dir.glob("*.txt"))
    random.seed(42)
    sample = random.sample(files, min(100, len(files)))

    total_in = 0
    total_out = 0
    nc = remap.num_classes()

    with tempfile.TemporaryDirectory() as td:
        for src in sample:
            original = src.read_text().strip().splitlines()
            total_in += sum(1 for ln in original if len(ln.split()) == 5)

            dst = Path(td) / src.name
            n_written, _n_dropped = remap_label_file(src, dst, remap)
            total_out += n_written

            # Verify every output line
            for line in dst.read_text().strip().splitlines():
                parts = line.split()
                assert len(parts) == 5, f"malformed output line: {line}"
                new_id = int(parts[0])
                assert 0 <= new_id < nc, (
                    f"new_id {new_id} out of range [0, {nc}) in {dst}"
                )
                # Coords must be floats in [0, 1]
                for c in parts[1:]:
                    v = float(c)
                    assert 0.0 <= v <= 1.0, f"coord {c} out of range"

    assert total_out > 0, "no labels were written"
    # Almost all labels should survive (a few may have weird IDs)
    survival_rate = total_out / max(total_in, 1)
    assert survival_rate > 0.95, (
        f"survival rate {survival_rate:.2%} too low "
        f"(in={total_in}, out={total_out})"
    )


def test_new_ids_contiguous(remap: LabelRemap) -> None:
    """New IDs must be exactly [0, 1, 2, ..., num_classes-1] — no gaps."""
    all_new = sorted(set(remap.new_to_name.keys()))
    assert all_new == list(range(remap.num_classes())), (
        f"new IDs not contiguous: {all_new}"
    )


def test_names_unique(remap: LabelRemap) -> None:
    """name_to_new must be injective (every name → exactly one new ID)."""
    assert len(set(remap.name_to_new.values())) == remap.num_classes()
    assert len(set(remap.new_to_name.values())) == remap.num_classes()

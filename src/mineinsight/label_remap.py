"""Class ID remap: MineInsight raw IDs [1..57] → Ultralytics YOLO IDs [0..33].

The MineInsight dataset assigns a unique numeric ID to every physical object
*instance* placed in the field — so every PFM-1 mine gets its own ID (22, 23,
24, ..., 36), even though they are all the same mine type. This makes the
detection task impossible: the model cannot learn to distinguish two physically
identical PFM-1 mines from each other just by their ID.

This module collapses instance IDs into **object-type IDs**. All PFM-1
instances → one class. All PMN → one class. Etc. Expected result:
`targets_list.yaml` has 51 ID entries → 34 unique object names → 34 new classes
`[0..33]`, which is the correct detection schema.

The remap is idempotent and round-trippable. A unit test verifies this.

Usage
-----
    from mineinsight.label_remap import LabelRemap

    remap = LabelRemap.from_targets_yaml(
        "/mnt/train-data/datasets/mineinsight/targets_list.yaml"
    )
    new_id = remap.raw_to_new[22]          # -> PFM-1 index (0..33)
    name   = remap.new_to_name[new_id]     # -> "PFM-1"
    remap.save("label_remap.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class LabelRemap:
    """Bidirectional mapping between raw MineInsight IDs and new contiguous IDs.

    Attributes
    ----------
    raw_to_name : dict[int, str]
        Every raw ID (1..57, with gaps) → canonical object name from
        ``targets_list.yaml``.
    name_to_new : dict[str, int]
        Each of the 34 unique object names → a contiguous new ID (0..33),
        assigned in sorted order by name for determinism.
    raw_to_new : dict[int, int]
        Convenience: raw ID → new ID.
    new_to_name : dict[int, str]
        Convenience: new ID → name.
    mine_new_ids : set[int]
        New IDs that correspond to actual landmines (for mine-specific mAP).
    """

    raw_to_name: dict[int, str] = field(default_factory=dict)
    name_to_new: dict[str, int] = field(default_factory=dict)
    raw_to_new: dict[int, int] = field(default_factory=dict)
    new_to_name: dict[int, str] = field(default_factory=dict)
    mine_new_ids: set[int] = field(default_factory=set)

    # Mines are identified by these substrings in the object name.
    # Everything else is a distractor.
    _MINE_KEYWORDS: tuple[str, ...] = (
        "PFM", "PMN", "PROM", "MON", "TC-", "TM-", "TMA", "TMM", "M6", "M-35",
        "Type 72", "VS-", "C-3",
    )

    @classmethod
    def from_targets_yaml(cls, yaml_path: str | Path) -> LabelRemap:
        """Build a remap from the dataset's canonical ``targets_list.yaml``.

        Parameters
        ----------
        yaml_path : str | Path
            Path to ``targets_list.yaml``. Must have top-level keys "Track1",
            "Track2", "Track3" (or subset), each a list of ``{id, name, text}``.

        Returns
        -------
        LabelRemap
            A fully populated remap ready for use.

        Raises
        ------
        FileNotFoundError
            If ``yaml_path`` does not exist.
        ValueError
            If ``yaml_path`` is malformed (no valid track entries).
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"targets_list.yaml not found at {yaml_path}")

        log.info(f"[LABEL_REMAP] loading from {yaml_path}")
        data = yaml.safe_load(yaml_path.read_text())
        if not data:
            raise ValueError(f"empty or invalid targets_list.yaml: {yaml_path}")

        remap = cls()

        # Pass 1: collect all (raw_id, name) pairs
        for track_key, entries in data.items():
            if not isinstance(entries, list):
                log.warning(f"[LABEL_REMAP] skipping non-list key: {track_key}")
                continue
            for entry in entries:
                raw_id = int(entry["id"])
                name = entry["name"].strip()
                # Clean trailing punctuation (yaml sometimes has "TC-3.6 ")
                name = name.rstrip(" ,;.")
                remap.raw_to_name[raw_id] = name

        if not remap.raw_to_name:
            raise ValueError(f"no entries parsed from {yaml_path}")

        # Pass 2: build contiguous new IDs from unique names (sorted for determinism)
        unique_names = sorted(set(remap.raw_to_name.values()))
        remap.name_to_new = {name: i for i, name in enumerate(unique_names)}
        remap.new_to_name = {i: name for name, i in remap.name_to_new.items()}

        # Pass 3: raw → new shortcut
        remap.raw_to_new = {
            raw_id: remap.name_to_new[name]
            for raw_id, name in remap.raw_to_name.items()
        }

        # Pass 4: mark which new IDs are mines
        for new_id, name in remap.new_to_name.items():
            if any(kw in name for kw in cls._MINE_KEYWORDS):
                remap.mine_new_ids.add(new_id)

        log.info(
            f"[LABEL_REMAP] parsed {len(remap.raw_to_name)} raw IDs → "
            f"{len(remap.name_to_new)} unique classes "
            f"({len(remap.mine_new_ids)} mines, "
            f"{len(remap.name_to_new) - len(remap.mine_new_ids)} distractors)",
        )
        return remap

    def remap_raw(self, raw_id: int) -> int:
        """Translate a raw MineInsight ID to a new contiguous ID.

        Raises
        ------
        KeyError
            If ``raw_id`` is not present in the mapping.
        """
        if raw_id not in self.raw_to_new:
            raise KeyError(
                f"raw class ID {raw_id} not found in remap "
                f"(known range: {min(self.raw_to_new)}..{max(self.raw_to_new)})",
            )
        return self.raw_to_new[raw_id]

    def num_classes(self) -> int:
        """Return the number of contiguous new classes (should be 34)."""
        return len(self.name_to_new)

    def save(self, out_path: str | Path) -> None:
        """Write the full remap as a JSON file for reproducibility."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "raw_to_name": {str(k): v for k, v in self.raw_to_name.items()},
            "name_to_new": self.name_to_new,
            "raw_to_new": {str(k): v for k, v in self.raw_to_new.items()},
            "new_to_name": {str(k): v for k, v in self.new_to_name.items()},
            "mine_new_ids": sorted(self.mine_new_ids),
            "num_classes": self.num_classes(),
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        log.info(f"[LABEL_REMAP] saved {out_path}")

    @classmethod
    def load(cls, in_path: str | Path) -> LabelRemap:
        """Load a remap previously written by :meth:`save`."""
        in_path = Path(in_path)
        payload = json.loads(in_path.read_text())
        return cls(
            raw_to_name={int(k): v for k, v in payload["raw_to_name"].items()},
            name_to_new=payload["name_to_new"],
            raw_to_new={int(k): v for k, v in payload["raw_to_new"].items()},
            new_to_name={int(k): v for k, v in payload["new_to_name"].items()},
            mine_new_ids=set(payload["mine_new_ids"]),
        )


def remap_label_file(
    in_path: Path, out_path: Path, remap: LabelRemap,
    drop_unknown: bool = True,
) -> tuple[int, int]:
    """Rewrite a single YOLO label file with remapped class IDs.

    The label format is:  ``class_id cx cy w h`` (all normalized 0-1).
    Only the ``class_id`` column is rewritten.

    Parameters
    ----------
    in_path : Path
        Source label file (YOLO format).
    out_path : Path
        Destination label file.
    remap : LabelRemap
        The bidirectional mapping.
    drop_unknown : bool
        If True, silently drop lines whose raw class ID is not in the remap.
        If False, raise KeyError.

    Returns
    -------
    (n_written, n_dropped) : tuple[int, int]
    """
    n_written = 0
    n_dropped = 0
    lines_out: list[str] = []
    for line in in_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            n_dropped += 1
            continue
        try:
            raw_id = int(parts[0])
        except ValueError:
            n_dropped += 1
            continue
        if raw_id not in remap.raw_to_new:
            if drop_unknown:
                n_dropped += 1
                continue
            raise KeyError(f"raw id {raw_id} not in remap (file={in_path})")
        new_id = remap.raw_to_new[raw_id]
        lines_out.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        n_written += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return n_written, n_dropped

"""Load extxyz frames into lightweight records, recovering the trajectory id.

The trajectory id is the extxyz `config_type` field (e.g. ``y640-1-gen4-2025``).
It is the correct unit for the train/val split: frames sharing a config_type are
one inherent structure vibrating (thermal rattle), so a frame-level split would
leak near-duplicates. ASE parses `config_type` cleanly into ``atoms.info`` for
every frame in our data; we fall back to a regex on any raw comment only if that
key is ever absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io import iread

from ..config import DataConfig, SourceConfig


@dataclass
class FrameRecord:
    """One structure: positions + cell + species, plus provenance for splitting."""

    positions: np.ndarray   # (N, 3) cartesian, angstrom
    numbers: np.ndarray     # (N,) atomic numbers
    cell: np.ndarray        # (3, 3) angstrom
    traj_id: str            # config_type — the split key
    source_path: str
    role: str               # "train_val" | "train_only"
    frame_index: int        # index within the source file

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]


def recover_traj_id(atoms: Atoms, regex: str) -> str:
    """Return the trajectory id for a frame.

    Primary source is ``atoms.info['config_type']``. If absent, apply ``regex``
    to a reconstructed ``key=value`` comment string as a fallback.
    """
    ct = atoms.info.get("config_type")
    if ct is not None:
        return str(ct)
    # Fallback: rebuild a comment-ish string and regex it.
    comment = " ".join(f"{k}={v}" for k, v in atoms.info.items())
    m = re.search(regex, comment)
    if m:
        return m.group(1)
    raise ValueError(
        f"could not recover trajectory id (no config_type, regex {regex!r} failed) "
        f"for frame with info keys {list(atoms.info)}"
    )


def load_source(source: SourceConfig, regex: str) -> list[FrameRecord]:
    """Read every frame of one extxyz file into FrameRecords (streamed)."""
    records: list[FrameRecord] = []
    for i, atoms in enumerate(iread(source.path)):
        records.append(
            FrameRecord(
                positions=atoms.get_positions().astype(np.float64),
                numbers=atoms.get_atomic_numbers().astype(np.int64),
                cell=np.asarray(atoms.get_cell()).astype(np.float64),
                traj_id=recover_traj_id(atoms, regex),
                source_path=source.path,
                role=source.role,
                frame_index=i,
            )
        )
    return records


def load_all(data_cfg: DataConfig) -> list[FrameRecord]:
    """Load and concatenate all configured sources (no thinning/dedup yet)."""
    records: list[FrameRecord] = []
    for source in data_cfg.sources:
        records.extend(load_source(source, data_cfg.traj_id_regex))
    return records

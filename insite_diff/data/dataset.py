"""Thinning, dedup, species mapping, and the torch Dataset.

Structures have variable atom counts (80 vs 640), so items are single frames;
the trainer loops over a batch as a list rather than stacking a ragged tensor.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from ase.data import atomic_numbers
from torch.utils.data import Dataset

from ..config import Config, DataConfig
from ..geometry import rmsd_same_atoms
from .extxyz import FrameRecord, load_all
from .split import DataSplit, trajectory_split


def species_to_index(species: list[str]) -> dict[int, int]:
    """Map atomic number -> contiguous type index in the order given by config.species."""
    return {atomic_numbers[s]: k for k, s in enumerate(species)}


def _minimage_rmsd(a: FrameRecord, b: FrameRecord) -> float:
    return rmsd_same_atoms(a.positions, b.positions, a.cell)


def _group_by_traj(records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    groups: dict[str, list[FrameRecord]] = defaultdict(list)
    for r in records:
        groups[r.traj_id].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.frame_index)
    return groups


def dedup_within_trajectory(records: list[FrameRecord], rmsd_thresh: float) -> list[FrameRecord]:
    """Drop frames within rmsd_thresh of ANY already-kept frame in the same trajectory.

    Compares against all kept frames (not just the previous one) because the k80
    duplicates are exact but non-adjacent (frames 0-4 repeated at 125-129).
    Trajectories are tiny (<=10 frames) so the O(k^2) compare is cheap. No-op when
    rmsd_thresh <= 0.
    """
    if rmsd_thresh <= 0:
        return list(records)
    kept: list[FrameRecord] = []
    for group in _group_by_traj(records).values():
        kept_in_group: list[FrameRecord] = []
        for r in group:
            if any(_minimage_rmsd(k, r) < rmsd_thresh for k in kept_in_group):
                continue
            kept_in_group.append(r)
        kept.extend(kept_in_group)
    return kept


def thin_within_trajectory(records: list[FrameRecord], stride: int) -> list[FrameRecord]:
    """Keep every ``stride``-th frame within each trajectory (order by frame_index)."""
    if stride <= 1:
        return list(records)
    kept: list[FrameRecord] = []
    for group in _group_by_traj(records).values():
        kept.extend(group[::stride])
    return kept


class StructureDataset(Dataset):
    """Yields per-frame tensors. Species are mapped to type indices via config.species."""

    def __init__(self, records: list[FrameRecord], species: list[str]):
        self.records = records
        self.species = species
        self.z_to_type = species_to_index(species)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        try:
            types = np.array([self.z_to_type[z] for z in r.numbers], dtype=np.int64)
        except KeyError as e:  # a species not declared in config.species
            raise KeyError(f"atomic number {e} not in configured species {self.species}") from e
        return {
            "positions": torch.tensor(r.positions, dtype=torch.float32),
            "numbers": torch.tensor(r.numbers, dtype=torch.long),
            "types": torch.tensor(types, dtype=torch.long),
            "cell": torch.tensor(r.cell, dtype=torch.float32),
            "n_atoms": r.n_atoms,
            "traj_id": r.traj_id,
        }


def prepare_records(data_cfg: DataConfig) -> list[FrameRecord]:
    """Load -> dedup -> thin (dedup before thinning so duplicates don't survive striding)."""
    records = load_all(data_cfg)
    records = dedup_within_trajectory(records, data_cfg.dedup_rmsd)
    records = thin_within_trajectory(records, data_cfg.stride)
    return records


def build_datasets(cfg: Config) -> tuple[StructureDataset, StructureDataset, DataSplit]:
    """Full pipeline: records -> trajectory split -> train/val datasets."""
    records = prepare_records(cfg.data)
    split = trajectory_split(records, cfg.split)
    train_ds = StructureDataset(split.train, cfg.data.species)
    val_ds = StructureDataset(split.val, cfg.data.species)
    return train_ds, val_ds, split

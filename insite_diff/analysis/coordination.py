"""Coordination-number distributions and mean bond length (PBC-aware).

The In-O coordination cutoff should sit at the first minimum of the In-O RDF
(~2.7 A for amorphous In2O3); it is a config parameter.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def coordination(frames: list[Atoms], z_center: int, z_neighbor: int, cutoff: float,
                 masks: list[np.ndarray] | None = None,
                 center_mobile_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (coordination_numbers, bond_lengths) pooled over frames.

    coordination_numbers: one entry per z_center atom (count of z_neighbor within cutoff).
    bond_lengths: every z_center-z_neighbor distance within cutoff.
    If ``center_mobile_only``, restrict the z_center atoms to mobile ones (per ``masks``).
    """
    if masks is None:
        masks = [None] * len(frames)
    cns: list[int] = []
    bonds: list[float] = []
    for atoms, mask in zip(frames, masks):
        num = atoms.get_atomic_numbers()
        i, j, d = neighbor_list("ijd", atoms, cutoff)
        sel = (num[i] == z_center) & (num[j] == z_neighbor)
        center_atoms = num == z_center
        if center_mobile_only:
            sel &= mask[i]
            center_atoms = center_atoms & mask
        bonds.extend(d[sel].tolist())
        counts = np.zeros(len(atoms), dtype=int)
        np.add.at(counts, i[sel], 1)
        cns.extend(counts[center_atoms].tolist())
    return np.array(cns), np.array(bonds)


def cn_histogram(cns: np.ndarray, max_cn: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Normalized histogram of coordination numbers over 0..max_cn."""
    bins = np.arange(0, max_cn + 2) - 0.5
    hist, _ = np.histogram(cns, bins=bins, density=True)
    return np.arange(0, max_cn + 1), hist


def summary(frames: list[Atoms], z_center: int = 49, z_neighbor: int = 8,
            cutoff: float = 2.7, masks: list[np.ndarray] | None = None,
            center_mobile_only: bool = False) -> dict:
    cns, bonds = coordination(frames, z_center, z_neighbor, cutoff, masks, center_mobile_only)
    return {
        "mean_cn": float(cns.mean()) if len(cns) else float("nan"),
        "mean_bond": float(bonds.mean()) if len(bonds) else float("nan"),
        "cns": cns,
        "bonds": bonds,
    }

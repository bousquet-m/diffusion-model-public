"""Small PBC geometry helpers shared across the package."""
from __future__ import annotations

import numpy as np


def min_image_displacement(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap Cartesian displacements into the minimum image for a given cell.

    ``delta`` is (..., 3) Cartesian; rows of ``cell`` are lattice vectors.
    """
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    frac -= np.round(frac)
    return frac @ cell


def rmsd_same_atoms(pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray) -> float:
    """Minimum-image RMSD between two frames with identical atom ordering."""
    disp = min_image_displacement(pos_b - pos_a, cell)
    return float(np.sqrt((disp ** 2).sum(axis=1).mean()))

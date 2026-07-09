"""Context/mobile partition for inpainting.

Two modes:
  - "interior": a contiguous cube (edge = region_frac x box edge) — atoms inside
    are mobile (regenerated), atoms outside are context (clamped). This is the
    milestone-one validation target; it needs a box large enough that the
    surrounding context is thicker than the denoiser receptive field.
  - "random_subset": a random fraction of atoms are mobile. No spatial buffer,
    so usable even on small cells.

Returns a boolean ``mobile_mask`` (True = mobile / to-be-generated).
"""
from __future__ import annotations

import numpy as np

from ..config import MaskConfig


def _fractional(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Cartesian -> fractional coords, wrapped into [0, 1). Rows of cell are lattice vectors."""
    frac = positions @ np.linalg.inv(cell)
    return frac - np.floor(frac)


def interior_mask(
    positions: np.ndarray,
    cell: np.ndarray,
    region_frac: float,
    center: str = "box_center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mobile = atoms inside a fractional cube of edge ``region_frac`` about a center."""
    frac = _fractional(positions, cell)
    if center == "box_center":
        center_frac = np.full(3, 0.5)
    elif center == "random":
        rng = rng or np.random.default_rng()
        center_frac = rng.random(3)
    else:
        raise ValueError(f"unknown interior center {center!r}")
    delta = frac - center_frac
    delta -= np.round(delta)                       # minimum image in fractional space
    half = region_frac / 2.0
    return np.all(np.abs(delta) < half, axis=1)


def random_subset_mask(
    n_atoms: int, mobile_frac: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n_mobile = int(round(mobile_frac * n_atoms))
    n_mobile = max(1, min(n_atoms - 1, n_mobile))  # keep at least one of each
    idx = rng.choice(n_atoms, size=n_mobile, replace=False)
    mask = np.zeros(n_atoms, dtype=bool)
    mask[idx] = True
    return mask


def partition(
    positions: np.ndarray,
    cell: np.ndarray,
    cfg: MaskConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dispatch to the configured masking mode. Returns ``mobile_mask`` (N,) bool."""
    if cfg.mode == "interior":
        return interior_mask(positions, cell, cfg.interior.region_frac, cfg.interior.center, rng)
    if cfg.mode == "random_subset":
        return random_subset_mask(positions.shape[0], cfg.random_subset.mobile_frac, rng)
    raise ValueError(f"unknown mask mode {cfg.mode!r}")

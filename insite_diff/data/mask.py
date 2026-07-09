"""Context/mobile partition for inpainting, parameterized by ATOM fraction.

We select exactly ``round(mask_frac * N)`` atoms as mobile (to regenerate); the
rest are context (clamped). The region is grown to that atom count in one of three
geometries:

  - "sphere": the K atoms nearest a center (minimum-image radial distance).
  - "cube":   the K atoms nearest by Chebyshev (L-inf) min-image distance -> a cube.
  - "slab":   the top (or bottom) K atoms along an axis. This is the geometry
              milestone two needs for surfaces; it is unused this milestone but
              implemented and unit-tested now.

``mask_frac = 1.0`` selects all atoms (pure unconditional generation, no context).
Returns a boolean ``mobile_mask`` (True = mobile / to-be-generated).
"""
from __future__ import annotations

import numpy as np

from ..config import MaskConfig


def n_mobile(mask_frac: float, n_atoms: int) -> int:
    """Number of mobile atoms for a given fraction (rounded, clamped to [0, N])."""
    return int(max(0, min(n_atoms, round(mask_frac * n_atoms))))


def _select_smallest(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the k atoms with the smallest scores (stable ties)."""
    mask = np.zeros(len(scores), dtype=bool)
    if k > 0:
        mask[np.argsort(scores, kind="stable")[:k]] = True
    return mask


def _center_frac(center: str, rng: np.random.Generator | None) -> np.ndarray:
    if center == "box_center":
        return np.full(3, 0.5)
    if center == "random":
        rng = rng or np.random.default_rng()
        return rng.random(3)
    raise ValueError(f"unknown mask center {center!r}")


def _min_image_delta_cart(positions: np.ndarray, cell: np.ndarray,
                          center_frac: np.ndarray) -> np.ndarray:
    """Cartesian minimum-image displacement of each atom from a fractional center."""
    frac = positions @ np.linalg.inv(cell)
    d = frac - center_frac
    d -= np.round(d)                       # minimum image in fractional space
    return d @ cell                        # back to Angstrom


def sphere_mask(positions, cell, mask_frac, center="box_center", rng=None) -> np.ndarray:
    d = _min_image_delta_cart(positions, cell, _center_frac(center, rng))
    return _select_smallest(np.linalg.norm(d, axis=1), n_mobile(mask_frac, len(positions)))


def cube_mask(positions, cell, mask_frac, center="box_center", rng=None) -> np.ndarray:
    d = _min_image_delta_cart(positions, cell, _center_frac(center, rng))
    return _select_smallest(np.max(np.abs(d), axis=1), n_mobile(mask_frac, len(positions)))


def slab_mask(positions, cell, mask_frac, axis=2, side="top") -> np.ndarray:
    """Top/bottom K atoms along ``axis`` (raw coordinate; surfaces are non-periodic there)."""
    coord = positions[:, axis]
    if side == "top":
        scores = -coord            # largest coordinate first
    elif side == "bottom":
        scores = coord
    else:
        raise ValueError(f"slab_side must be top|bottom, got {side!r}")
    return _select_smallest(scores, n_mobile(mask_frac, len(positions)))


def partition(positions: np.ndarray, cell: np.ndarray, cfg: MaskConfig,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Dispatch to the configured geometry. Returns ``mobile_mask`` (N,) bool."""
    if cfg.geometry == "sphere":
        return sphere_mask(positions, cell, cfg.mask_frac, cfg.center, rng)
    if cfg.geometry == "cube":
        return cube_mask(positions, cell, cfg.mask_frac, cfg.center, rng)
    if cfg.geometry == "slab":
        return slab_mask(positions, cell, cfg.mask_frac, cfg.slab_axis, cfg.slab_side)
    raise ValueError(f"unknown mask geometry {cfg.geometry!r}")

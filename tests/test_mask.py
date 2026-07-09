"""Atom-fraction masking: exact counts per geometry, mask_frac=1.0 -> no context."""
import numpy as np
import pytest

from insite_diff.config import MaskConfig
from insite_diff.data.mask import (cube_mask, n_mobile, partition, slab_mask,
                                    sphere_mask)


def _structure(n=200, box=20.25, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * box
    pos = rng.random((n, 3)) @ cell
    return pos, cell


@pytest.mark.parametrize("mask_frac", [0.0, 0.016, 0.05, 0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("geometry", ["sphere", "cube", "slab"])
def test_mask_frac_gives_exact_atom_count(mask_frac, geometry):
    pos, cell = _structure()
    n = len(pos)
    expected = n_mobile(mask_frac, n)
    if geometry == "sphere":
        m = sphere_mask(pos, cell, mask_frac)
    elif geometry == "cube":
        m = cube_mask(pos, cell, mask_frac)
    else:
        m = slab_mask(pos, cell, mask_frac)
    assert int(m.sum()) == expected


def test_mask_frac_one_has_zero_context():
    pos, cell = _structure()
    for geom in ("sphere", "cube", "slab"):
        cfg = MaskConfig(mask_frac=1.0, geometry=geom)
        m = partition(pos, cell, cfg)
        assert m.all() and int((~m).sum()) == 0


def test_sphere_selects_the_nearest_atoms():
    # nearest-to-center atoms must all be closer than any unselected atom
    pos, cell = _structure()
    center = np.full(3, 0.5) @ cell
    m = sphere_mask(pos, cell, 0.2, center="box_center")
    r = np.linalg.norm(pos - center, axis=1)   # box_center, no wrap needed here
    assert r[m].max() <= r[~m].min() + 1e-9


def test_slab_top_selects_largest_z():
    pos, cell = _structure()
    m = slab_mask(pos, cell, 0.25, axis=2, side="top")
    assert pos[m, 2].min() >= pos[~m, 2].max() - 1e-9


def test_slab_bottom_selects_smallest_z():
    pos, cell = _structure()
    m = slab_mask(pos, cell, 0.25, axis=2, side="bottom")
    assert pos[m, 2].max() <= pos[~m, 2].min() + 1e-9


def test_partition_dispatch_matches_geometry_helpers():
    pos, cell = _structure()
    cfg = MaskConfig(mask_frac=0.15, geometry="cube", center="box_center")
    assert np.array_equal(partition(pos, cell, cfg),
                          cube_mask(pos, cell, 0.15, center="box_center"))

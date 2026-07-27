"""Slab-with-vacuum machinery: per-axis periodicity in the graph / wrap / min-image,
and a region-confined prior. The invariant that matters for surfaces is that an OPEN
(non-periodic) axis behaves like open space — no bonding, wrapping, or spawning across
the vacuum — while the in-plane axes stay periodic, and bulk (all-periodic) is unchanged.
"""
import numpy as np
import torch

from insite_diff.data.graph import build_graph_torch
from insite_diff.diffusion.noising import min_image, uniform_init
from insite_diff.diffusion.sampler import _wrap


def test_graph_open_axis_does_not_bond_across_vacuum():
    # Two atoms near opposite z-faces of the cell: ~1 Å apart through the periodic image
    # in z, but ~9 Å apart within the cell. Periodic z -> they bond; open z -> they don't.
    cell = torch.eye(3) * 10.0
    pos = torch.tensor([[5.0, 5.0, 0.5], [5.0, 5.0, 9.5]])
    cutoff = 2.0

    g_pbc = build_graph_torch(pos, cell, cutoff, pbc=[True, True, True])
    assert g_pbc.edge_index.shape[1] == 2, "periodic z should bond across the boundary"

    g_open = build_graph_torch(pos, cell, cutoff, pbc=[True, True, False])
    assert g_open.edge_index.shape[1] == 0, "open z must NOT bond across the vacuum"


def test_graph_open_axis_keeps_inplane_periodicity():
    # Atoms near opposite x-faces (in-plane, still periodic) must bond even with open z.
    cell = torch.eye(3) * 10.0
    pos = torch.tensor([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    g = build_graph_torch(pos, cell, 2.0, pbc=[True, True, False])
    assert g.edge_index.shape[1] == 2, "in-plane x stays periodic"


def test_graph_all_periodic_unchanged():
    # pbc=None and pbc=[True,True,True] must be identical to the bulk default.
    torch.manual_seed(0)
    cell = torch.eye(3) * 12.0
    pos = torch.rand(50, 3) * 12.0
    a = build_graph_torch(pos, cell, 4.0)
    b = build_graph_torch(pos, cell, 4.0, pbc=[True, True, True])
    assert a.edge_index.shape == b.edge_index.shape
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.allclose(a.edge_vec, b.edge_vec)


def test_min_image_open_axis_keeps_raw_displacement():
    cell = torch.eye(3) * 10.0
    delta = torch.tensor([[0.0, 0.0, 9.0]])           # 9 Å apart in z
    # periodic: min image is -1 Å; open: stays +9 Å
    assert torch.allclose(min_image(delta, cell)[0, 2], torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(min_image(delta, cell, pbc=[True, True, False])[0, 2],
                          torch.tensor(9.0), atol=1e-5)


def test_wrap_leaves_open_axis_unwrapped():
    cell = torch.eye(3) * 10.0
    pos = torch.tensor([[1.0, 1.0, 12.0]])            # z above the cell (into "vacuum")
    assert torch.allclose(_wrap(pos, cell)[0, 2], torch.tensor(2.0), atol=1e-5)   # periodic folds
    assert torch.allclose(_wrap(pos, cell, pbc=[True, True, False])[0, 2],
                          torch.tensor(12.0), atol=1e-5)                          # open: untouched


def test_uniform_init_bounds_confine_to_band():
    cell = torch.eye(3) * 10.0
    # confine z to the lower 40% (material slab), x,y full
    bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.4]])
    x = uniform_init(500, cell, generator=torch.Generator().manual_seed(0), bounds=bounds)
    assert x[:, 2].max() <= 4.0 + 1e-4 and x[:, 2].min() >= -1e-4
    assert x[:, 0].max() > 5.0                        # x still fills the cell


def test_uniform_init_no_bounds_unchanged():
    cell = torch.eye(3) * 8.0
    a = uniform_init(100, cell, generator=torch.Generator().manual_seed(3))
    b = uniform_init(100, cell, generator=torch.Generator().manual_seed(3), bounds=None)
    assert torch.allclose(a, b)

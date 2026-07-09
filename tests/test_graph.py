"""The GPU-native torch neighbor list must match ASE's minimum-image graph."""
import numpy as np
import torch

from insite_diff.data.graph import build_graph, build_graph_torch


def _structure(n=120, box=15.0, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * box
    pos = rng.random((n, 3)) @ cell
    return pos, cell


def _edge_set(edge_index):
    return {(int(s), int(d)) for s, d in edge_index.t().tolist()}


def test_torch_graph_matches_ase_edge_set():
    pos, cell = _structure()
    cutoff = 4.0
    g_ase = build_graph(pos, cell, cutoff, max_neighbors=0)
    g_torch = build_graph_torch(torch.tensor(pos, dtype=torch.float64),
                                torch.tensor(cell, dtype=torch.float64), cutoff, max_neighbors=0)
    assert _edge_set(g_ase.edge_index) == _edge_set(g_torch.edge_index)
    assert g_ase.edge_index.shape[1] == g_torch.edge_index.shape[1]


def test_torch_graph_edge_vectors_match_ase():
    pos, cell = _structure()
    cutoff = 4.0
    g_ase = build_graph(pos, cell, cutoff, max_neighbors=0)
    g_torch = build_graph_torch(torch.tensor(pos, dtype=torch.float64),
                                torch.tensor(cell, dtype=torch.float64), cutoff, max_neighbors=0)
    # index edges by (src,dst) and compare edge vectors / lengths
    ase = {(int(s), int(d)): v for (s, d), v in
           zip(g_ase.edge_index.t().tolist(), g_ase.edge_vec.tolist())}
    for (s, d), v in zip(g_torch.edge_index.t().tolist(), g_torch.edge_vec.tolist()):
        assert np.allclose(ase[(s, d)], v, atol=1e-6)


def test_torch_graph_respects_max_neighbors():
    pos, cell = _structure(n=200, box=15.0)
    g = build_graph_torch(torch.tensor(pos), torch.tensor(cell, dtype=torch.float32),
                          cutoff=5.0, max_neighbors=8)
    # per-center degree must not exceed the cap
    _, counts = torch.unique(g.edge_index[1], return_counts=True)
    assert int(counts.max()) <= 8

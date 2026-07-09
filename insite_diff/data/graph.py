"""PBC minimum-image neighbor graph construction.

Built fresh on (noised) positions at every use. We delegate the periodic
bookkeeping to ASE's ``neighbor_list`` (``ijD``), which returns, for cutoff
``r``, all directed pairs (i, j) within ``r`` together with the minimum-image
displacement vector D = r_j - r_i (including periodic offsets). This is exactly
the edge set a message-passing denoiser consumes, and it handles receptive-field
wrap-around correctly for small cells.

Edge vectors are geometric inputs to the (equivariant) network, not something we
backprop position gradients through, so numpy construction per call is fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list


@dataclass
class Graph:
    edge_index: torch.Tensor   # (2, E) long — rows [src (j), dst (i)]
    edge_vec: torch.Tensor     # (E, 3) float — r_i - r_j (points into dst)
    edge_len: torch.Tensor     # (E,) float
    n_nodes: int


def _cap_neighbors(i: np.ndarray, j: np.ndarray, d: np.ndarray, D: np.ndarray,
                   max_neighbors: int) -> tuple[np.ndarray, ...]:
    """Keep only the ``max_neighbors`` nearest j for each center i."""
    order = np.lexsort((d, i))            # sort by center, then distance
    i, j, d, D = i[order], j[order], d[order], D[order]
    keep = np.ones(len(i), dtype=bool)
    # rank within each center group
    _, starts = np.unique(i, return_index=True)
    rank = np.arange(len(i)) - np.repeat(starts, np.diff(np.append(starts, len(i))))
    keep &= rank < max_neighbors
    return i[keep], j[keep], d[keep], D[keep]


def build_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    numbers: np.ndarray | None = None,
    max_neighbors: int = 0,
    pbc: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Graph:
    """Build a minimum-image neighbor graph.

    Returns edges as (src=j, dst=i) with ``edge_vec = r_i - r_j`` so that a
    message from neighbor j to center i carries the displacement pointing at i.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    n = positions.shape[0]
    atoms = Atoms(
        numbers=numbers if numbers is not None else np.ones(n, dtype=int),
        positions=positions,
        cell=np.asarray(cell, dtype=np.float64),
        pbc=pbc,
    )
    # i = center, j = neighbor, D = r_j - r_i (min image, with periodic offset)
    i, j, d, D = neighbor_list("ijdD", atoms, cutoff)
    if max_neighbors and len(i):
        i, j, d, D = _cap_neighbors(i, j, d, D, max_neighbors)

    edge_index = torch.tensor(np.stack([j, i]), dtype=torch.long, device=device)  # (2,E): src=j, dst=i
    edge_vec = torch.tensor(-D, dtype=dtype, device=device)   # r_i - r_j points into the center i
    edge_len = torch.tensor(d, dtype=dtype, device=device)
    return Graph(edge_index=edge_index, edge_vec=edge_vec, edge_len=edge_len, n_nodes=n)

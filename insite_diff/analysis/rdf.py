"""Partial radial distribution functions (In-O, In-In, O-O), PBC-aware.

Uses ASE's neighbor_list so periodic images out to r_max are counted correctly
(important where r_max exceeds box/2). g(r) is normalized per frame against the
ideal-gas expectation and averaged over frames.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def partial_rdf(frames: list[Atoms], z1: int, z2: int, rmax: float = 6.0,
                nbins: int = 240, masks: list[np.ndarray] | None = None,
                restrict: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Return (r_centers, g) for the z1-z2 partial RDF averaged over frames.

    ``restrict`` selects which atoms count as the z1 *center* (and, for
    "both_mobile", the z2 *neighbor*), using per-frame boolean ``masks``:
      - "all":           whole structure (non-discriminating when the mask is small)
      - "center_mobile": z1 center must be mobile ("at least one atom mobile")
      - "both_mobile":   both z1 center and z2 neighbor mobile
    Normalization uses the eligible center/neighbor counts, so gen and ref are
    compared on the same footing.
    """
    edges = np.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    if masks is None:
        masks = [None] * len(frames)

    g_accum = np.zeros(nbins)
    n_frames = 0
    for atoms, mask in zip(frames, masks):
        num = atoms.get_atomic_numbers()
        i, j, d = neighbor_list("ijd", atoms, rmax)
        sel = (num[i] == z1) & (num[j] == z2)
        z1_atoms = num == z1
        z2_atoms = num == z2
        if restrict in ("center_mobile", "both_mobile"):
            sel &= mask[i]
            z1_atoms = z1_atoms & mask
        if restrict == "both_mobile":
            sel &= mask[j]
            z2_atoms = z2_atoms & mask
        n1, n2 = int(z1_atoms.sum()), int(z2_atoms.sum())
        if n1 == 0 or n2 == 0:
            continue
        hist, _ = np.histogram(d[sel], bins=edges)
        pair_density = n1 * (n2 - (1 if z1 == z2 else 0)) / atoms.get_volume()
        ideal = pair_density * shell
        g_accum += hist / np.maximum(ideal, 1e-12)
        n_frames += 1
    return centers, g_accum / max(n_frames, 1)


def all_partials(frames: list[Atoms], z_in: int = 49, z_o: int = 8,
                 rmax: float = 6.0, nbins: int = 240,
                 masks: list[np.ndarray] | None = None,
                 restrict: str = "all") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute In-O, In-In, O-O partials in one call (with optional mobile restriction)."""
    return {
        "In-O": partial_rdf(frames, z_in, z_o, rmax, nbins, masks, restrict),
        "In-In": partial_rdf(frames, z_in, z_in, rmax, nbins, masks, restrict),
        "O-O": partial_rdf(frames, z_o, z_o, rmax, nbins, masks, restrict),
    }


def first_peak(centers: np.ndarray, g: np.ndarray, lo: float = 1.5, hi: float = 3.0) -> float:
    """Location of the first RDF peak within [lo, hi]."""
    m = (centers > lo) & (centers < hi)
    return float(centers[m][np.argmax(g[m])])

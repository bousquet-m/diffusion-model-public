"""NVT post-refinement mechanics (mace_relax.nvt_refine).

Uses a Lennard-Jones calculator as a cheap stand-in for the MACE potential so the MD
plumbing is tested without the model. The load-once-run-many MACE path is exercised on
the cluster; here we lock the physical contract that matters: NVT holds the cell fixed
(this project cannot do NPT), the run is deterministic under a seed, and atoms actually
move.
"""
import numpy as np
import torch
from ase import Atoms
from ase.calculators.lj import LennardJones

from insite_diff.analysis.mace_relax import nvt_refine


def _cell_of_atoms():
    # Slightly rattled simple-cubic Ar-like lattice, periodic — enough for a stable MD step.
    rng = np.random.default_rng(0)
    n = 3
    base = np.array([[x, y, z] for x in range(n) for y in range(n) for z in range(n)],
                    dtype=float) * 3.0
    base += rng.normal(scale=0.1, size=base.shape)
    return Atoms(numbers=[18] * len(base), positions=base, cell=np.eye(3) * (3.0 * n), pbc=True)


def test_nvt_refine_keeps_cell_fixed():
    """NVT, not NPT: the box must be identical before and after."""
    atoms = _cell_of_atoms()
    calc = LennardJones(epsilon=0.01, sigma=3.0, rc=8.0)
    cell0 = atoms.get_cell().array.copy()
    out, e0, e1 = nvt_refine(atoms, calc, temperature_K=300.0, timestep_fs=1.0, steps=20,
                             quench_steps=10, seed=0)
    assert np.allclose(out.get_cell().array, cell0), "NVT must not change the cell"
    assert np.isfinite(e0) and np.isfinite(e1)
    assert len(out) == len(atoms)


def test_nvt_refine_moves_atoms_and_is_deterministic():
    atoms = _cell_of_atoms()
    calc = LennardJones(epsilon=0.01, sigma=3.0, rc=8.0)
    a = nvt_refine(atoms, calc, temperature_K=300.0, steps=20, quench_steps=0, seed=7)[0]
    b = nvt_refine(atoms, calc, temperature_K=300.0, steps=20, quench_steps=0, seed=7)[0]
    assert np.allclose(a.get_positions(), b.get_positions()), "same seed -> same trajectory"
    assert not np.allclose(a.get_positions(), atoms.get_positions()), "MD should move atoms"


def test_nvt_refine_steps_zero_is_quench_only():
    """steps=0 skips MD; with quench_steps=0 too it is a pure single point (no move)."""
    atoms = _cell_of_atoms()
    calc = LennardJones(epsilon=0.01, sigma=3.0, rc=8.0)
    out, e0, e1 = nvt_refine(atoms, calc, steps=0, quench_steps=0, seed=0)
    assert np.allclose(out.get_positions(), atoms.get_positions())
    assert e0 == e1

"""Minimal MACE integration: load the potential, relax a structure, report energy.

The trained ``scan_v3_swa.model`` was serialized on CUDA. e3nn's
``CodeGenMixin.__setstate__`` reloads its TorchScript submodules with a bare
``torch.jit.load`` (no map_location), so loading on a CPU/MPS box fails with a
CUDA-backend error. We patch ``torch.jit.load`` to inject the target device for
the duration of model construction only.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import LBFGS


@contextlib.contextmanager
def _force_jit_map_location(device: str):
    orig = torch.jit.load

    def patched(f, map_location=None, *args, **kwargs):
        return orig(f, map_location=map_location or device, *args, **kwargs)

    torch.jit.load = patched
    try:
        yield
    finally:
        torch.jit.load = orig


def load_calculator(model_path: str, device: str = "cpu", dtype: str = "float32"):
    """Load the MACE potential as an ASE calculator (CPU by default)."""
    from mace.calculators import MACECalculator
    with _force_jit_map_location(device):
        return MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)


def potential_energy(atoms: Atoms, calc) -> float:
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy())


def relax(atoms: Atoms, calc, fmax: float = 0.05, steps: int = 200) -> tuple[Atoms, float, float]:
    """Relax with LBFGS. Returns (relaxed_atoms, energy_before, energy_after)."""
    a = atoms.copy()
    a.calc = calc
    e0 = float(a.get_potential_energy())
    dyn = LBFGS(a, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    e1 = float(a.get_potential_energy())
    return a, e0, e1


def nvt_refine(atoms: Atoms, calc, temperature_K: float = 500.0, timestep_fs: float = 1.0,
               steps: int = 200, friction_per_fs: float = 0.01, quench_fmax: float = 0.05,
               quench_steps: int = 200, seed: int = 0) -> tuple[Atoms, float, float]:
    """Short NVT MD + 0 K quench — the paper's post-sampling cleanup for rare bad atoms.

    NVT (fixed cell) on purpose: this potential is not reliable for the stress/cell
    dynamics NPT needs, and the box comes from the generated structure anyway. The Langevin
    thermostat lets under-coordinated atoms find neighbors over a few hundred fs; the final
    LBFGS quench then reports the inherent-structure (0 K) energy so it is comparable to a
    relaxed reference.

    ``friction_per_fs`` and ``timestep_fs`` are in fs / fs^-1 (converted via ``units.fs``).
    Returns (refined_atoms, energy_before, energy_after) like ``relax``.
    """
    a = atoms.copy()
    a.calc = calc
    e0 = float(a.get_potential_energy())

    if steps > 0:
        # Deterministic under `seed`: one generator feeds both the initial Maxwell-Boltzmann
        # velocities and the Langevin random force. Kill net momentum (drift is spurious in a
        # periodic box); no ZeroRotation under PBC.
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(a, temperature_K=temperature_K, rng=rng)
        Stationary(a)
        dyn = Langevin(a, timestep=timestep_fs * units.fs, temperature_K=temperature_K,
                       friction=friction_per_fs / units.fs, rng=rng, logfile=None)
        dyn.run(steps)

    if quench_steps > 0:
        LBFGS(a, logfile=None).run(fmax=quench_fmax, steps=quench_steps)
    e1 = float(a.get_potential_energy())
    return a, e0, e1

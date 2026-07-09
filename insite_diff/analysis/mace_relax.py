"""Minimal MACE integration: load the potential, relax a structure, report energy.

The trained ``scan_v3_swa.model`` was serialized on CUDA. e3nn's
``CodeGenMixin.__setstate__`` reloads its TorchScript submodules with a bare
``torch.jit.load`` (no map_location), so loading on a CPU/MPS box fails with a
CUDA-backend error. We patch ``torch.jit.load`` to inject the target device for
the duration of model construction only.
"""
from __future__ import annotations

import contextlib

import torch
from ase import Atoms
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

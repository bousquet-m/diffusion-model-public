"""Forward noising: CoM-free, PBC-aware, density-normalized VP diffusion.

Formulation (documented so the choices are auditable)
-----------------------------------------------------
We diffuse Cartesian positions, but in a frame chosen so VP/DDPM behaves well
under periodic boundary conditions and across the multiple cell sizes we train on:

1. **CoM-free frame.** For each structure we remove the center of mass using the
   *periodic* (circular-mean) CoM, then express atoms by their minimum-image
   displacement from that CoM. This kills the translational zero mode so the
   Gaussian prior lives on the same (sum_i v_i = 0) subspace the equivariant
   denoiser can represent. All injected noise is likewise CoM-free.

2. **Density normalization.** Centered coordinates are divided by a per-structure
   length scale ``s = V^(1/3) / sqrt(12)`` (the std of a uniform fill of the box).
   This maps every cell size — the 80- and 640-atom boxes alike — into a common
   ~unit-variance frame, so one VP schedule and one model are consistent across
   densities. We multiply back by ``s`` to return to Angstrom.

   ASSUMPTION: this scale assumes the box is **uniformly filled by atoms**. It is
   correct for the bulk periodic cells here. A later slab/surface cell contains
   vacuum, so V overstates the occupied volume and ``s`` would be too large — the
   normalization must be revisited (e.g. scale by the occupied sub-volume, or by a
   fixed per-species length) before this code is used on slabs. (milestone 2)

3. **VP/DDPM.** In the normalized CoM-free frame, ``z_t = sqrt(abar_t) z_0 +
   sqrt(1-abar_t) eps`` with ``eps`` CoM-free. The network sees real-Angstrom
   geometry (z_t * s) for its neighbor graph and predicts ``eps``.

PBC enters through (a) the circular-mean CoM, (b) minimum-image centering, and
(c) the neighbor graph built downstream — all translation/wrap invariant, so the
equivariant network is unaffected by which periodic image a coordinate sits in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .schedule import VPSchedule


# --------------------------------------------------------------------------- #
# PBC / CoM helpers (torch)
# --------------------------------------------------------------------------- #
def pbc_mask(pbc, frac: torch.Tensor) -> torch.Tensor | None:
    """Per-axis periodicity as a (3,) float in frac's dtype/device, or None if fully
    periodic. ``pbc`` may be None/True (all periodic) or a length-3 bool sequence."""
    if pbc is None or pbc is True or all(bool(p) for p in pbc):
        return None
    return torch.as_tensor([1.0 if p else 0.0 for p in pbc], dtype=frac.dtype, device=frac.device)


def min_image(delta: torch.Tensor, cell: torch.Tensor, pbc=None) -> torch.Tensor:
    """Minimum-image Cartesian displacements (..., 3) for ``cell`` (3,3).

    With ``pbc`` (length-3 bool) a False axis is treated as open (a slab's vacuum
    direction): the raw displacement is kept there instead of wrapping to the nearest
    periodic image. Default (None) = fully periodic, identical to before."""
    inv = torch.linalg.inv(cell)
    frac = delta @ inv
    shift = torch.round(frac)
    p = pbc_mask(pbc, frac)
    if p is not None:
        shift = shift * p                  # open axes (p=0): no wrap, keep raw displacement
    return (frac - shift) @ cell


def pbc_center(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Periodic center of mass (unit masses) via per-dimension circular mean -> (3,)."""
    inv = torch.linalg.inv(cell)
    frac = positions @ inv
    theta = 2 * math.pi * frac
    ang = torch.atan2(torch.sin(theta).mean(0), torch.cos(theta).mean(0))  # (3,) in [-pi,pi]
    frac_c = ang / (2 * math.pi)
    return frac_c @ cell


def to_com_free(positions: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centered positions, effective CoM) with *exact* zero arithmetic mean.

    The circular-mean CoM robustly centers the periodic distribution (so min-image
    unwrapping doesn't split a cluster), but the arithmetic mean of the wrapped
    coordinates is not exactly zero. The equivariant CoM-free subspace needs an
    exact zero mean, so we subtract the residual and fold it back into the stored
    CoM — reconstruction ``centered + com`` then returns the original (periodic
    image of the) positions.
    """
    com = pbc_center(positions, cell)
    centered = min_image(positions - com, cell)
    residual = centered.mean(dim=0)
    centered = centered - residual
    com = com + residual
    return centered, com


def structure_scale(cell: torch.Tensor) -> torch.Tensor:
    """Per-structure length scale s = V^(1/3)/sqrt(12) used to normalize coordinates."""
    vol = torch.abs(torch.linalg.det(cell))
    return vol ** (1.0 / 3.0) / math.sqrt(12.0)


def draw_randn(shape, device, dtype, generator: torch.Generator | None = None) -> torch.Tensor:
    """torch.randn that tolerates a generator on a different device than ``device``.

    A ``generator`` must live on the same device as the tensor being drawn, so we
    draw on the generator's device (CPU for reproducible sampling) and move to the
    compute device. With no generator we draw directly on ``device``.
    """
    gdev = generator.device if generator is not None else device
    x = torch.randn(*shape, generator=generator, device=gdev, dtype=dtype)
    return x.to(device)


def draw_rand(shape, device, dtype, generator: torch.Generator | None = None) -> torch.Tensor:
    """torch.rand (uniform) counterpart of ``draw_randn`` — same cross-device rule.

    Every seeded RNG draw must go through these two helpers. The eval scripts build a
    CPU generator for reproducibility (``torch.Generator(device="cpu")``) and pass it
    alongside a CUDA compute device; drawing directly with that pair raises
    "Expected a 'cuda' device type for generator but found 'cpu'" — a crash that is
    invisible on a CPU-only box.
    """
    gdev = generator.device if generator is not None else device
    x = torch.rand(*shape, generator=generator, device=gdev, dtype=dtype)
    return x.to(device)


def com_free_noise(n_atoms: int, cell: torch.Tensor | None = None,
                   generator: torch.Generator | None = None,
                   device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Gaussian noise (n_atoms, 3) projected to zero mean over the atom axis."""
    eps = draw_randn((n_atoms, 3), device, dtype, generator)
    return eps - eps.mean(dim=0, keepdim=True)


# --------------------------------------------------------------------------- #
# Forward process
# --------------------------------------------------------------------------- #
@dataclass
class NoisedSample:
    z_t: torch.Tensor       # noised coords, normalized CoM-free frame (N,3)
    x_t_cart: torch.Tensor  # z_t * s, centered Angstrom coords for graph building (N,3)
    noise: torch.Tensor     # the CoM-free eps target (N,3)
    z0: torch.Tensor        # clean normalized coords (N,3)
    com: torch.Tensor       # (3,) periodic CoM removed
    scale: torch.Tensor     # () length scale s


def normalize(positions: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """positions (Angstrom) -> (z0 normalized CoM-free, com, scale)."""
    centered, com = to_com_free(positions, cell)
    s = structure_scale(cell)
    return centered / s, com, s


def denormalize(z: torch.Tensor, com: torch.Tensor, scale: torch.Tensor,
                cell: torch.Tensor, wrap: bool = True) -> torch.Tensor:
    """Normalized CoM-free coords -> Angstrom positions (optionally wrapped into cell)."""
    pos = z * scale + com
    if wrap:
        inv = torch.linalg.inv(cell)
        frac = pos @ inv
        frac = frac - torch.floor(frac)
        pos = frac @ cell
    return pos


def q_sample(schedule: VPSchedule, z0: torch.Tensor, t: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """DDPM forward q(z_t | z_0) in the normalized frame. ``t`` is a scalar/1-elem tensor."""
    a = schedule.sqrt_alpha_bar.to(z0.dtype)[t]
    b = schedule.sqrt_one_minus_alpha_bar.to(z0.dtype)[t]
    return a * z0 + b * noise


def add_noise(schedule: VPSchedule, positions: torch.Tensor, cell: torch.Tensor,
              t: torch.Tensor, generator: torch.Generator | None = None) -> NoisedSample:
    """Full forward pass for one structure: normalize -> CoM-free noise -> q_sample."""
    z0, com, s = normalize(positions, cell)
    noise = com_free_noise(z0.shape[0], cell, generator=generator,
                           device=z0.device, dtype=z0.dtype)
    z_t = q_sample(schedule, z0, t, noise)
    return NoisedSample(z_t=z_t, x_t_cart=z_t * s, noise=noise, z0=z0, com=com, scale=s)


# --------------------------------------------------------------------------- #
# Variance-exploding (VE) forward process — amorphous recipe
# --------------------------------------------------------------------------- #
def uniform_init(n_atoms: int, cell: torch.Tensor,
                 generator: torch.Generator | None = None,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 bounds: torch.Tensor | None = None) -> torch.Tensor:
    """Atoms uniformly distributed in the cell (fractional coords ~ U(0,1)).

    This is the VE prior: it already fills the box at the right density, so the
    diffusion only has to fix *local* order — unlike the VP Gaussian blob, which
    concentrates atoms in the center and collapses.

    ``bounds`` (3,2) gives per-axis fractional [lo, hi] so the prior fills only a
    sub-region — e.g. the material slab, not the vacuum, along a non-periodic axis.
    Default (None) = full cell on every axis, identical to before.
    """
    frac = draw_rand((n_atoms, 3), device, dtype, generator)
    if bounds is not None:
        b = torch.as_tensor(bounds, dtype=dtype, device=frac.device)     # (3,2)
        frac = b[:, 0] + (b[:, 1] - b[:, 0]) * frac
    return frac @ cell.to(device=frac.device, dtype=dtype)


def ve_add_noise(positions: torch.Tensor, sigma: torch.Tensor, cell: torch.Tensor,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive VE noising at physical (Angstrom) scale: x_sigma = x0 + sigma * eps.

    eps is CoM-free (translation-invariant, matching the equivariant denoiser).
    Returns (x_sigma, eps). No coordinate normalization — we work in real Angstrom.
    """
    eps = com_free_noise(positions.shape[0], cell, generator=generator,
                         device=positions.device, dtype=positions.dtype)
    return positions + sigma * eps, eps

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
def min_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap Cartesian displacements (..., 3) into the minimum image for ``cell`` (3,3)."""
    inv = torch.linalg.inv(cell)
    frac = delta @ inv
    frac = frac - torch.round(frac)
    return frac @ cell


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


def com_free_noise(n_atoms: int, cell: torch.Tensor | None = None,
                   generator: torch.Generator | None = None,
                   device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Gaussian noise (n_atoms, 3) projected to zero mean over the atom axis."""
    eps = torch.randn(n_atoms, 3, generator=generator, device=device, dtype=dtype)
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

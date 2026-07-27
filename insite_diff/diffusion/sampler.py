"""Reverse (ancestral) DDPM sampling in the normalized CoM-free frame.

Decoupled from the denoiser: callers pass a ``predict_noise(z_t, t) -> eps``
callable (which internally builds the Angstrom neighbor graph from ``z_t * s`` and
runs the network). The RePaint inpainting variant lives in ``sampling/inpaint.py``.

All coordinates here are in the normalized CoM-free frame; every update keeps the
CoM-free constraint by projecting both the predicted noise and the injected noise
to zero mean over atoms.
"""
from __future__ import annotations

from typing import Callable

import torch

from .noising import com_free_noise, denormalize, uniform_init
from .schedule import VESchedule, VPSchedule

PredictNoise = Callable[[torch.Tensor, int], torch.Tensor]


def _com_free(v: torch.Tensor) -> torch.Tensor:
    return v - v.mean(dim=0, keepdim=True)


def p_mean(schedule: VPSchedule, z_t: torch.Tensor, t: int, eps_hat: torch.Tensor) -> torch.Tensor:
    """Posterior mean of z_{t-1} from eps-prediction (DDPM)."""
    eps_hat = _com_free(eps_hat)
    coef = (schedule.betas[t] / schedule.sqrt_one_minus_alpha_bar[t]).to(z_t.dtype)
    return schedule.sqrt_recip_alphas[t].to(z_t.dtype) * (z_t - coef * eps_hat)


def p_sample_step(schedule: VPSchedule, z_t: torch.Tensor, t: int, eps_hat: torch.Tensor,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """One reverse step z_t -> z_{t-1}. No noise added at t == 0."""
    mean = p_mean(schedule, z_t, t, eps_hat)
    if t == 0:
        return mean
    var = schedule.posterior_variance[t].to(z_t.dtype)
    noise = com_free_noise(z_t.shape[0], generator=generator, device=z_t.device, dtype=z_t.dtype)
    return mean + torch.sqrt(var) * noise


@torch.no_grad()
def sample(
    schedule: VPSchedule,
    predict_noise: PredictNoise,
    n_atoms: int,
    cell: torch.Tensor,
    com: torch.Tensor,
    scale: torch.Tensor,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    return_traj: bool = False,
) -> torch.Tensor:
    """Ancestral sampling from the CoM-free Gaussian prior to z_0, returned as
    Angstrom positions (wrapped into ``cell``)."""
    z_t = com_free_noise(n_atoms, generator=generator, device=device)
    traj = []
    for t in reversed(range(schedule.timesteps)):
        eps_hat = predict_noise(z_t, t)
        z_t = p_sample_step(schedule, z_t, t, eps_hat, generator=generator)
        if return_traj:
            traj.append(z_t.clone())
    positions = denormalize(z_t, com, scale, cell, wrap=True)
    return (positions, traj) if return_traj else positions


# --------------------------------------------------------------------------- #
# VE annealed Langevin sampling (amorphous recipe)
# --------------------------------------------------------------------------- #
PredictEps = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, sigma) -> eps_hat


def _wrap(pos: torch.Tensor, cell: torch.Tensor, pbc=None) -> torch.Tensor:
    """Wrap positions into the cell. A non-periodic axis (``pbc`` False) is left
    unwrapped so a slab's surface atoms are not folded through the vacuum gap."""
    from .noising import pbc_mask
    cell = cell.to(pos.dtype)
    frac = pos @ torch.linalg.inv(cell)
    shift = torch.floor(frac)
    p = pbc_mask(pbc, frac)
    if p is not None:
        shift = shift * p
    return (frac - shift) @ cell


@torch.no_grad()
def annealed_langevin(
    schedule: VESchedule,
    predict_eps: PredictEps,
    n_atoms: int,
    cell: torch.Tensor,
    langevin_steps: int = 10,
    step_lr: float = 2.0e-5,
    refine_steps: int = 100,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    x_init: torch.Tensor | None = None,
    pbc=None,
    prior_bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """NCSN annealed Langevin from a uniform-in-cell prior.

    ``predict_eps(x, sigma)`` returns the CoM-free noise estimate; the score is
    ``-eps_hat / sigma``. For each sigma (high->low) we take ``langevin_steps``
    Langevin steps with step size ``step_lr * (sigma/sigma_min)^2`` and external
    CoM-free noise, then a final ``refine_steps`` at sigma_min without external
    noise. Positions are wrapped into the cell each step (open axes per ``pbc`` are
    not wrapped). ``prior_bounds`` (3,2) confines the uniform prior to a sub-region
    (e.g. the material slab, not the vacuum)."""
    cell = cell.to(device)
    x = uniform_init(n_atoms, cell, generator=generator, device=device, dtype=cell.dtype,
                     bounds=prior_bounds) \
        if x_init is None else x_init.to(device)
    sigmas = schedule.sigmas.to(device)
    sigma_min = sigmas[-1]

    def langevin(x, sigma, add_noise):
        step = step_lr * (sigma / sigma_min) ** 2
        eps_hat = predict_eps(x, sigma)
        x = x + step * (-eps_hat / sigma)
        if add_noise:
            z = com_free_noise(n_atoms, generator=generator, device=device, dtype=x.dtype)
            x = x + torch.sqrt(2 * step) * z
        return _wrap(x, cell, pbc)

    for sigma in sigmas:
        for _ in range(langevin_steps):
            x = langevin(x, sigma, add_noise=True)
    for _ in range(refine_steps):
        x = langevin(x, sigma_min, add_noise=False)
    return x

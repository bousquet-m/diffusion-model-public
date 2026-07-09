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

from .noising import com_free_noise, denormalize
from .schedule import VPSchedule

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

"""Denoising score matching loss (epsilon-prediction MSE).

For each structure we noise it at a random timestep in the normalized CoM-free
frame, build the neighbor graph on the noised Angstrom coordinates, and ask the
network to predict the injected CoM-free noise. A batch is a list of structures
(atom counts vary), so we average the per-structure losses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.graph import build_graph_torch
from ..diffusion.noising import add_noise, ve_add_noise
from ..diffusion.schedule import VESchedule, VPSchedule


def structure_eps_loss(model, schedule: VPSchedule, item: dict, cutoff: float,
                       max_neighbors: int, device, generator=None,
                       loss_weighting: str = "uniform", min_snr_gamma: float = 5.0) -> torch.Tensor:
    pos = item["positions"].to(device)
    cell = item["cell"].to(device)
    types = item["types"].to(device)
    n = pos.shape[0]

    t = torch.randint(0, schedule.timesteps, (1,), device=device)
    ns = add_noise(schedule, pos, cell, t, generator=generator)

    g = build_graph_torch(ns.x_t_cart.detach(), cell, cutoff, max_neighbors=max_neighbors)
    eps_hat = model(types, g.edge_index, g.edge_vec, t.float() / schedule.timesteps, n)
    mse = F.mse_loss(eps_hat, ns.noise)
    if loss_weighting == "min_snr":
        mse = schedule.min_snr_eps_weight(t, min_snr_gamma).squeeze() * mse
    return mse


def batch_eps_loss(model, schedule: VPSchedule, batch: list[dict], cutoff: float,
                   max_neighbors: int, device, generator=None,
                   loss_weighting: str = "uniform", min_snr_gamma: float = 5.0) -> torch.Tensor:
    losses = [
        structure_eps_loss(model, schedule, item, cutoff, max_neighbors, device, generator,
                           loss_weighting, min_snr_gamma)
        for item in batch
    ]
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# VE denoising score matching (amorphous recipe)
# --------------------------------------------------------------------------- #
def structure_ve_loss(model, schedule: VESchedule, item: dict, cutoff: float,
                      max_neighbors: int, device, generator=None) -> torch.Tensor:
    """Denoising score matching: noise at a random sigma level, predict the eps.

    The network is conditioned on normalized log(sigma) and predicts the injected
    CoM-free noise; MSE(eps_hat, eps) is the sigma^2-weighted DSM objective.
    """
    pos = item["positions"].to(device)
    cell = item["cell"].to(device)
    types = item["types"].to(device)
    n = pos.shape[0]

    sigma = schedule.sample_sigma(generator=generator)          # (1,)
    x_sigma, eps = ve_add_noise(pos, sigma, cell, generator=generator)
    g = build_graph_torch(x_sigma.detach(), cell, cutoff, max_neighbors=max_neighbors)
    eps_hat = model(types, g.edge_index, g.edge_vec, schedule.cond(sigma), n)
    return F.mse_loss(eps_hat, eps)


def batch_ve_loss(model, schedule: VESchedule, batch: list[dict], cutoff: float,
                  max_neighbors: int, device, generator=None) -> torch.Tensor:
    losses = [structure_ve_loss(model, schedule, item, cutoff, max_neighbors, device, generator)
              for item in batch]
    return torch.stack(losses).mean()

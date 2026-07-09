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
from ..diffusion.noising import add_noise
from ..diffusion.schedule import VPSchedule


def structure_eps_loss(model, schedule: VPSchedule, item: dict, cutoff: float,
                       max_neighbors: int, device, generator=None) -> torch.Tensor:
    pos = item["positions"].to(device)
    cell = item["cell"].to(device)
    types = item["types"].to(device)
    n = pos.shape[0]

    t = torch.randint(0, schedule.timesteps, (1,), device=device)
    ns = add_noise(schedule, pos, cell, t, generator=generator)

    g = build_graph_torch(ns.x_t_cart.detach(), cell, cutoff, max_neighbors=max_neighbors)
    eps_hat = model(types, g.edge_index, g.edge_vec, t.float() / schedule.timesteps, n)
    return F.mse_loss(eps_hat, ns.noise)


def batch_eps_loss(model, schedule: VPSchedule, batch: list[dict], cutoff: float,
                   max_neighbors: int, device, generator=None) -> torch.Tensor:
    losses = [
        structure_eps_loss(model, schedule, item, cutoff, max_neighbors, device, generator)
        for item in batch
    ]
    return torch.stack(losses).mean()

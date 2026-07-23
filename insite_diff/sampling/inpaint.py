"""RePaint-style inpainting: clamp context atoms, generate mobile atoms.

Modelled on RePaint (Lugmayr 2022) / CHGGen host-guided inpainting. At every
reverse step the mobile atoms take a denoising step while the context atoms are
overwritten with a freshly re-noised copy of their known positions, so the
generated region is conditioned on the true surroundings.

Frame handling (the CoM-free subtlety the brief flags)
------------------------------------------------------
Equivariant diffusion assumes a CoM-free frame. We anchor that frame to the
**context block**: coordinates are expressed relative to the context's periodic
CoM and normalized by the cell length scale, so the context has zero mean by
construction. After every substitution we re-subtract the context-block mean
("realign the fixed block's CoM") so re-noising and denoising cannot let the
anchor drift. Because the context CoM is the frame origin, the clamped atoms
return to exactly their known positions at t=0.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.graph import build_graph_torch
from ..diffusion.noising import (com_free_noise, draw_randn, min_image, pbc_center,
                                 q_sample, structure_scale, uniform_init)
from ..diffusion.sampler import _wrap
from ..diffusion.schedule import VESchedule, VPSchedule


def _subset_com_free_noise(mask: torch.Tensor, device, dtype, generator=None) -> torch.Tensor:
    """Gaussian noise (N,3) whose mean over the ``mask`` atoms is zero."""
    eps = draw_randn((mask.shape[0], 3), device, dtype, generator)
    eps[mask] = eps[mask] - eps[mask].mean(dim=0, keepdim=True)
    return eps


@torch.no_grad()
def inpaint(
    schedule: VPSchedule,
    model,
    positions: torch.Tensor,     # (N,3) known coords; mobile entries are ignored/overwritten
    cell: torch.Tensor,
    types: torch.Tensor,
    mobile_mask: torch.Tensor,   # (N,) bool, True = generate
    cutoff: float,
    max_neighbors: int = 0,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    n_resample: int = 8,
) -> torch.Tensor:
    """Return a full (N,3) structure with context clamped and mobile generated."""
    positions = positions.to(device)
    cell = cell.to(device)
    types = types.to(device)
    mobile_mask = mobile_mask.to(device)
    context = ~mobile_mask
    dtype = positions.dtype
    N = positions.shape[0]

    # --- context-anchored normalized frame ---
    s = structure_scale(cell)
    com_ctx = pbc_center(positions[context], cell)
    z_ref = min_image(positions - com_ctx, cell) / s      # all atoms in context frame
    ctx_mean = z_ref[context].mean(dim=0)
    z_ref = z_ref - ctx_mean                               # context mean now exactly 0
    com_eff = com_ctx + ctx_mean * s                       # for denormalization

    def predict(z_t: torch.Tensor, t: int) -> torch.Tensor:
        g = build_graph_torch(z_t * s, cell, cutoff, max_neighbors=max_neighbors)
        return model(types, g.edge_index, g.edge_vec,
                     torch.tensor(t / schedule.timesteps, device=device), N)

    def reverse_mean(z_t, t, eps_hat):
        eps_hat = eps_hat - eps_hat.mean(0, keepdim=True)
        coef = (schedule.betas[t] / schedule.sqrt_one_minus_alpha_bar[t]).to(dtype)
        return schedule.sqrt_recip_alphas[t].to(dtype) * (z_t - coef * eps_hat)

    def renoise_context(t: int) -> torch.Tensor:
        """z_ref context re-noised to level t (exact reference at t == 0)."""
        if t <= 0:
            return z_ref.clone()
        eps = _subset_com_free_noise(context, device, dtype, generator)
        return q_sample(schedule, z_ref, torch.tensor(t), eps)

    def realign(z: torch.Tensor) -> torch.Tensor:
        return z - z[context].mean(dim=0, keepdim=True)

    # --- initialize z at t = T-1: context noised, mobile from prior ---
    T = schedule.timesteps
    z_t = draw_randn((N, 3), device, dtype, generator)
    z_t[mobile_mask] = z_t[mobile_mask] - z_t[mobile_mask].mean(0, keepdim=True)
    z_t[context] = renoise_context(T - 1)[context]
    z_t = realign(z_t)

    for t in reversed(range(T)):
        for u in range(n_resample):
            eps_hat = predict(z_t, t)
            mean = reverse_mean(z_t, t, eps_hat)
            if t > 0:
                var = schedule.posterior_variance[t].to(dtype)
                noise = _subset_com_free_noise(mobile_mask, device, dtype, generator)
                z_prev = mean + torch.sqrt(var) * noise
            else:
                z_prev = mean

            z_new = z_prev.clone()
            z_new[context] = renoise_context(t - 1)[context]   # clamp context (RePaint)
            z_new = realign(z_new)                              # realign fixed block CoM

            # RePaint resampling: jump back t-1 -> t and redo, except on the last pass
            if u < n_resample - 1 and t > 0:
                beta = schedule.betas[t].to(dtype)
                jump_noise = _subset_com_free_noise(mobile_mask, device, dtype, generator)
                z_t = torch.sqrt(1 - beta) * z_new + torch.sqrt(beta) * jump_noise
                z_t = realign(z_t)
            else:
                z_t = z_new

    positions_out = z_t * s + com_eff
    # wrap into cell
    inv = torch.linalg.inv(cell)
    frac = positions_out @ inv
    positions_out = (frac - torch.floor(frac)) @ cell
    return positions_out


# --------------------------------------------------------------------------- #
# VE (amorphous recipe) inpainting — RePaint in annealed Langevin
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ve_inpaint(schedule: VESchedule, model, positions: torch.Tensor, cell: torch.Tensor,
               types: torch.Tensor, mobile_mask: torch.Tensor, cutoff: float,
               max_neighbors: int, langevin_steps: int, step_lr: float, refine_steps: int,
               generator: torch.Generator | None = None, device="cpu",
               start_sigma: float | None = None) -> torch.Tensor:
    """Clamp context, generate mobile via annealed Langevin.

    In physical Cartesian coordinates (no normalized frame), so context atoms are
    simply held at their known positions (re-noised to the current sigma each step,
    RePaint-style); no CoM realignment is needed. mask_frac=1.0 (empty context)
    reduces to unconditional annealed_langevin.

    SDEdit augmentation (``start_sigma`` > 0): instead of the full anneal from the
    uniform prior at sigma_max, begin partway down the ladder from the REAL structure
    (``positions``) plus ``start_sigma`` of noise, iterating only the sigma levels at or
    below ``start_sigma``. This is the similarity<->diversity dial for making "more like
    these" structures from few training points: smaller start_sigma stays closer to the
    seed. At mask_frac=1.0 the whole structure is the seed (context empty).
    """
    positions = positions.to(device)
    cell = cell.to(device)
    types = types.to(device)
    mobile_mask = mobile_mask.to(device)
    context = ~mobile_mask
    n, dtype = positions.shape[0], positions.dtype
    sigmas = schedule.sigmas.to(device)
    sigma_min = sigmas[-1]

    def predict_eps(x, sigma):
        g = build_graph_torch(x, cell, cutoff, max_neighbors=max_neighbors)
        return model(types, g.edge_index, g.edge_vec, schedule.cond(sigma), n)

    def clamp_context(x, sigma):
        if context.any():
            eps_c = com_free_noise(n, generator=generator, device=device, dtype=dtype)
            x = x.clone()
            x[context] = positions[context] + sigma * eps_c[context]
        return x

    if start_sigma is not None and start_sigma > 0:
        ladder = sigmas[sigmas <= float(start_sigma)]
        if ladder.numel() == 0:
            raise ValueError(f"start_sigma {start_sigma} is below sigma_min {float(sigma_min)}")
        eps0 = com_free_noise(n, generator=generator, device=device, dtype=dtype)
        x = _wrap(positions + ladder[0] * eps0, cell)   # SDEdit: real seed + intermediate noise
    else:
        ladder = sigmas
        x = uniform_init(n, cell, generator=generator, device=device, dtype=dtype)
    x = clamp_context(x, ladder[0])
    for sigma in ladder:
        step = step_lr * (sigma / sigma_min) ** 2
        for _ in range(langevin_steps):
            eps_hat = predict_eps(x, sigma)
            z = com_free_noise(n, generator=generator, device=device, dtype=dtype)
            x = x + step * (-eps_hat / sigma) + torch.sqrt(2 * step) * z
            x = clamp_context(x, sigma)          # RePaint: re-noised known context
            x = _wrap(x, cell)
    for _ in range(refine_steps):                # final refine, context exact
        eps_hat = predict_eps(x, sigma_min)
        x = x + step_lr * (-eps_hat / sigma_min)
        if context.any():
            x = x.clone()
            x[context] = positions[context]
        x = _wrap(x, cell)
    return x


def inpaint_dispatch(cfg, model, positions, cell, types, mobile_mask, device,
                     generator=None) -> torch.Tensor:
    """Run the inpainting sampler for the configured diffusion type (vp|ve)."""
    d = cfg.diffusion
    if d.type == "ve":
        sched = VESchedule(d.sigma_min, d.sigma_max, d.n_sigma_levels).to(device)
        start_sigma = getattr(d, "sdedit_sigma", 0.0) or None
        return ve_inpaint(sched, model, positions, cell, types, mobile_mask,
                          cfg.graph.cutoff, cfg.graph.max_neighbors,
                          d.langevin_steps, d.langevin_step_lr, d.refine_steps,
                          generator=generator, device=device, start_sigma=start_sigma)
    sched = VPSchedule(d.timesteps, d.beta_schedule).to(device)
    return inpaint(sched, model, positions, cell, types, mobile_mask,
                   cutoff=cfg.graph.cutoff, max_neighbors=cfg.graph.max_neighbors,
                   device=device, generator=generator, n_resample=cfg.sampling.n_resample)

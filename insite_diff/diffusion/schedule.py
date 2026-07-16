"""Variance-preserving (VP / DDPM) noise schedule.

We use VP with epsilon-prediction. The cosine schedule (Nichol & Dhariwal 2021)
is the default; a linear schedule is available for comparison. All per-timestep
quantities are precomputed as tensors indexed by integer t in [0, T).
"""
from __future__ import annotations

import torch


def _cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar(t) for the cosine schedule, length ``timesteps`` (t = 0..T-1)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    # betas_t = 1 - abar_t/abar_{t-1}; clip for numerical stability, then rebuild abar.
    betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)


def _linear_alpha_bar(timesteps: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


class VPSchedule:
    """Precomputed VP/DDPM coefficients for forward noising and reverse sampling."""

    def __init__(self, timesteps: int, beta_schedule: str = "cosine"):
        self.timesteps = timesteps
        if beta_schedule == "cosine":
            alpha_bar = _cosine_alpha_bar(timesteps)
        elif beta_schedule == "linear":
            alpha_bar = _linear_alpha_bar(timesteps)
        else:
            raise ValueError(f"unknown beta_schedule {beta_schedule!r}")

        alpha_bar = alpha_bar.to(torch.float64)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])
        alphas = alpha_bar / alpha_bar_prev
        betas = 1.0 - alphas

        # Forward q(x_t | x_0)
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        # Reverse p(x_{t-1} | x_t)
        self.betas = betas
        self.alphas = alphas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.alpha_bar_prev = alpha_bar_prev
        # DDPM posterior variance: beta_t * (1 - abar_{t-1}) / (1 - abar_t)
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        # Signal-to-noise ratio per step, for min-SNR-gamma loss weighting.
        self.snr = alpha_bar / (1.0 - alpha_bar)

    def min_snr_eps_weight(self, t: torch.Tensor, gamma: float) -> torch.Tensor:
        """min-SNR-gamma loss weight for eps-prediction: min(SNR, gamma)/SNR.

        Equals 1 at high noise (SNR<=gamma) and gamma/SNR<1 at low noise, so it
        down-weights the easy low-noise steps and shifts learning toward the
        harder mid/high-noise regime.
        """
        snr = self.snr.to(torch.float32)[t]
        return torch.clamp(snr, max=gamma) / snr

    def to(self, device: torch.device | str) -> "VPSchedule":
        for name, val in vars(self).items():
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather per-sample coefficients and shape them to (B, 1, 1) for broadcasting."""
        out = buf.to(torch.float32)[t]
        return out.view(-1, *([1] * 2))


class VESchedule:
    """Variance-exploding noise ladder for score-based / annealed-Langevin diffusion.

    A geometric sequence of noise levels sigma from sigma_max (index 0) down to
    sigma_min (index -1), in Angstrom. Small max sigma keeps the forward process in
    the physically-meaningful local-perturbation regime (no full destruction), and
    the model is conditioned on log(sigma).
    """

    def __init__(self, sigma_min: float, sigma_max: float, n_levels: int):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_levels = n_levels
        # high -> low, geometric
        self.sigmas = torch.exp(torch.linspace(
            torch.log(torch.tensor(sigma_max)), torch.log(torch.tensor(sigma_min)), n_levels))

    def to(self, device: torch.device | str) -> "VESchedule":
        self.sigmas = self.sigmas.to(device)
        return self

    def sample_sigma(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Random noise level from the ladder (for training).

        Draws on the generator's device, not the ladder's: a CPU generator paired with a
        CUDA schedule otherwise raises "Expected a 'cuda' device type for generator".
        See ``noising.draw_randn`` / ``draw_rand`` for the same rule.
        """
        gdev = generator.device if generator is not None else self.sigmas.device
        idx = torch.randint(0, self.n_levels, (1,), generator=generator, device=gdev)
        return self.sigmas[idx.to(self.sigmas.device)]

    def cond(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map sigma -> [0,1] conditioning scalar via normalized log(sigma)."""
        lo, hi = torch.log(torch.tensor(self.sigma_min)), torch.log(torch.tensor(self.sigma_max))
        return ((torch.log(sigma) - lo) / (hi - lo)).clamp(0, 1).to(sigma.device)

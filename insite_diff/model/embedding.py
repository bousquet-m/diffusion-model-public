"""Noise-level embedding (scalar / 0e features), conditioning the denoiser."""
from __future__ import annotations

import math

import torch
from torch import nn


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding of a normalized noise level t/T in [0, 1] -> MLP -> dim scalars.

    The output is rotation-invariant (0e), so it can be concatenated into node
    scalar features and the radial network without breaking equivariance.
    """

    def __init__(self, dim: int, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """t_norm: scalar or (B,) in [0,1]. Returns (dim,) or (B, dim)."""
        t = torch.atleast_1d(t_norm).float()
        freqs = torch.exp(
            torch.linspace(0, math.log(1000.0), self.n_freq, device=t.device)
        )
        ang = t[:, None] * freqs[None, :]  # (B, n_freq)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, 2*n_freq)
        out = self.mlp(feats)
        return out.squeeze(0) if t_norm.ndim == 0 else out

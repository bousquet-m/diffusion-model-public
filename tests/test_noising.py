"""CoM-free + PBC-aware noising invariants."""
import math

import numpy as np
import pytest
import torch

from insite_diff.diffusion.noising import (
    add_noise, com_free_noise, denormalize, min_image, normalize,
    pbc_center, structure_scale, to_com_free,
)
from insite_diff.diffusion.schedule import VPSchedule

torch.manual_seed(0)


def _random_structure(n=64, box=20.25):
    cell = torch.eye(3, dtype=torch.float64) * box
    frac = torch.rand(n, 3, dtype=torch.float64)
    pos = frac @ cell
    return pos, cell


def test_com_free_noise_is_zero_mean():
    eps = com_free_noise(128)
    assert torch.allclose(eps.mean(0), torch.zeros(3), atol=1e-6)


def test_centered_coords_have_exact_zero_mean():
    pos, cell = _random_structure()
    centered, com = to_com_free(pos, cell)
    assert torch.allclose(centered.mean(0), torch.zeros(3, dtype=pos.dtype), atol=1e-10)


def test_normalize_denormalize_roundtrip():
    pos, cell = _random_structure()
    z0, com, s = normalize(pos, cell)
    recon = denormalize(z0, com, s, cell, wrap=True)
    # recon should equal pos up to a periodic image -> min-image displacement ~ 0
    disp = min_image(recon - pos, cell)
    assert torch.allclose(disp, torch.zeros_like(disp), atol=1e-8)


def test_centering_is_translation_invariant():
    pos, cell = _random_structure()
    centered_a, com_a = to_com_free(pos, cell)
    shift = torch.tensor([3.1, -7.4, 12.0], dtype=pos.dtype)
    centered_b, com_b = to_com_free(pos + shift, cell)
    # Centered (internal) coordinates are unchanged by a global translation.
    assert torch.allclose(centered_a, centered_b, atol=1e-8)
    # CoM tracks the shift (modulo the cell).
    assert torch.allclose(min_image(com_b - com_a - shift, cell),
                          torch.zeros(3, dtype=pos.dtype), atol=1e-8)


def test_centering_is_wrap_invariant():
    pos, cell = _random_structure()
    centered_a, _ = to_com_free(pos, cell)
    # Wrap every atom by a random integer number of lattice vectors.
    offsets = torch.randint(-2, 3, (pos.shape[0], 3)).to(pos.dtype)
    pos_wrapped = pos + offsets @ cell
    centered_b, _ = to_com_free(pos_wrapped, cell)
    assert torch.allclose(centered_a, centered_b, atol=1e-8)


def test_forward_noising_stays_com_free():
    pos, cell = _random_structure()
    sched = VPSchedule(timesteps=100, beta_schedule="cosine")
    for t in [0, 25, 50, 99]:
        ns = add_noise(sched, pos.float(), cell.float(), torch.tensor(t))
        assert torch.allclose(ns.noise.mean(0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(ns.z_t.mean(0), torch.zeros(3), atol=1e-5)


def test_scale_matches_uniform_fill_std():
    # s = V^(1/3)/sqrt(12); for a cube that is box/sqrt(12).
    _, cell = _random_structure(box=10.0)
    assert math.isclose(float(structure_scale(cell)), 10.0 / math.sqrt(12.0), rel_tol=1e-6)


def test_two_cell_sizes_map_to_similar_variance():
    # Density normalization: 80- and 640-box structures land at ~unit variance.
    torch.manual_seed(1)
    for box in (10.084, 20.252):
        pos, cell = _random_structure(n=200, box=box)
        z0, _, _ = normalize(pos, cell)
        std = z0.std().item()
        assert 0.7 < std < 1.3, f"box {box}: normalized std {std}"

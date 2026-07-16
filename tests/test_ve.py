"""VE (variance-exploding) core: schedule ladder, additive noising, uniform init."""
import math

import numpy as np
import pytest
import torch

from insite_diff.diffusion.noising import min_image, ve_add_noise, uniform_init
from insite_diff.diffusion.sampler import annealed_langevin
from insite_diff.diffusion.schedule import VESchedule
from insite_diff.geometry import rmsd_same_atoms


def test_sigma_ladder_is_geometric_high_to_low():
    s = VESchedule(sigma_min=0.01, sigma_max=0.75, n_levels=100)
    assert s.sigmas[0].item() == max(s.sigmas).item()          # index 0 = highest
    assert abs(s.sigmas[0].item() - 0.75) < 1e-5
    assert abs(s.sigmas[-1].item() - 0.01) < 1e-5
    ratios = (s.sigmas[:-1] / s.sigmas[1:])
    assert torch.allclose(ratios, ratios[0] * torch.ones_like(ratios), rtol=1e-4)  # geometric


def test_cond_maps_sigma_to_unit_interval():
    s = VESchedule(0.01, 0.75, 100)
    assert abs(s.cond(torch.tensor([0.75])).item() - 1.0) < 1e-4
    assert abs(s.cond(torch.tensor([0.01])).item() - 0.0) < 1e-4


def test_uniform_init_fills_the_box():
    torch.manual_seed(0)
    cell = torch.eye(3) * 20.25
    x = uniform_init(2000, cell)
    frac = x @ torch.linalg.inv(cell)
    assert frac.min() >= 0.0 and frac.max() <= 1.0
    # roughly uniform: mean fractional ~0.5, std ~1/sqrt(12)
    assert abs(frac.mean().item() - 0.5) < 0.05
    assert abs(frac.std().item() - 1 / math.sqrt(12)) < 0.03


def test_ve_noise_is_additive_com_free_and_physical_scale():
    torch.manual_seed(0)
    cell = torch.eye(3) * 20.25
    x0 = torch.rand(200, 3) @ cell
    sigma = torch.tensor(0.5)
    x_sigma, eps = ve_add_noise(x0, sigma, cell)
    # additive: x_sigma - x0 == sigma * eps
    assert torch.allclose(x_sigma - x0, sigma * eps, atol=1e-6)
    # eps CoM-free
    assert torch.allclose(eps.mean(0), torch.zeros(3), atol=1e-5)
    # displacement scale ~ sigma (physical Angstrom), not unit-normalized
    disp = (x_sigma - x0).norm(dim=1)
    assert 0.3 < disp.mean().item() < 1.5      # ~sigma * sqrt(3-ish), O(Angstrom)


def test_annealed_langevin_output_in_box():
    cell = torch.eye(3) * 15.0
    def zero(x, sigma):
        return torch.zeros_like(x)
    out = annealed_langevin(VESchedule(0.01, 0.5, 10), zero, 30, cell,
                            langevin_steps=2, refine_steps=5,
                            generator=torch.Generator().manual_seed(0))
    assert out.shape == (30, 3)
    frac = out @ torch.linalg.inv(cell)
    assert frac.min() >= -1e-4 and frac.max() <= 1 + 1e-4


def test_annealed_langevin_oracle_reconstructs():
    """With the true eps (perfect score), Langevin from a uniform prior must
    recover the real structure — validates the VE sampler/formulation."""
    torch.manual_seed(0)
    cell = torch.eye(3, dtype=torch.float64) * 12.0
    x0 = torch.rand(24, 3, dtype=torch.float64) @ cell
    sched = VESchedule(0.01, 0.5, 80)

    def oracle(x, sigma):
        return min_image(x - x0, cell) / sigma

    out = annealed_langevin(sched, oracle, 24, cell, langevin_steps=12, step_lr=2e-5,
                            refine_steps=120, generator=torch.Generator().manual_seed(0))
    assert rmsd_same_atoms(out.numpy(), x0.numpy(), cell.numpy()) < 0.3


# --------------------------------------------------------------------------- #
# Seeded-RNG device discipline
#
# The eval scripts pair a CPU generator (for reproducibility) with a CUDA compute
# device. Drawing with that pair raises "Expected a 'cuda' device type for generator
# but found 'cpu'" — which a CPU-only dev box can never reproduce, so it reached the
# cluster and killed a sweep at uniform_init. draw_randn/draw_rand encapsulate the
# rule; the guard below is the part that actually runs without a GPU.
# --------------------------------------------------------------------------- #
def test_no_raw_seeded_rng_outside_helpers():
    """Every seeded draw must go through draw_randn/draw_rand (or mirror their rule).

    A raw torch.rand/randn/randint taking `generator=` and an explicit `device=` is the
    exact shape of the bug; catching it by source inspection is the only way to catch it
    on a CPU-only machine.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "insite_diff"
    allowed = {("noising.py", "draw_randn"), ("noising.py", "draw_rand"),
               ("schedule.py", "sample_sigma")}          # these implement the rule
    offenders = []
    for path in root.rglob("*.py"):
        src = path.read_text()
        fn = None
        for i, line in enumerate(src.splitlines(), 1):
            m = re.match(r"\s*def (\w+)", line)
            if m:
                fn = m.group(1)
            if re.search(r"torch\.(rand|randn|randint)\(", line) and "generator=" in line:
                if (path.name, fn) not in allowed:
                    offenders.append(f"{path.name}:{i} in {fn}(): {line.strip()}")
    assert not offenders, (
        "raw seeded RNG draw bypassing draw_randn/draw_rand:\n  " + "\n  ".join(offenders))


def test_uniform_init_is_reproducible_and_in_box():
    cell = torch.eye(3) * 9.0
    a = uniform_init(20, cell, generator=torch.Generator().manual_seed(7))
    b = uniform_init(20, cell, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a, b)                      # generator is actually honored
    frac = a @ torch.linalg.inv(cell)
    assert frac.min() >= -1e-6 and frac.max() <= 1 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reproduces a CUDA-only crash")
def test_cpu_generator_with_cuda_device():
    """The gen5-sweep crash: CPU generator + CUDA device must not raise, and the
    result must land on the compute device."""
    dev = torch.device("cuda")
    cell = (torch.eye(3) * 10.0).to(dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = uniform_init(16, cell, generator=g, device=dev, dtype=torch.float32)
    assert x.device.type == "cuda" and x.shape == (16, 3)

    sched = VESchedule(0.01, 0.75, 50).to(dev)
    s = sched.sample_sigma(generator=torch.Generator(device="cpu").manual_seed(0))
    assert s.device.type == "cuda"

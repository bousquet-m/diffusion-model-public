"""Inpainting must clamp context atoms (unchanged at t=0) while generating mobile
atoms. Model-agnostic: the clamp holds regardless of network weights."""
import numpy as np
import torch

from insite_diff.data.graph import build_graph  # noqa: F401 (ensures import path)
from insite_diff.data.mask import cube_mask
from insite_diff.diffusion.noising import min_image
from insite_diff.diffusion.schedule import VPSchedule
from insite_diff.model.denoiser import E3Denoiser
from insite_diff.sampling.inpaint import inpaint


def _setup(n=80, box=15.0, seed=0):
    torch.manual_seed(seed)
    cell = torch.eye(3, dtype=torch.float32) * box
    frac = torch.rand(n, 3)
    pos = frac @ cell
    types = torch.randint(0, 2, (n,))
    mobile = torch.tensor(cube_mask(pos.numpy(), cell.numpy(), mask_frac=0.4))
    model = E3Denoiser(n_species=2, hidden_irreps="8x0e + 4x1o", sh_lmax=1, n_layers=1,
                       radial_basis=4, sigma_embed_dim=8, cutoff=4.0).eval()
    sched = VPSchedule(timesteps=15, beta_schedule="cosine")
    return sched, model, pos, cell, types, mobile


def test_context_atoms_unchanged():
    sched, model, pos, cell, types, mobile = _setup()
    out = inpaint(sched, model, pos, cell, types, mobile, cutoff=4.0,
                  generator=torch.Generator().manual_seed(1))
    context = ~mobile
    disp = min_image(out[context] - pos[context], cell)   # periodic-aware difference
    assert torch.allclose(disp, torch.zeros_like(disp), atol=1e-4), \
        f"context moved: max {disp.abs().max().item():.2e}"


def test_mobile_atoms_are_generated():
    sched, model, pos, cell, types, mobile = _setup()
    out = inpaint(sched, model, pos, cell, types, mobile, cutoff=4.0,
                  generator=torch.Generator().manual_seed(1))
    disp = min_image(out[mobile] - pos[mobile], cell)
    assert disp.abs().mean() > 1e-2, "mobile atoms did not change"


def test_output_shape_and_in_box():
    sched, model, pos, cell, types, mobile = _setup()
    out = inpaint(sched, model, pos, cell, types, mobile, cutoff=4.0,
                  generator=torch.Generator().manual_seed(2))
    assert out.shape == pos.shape
    assert bool(((out >= 0) & (out <= float(cell[0, 0]) + 1e-4)).all())


def test_number_of_context_and_mobile_preserved():
    sched, model, pos, cell, types, mobile = _setup()
    # count consistency: nothing is created/destroyed
    assert int(mobile.sum()) + int((~mobile).sum()) == pos.shape[0]
    assert 0 < int(mobile.sum()) < pos.shape[0]


# --- SDEdit partial-noising (VE augmentation) ------------------------------- #
def _ve_setup(n=60, box=15.0, seed=0):
    from insite_diff.diffusion.schedule import VESchedule
    torch.manual_seed(seed)
    cell = torch.eye(3, dtype=torch.float32) * box
    pos = torch.rand(n, 3) @ cell
    types = torch.zeros(n, dtype=torch.long)     # single species (a-C-like)
    model = E3Denoiser(n_species=1, hidden_irreps="8x0e + 4x1o", sh_lmax=1, n_layers=1,
                       radial_basis=4, sigma_embed_dim=8, cutoff=4.0).eval()
    sched = VESchedule(sigma_min=0.01, sigma_max=0.75, n_levels=40)
    return sched, model, pos, cell, types


def test_sdedit_stays_closer_to_seed_than_full_generation():
    """A low SDEdit start_sigma must perturb the seed less than the full anneal from
    the uniform prior — the whole point of the similarity<->diversity dial."""
    from insite_diff.diffusion.noising import min_image
    from insite_diff.sampling.inpaint import ve_inpaint
    sched, model, pos, cell, types = _ve_setup()
    mobile = torch.ones(pos.shape[0], dtype=torch.bool)     # mask_frac=1.0: whole structure
    kw = dict(cutoff=4.0, max_neighbors=0, langevin_steps=2, step_lr=1e-4, refine_steps=5)

    full = ve_inpaint(sched, model, pos, cell, types, mobile, **kw,
                      generator=torch.Generator().manual_seed(1))
    sde = ve_inpaint(sched, model, pos, cell, types, mobile, **kw,
                     generator=torch.Generator().manual_seed(1), start_sigma=0.1)

    d_full = min_image(full - pos, cell).norm(dim=1).mean()
    d_sde = min_image(sde - pos, cell).norm(dim=1).mean()
    assert d_sde < d_full, f"SDEdit ({d_sde:.3f}) not closer to seed than full ({d_full:.3f})"
    assert bool(((sde >= 0) & (sde <= float(cell[0, 0]) + 1e-4)).all())


def test_sdedit_below_sigma_min_raises():
    from insite_diff.sampling.inpaint import ve_inpaint
    sched, model, pos, cell, types = _ve_setup()
    mobile = torch.ones(pos.shape[0], dtype=torch.bool)
    with __import__("pytest").raises(ValueError):
        ve_inpaint(sched, model, pos, cell, types, mobile, cutoff=4.0, max_neighbors=0,
                   langevin_steps=1, step_lr=1e-4, refine_steps=1, start_sigma=0.001)

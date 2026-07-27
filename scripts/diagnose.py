"""One-step denoising diagnostic: is the failure in the MODEL or the SAMPLER?

Loads a trained checkpoint and a real structure, then at a range of noise levels t:
  - predicts eps and compares to the TRUE injected eps:
      cos(eps_hat, eps)   ~1 = perfect direction, ~0 = uninformative
      MSE(eps_hat, eps)   1.0 = predicting zero (the trivial baseline)
      ||eps_hat|| / ||eps||   ~0 = model outputs ~nothing (the "predict zero" failure)
  - reconstructs x0 from eps_hat and reports its RMSD to the true x0 (Angstrom).

Reading:
  - High cos / low x0-RMSD at LOW noise (small t)  -> the model denoises fine;
    if full generation still fails, suspect the SAMPLER.
  - cos ~0, MSE ~1, ratio ~0 at ALL t             -> the model never learned;
    this is a TRAINING problem, not the sampler.

    python scripts/diagnose.py --checkpoint <final.pt> --structure <640.extxyz>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from ase.io import read

from insite_diff.data.dataset import species_to_index
from insite_diff.data.graph import build_graph_torch
from insite_diff.diffusion.noising import (com_free_noise, min_image, normalize,
                                           q_sample, ve_add_noise)
from insite_diff.diffusion.schedule import VESchedule, VPSchedule
from insite_diff.training.trainer import load_model


def diagnose_ve(model, cfg, pos, cell, types, n, device, repeats):
    """VE diagnostic: predict eps across the sigma ladder. A working VE model keeps
    cos high and x0-RMSD small at EVERY sigma — there is no dead zone, because the
    noise never fully destroys the structure."""
    d = cfg.diffusion
    sched = VESchedule(d.sigma_min, d.sigma_max, d.n_sigma_levels).to(device)
    hdr = f"{'sigma(A)':>9} {'cos':>7} {'MSE':>7} {'|eps_hat|/|eps|':>15} {'x0_RMSD(A)':>11}"
    print(hdr + "\n" + "-" * len(hdr))
    idx = np.linspace(0, d.n_sigma_levels - 1, 8).round().astype(int)   # sample the ladder
    for i in idx:
        sigma = sched.sigmas[i]
        coss, mses, ratios, rmsds = [], [], [], []
        for _ in range(repeats):
            x_sigma, eps = ve_add_noise(pos, sigma, cell)
            with torch.no_grad():
                g = build_graph_torch(x_sigma, cell, cfg.graph.cutoff, cfg.graph.max_neighbors,
                                      pbc=cfg.graph.pbc)
                eps_hat = model(types, g.edge_index, g.edge_vec, sched.cond(sigma), n)
            coss.append(torch.cosine_similarity(eps_hat.flatten(), eps.flatten(), dim=0).item())
            mses.append(torch.mean((eps_hat - eps) ** 2).item())
            ratios.append((eps_hat.norm() / eps.norm()).item())
            x0_hat = x_sigma - sigma * eps_hat
            rmsds.append(min_image(x0_hat - pos, cell).pow(2).sum(1).mean().sqrt().item())
        print(f"{float(sigma):>9.4f} {np.mean(coss):>7.3f} {np.mean(mses):>7.3f} "
              f"{np.mean(ratios):>15.3f} {np.mean(rmsds):>11.3f}")
    print("\nWorking VE model: cos ~high and x0_RMSD small at ALL sigma (no dead zone).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--structure", required=True, help="extxyz file")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=4, help="noise draws averaged per t")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg, avg = load_model(args.checkpoint, device=device)
    print(f"loaded model: type={cfg.diffusion.type}, avg_neighbors={avg}, "
          f"cutoff={cfg.graph.cutoff}, n_layers={cfg.model.n_layers}, hidden={cfg.model.hidden_irreps}")

    atoms = read(args.structure, index=args.frame)
    z2i = species_to_index(cfg.data.species)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=torch.float32, device=device)
    types = torch.tensor([z2i[z] for z in atoms.get_atomic_numbers()], dtype=torch.long, device=device)
    n = pos.shape[0]
    print(f"structure: N={n}, box={float(cell[0,0]):.2f} A\n")

    if cfg.diffusion.type == "ve":
        diagnose_ve(model, cfg, pos, cell, types, n, device, args.repeats)
        return

    schedule = VPSchedule(cfg.diffusion.timesteps, cfg.diffusion.beta_schedule).to(device)
    z0, com, s = normalize(pos, cell)

    torch.manual_seed(0)
    hdr = f"{'t':>5} {'t/T':>6} {'sqrt_abar':>9} {'cos':>7} {'MSE':>7} {'|eps_hat|/|eps|':>15} {'x0_RMSD(A)':>11}"
    print(hdr)
    print("-" * len(hdr))
    for t in [1, 10, 50, 100, 200, 500, 800, 999]:
        coss, mses, ratios, rmsds = [], [], [], []
        for _ in range(args.repeats):
            tt = torch.tensor(t, device=device)
            eps = com_free_noise(n, device=device)
            z_t = q_sample(schedule, z0, tt, eps)
            with torch.no_grad():
                g = build_graph_torch(z_t * s, cell, cfg.graph.cutoff, cfg.graph.max_neighbors)
                eps_hat = model(types, g.edge_index, g.edge_vec,
                                tt.float() / schedule.timesteps, n)
            coss.append(torch.cosine_similarity(eps_hat.flatten(), eps.flatten(), dim=0).item())
            mses.append(torch.mean((eps_hat - eps) ** 2).item())
            ratios.append((eps_hat.norm() / eps.norm()).item())
            sa = schedule.sqrt_alpha_bar[t].item()
            sb = schedule.sqrt_one_minus_alpha_bar[t].item()
            z0_hat = (z_t - sb * eps_hat) / sa
            rmsds.append((((z0_hat - z0) * s).pow(2).sum(1).mean().sqrt()).item())
        print(f"{t:>5} {t/schedule.timesteps:>6.3f} {schedule.sqrt_alpha_bar[t].item():>9.4f} "
              f"{np.mean(coss):>7.3f} {np.mean(mses):>7.3f} {np.mean(ratios):>15.3f} {np.mean(rmsds):>11.3f}")

    print("\nBaselines: predict-zero -> cos 0, MSE ~1, ratio 0, x0_RMSD = clean spread.")
    print("A working denoiser should show cos -> 1 and x0_RMSD -> 0 as t -> 1 (low noise).")


if __name__ == "__main__":
    main()

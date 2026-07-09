"""Milestone-one validation.

Mask an interior region of held-out (640) structures, regenerate it by inpainting,
and check that the result reproduces reference structural statistics:
  - In-O / In-In / O-O partial RDFs (and first-peak positions),
  - In-O coordination-number distribution and mean In-O bond length,
comparing generated vs the masked reference within configured tolerances.
Writes comparison plots and a JSON summary.

Optional --mace: report MACE energy distributions of generated (and optionally
relaxed) structures vs the reference set. This is slow on CPU for 640 atoms
(~40s/energy eval), so keep --n small.

    python scripts/validate.py --config configs/base.yaml \
        --checkpoint checkpoints/final.pt --n 8 --out outputs/validation
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from insite_diff.analysis import coordination, plots
from insite_diff.analysis.rdf import all_partials, first_peak
from insite_diff.config import load_config
from insite_diff.data.dataset import build_datasets
from insite_diff.data.mask import partition
from insite_diff.diffusion.schedule import VPSchedule
from insite_diff.sampling.inpaint import inpaint
from insite_diff.training.trainer import load_model
from insite_diff.utils import seed_everything, select_device

Z_IN, Z_O = atomic_numbers["In"], atomic_numbers["O"]


def make_atoms(numbers, positions, cell) -> Atoms:
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def generate(cfg, model, schedule, val_ds, n, device):
    """Inpaint n held-out structures; return (generated, reference) Atoms lists + masks."""
    rng = np.random.default_rng(cfg.seed)
    gen_frames, ref_frames, masks = [], [], []
    for i in range(min(n, len(val_ds))):
        item = val_ds[i]
        pos, cell = item["positions"], item["cell"]
        mobile = partition(pos.numpy(), cell.numpy(), cfg.mask, rng)
        out = inpaint(schedule, model, pos, cell, item["types"], torch.tensor(mobile),
                      cutoff=cfg.graph.cutoff, max_neighbors=cfg.graph.max_neighbors,
                      device=device,
                      generator=torch.Generator(device="cpu").manual_seed(cfg.seed + i),
                      repaint_jumps=cfg.sampling.repaint_jumps)
        numbers = item["numbers"].numpy()
        gen_frames.append(make_atoms(numbers, out.cpu().numpy(), cell.numpy()))
        ref_frames.append(make_atoms(numbers, pos.numpy(), cell.numpy()))
        masks.append(mobile)
        print(f"[validate] inpainted {i}: traj={item['traj_id']} mobile={int(mobile.sum())}")
    return gen_frames, ref_frames, masks


def structural_report(cfg, gen_frames, ref_frames, out_dir):
    rmax, nbins, cn_cut = cfg.validation.rdf_rmax, cfg.validation.rdf_bins, cfg.validation.cn_cutoff_ino
    ref_rdf = all_partials(ref_frames, Z_IN, Z_O, rmax, nbins)
    gen_rdf = all_partials(gen_frames, Z_IN, Z_O, rmax, nbins)
    plots.plot_rdfs(ref_rdf, gen_rdf, os.path.join(out_dir, "rdf_comparison.png"))

    ref_c = coordination.summary(ref_frames, Z_IN, Z_O, cn_cut)
    gen_c = coordination.summary(gen_frames, Z_IN, Z_O, cn_cut)
    plots.plot_cn(coordination.cn_histogram(ref_c["cns"]),
                  coordination.cn_histogram(gen_c["cns"]),
                  os.path.join(out_dir, "coordination_comparison.png"))

    peak_ref = first_peak(*ref_rdf["In-O"])
    peak_gen = first_peak(*gen_rdf["In-O"])
    tol = cfg.validation.tolerances
    checks = {
        "In-O_first_peak": {
            "ref": peak_ref, "gen": peak_gen, "abs_diff": abs(peak_gen - peak_ref),
            "tol": tol.rdf_peak_A, "pass": abs(peak_gen - peak_ref) <= tol.rdf_peak_A,
        },
        "mean_In-O_bond": {
            "ref": ref_c["mean_bond"], "gen": gen_c["mean_bond"],
            "abs_diff": abs(gen_c["mean_bond"] - ref_c["mean_bond"]),
            "tol": tol.mean_bond_A, "pass": abs(gen_c["mean_bond"] - ref_c["mean_bond"]) <= tol.mean_bond_A,
        },
        "mean_In-O_coordination": {"ref": ref_c["mean_cn"], "gen": gen_c["mean_cn"]},
    }
    return checks


def mace_report(cfg, gen_frames, ref_frames, out_dir, relax_steps, device):
    from insite_diff.analysis.mace_relax import load_calculator, potential_energy, relax
    mace_device = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    calc = load_calculator(cfg.mace.model_path, device=mace_device)
    e_ref = [potential_energy(a, calc) / len(a) for a in ref_frames]
    e_gen = [potential_energy(a, calc) / len(a) for a in gen_frames]
    energies = {"reference": e_ref, "generated": e_gen}
    if relax_steps > 0:
        e_relaxed = []
        for a in gen_frames:
            _, _, e1 = relax(a, calc, fmax=cfg.mace.relax_fmax, steps=relax_steps)
            e_relaxed.append(e1 / len(a))
        energies["generated_relaxed"] = e_relaxed
    plots.plot_energy_hist(energies, os.path.join(out_dir, "energy_comparison.png"))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for k, v in energies.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs/validation")
    ap.add_argument("--mace", action="store_true", help="also compute MACE energy distributions (slow)")
    ap.add_argument("--relax-steps", type=int, default=None, help="override MACE relax steps")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model, _, _ = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    schedule = VPSchedule(cfg.diffusion.timesteps, cfg.diffusion.beta_schedule).to(device)
    _, val_ds, split = build_datasets(cfg)
    print(f"[validate] {len(val_ds)} held-out frames / {len(split.val_traj_ids)} trajectories")

    gen_frames, ref_frames, _ = generate(cfg, model, schedule, val_ds, args.n, device)
    report = {"n_structures": len(gen_frames), "structural": structural_report(cfg, gen_frames, ref_frames, out_dir)}

    for name, c in report["structural"].items():
        if "pass" in c:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"[validate] {name}: ref={c['ref']:.3f} gen={c['gen']:.3f} "
                  f"|d|={c['abs_diff']:.3f} tol={c['tol']:.3f} -> {status}")
        else:
            print(f"[validate] {name}: ref={c['ref']:.3f} gen={c['gen']:.3f}")

    if args.mace:
        relax_steps = args.relax_steps if args.relax_steps is not None else cfg.mace.relax_steps
        print(f"[validate] MACE energies (relax_steps={relax_steps}) — slow on CPU...")
        report["mace"] = mace_report(cfg, gen_frames, ref_frames, out_dir, relax_steps, device)
        for k, v in report["mace"].items():
            print(f"[validate] energy/atom {k}: {v['mean']:.4f} +/- {v['std']:.4f} eV")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"[validate] wrote plots + summary.json to {out_dir}/")


if __name__ == "__main__":
    main()

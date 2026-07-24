"""Mask-fraction sweep: the core milestone-1b evaluation of a trained model.

For each mask_frac on a ladder, regenerate masked regions of held-out-trajectory
structures and report, per fraction:
  - mobile-only In-O / In-In / O-O RDFs, In-O coordination, mean In-O bond length
  - multi-seed SPREAD: sample the same masked structure n_seeds times; permutation-
    aware RMSD over mobile atoms (Hungarian within species). ~0 spread => mobile
    atoms were never noised (bug); small spread at low mask_frac is expected.
  - MEMORIZATION: SOAP nearest-neighbor cosine similarity to the training set.
  - optional --mace: energy of generated structures before/after relaxation vs ref.

Writes one summary_<frac>.json per fraction plus a combined metric-vs-mask_frac plot.

    python scripts/sweep.py --config configs/base.yaml --checkpoint checkpoints/final.pt \
        --n-structures 3 --n-seeds 10 --out outputs/sweep
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from insite_diff.analysis import carbon, coordination, plots, similarity
from insite_diff.analysis.rdf import all_partials, first_peak, partial_rdf
from insite_diff.config import load_config
from insite_diff.data.dataset import build_datasets
from insite_diff.data.graph import build_graph_torch
from insite_diff.data.mask import partition
from insite_diff.diffusion.noising import structure_scale
from insite_diff.diffusion.sampler import sample as sample_unconditional
from insite_diff.diffusion.schedule import VPSchedule
from insite_diff.sampling.inpaint import inpaint_dispatch
from insite_diff.training.trainer import load_model
from insite_diff.utils import seed_everything, select_device

Z_IN, Z_O, Z_C = atomic_numbers["In"], atomic_numbers["O"], atomic_numbers["C"]
DEFAULT_LADDER = [0.016, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]


def _is_carbon(cfg) -> bool:
    """a-C run: single carbon species. Selects the sp2/sp3 metric over the In-O one."""
    return cfg.data.species == ["C"]


def _soap_species(cfg):
    return ("C",) if _is_carbon(cfg) else ("In", "O")


def _atoms(numbers, positions, cell):
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _generate_full_vp(cfg, model, cell, types, n_atoms, device, seed):
    """VP pure unconditional generation (mask_frac == 1.0, no context)."""
    schedule = VPSchedule(cfg.diffusion.timesteps, cfg.diffusion.beta_schedule).to(device)
    s = structure_scale(cell)

    def predict(z_t, t):
        g = build_graph_torch(z_t * s, cell, cfg.graph.cutoff, cfg.graph.max_neighbors)
        return model(types, g.edge_index, g.edge_vec,
                     torch.tensor(t / schedule.timesteps, device=device), n_atoms)

    return sample_unconditional(schedule, predict, n_atoms, cell,
                                com=torch.zeros(3, device=device), scale=s,
                                generator=torch.Generator(device="cpu").manual_seed(seed),
                                device=device)


def _seeds_for_structure(cfg, model, item, mask_frac, device, n_seeds):
    """n_seeds generated samples of ONE masked structure; returns (samples, mobile_mask).

    Uses inpaint_dispatch (vp|ve). For VE, mask_frac=1.0 reduces to unconditional
    annealed Langevin (empty context). For VP, mask_frac=1.0 uses the sampler.
    """
    pos, cell, types = item["positions"], item["cell"], item["types"]
    n = pos.shape[0]

    if mask_frac >= 1.0:
        mobile = np.ones(n, dtype=bool)
        if cfg.diffusion.type == "ve":
            gen = [inpaint_dispatch(cfg, model, pos, cell, types, torch.tensor(mobile), device,
                                    generator=torch.Generator(device="cpu").manual_seed(cfg.seed + s)
                                    ).cpu().numpy() for s in range(n_seeds)]
        else:
            gen = [_generate_full_vp(cfg, model, cell.to(device), types.to(device), n,
                                     device, seed=cfg.seed + s).cpu().numpy() for s in range(n_seeds)]
        return gen, mobile

    rng = np.random.default_rng(cfg.seed)          # same mask across seeds
    mobile = partition(pos.numpy(), cell.numpy(), dataclasses.replace(cfg.mask, mask_frac=mask_frac), rng)
    gen = [inpaint_dispatch(cfg, model, pos, cell, types, torch.tensor(mobile), device,
                            generator=torch.Generator(device="cpu").manual_seed(cfg.seed + s)
                            ).cpu().numpy() for s in range(n_seeds)]
    return gen, mobile


def run_fraction(cfg, model, val_ds, mask_frac, device, args, train_desc):
    carbon_run = _is_carbon(cfg)
    rmax, nbins = cfg.validation.rdf_rmax, cfg.validation.rdf_bins
    cn = cfg.validation.cn_cutoff_cc if carbon_run else cfg.validation.cn_cutoff_ino
    gen_frames, ref_frames, masks = [], [], []
    spreads = []
    for i in range(min(args.n_structures, len(val_ds))):
        item = val_ds[i]
        numbers = item["numbers"].numpy()
        cell = item["cell"].numpy()
        samples, mobile = _seeds_for_structure(cfg, model, item, mask_frac, device, args.n_seeds)
        spreads.append(similarity.multi_seed_spread(samples, numbers, item["cell"].numpy(), mobile)["mean_rmsd"])
        for sp in samples:
            gen_frames.append(_atoms(numbers, sp, cell))
            ref_frames.append(_atoms(numbers, item["positions"].numpy(), cell))
            masks.append(mobile)
        print(f"[sweep]   frac={mask_frac:<5} struct {i}: mobile={int(mobile.sum())}/{len(mobile)} "
              f"spread={spreads[-1]:.3f} A")

    restrict = "both_mobile" if mask_frac < 1.0 else "all"
    cmo = mask_frac < 1.0
    nn_sim = similarity.memorization_similarity(gen_frames, masks, train_desc,
                                                species=_soap_species(cfg))
    result = {
        "mask_frac": mask_frac,
        "n_samples": len(gen_frames),
        "multiseed_spread_rmsd": float(np.mean(spreads)),
        "memorization_nn_similarity": {"mean": float(nn_sim.mean()), "max": float(nn_sim.max())},
    }

    if carbon_run:
        # a-C: C-C RDF first peak + the sp2/sp3 hybridization metric (mobile atoms), gen vs
        # the like-masked reference. sp3% is judged as a DISTRIBUTION (mean AND spread).
        gen_rdf = partial_rdf(gen_frames, Z_C, Z_C, rmax, nbins, masks, restrict)
        ref_rdf = partial_rdf(ref_frames, Z_C, Z_C, rmax, nbins, masks, restrict)
        gen_c = carbon.summary(gen_frames, cn, masks, center_mobile_only=cmo)
        ref_c = carbon.summary(ref_frames, cn, masks, center_mobile_only=cmo)
        result["CC_peak"] = {"gen": first_peak(*gen_rdf, lo=1.0, hi=2.0),
                             "ref": first_peak(*ref_rdf, lo=1.0, hi=2.0)}
        result["sp3_pct"] = {"gen_mean": gen_c["sp3_mean"], "gen_std": gen_c["sp3_std"],
                             "ref_mean": ref_c["sp3_mean"], "ref_std": ref_c["sp3_std"]}
        result["sp2_pct"] = {"gen_mean": gen_c["sp2_mean"], "ref_mean": ref_c["sp2_mean"]}
        result["mean_C_coordination"] = {"gen": gen_c["mean_cn"], "ref": ref_c["mean_cn"]}
    else:
        gen_rdf = all_partials(gen_frames, Z_IN, Z_O, rmax, nbins, masks, restrict=restrict)
        ref_rdf = all_partials(ref_frames, Z_IN, Z_O, rmax, nbins, masks, restrict=restrict)
        gen_c = coordination.summary(gen_frames, Z_IN, Z_O, cn, masks, center_mobile_only=cmo)
        ref_c = coordination.summary(ref_frames, Z_IN, Z_O, cn, masks, center_mobile_only=cmo)
        result["In-O_peak"] = {"gen": first_peak(*gen_rdf["In-O"]), "ref": first_peak(*ref_rdf["In-O"])}
        result["mean_In-O_bond"] = {"gen": gen_c["mean_bond"], "ref": ref_c["mean_bond"]}
        result["mean_In-O_coordination"] = {"gen": gen_c["mean_cn"], "ref": ref_c["mean_cn"]}

    if args.mace:
        from insite_diff.analysis.mace_relax import (load_calculator, nvt_refine,
                                                     potential_energy, relax)
        dev = "cuda" if getattr(device, "type", "") == "cuda" else "cpu"
        calc = load_calculator(cfg.mace.model_path, device=dev, head=cfg.mace.head)
        e_gen = [potential_energy(a, calc) / len(a) for a in gen_frames]
        e_ref = [potential_energy(a, calc) / len(a) for a in ref_frames]
        rec = {"gen_before": float(np.mean(e_gen)), "ref": float(np.mean(e_ref))}
        if args.relax_steps > 0:
            e_rel = [relax(a, calc, cfg.mace.relax_fmax, args.relax_steps)[2] / len(a) for a in gen_frames]
            rec["gen_after_relax"] = float(np.mean(e_rel))

        # NVT post-refinement (step 4): does a short MD+quench close the under-coordination?
        # Recompute the structural metrics on the refined frames, not just energy.
        m = cfg.mace
        if m.nvt_steps > 0:
            print(f"[sweep]   NVT refine: T={m.nvt_temperature_K}K dt={m.nvt_timestep_fs}fs "
                  f"steps={m.nvt_steps} on {len(gen_frames)} frames...")
            refined = [nvt_refine(a, calc, temperature_K=m.nvt_temperature_K,
                                  timestep_fs=m.nvt_timestep_fs, steps=m.nvt_steps,
                                  friction_per_fs=m.nvt_friction_per_fs,
                                  quench_fmax=m.relax_fmax, quench_steps=m.relax_steps,
                                  seed=cfg.seed + i)[0] for i, a in enumerate(gen_frames)]
            rec["gen_after_nvt"] = float(np.mean([potential_energy(a, calc) / len(a)
                                                  for a in refined]))
            if carbon_run:
                nvt_rdf = partial_rdf(refined, Z_C, Z_C, rmax, nbins, masks, restrict)
                nvt_c = carbon.summary(refined, cn, masks, center_mobile_only=cmo)
                result["CC_peak"]["gen_nvt"] = first_peak(*nvt_rdf, lo=1.0, hi=2.0)
                result["sp3_pct"]["gen_nvt_mean"] = nvt_c["sp3_mean"]
                result["sp3_pct"]["gen_nvt_std"] = nvt_c["sp3_std"]
                result["mean_C_coordination"]["gen_nvt"] = nvt_c["mean_cn"]
            else:
                nvt_rdf = all_partials(refined, Z_IN, Z_O, rmax, nbins, masks, restrict=restrict)
                nvt_c = coordination.summary(refined, Z_IN, Z_O, cn, masks, center_mobile_only=cmo)
                result["In-O_peak"]["gen_nvt"] = first_peak(*nvt_rdf["In-O"])
                result["mean_In-O_bond"]["gen_nvt"] = nvt_c["mean_bond"]
                result["mean_In-O_coordination"]["gen_nvt"] = nvt_c["mean_cn"]
        result["energy_per_atom"] = rec
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fractions", default=",".join(map(str, DEFAULT_LADDER)),
                    help="comma-separated mask_frac ladder")
    ap.add_argument("--n-structures", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--train-sample", type=int, default=200, help="training structures for SOAP reference")
    ap.add_argument("--max-envs", type=int, default=5000, help="cap on training SOAP environments")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--mace", action="store_true")
    ap.add_argument("--relax-steps", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    # VE Langevin overrides (sampling-time only — no retraining). Left as None they
    # keep the config value; set them to tune the under-relaxation the gen6 sweep showed
    # without editing a config per grid point. See HANDOFF "Langevin tuning".
    ap.add_argument("--langevin-step-lr", type=float, default=None)
    ap.add_argument("--langevin-steps", type=int, default=None)
    ap.add_argument("--refine-steps", type=int, default=None)
    # SDEdit augmentation (VE): start the anneal from a real structure + this much noise,
    # iterating only the ladder <= this sigma. The similarity<->diversity dial for making
    # "more like these 10" without wandering off-distribution. 0/None = full generation.
    ap.add_argument("--sdedit-sigma", type=float, default=None)
    # MACE NVT post-refinement (step 4). --nvt-steps > 0 (with --mace) runs a short NVT MD +
    # 0 K quench on each generated frame and reports post-refinement coordination/bond/energy,
    # so we can see whether it closes the under-coordination. None = keep config value.
    ap.add_argument("--nvt-steps", type=int, default=None)
    ap.add_argument("--nvt-temp", type=float, default=None, help="NVT temperature (K)")
    ap.add_argument("--nvt-timestep", type=float, default=None, help="NVT timestep (fs)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # stream progress to redirected logs
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    # Apply VE Langevin overrides before anything reads cfg.diffusion.
    if args.langevin_step_lr is not None:
        cfg.diffusion.langevin_step_lr = args.langevin_step_lr
    if args.langevin_steps is not None:
        cfg.diffusion.langevin_steps = args.langevin_steps
    if args.refine_steps is not None:
        cfg.diffusion.refine_steps = args.refine_steps
    if args.sdedit_sigma is not None:
        cfg.diffusion.sdedit_sigma = args.sdedit_sigma
    if args.nvt_steps is not None:
        cfg.mace.nvt_steps = args.nvt_steps
    if args.nvt_temp is not None:
        cfg.mace.nvt_temperature_K = args.nvt_temp
    if args.nvt_timestep is not None:
        cfg.mace.nvt_timestep_fs = args.nvt_timestep
    if cfg.diffusion.type == "ve":
        d = cfg.diffusion
        print(f"[sweep] VE Langevin: step_lr={d.langevin_step_lr} steps={d.langevin_steps} "
              f"refine={d.refine_steps} (sigma_levels={d.n_sigma_levels})")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fractions = [float(x) for x in args.fractions.split(",")]

    model, _, _ = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    train_ds, val_ds, split = build_datasets(cfg)
    print(f"[sweep] {len(val_ds)} held-out frames / {len(split.val_traj_ids)} trajectories; "
          f"ladder={fractions}")

    # SOAP reference from a sample of the training set (computed once).
    k = min(args.train_sample, len(train_ds))
    train_frames = [_atoms(train_ds[i]["numbers"].numpy(), train_ds[i]["positions"].numpy(),
                           train_ds[i]["cell"].numpy()) for i in range(k)]
    print(f"[sweep] building SOAP environment bank from {k} training structures...")
    train_desc = similarity.atom_environments(train_frames, masks=None,
                                              species=_soap_species(cfg), max_envs=args.max_envs)

    # a-C: the reference sp3% DISTRIBUTION (mean + spread) over ALL real structures — the
    # target the generated structures must reproduce. Computed once, reported for context.
    ref_dist = None
    if _is_carbon(cfg):
        all_real = [_atoms(ds[i]["numbers"].numpy(), ds[i]["positions"].numpy(), ds[i]["cell"].numpy())
                    for ds in (train_ds, val_ds) for i in range(len(ds))]
        rc = carbon.summary(all_real, cfg.validation.cn_cutoff_cc)
        ref_dist = {"sp3_mean": rc["sp3_mean"], "sp3_std": rc["sp3_std"],
                    "sp2_mean": rc["sp2_mean"], "mean_cn": rc["mean_cn"], "n_structures": len(all_real)}
        print(f"[sweep] a-C reference (n={ref_dist['n_structures']}): "
              f"sp3% = {ref_dist['sp3_mean']:.1f} +/- {ref_dist['sp3_std']:.1f}, "
              f"mean C-coordination = {ref_dist['mean_cn']:.2f}")

    results = []
    for frac in fractions:
        print(f"[sweep] mask_frac = {frac}")
        res = run_fraction(cfg, model, val_ds, frac, device, args, train_desc)
        results.append(res)
        if _is_carbon(cfg):
            print(f"[sweep]   frac={frac}: sp3% gen {res['sp3_pct']['gen_mean']:.1f}+/-"
                  f"{res['sp3_pct']['gen_std']:.1f} vs ref {res['sp3_pct']['ref_mean']:.1f}+/-"
                  f"{res['sp3_pct']['ref_std']:.1f} | spread {res['multiseed_spread_rmsd']:.2f} A "
                  f"| memNN {res['memorization_nn_similarity']['mean']:.4f}")
        with open(os.path.join(args.out, f"summary_{frac}.json"), "w") as f:
            json.dump(res, f, indent=2)

    # combined plot
    if _is_carbon(cfg):
        panels = {
            "C-C first peak (mobile)": {"gen": [r["CC_peak"]["gen"] for r in results],
                                        "ref": [r["CC_peak"]["ref"] for r in results], "ylabel": "A"},
            "sp3 % (mean)": {"gen": [r["sp3_pct"]["gen_mean"] for r in results],
                             "ref": [r["sp3_pct"]["ref_mean"] for r in results], "ylabel": "%"},
            "sp3 % spread (std)": {"gen": [r["sp3_pct"]["gen_std"] for r in results],
                                   "ref": [r["sp3_pct"]["ref_std"] for r in results], "ylabel": "%"},
            "mean C coordination": {"gen": [r["mean_C_coordination"]["gen"] for r in results],
                                    "ref": [r["mean_C_coordination"]["ref"] for r in results], "ylabel": "CN"},
            "multi-seed spread": {"gen": [r["multiseed_spread_rmsd"] for r in results], "ylabel": "RMSD (A)"},
            "memorization (NN SOAP sim)": {"gen": [r["memorization_nn_similarity"]["mean"] for r in results],
                                           "ylabel": "cosine"},
        }
    else:
        panels = {
            "In-O first peak (mobile)": {"gen": [r["In-O_peak"]["gen"] for r in results],
                                         "ref": [r["In-O_peak"]["ref"] for r in results], "ylabel": "A"},
            "mean In-O bond (mobile)": {"gen": [r["mean_In-O_bond"]["gen"] for r in results],
                                        "ref": [r["mean_In-O_bond"]["ref"] for r in results], "ylabel": "A"},
            "mean In-O coordination": {"gen": [r["mean_In-O_coordination"]["gen"] for r in results],
                                       "ref": [r["mean_In-O_coordination"]["ref"] for r in results], "ylabel": "CN"},
            "multi-seed spread": {"gen": [r["multiseed_spread_rmsd"] for r in results], "ylabel": "RMSD (A)"},
            "memorization (NN SOAP sim)": {"gen": [r["memorization_nn_similarity"]["mean"] for r in results],
                                           "ylabel": "cosine"},
        }
    if args.mace:
        panels["energy/atom"] = {"gen": [r["energy_per_atom"]["gen_before"] for r in results],
                                 "ref": [r["energy_per_atom"]["ref"] for r in results], "ylabel": "eV"}
    plots.plot_sweep(fractions, panels, os.path.join(args.out, "sweep.png"))
    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump({"fractions": fractions, "results": results, "reference_distribution": ref_dist}, f, indent=2)
    print(f"[sweep] wrote {len(results)} fractions + sweep.png to {args.out}/")


if __name__ == "__main__":
    main()

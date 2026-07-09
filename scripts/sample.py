"""Inpaint held-out structures: clamp context, regenerate a masked interior region.

    python scripts/sample.py --config configs/smoke.yaml \
        --checkpoint checkpoints/smoke/final.pt --n 2 --out outputs/smoke

Writes, per sample, an extxyz with the regenerated structure and a companion
extxyz of the masked reference, plus a boolean 'mobile' array for downstream
analysis.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from insite_diff.config import load_config
from insite_diff.data.dataset import build_datasets
from insite_diff.data.mask import partition
from insite_diff.diffusion.schedule import VPSchedule
from insite_diff.sampling.inpaint import inpaint
from insite_diff.training.trainer import load_model
from insite_diff.utils import seed_everything, select_device


def to_atoms(numbers, positions, cell, mobile):
    a = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    a.set_array("mobile", mobile.astype(int))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2, help="number of held-out structures to inpaint")
    ap.add_argument("--out", default="outputs/samples")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model, ckpt_cfg, _ = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    schedule = VPSchedule(cfg.diffusion.timesteps, cfg.diffusion.beta_schedule).to(device)

    _, val_ds, split = build_datasets(cfg)
    if len(val_ds) == 0:
        raise SystemExit("no validation structures available to inpaint")
    print(f"[sample] {len(val_ds)} held-out frames from {len(split.val_traj_ids)} trajectories")

    rng = np.random.default_rng(cfg.seed)
    n = min(args.n, len(val_ds))
    for i in range(n):
        item = val_ds[i]
        pos = item["positions"]
        cell = item["cell"]
        mobile = partition(pos.numpy(), cell.numpy(), cfg.mask, rng)
        mobile_t = torch.tensor(mobile)

        out = inpaint(schedule, model, pos, cell, item["types"], mobile_t,
                      cutoff=cfg.graph.cutoff, max_neighbors=cfg.graph.max_neighbors,
                      device=device, generator=torch.Generator(device="cpu").manual_seed(cfg.seed + i),
                      repaint_jumps=cfg.sampling.repaint_jumps)

        numbers = item["numbers"].numpy()
        gen = to_atoms(numbers, out.cpu().numpy(), cell.numpy(), mobile)
        ref = to_atoms(numbers, pos.numpy(), cell.numpy(), mobile)
        base = os.path.join(args.out, f"sample_{i}_{item['traj_id']}")
        write(base + "_generated.extxyz", gen)
        write(base + "_reference.extxyz", ref)
        print(f"[sample] {i}: traj={item['traj_id']} mobile={int(mobile.sum())}/{len(mobile)} "
              f"-> {base}_generated.extxyz")

    print(f"[sample] wrote {n} samples to {args.out}/")


if __name__ == "__main__":
    main()

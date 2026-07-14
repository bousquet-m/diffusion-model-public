"""Oracle-sampler test: is the Cartesian-VP formulation + sampler correct, or
fundamentally broken — independent of any trained model?

We feed the reverse sampler the TRUE eps at every step (a perfect denoiser):
    eps_oracle(z_t, t) = (z_t - sqrt(abar_t) * z0) / sqrt(1 - abar_t)
which is exactly the noise that maps the known clean z0 to the current z_t. With a
correct sampler this drives the reverse chain back to z0, so it should reconstruct
the real structure. A "predict-zero" control should instead collapse to a blob.

Reading:
  - oracle RECONSTRUCTS (small RMSD, no atom overlaps) -> sampler/formulation OK;
    the failure is learnability (the model can't predict eps), not the process.
  - oracle ALSO COLLAPSES (large RMSD / overlaps like the zero control) -> the
    Cartesian-VP formulation is broken for periodic structures -> need fractional
    + wrapped-normal.

Model-free: no checkpoint required.

    python scripts/oracle_test.py --structure /path/640.extxyz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from ase.io import read

from insite_diff.diffusion.noising import min_image, normalize
from insite_diff.diffusion.sampler import sample
from insite_diff.diffusion.schedule import VPSchedule
from insite_diff.geometry import rmsd_same_atoms


def min_pair_distance(pos: np.ndarray, cell: np.ndarray) -> float:
    """Smallest nonzero min-image interatomic distance (Angstrom). ~0 => overlaps."""
    d = pos[None, :, :] - pos[:, None, :]
    frac = d @ np.linalg.inv(cell)
    d = (frac - np.round(frac)) @ cell
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    atoms = read(args.structure, index=args.frame)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=torch.float64)
    n = pos.shape[0]
    z0, com, s = normalize(pos, cell)

    schedule = VPSchedule(args.timesteps, "cosine")
    # cast schedule buffers to float64 for a clean numerical test
    schedule.to("cpu")

    sqrt_ab = schedule.sqrt_alpha_bar.double()
    sqrt_1m = schedule.sqrt_one_minus_alpha_bar.double()

    def predict_oracle(z_t, t):
        return (z_t - sqrt_ab[t] * z0) / sqrt_1m[t]

    def predict_zero(z_t, t):
        return torch.zeros_like(z_t)

    ref_np = pos.numpy()
    cell_np = cell.numpy()
    print(f"structure: N={n}  box={float(cell[0,0]):.2f} A  scale s={float(s):.3f} A")
    print(f"reference: min interatomic distance = {min_pair_distance(ref_np, cell_np):.3f} A\n")

    gen = torch.Generator().manual_seed(args.seed)
    for name, fn in [("ORACLE (true eps)", predict_oracle), ("ZERO control", predict_zero)]:
        out = sample(schedule, fn, n, cell, com, s, generator=gen, device="cpu").double()
        rmsd = rmsd_same_atoms(out.numpy(), ref_np, cell_np)
        mind = min_pair_distance(out.numpy(), cell_np)
        verdict = "reconstructs" if (rmsd < 0.5 and mind > 1.0) else "COLLAPSES/wrong"
        print(f"{name:20s}: min-image RMSD to ref = {rmsd:8.3f} A | "
              f"min pair dist = {mind:6.3f} A  -> {verdict}")

    print("\nInterpretation:")
    print("  oracle reconstructs (RMSD~0, no overlaps) -> sampler/formulation OK -> learnability problem.")
    print("  oracle collapses like the zero control     -> Cartesian-VP formulation broken -> go fractional.")


if __name__ == "__main__":
    main()

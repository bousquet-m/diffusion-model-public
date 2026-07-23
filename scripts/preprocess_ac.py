"""Relabel and consolidate the raw amorphous-carbon AIMD snapshots into one extxyz.

The raw data (aC/extxyz/1.extxyz ... 10.extxyz) has two problems that would make the
pipeline chemically and statistically wrong if fed in as-is (both verified in the a-C
handoff, do not rediscover):

  1. Species mislabel. The species column says "H", but the atoms are carbon: 216 atoms
     in a ~10.88 A box is ~3.3 g/cc (a-C), not the 0.28 g/cc that 216 H would be. We
     relabel H -> C so the data, MACE, and any DFT are chemically correct.
  2. One trajectory id for all ten. Every file carries config_type=ac_325, so a
     split-by-trajectory (the only allowed split — frames of one trajectory are thermal
     near-duplicates) would see a single trajectory and be unable to hold anything out.
     These are ten INDEPENDENT AIMD final snapshots, so each gets a unique id
     ac_325_1 ... ac_325_10.

Output is a single aC.extxyz with ten frames, mirroring how 640.extxyz packs independent
In2O3 structures into one file. Adapted from the raw aC/xyz2extxyz.py.

    python scripts/preprocess_ac.py \
        --src /Users/matt/Desktop/data/aC/extxyz --out /Users/matt/Desktop/data/aC/aC.extxyz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write


def relabel_to_carbon(atoms):
    """Return a copy with every atom set to carbon, preserving positions/cell/pbc."""
    a = atoms.copy()
    a.set_chemical_symbols(["C"] * len(a))
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Users/matt/Desktop/data/aC/extxyz",
                    help="directory holding the raw 1.extxyz ... N.extxyz snapshots")
    ap.add_argument("--out", default="/Users/matt/Desktop/data/aC/aC.extxyz",
                    help="combined output extxyz (one frame per snapshot)")
    ap.add_argument("--n", type=int, default=10, help="number of snapshots (1..N)")
    ap.add_argument("--prefix", default="ac_325", help="trajectory-id prefix")
    args = ap.parse_args()

    src = Path(args.src)
    frames = []
    for i in range(1, args.n + 1):
        atoms = read(src / f"{i}.extxyz", index=-1)   # final snapshot of the file
        atoms = relabel_to_carbon(atoms)
        # Unique, stable trajectory id so the split can hold snapshots out (see module docstring).
        atoms.info["config_type"] = f"{args.prefix}_{i}"
        L = float(atoms.get_cell()[0, 0])
        dens = len(atoms) * 12.011 / 6.02214076e23 / (L * 1e-8) ** 3
        print(f"{i}: N={len(atoms)} L={L:.3f} A  rho={dens:.3f} g/cc  id={atoms.info['config_type']}")
        frames.append(atoms)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, frames, format="extxyz")
    print(f"\nwrote {len(frames)} carbon-relabelled frames -> {out}")


if __name__ == "__main__":
    main()

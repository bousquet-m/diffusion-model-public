"""Data report + pipeline sanity check.

Loads data through the real pipeline (dedup -> thin -> trajectory split) and
reports: per-source frame/atom/cell facts, trajectory grouping, In-O RDF first
peak, the train/val split (with a zero-shared-trajectory assertion), a
neighbor-graph edge count with the receptive-field-vs-box check, an example
interior mask, and a within-trajectory RMSD-vs-separation table that justifies
the thinning stride.

Usage:
    python scripts/inspect_data.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
from ase.geometry import get_distances

from insite_diff.config import load_config
from insite_diff.data.dataset import build_datasets, prepare_records, _group_by_traj
from insite_diff.data.extxyz import load_all
from insite_diff.data.graph import build_graph
from insite_diff.data.mask import partition
from insite_diff.geometry import rmsd_same_atoms


def rdf_first_peak(positions, cell, numbers, z1, z2, rmax=4.0, nbins=160):
    sym = numbers
    p1 = positions[sym == z1]
    p2 = positions[sym == z2]
    _, d_len = get_distances(p1, p2, cell=cell, pbc=True)  # 2nd return = distances (n1,n2)
    d = d_len.flatten()
    d = d[d > 1e-6]
    hist, edges = np.histogram(d, bins=nbins, range=(0.0, rmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers > 1.5) & (centers < 3.0)
    return centers[mask][np.argmax(hist[mask])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("=== RAW LOAD (per source) ===")
    raw = load_all(cfg.data)
    by_src = defaultdict(list)
    for r in raw:
        by_src[r.source_path].append(r)
    for path, recs in by_src.items():
        trajs = {r.traj_id for r in recs}
        boxes = Counter(round(float(r.cell[0, 0]), 3) for r in recs)
        n_atoms = Counter(r.n_atoms for r in recs)
        print(f"  {path.split('/')[-1]}: frames={len(recs)} trajs={len(trajs)} "
              f"N={dict(n_atoms)} boxes={dict(boxes)} role={recs[0].role}")
        peak = rdf_first_peak(recs[0].positions, recs[0].cell, recs[0].numbers, 49, 8)
        print(f"      In-O RDF first peak (frame 0) ~ {peak:.3f} A")

    print("\n=== DEDUP + THIN ===")
    processed = prepare_records(cfg.data)
    print(f"  raw={len(raw)} -> after dedup(rmsd<{cfg.data.dedup_rmsd}) + stride({cfg.data.stride}) = {len(processed)}")

    print("\n=== TRAJECTORY SPLIT ===")
    train_ds, val_ds, split = build_datasets(cfg)
    print(f"  train: {len(train_ds)} frames / {len(split.train_traj_ids)} trajectories")
    print(f"  val:   {len(val_ds)} frames / {len(split.val_traj_ids)} trajectories")
    assert not (split.train_traj_ids & split.val_traj_ids), "LEAK: shared trajectory id!"
    val_sources = {r.source_path.split('/')[-1] for r in split.val}
    print(f"  val drawn only from: {val_sources}  (expected: 640.extxyz)")
    print(f"  example val traj ids: {sorted(split.val_traj_ids)[:5]}")

    print("\n=== NEIGHBOR GRAPH + RECEPTIVE FIELD ===")
    sample = val_ds[0] if len(val_ds) else train_ds[0]
    g = build_graph(sample["positions"].numpy(), sample["cell"].numpy(),
                    cfg.graph.cutoff, sample["numbers"].numpy(), cfg.graph.max_neighbors)
    box = float(sample["cell"][0, 0])
    rf = cfg.graph.cutoff * cfg.model.n_layers
    print(f"  frame N={sample['n_atoms']} box={box:.2f} A: edges={g.edge_index.shape[1]} "
          f"(avg degree {g.edge_index.shape[1]/sample['n_atoms']:.1f})")
    print(f"  receptive field = cutoff*n_layers = {rf:.1f} A vs box/2 = {box/2:.1f} A -> "
          f"{'OK (clean buffer)' if rf < box/2 else 'WRAPS — no clean buffer'}")

    print("\n=== EXAMPLE INTERIOR MASK (val frame) ===")
    rng = np.random.default_rng(cfg.seed)
    mob = partition(sample["positions"].numpy(), sample["cell"].numpy(), cfg.mask, rng)
    print(f"  geometry={cfg.mask.geometry} mask_frac={cfg.mask.mask_frac}: "
          f"mobile={int(mob.sum())} context={int((~mob).sum())} of {len(mob)}")

    print("\n=== STRIDE JUSTIFICATION: within-trajectory RMSD vs frame separation ===")
    groups = _group_by_traj(raw)
    seps = defaultdict(list)
    for group in groups.values():
        for k in range(len(group)):
            for lag in range(1, len(group) - k):
                seps[lag].append(
                    rmsd_same_atoms(group[k].positions, group[k + lag].positions, group[k].cell)
                )
    for lag in sorted(seps)[:6]:
        vals = np.array(seps[lag])
        print(f"  lag {lag}: mean RMSD {vals.mean():.3f} A (n={len(vals)})")
    print("  -> within a trajectory frames are one vibrating inherent structure "
          "(sub-angstrom, non-diffusive); stride mainly reduces redundancy, not decorrelation.")


if __name__ == "__main__":
    main()

"""Summarize a Langevin-tuning run (tune_langevin_gen6.job) into one comparison table.

Each subdirectory of the tuning output holds a normal sweep's summary_<frac>.json files.
This walks them and prints, per (run, mask_frac): mean In-O coordination gen/ref, mean
bond gen/ref, energy/atom gen-ref (meV), and multi-seed spread — the four numbers that
say whether a knob setting fixed the under-relaxation without collapsing diversity.

    python scripts/collect_langevin_tuning.py outputs/tune_gen6
"""
from __future__ import annotations

import glob
import json
import os
import sys


def _load_run(run_dir: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "summary_*.json")),
                    key=lambda p: float(os.path.basename(p)[len("summary_"):-len(".json")])):
        out.append(json.load(open(f)))
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs/tune_gen6"
    run_dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"no run subdirectories under {root}")
        return

    # Post-NVT columns only appear if any run enabled --nvt-steps (step 4).
    has_nvt = any("gen_nvt" in r.get("mean_In-O_coordination", {})
                  for rd in run_dirs for r in _load_run(rd))
    hdr = (f"{'run':>10} {'frac':>5} {'coord gen/ref':>15} {'d_coord':>8} "
           f"{'bond gen/ref':>17} {'E gen-ref(meV)':>15} {'spread(A)':>10}")
    if has_nvt:
        # E_nvt is vs the QUENCHED ref (fair: inherent structure vs inherent structure);
        # falls back to the raw ref for older runs that predate ref_quenched.
        hdr += f" {'coord_nvt':>9} {'d_coord_nvt':>11} {'E_nvt-refq(meV)':>15} {'spread_nvt':>10}"
    print(hdr)
    print("-" * len(hdr))
    for rd in run_dirs:
        name = os.path.basename(rd)
        for r in _load_run(rd):
            c = r["mean_In-O_coordination"]
            b = r["mean_In-O_bond"]
            e = r.get("energy_per_atom")
            de = (e["gen_before"] - e["ref"]) * 1000 if e else float("nan")
            dcoord = c["gen"] - c["ref"]
            line = (f"{name:>10} {r['mask_frac']:>5} {c['gen']:>7.3f}/{c['ref']:<7.3f} "
                    f"{dcoord:>+8.3f} {b['gen']:>8.4f}/{b['ref']:<8.4f} "
                    f"{de:>+15.1f} {r['multiseed_spread_rmsd']:>10.3f}")
            if has_nvt and "gen_nvt" in c:
                cn = c["gen_nvt"]
                refq = e.get("ref_quenched", e["ref"]) if e else None
                enr = (e["gen_after_nvt"] - refq) * 1000 if (e and "gen_after_nvt" in e) else float("nan")
                spn = r.get("multiseed_spread_rmsd_nvt", float("nan"))
                line += f" {cn:>9.3f} {cn - c['ref']:>+11.3f} {enr:>+15.1f} {spn:>10.3f}"
            print(line)
    tail = ("\nbetter = d_coord nearer 0 and |E gen-ref| smaller, spread NOT collapsed "
            "(that would be relaxation bought with diversity).")
    if has_nvt:
        tail += ("\nNVT (step 4) works if d_coord_nvt is nearer 0 than d_coord, E_nvt-refq "
                 "(gen vs QUENCHED ref) is small, and spread_nvt has not collapsed vs spread.")
    print(tail)


if __name__ == "__main__":
    main()

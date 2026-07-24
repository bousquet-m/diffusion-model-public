"""Summarize a-C screen/tune runs into one sp3 comparison table.

Walks each subdirectory (one hyperparameter setting) of a screen/tune output root and prints,
per (run, mask_frac): sp3% gen vs ref (the target distribution mean 91.9), mean C-coordination
gen/ref, multi-seed spread, SOAP memorization, and — when present — MACE energy gen-ref (meV)
and the post-NVT sp3/energy. The winning setting is the one whose sp3 gen mean approaches ~92
with spread NOT collapsed to 0 (diversity kept) and memorization not saturating.

    python scripts/collect_ac.py outputs/screen_ac
    python scripts/collect_ac.py outputs/tune_ac
    python scripts/collect_ac.py outputs/nvt_ac
"""
from __future__ import annotations

import glob
import json
import os
import sys


def _load_run(run_dir: str) -> list[dict]:
    files = glob.glob(os.path.join(run_dir, "summary_*.json"))
    if not files:
        f = os.path.join(run_dir, "sweep.json")
        return json.load(open(f))["results"] if os.path.exists(f) else []
    return [json.load(open(p)) for p in sorted(
        files, key=lambda p: float(os.path.basename(p)[len("summary_"):-len(".json")]))]


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs/screen_ac"
    run_dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"no run subdirectories under {root}")
        return

    runs = {os.path.basename(d): _load_run(d) for d in run_dirs}
    has_nvt = any("gen_nvt_mean" in r.get("sp3_pct", {}) for rs in runs.values() for r in rs)
    has_e = any("energy_per_atom" in r for rs in runs.values() for r in rs)

    hdr = (f"{'run':>16} {'frac':>5} {'sp3 gen/ref':>17} {'d_sp3':>7} "
           f"{'CN gen/ref':>13} {'spread':>7} {'memNN':>7}")
    if has_e:
        hdr += f" {'dE(meV)':>9}"
    if has_nvt:
        hdr += f" {'sp3_nvt':>8} {'dE_nvt':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, results in runs.items():
        for r in results:
            s = r["sp3_pct"]
            c = r["mean_C_coordination"]
            dsp3 = s["gen_mean"] - s["ref_mean"]
            line = (f"{name:>16} {r['mask_frac']:>5} "
                    f"{s['gen_mean']:>6.1f}±{s['gen_std']:<3.1f}/{s['ref_mean']:<5.1f} "
                    f"{dsp3:>+7.1f} {c['gen']:>5.2f}/{c['ref']:<5.2f} "
                    f"{r['multiseed_spread_rmsd']:>7.2f} "
                    f"{r['memorization_nn_similarity']['mean']:>7.4f}")
            if has_e:
                e = r.get("energy_per_atom")
                de = (e["gen_before"] - e["ref"]) * 1000 if e else float("nan")
                line += f" {de:>+9.0f}"
            if has_nvt:
                snvt = s.get("gen_nvt_mean", float("nan"))
                e = r.get("energy_per_atom", {})
                denvt = (e["gen_after_nvt"] - e["ref"]) * 1000 if "gen_after_nvt" in e else float("nan")
                line += f" {snvt:>8.1f} {denvt:>+8.0f}"
            print(line)
    print("\ntarget: sp3 gen -> ~92 (ref mean), spread NOT ~0 (keep diversity), memNN not saturated.")
    if has_nvt:
        print("NVT works if sp3_nvt rises toward ref AND dE_nvt < dE (energy down); too-high T restructures.")


if __name__ == "__main__":
    main()

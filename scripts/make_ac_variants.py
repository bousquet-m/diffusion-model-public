"""Generate the a-C hyperparameter-screen configs from configs/ac.yaml.

The ac1 run showed the model LEARNED (clean diagnose, generalizes to held-out) but the
sampler produces sp²-rich structures (sp³ ~74% vs ref ~92%; a 0 K quench drops energy
~270 meV but does NOT recover sp³ → genuine sp²-rich minima, a topology problem). Before
concluding 10 structures is too few, screen the training hyperparameters most likely to let
the anneal reach the tetrahedral network:
  - sigma_max: 0.6 Å moves barely reorganize a C network; larger max noise permits the
    bond-topology rearrangement sp²→sp³ needs. (ac_s09, ac_s12)
  - capacity: the sp³ network is highly structured; 74k params may under-sharpen the score
    (unlike In2O3's soft ionic coordination). (ac_cap)
  - both together (ac_cap_s09).

Each variant differs from the baseline in exactly the stated knobs (generated here so they
cannot drift from ac.yaml), and gets its own checkpoint/log dir. Regenerate with:

    python scripts/make_ac_variants.py
"""
from __future__ import annotations

from pathlib import Path

BASE = Path("configs/ac.yaml")
BIG_IRREPS = "128x0e + 64x1o + 16x2e"     # ~2x width per irrep vs the 64/32/8 baseline

# name -> list of (old_substring, new_substring) edits applied to the baseline text.
VARIANTS = {
    "ac_s09": [("sigma_max: 0.60", "sigma_max: 0.90")],
    "ac_s12": [("sigma_max: 0.60", "sigma_max: 1.20")],
    "ac_cap": [('hidden_irreps: "64x0e + 32x1o + 8x2e"', f'hidden_irreps: "{BIG_IRREPS}"')],
    "ac_cap_s09": [("sigma_max: 0.60", "sigma_max: 0.90"),
                   ('hidden_irreps: "64x0e + 32x1o + 8x2e"', f'hidden_irreps: "{BIG_IRREPS}"')],
}


def main() -> None:
    base = BASE.read_text()
    for name, edits in VARIANTS.items():
        text = base
        for old, new in edits:
            if old not in text:
                raise SystemExit(f"[{name}] pattern not found in ac.yaml: {old!r}")
            text = text.replace(old, new)
        # Redirect checkpoint/log dirs to this variant (anchored to line end).
        text = text.replace("checkpoint_dir: checkpoints/ac\n", f"checkpoint_dir: checkpoints/{name}\n")
        text = text.replace("log_dir: runs/ac\n", f"log_dir: runs/{name}\n")
        header = (f"# {name}: a-C hyperparameter-screen variant, generated from ac.yaml by\n"
                  f"# scripts/make_ac_variants.py (edits: {edits}). Do not hand-edit; regenerate.\n")
        out = Path(f"configs/{name}.yaml")
        out.write_text(header + text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Comparison plots: regenerated vs reference structural statistics."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rdfs(ref: dict, gen: dict, out_path: str):
    """ref/gen: {pair_label: (r, g)}. One panel per partial."""
    pairs = list(ref.keys())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        r, g_ref = ref[pair]
        _, g_gen = gen[pair]
        ax.plot(r, g_ref, label="reference", lw=2, color="k")
        ax.plot(r, g_gen, label="generated", lw=2, color="tab:red", ls="--")
        ax.set_title(f"{pair} RDF")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cn(ref_cn, gen_cn, out_path: str, max_cn: int = 10):
    """ref_cn/gen_cn: (values, hist) from coordination.cn_histogram."""
    fig, ax = plt.subplots(figsize=(5, 4))
    x_ref, h_ref = ref_cn
    x_gen, h_gen = gen_cn
    w = 0.4
    ax.bar(x_ref - w / 2, h_ref, width=w, label="reference", color="k", alpha=0.7)
    ax.bar(x_gen + w / 2, h_gen, width=w, label="generated", color="tab:red", alpha=0.7)
    ax.set_xlabel("In-O coordination number")
    ax.set_ylabel("probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_energy_hist(energies: dict, out_path: str, per_atom: bool = True):
    """energies: {label: array of energies}. Overlaid histograms."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"reference": "k", "generated": "tab:red", "generated_relaxed": "tab:blue"}
    for label, vals in energies.items():
        vals = np.asarray(vals)
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=min(20, max(3, len(vals))), alpha=0.5,
                label=label, color=colors.get(label))
    ax.set_xlabel("energy per atom (eV)" if per_atom else "energy (eV)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

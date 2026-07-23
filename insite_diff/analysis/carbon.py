"""Amorphous-carbon hybridization metric: sp / sp2 / sp3 fractions.

This is THE validation metric for a-C (the analog of In-O coordination for In2O3).
It wraps the logic of the reference ``coordination_numbers.py`` shipped with the data:

  - Count each carbon's C-C neighbors within ``cutoff`` (rcut = 1.95 A, the first RDF
    minimum for a-C). This coordination cutoff is a physical property of the metric and
    is SEPARATE from the model's graph/message-passing cutoff.
  - Classify by coordination number: cn == 2 -> sp, cn == 3 -> sp2, cn == 4 -> sp3.
  - Report each hybridization as a PERCENT of the considered atoms (the reference script
    divides by the fixed 216; here we divide by the number of atoms actually considered —
    all atoms, or the mobile subset — so the same code works for inpainted generation).

The generated distribution is judged against the 10-structure reference distribution:
its MEAN and SPREAD of sp3%, not a single number (the 10 AIMD snapshots span sp3%
84-95, so a generator that only hits the mean has not reproduced the distribution).

Neighbor counting uses ASE's PBC-aware ``neighbor_list`` (same as ``coordination.py``);
it agrees with the reference's orthorhombic minimum-image loop on this data.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

# Coordination-number -> hybridization label. sp3 (4-coordinated, tetrahedral) is the
# target that ta-C is defined by; sp2 (3) is graphitic; sp (2) is chain-like.
HYB_BY_CN = {2: "sp", 3: "sp2", 4: "sp3"}
CN_CC_CUTOFF = 1.95   # A — the a-C coordination cutoff from the reference metric


def coordination_numbers(atoms: Atoms, cutoff: float = CN_CC_CUTOFF) -> np.ndarray:
    """Per-atom count of neighbors within ``cutoff`` (single-species C, PBC-aware)."""
    i = neighbor_list("i", atoms, cutoff)
    return np.bincount(i, minlength=len(atoms))


def frame_fractions(atoms: Atoms, cutoff: float = CN_CC_CUTOFF,
                    mask: np.ndarray | None = None) -> dict:
    """sp/sp2/sp3 percentages for ONE structure.

    ``mask`` (if given and ``mobile_only`` upstream) restricts the *centers* whose
    hybridization is counted to the mobile atoms; each such atom's coordination still
    counts all C neighbors. Denominator is the number of considered centers.
    """
    cn = coordination_numbers(atoms, cutoff)
    centers = np.ones(len(atoms), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n = int(centers.sum())
    if n == 0:
        return {"sp": float("nan"), "sp2": float("nan"), "sp3": float("nan"),
                "mean_cn": float("nan"), "n_centers": 0}
    cn_c = cn[centers]
    return {
        "sp": 100.0 * np.mean(cn_c == 2),
        "sp2": 100.0 * np.mean(cn_c == 3),
        "sp3": 100.0 * np.mean(cn_c == 4),
        "mean_cn": float(cn_c.mean()),
        "n_centers": n,
    }


def summary(frames: list[Atoms], cutoff: float = CN_CC_CUTOFF,
            masks: list[np.ndarray] | None = None,
            center_mobile_only: bool = False) -> dict:
    """Per-frame sp/sp2/sp3 distribution over ``frames``.

    Returns the per-frame arrays plus their mean and spread (std). The distribution —
    mean AND spread of sp3% — is the a-C success criterion, so both are surfaced. Signature
    mirrors ``coordination.summary`` (masks + ``center_mobile_only``) so sweep.py can call
    it the same way it calls the In-O metric.
    """
    if masks is None:
        masks = [None] * len(frames)
    sp, sp2, sp3, mean_cn = [], [], [], []
    for atoms, mask in zip(frames, masks):
        use_mask = mask if (center_mobile_only and mask is not None) else None
        f = frame_fractions(atoms, cutoff, use_mask)
        sp.append(f["sp"]); sp2.append(f["sp2"]); sp3.append(f["sp3"]); mean_cn.append(f["mean_cn"])
    sp, sp2, sp3, mean_cn = map(np.asarray, (sp, sp2, sp3, mean_cn))
    return {
        "sp3_per_frame": sp3, "sp2_per_frame": sp2, "sp_per_frame": sp,
        "sp3_mean": float(np.nanmean(sp3)) if len(sp3) else float("nan"),
        "sp3_std": float(np.nanstd(sp3)) if len(sp3) else float("nan"),
        "sp2_mean": float(np.nanmean(sp2)) if len(sp2) else float("nan"),
        "sp2_std": float(np.nanstd(sp2)) if len(sp2) else float("nan"),
        "mean_cn": float(np.nanmean(mean_cn)) if len(mean_cn) else float("nan"),
    }

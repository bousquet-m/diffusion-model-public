"""Diversity (multi-seed spread) and memorization (SOAP NN-similarity) metrics.

- ``permutation_rmsd``: RMSD over mobile atoms with an optimal assignment WITHIN
  each species (same-species atoms are interchangeable, so a raw index-wise RMSD
  would overstate the difference). Used to measure spread across seeds: zero
  spread means the mobile atoms were never really noised (a bug); small nonzero
  spread at low mask_frac is expected.
- ``memorization_similarity``: average-SOAP descriptor per structure, then the
  cosine similarity of each generated structure to its nearest training structure.
  High values flag possible memorization (our dataset is thin).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from ..geometry import min_image_displacement


def permutation_rmsd(pos_a: np.ndarray, pos_b: np.ndarray, numbers: np.ndarray,
                     cell: np.ndarray, mobile_mask: np.ndarray) -> float:
    """Min-image RMSD over mobile atoms, optimally matched within each species."""
    total_sq, count = 0.0, 0
    for z in np.unique(numbers[mobile_mask]):
        sel = mobile_mask & (numbers == z)
        A, B = pos_a[sel], pos_b[sel]
        if len(A) == 0:
            continue
        delta = A[:, None, :] - B[None, :, :]                      # (n,n,3)
        disp = min_image_displacement(delta.reshape(-1, 3), cell).reshape(len(A), len(B), 3)
        d2 = (disp ** 2).sum(-1)                                   # (n,n) squared distances
        r, c = linear_sum_assignment(d2)                          # Hungarian within species
        total_sq += float(d2[r, c].sum())
        count += len(A)
    return float(np.sqrt(total_sq / max(count, 1)))


def multi_seed_spread(samples: list[np.ndarray], numbers: np.ndarray, cell: np.ndarray,
                      mobile_mask: np.ndarray) -> dict:
    """Mean/std of pairwise permutation-RMSD across generated samples of one masked structure."""
    rmsds = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            rmsds.append(permutation_rmsd(samples[i], samples[j], numbers, cell, mobile_mask))
    rmsds = np.array(rmsds) if rmsds else np.array([0.0])
    return {"mean_rmsd": float(rmsds.mean()), "std_rmsd": float(rmsds.std()),
            "min_rmsd": float(rmsds.min()), "max_rmsd": float(rmsds.max())}


def _soap(species: list[str], r_cut: float = 5.0, n_max: int = 6, l_max: int = 4):
    from dscribe.descriptors import SOAP
    return SOAP(species=species, r_cut=r_cut, n_max=n_max, l_max=l_max,
                periodic=True, average="off")          # per-atom environments


def atom_environments(frames: list[Atoms], masks: list[np.ndarray] | None = None,
                      species=("In", "O"), max_envs: int | None = None,
                      seed: int = 0, **kw) -> np.ndarray:
    """Stacked L2-normalized per-atom SOAP environments (optionally only masked atoms).

    Memorization must be judged on the GENERATED atoms only: a whole-structure
    descriptor of an inpainted cell is dominated by the frozen context (which is
    real training data), so it would look ~identical to training regardless of the
    generated region. Restricting to mobile-atom environments fixes that.
    """
    soap = _soap(list(species), **kw)
    rows = []
    for k, a in enumerate(frames):
        d = soap.create(a)
        if masks is not None:
            d = d[masks[k]]
        rows.append(d)
    env = np.vstack(rows) if rows else np.zeros((0, 1))
    env = env / np.maximum(np.linalg.norm(env, axis=1, keepdims=True), 1e-12)
    if max_envs and len(env) > max_envs:
        env = env[np.random.default_rng(seed).choice(len(env), max_envs, replace=False)]
    return env


def memorization_similarity(gen_frames: list[Atoms], gen_masks: list[np.ndarray],
                            train_bank: np.ndarray, species=("In", "O"), **kw) -> np.ndarray:
    """NN cosine similarity of each generated (mobile) atom environment to the training
    environment bank. High values flag possible memorization of local structure."""
    gen_env = atom_environments(gen_frames, gen_masks, species, **kw)
    if len(gen_env) == 0 or len(train_bank) == 0:
        return np.zeros(len(gen_env))
    return (gen_env @ train_bank.T).max(axis=1)        # per-mobile-atom NN similarity

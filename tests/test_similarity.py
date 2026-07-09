"""Permutation-aware RMSD must be invariant to relabeling same-species atoms."""
import numpy as np

from insite_diff.analysis.similarity import multi_seed_spread, permutation_rmsd


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 15.0
    numbers = np.array([49] * 6 + [8] * 6)          # 6 In, 6 O
    pos_a = rng.random((12, 3)) @ cell
    pos_b = rng.random((12, 3)) @ cell
    return pos_a, pos_b, numbers, cell, rng


def test_permutation_rmsd_invariant_to_same_species_relabeling():
    pos_a, pos_b, numbers, cell, rng = _setup()
    mobile = np.ones(12, dtype=bool)
    r0 = permutation_rmsd(pos_a, pos_b, numbers, cell, mobile)
    # permute In among In and O among O in structure b (species layout preserved)
    perm = np.concatenate([rng.permutation(6), 6 + rng.permutation(6)])
    r1 = permutation_rmsd(pos_a, pos_b[perm], numbers[perm], cell, mobile)
    assert abs(r0 - r1) < 1e-9


def test_permutation_rmsd_zero_for_identical():
    pos_a, _, numbers, cell, _ = _setup()
    mobile = np.ones(12, dtype=bool)
    assert permutation_rmsd(pos_a, pos_a.copy(), numbers, cell, mobile) < 1e-9


def test_permutation_rmsd_no_worse_than_identity():
    # optimal assignment can only reduce cost vs the index-wise pairing
    pos_a, pos_b, numbers, cell, _ = _setup()
    mobile = np.ones(12, dtype=bool)
    from insite_diff.geometry import rmsd_same_atoms
    assert permutation_rmsd(pos_a, pos_b, numbers, cell, mobile) <= \
        rmsd_same_atoms(pos_a, pos_b, cell) + 1e-9


def test_multiseed_spread_zero_for_identical_samples():
    pos_a, _, numbers, cell, _ = _setup()
    mobile = np.ones(12, dtype=bool)
    s = multi_seed_spread([pos_a, pos_a.copy(), pos_a.copy()], numbers, cell, mobile)
    assert s["mean_rmsd"] < 1e-9

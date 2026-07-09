"""Sanity checks for RDF and coordination on synthetic structures."""
import numpy as np
from ase import Atoms

from insite_diff.analysis.coordination import coordination, summary
from insite_diff.analysis.rdf import first_peak, partial_rdf


def _dimer_lattice(d=2.15, box=12.0):
    # One In and one O separated by d, in a large box (isolated pair within rmax).
    return Atoms("InO", positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0 + d]],
                 cell=np.eye(3) * box, pbc=True)


def test_rdf_peak_at_known_distance():
    frames = [_dimer_lattice(d=2.15)]
    r, g = partial_rdf(frames, 49, 8, rmax=6.0, nbins=240)
    assert abs(first_peak(r, g) - 2.15) < 0.05


def test_coordination_counts_neighbors():
    frames = [_dimer_lattice(d=2.15)]
    cns, bonds = coordination(frames, 49, 8, cutoff=2.7)
    assert cns.tolist() == [1]          # the single In has exactly one O within cutoff
    assert abs(bonds[0] - 2.15) < 1e-6


def test_summary_mean_bond():
    frames = [_dimer_lattice(d=2.2)]
    s = summary(frames, 49, 8, cutoff=2.7)
    assert abs(s["mean_bond"] - 2.2) < 1e-6
    assert s["mean_cn"] == 1.0

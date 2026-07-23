"""Sanity checks for RDF and coordination on synthetic structures."""
import numpy as np
from ase import Atoms

from insite_diff.analysis import carbon
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


def test_coordination_center_mobile_only_restricts_centers():
    # Two In and two O; only the first In is mobile -> only its CN is counted.
    atoms = Atoms("In2O2", positions=[[1, 1, 1], [8, 8, 8], [1, 1, 3.15], [8, 8, 10.15]],
                  cell=np.eye(3) * 14.0, pbc=True)
    mask = np.array([True, False, False, False])   # In #0 mobile
    cns, bonds = coordination([atoms], 49, 8, cutoff=2.7, masks=[mask],
                              center_mobile_only=True)
    assert cns.tolist() == [1]           # only mobile In #0's coordination
    assert len(bonds) == 1


# --- amorphous-carbon sp2/sp3 metric ---------------------------------------- #
def _tetrahedron(bond=1.5, box=20.0):
    """One central C sp3-coordinated to 4 others (isolated in a large box)."""
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    dirs *= bond / np.sqrt(3)
    pos = np.vstack([[0, 0, 0], dirs]) + box / 2
    return Atoms(numbers=[6] * 5, positions=pos, cell=np.eye(3) * box, pbc=True)


def test_carbon_coordination_counts_cc_neighbors():
    cn = carbon.coordination_numbers(_tetrahedron(), cutoff=1.95)
    assert cn.tolist() == [4, 1, 1, 1, 1]     # center is 4-coordinated, arms are 1


def test_carbon_fractions_center_sp3():
    # Over all 5 atoms only the center is sp3 (4-coord) -> 20%.
    f = carbon.frame_fractions(_tetrahedron(), cutoff=1.95)
    assert abs(f["sp3"] - 20.0) < 1e-9
    assert abs(f["sp2"] - 0.0) < 1e-9


def test_carbon_fractions_mobile_only_restricts_centers():
    # Restrict centers to the mobile central atom -> it is the only considered atom, 100% sp3.
    mask = np.array([True, False, False, False, False])
    f = carbon.frame_fractions(_tetrahedron(), cutoff=1.95, mask=mask)
    assert abs(f["sp3"] - 100.0) < 1e-9
    assert f["n_centers"] == 1


def test_carbon_summary_reports_mean_and_spread():
    # Two frames with different arm distances: one all sp3-center, one where arms are pushed
    # out of the cutoff so the center is 0-coordinated. sp3% per frame = [20, 0].
    s = carbon.summary([_tetrahedron(bond=1.5), _tetrahedron(bond=2.5)], cutoff=1.95)
    assert np.allclose(sorted(s["sp3_per_frame"]), [0.0, 20.0])
    assert abs(s["sp3_mean"] - 10.0) < 1e-9
    assert abs(s["sp3_std"] - 10.0) < 1e-9

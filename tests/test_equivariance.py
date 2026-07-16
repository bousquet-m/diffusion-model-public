"""The denoiser must be E(3)-equivariant: rotating the input rotates the predicted
score, translations leave it unchanged (relative-vector inputs), and the output is
CoM-free."""
import numpy as np
import pytest
import torch

from insite_diff.data.graph import build_graph
from insite_diff.model.denoiser import E3Denoiser
from insite_diff import e3nn_compat  # noqa: F401
from e3nn import o3


TP_MODES = ["fc", "uvu"]


def _model(tp_mode="fc"):
    torch.manual_seed(0)
    m = E3Denoiser(n_species=2, hidden_irreps="16x0e + 8x1o", sh_lmax=2, n_layers=2,
                   radial_basis=6, sigma_embed_dim=8, cutoff=4.0, avg_neighbors=20.0,
                   tp_mode=tp_mode)
    return m.eval()


def _graph(pos, box=10.0, cutoff=4.0):
    cell = np.eye(3) * box
    g = build_graph(pos.numpy(), cell, cutoff, numbers=None, max_neighbors=0)
    return g


@pytest.mark.parametrize("tp_mode", TP_MODES)
def test_output_shape_and_com_free(tp_mode):
    m = _model(tp_mode)
    torch.manual_seed(1)
    n = 40
    pos = torch.rand(n, 3) * 10
    g = _graph(pos)
    types = torch.randint(0, 2, (n,))
    eps = m(types, g.edge_index, g.edge_vec, torch.tensor(0.3), n)
    assert eps.shape == (n, 3)
    assert torch.allclose(eps.mean(0), torch.zeros(3), atol=1e-5)


@pytest.mark.parametrize("tp_mode", TP_MODES)
def test_rotation_equivariance(tp_mode):
    m = _model(tp_mode)
    torch.manual_seed(2)
    n = 48
    pos = torch.rand(n, 3) * 10
    g = _graph(pos)
    types = torch.randint(0, 2, (n,))
    t = torch.tensor(0.4)

    eps = m(types, g.edge_index, g.edge_vec, t, n)
    R = o3.rand_matrix().to(g.edge_vec.dtype)          # random proper rotation
    eps_rot = m(types, g.edge_index, g.edge_vec @ R.T, t, n)

    assert torch.allclose(eps_rot, eps @ R.T, atol=1e-4), \
        f"max err {(eps_rot - eps @ R.T).abs().max().item():.2e}"


@pytest.mark.parametrize("tp_mode", TP_MODES)
def test_inversion_equivariance(tp_mode):
    # Parity: inverting coordinates should invert the (odd) vector output.
    m = _model(tp_mode)
    torch.manual_seed(3)
    n = 32
    pos = torch.rand(n, 3) * 10
    g = _graph(pos)
    types = torch.randint(0, 2, (n,))
    t = torch.tensor(0.5)
    eps = m(types, g.edge_index, g.edge_vec, t, n)
    eps_inv = m(types, g.edge_index, -g.edge_vec, t, n)
    assert torch.allclose(eps_inv, -eps, atol=1e-4)

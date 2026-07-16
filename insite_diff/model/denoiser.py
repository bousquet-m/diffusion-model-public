"""E(3)-equivariant, noise-conditioned denoiser (e3nn).

**Unconditional.** Every atom is noised at training time and the network predicts
the noise on every atom; there is no context/conditioning channel. Fixed-context
inpainting is done entirely at sampling time via RePaint masking (see
``sampling/inpaint.py``), so one trained model serves any mask geometry.

A compact NequIP-style message-passing network. Nodes carry equivariant features
(scalars + vectors); messages are tensor products of neighbor features with the
spherical harmonics of the edge direction, weighted by a radial network that is
also conditioned on the noise level. The readout emits one 1o (vector) per atom —
the predicted CoM-free noise ``eps``.

The network consumes a prebuilt graph (edge_index + Cartesian edge vectors), so it
is agnostic to how neighbors were found and trivially testable for equivariance:
rotating the edge vectors rotates the output.
"""
from __future__ import annotations

import torch
from torch import nn

from .. import e3nn_compat  # noqa: F401  (must precede e3nn import)
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.o3 import (FullyConnectedTensorProduct, Irreps, Linear, SphericalHarmonics,
                     TensorProduct)

from .embedding import NoiseEmbedding


def _build_gate(irreps_out: Irreps) -> tuple[Gate, Irreps]:
    """Gate nonlinearity for ``irreps_out``: SiLU on scalars, sigmoid gates on vectors.

    Returns (gate, tp_out_irreps) where tp_out_irreps is what the tensor product
    must produce (scalars + gate-scalars + gated) and gate.irreps_out == irreps_out.
    """
    scalars = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    gated = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    n_gates = sum(mul for mul, _ in gated)
    gates = Irreps(f"{n_gates}x0e") if n_gates else Irreps("")
    gate = Gate(
        scalars, [torch.nn.functional.silu] * (1 if len(scalars) else 0),
        gates, [torch.sigmoid] * (1 if len(gates) else 0),
        gated,
    )
    return gate, gate.irreps_in


def _uvu_tensor_product(irreps_in: Irreps, irreps_sh: Irreps,
                        irreps_target: Irreps) -> tuple[TensorProduct, Irreps]:
    """NequIP-style message tensor product: one ``uvu`` path per (input, sh) -> output
    coupling that the target needs.

    Under ``uvu`` the per-edge weights are indexed by the input multiplicity alone
    (``mul`` weights per path) rather than the full ``mul_in x mul_sh x mul_out`` block
    a FullyConnectedTensorProduct requires — ~46x fewer weights for the gen5 irreps.
    The output carries one entry per path (duplicated/unsorted irreps), so callers mix
    it into the target with a Linear.
    """
    target_irs = {ir for _, ir in irreps_target}
    irreps_mid, instructions = [], []
    for i, (mul, ir_in) in enumerate(irreps_in):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                if ir_out in target_irs:
                    instructions.append((i, j, len(irreps_mid), "uvu", True))
                    irreps_mid.append((mul, ir_out))
    irreps_mid = Irreps(irreps_mid)
    tp = TensorProduct(irreps_in, irreps_sh, irreps_mid, instructions,
                       shared_weights=False, internal_weights=False)
    return tp, irreps_mid


class Interaction(nn.Module):
    """One equivariant message-passing layer with a noise-conditioned radial net.

    ``tp_mode`` selects the message parameterization; see ``ModelConfig.tp_mode``.
    Both modes are equivariant and produce the same irreps — they differ only in how
    many weights the radial net must emit per edge (and so in speed/param count).
    """

    def __init__(self, irreps_in: Irreps, irreps_sh: Irreps, irreps_out: Irreps,
                 radial_in: int, radial_hidden: int, avg_neighbors: float,
                 tp_mode: str = "fc"):
        super().__init__()
        self.gate, tp_out = _build_gate(irreps_out)
        if tp_mode == "uvu":
            self.tp, irreps_mid = _uvu_tensor_product(irreps_in, irreps_sh, tp_out)
            self.mix = Linear(irreps_mid, tp_out)   # mix duplicated paths into the gate input
        else:
            self.tp = FullyConnectedTensorProduct(
                irreps_in, irreps_sh, tp_out, shared_weights=False, internal_weights=False
            )
            self.mix = None
        self.radial = nn.Sequential(
            nn.Linear(radial_in, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, self.tp.weight_numel),
        )
        self.skip = Linear(irreps_in, irreps_out)
        self.avg_neighbors = avg_neighbors

    def forward(self, node, edge_index, edge_sh, edge_scalars):
        src, dst = edge_index[0], edge_index[1]
        w = self.radial(edge_scalars)                      # (E, weight_numel)
        msg = self.tp(node[src], edge_sh, w)               # (E, tp_out | irreps_mid)
        agg = node.new_zeros(node.shape[0], msg.shape[1])
        agg.index_add_(0, dst, msg)
        agg = agg / (self.avg_neighbors ** 0.5)
        if self.mix is not None:
            agg = self.mix(agg)                            # (N, tp_out)
        return self.gate(agg) + self.skip(node)


class E3Denoiser(nn.Module):
    def __init__(self, n_species: int, hidden_irreps: str = "32x0e + 16x1o",
                 sh_lmax: int = 2, n_layers: int = 2, radial_basis: int = 8,
                 sigma_embed_dim: int = 32, cutoff: float = 4.0,
                 avg_neighbors: float = 20.0, tp_mode: str = "fc"):
        super().__init__()
        self.cutoff = cutoff
        self.radial_basis = radial_basis
        hidden = Irreps(hidden_irreps)
        self.irreps_sh = Irreps.spherical_harmonics(sh_lmax)
        self.sh = SphericalHarmonics(self.irreps_sh, normalize=True, normalization="component")

        self.type_embed = nn.Embedding(n_species, sigma_embed_dim)
        self.noise_embed = NoiseEmbedding(sigma_embed_dim)
        # Initial node features: [type ++ noise] scalars -> hidden (fills 0e, 1o start at 0).
        self.embed_lin = Linear(Irreps(f"{2 * sigma_embed_dim}x0e"), hidden)

        radial_in = radial_basis + sigma_embed_dim
        self.layers = nn.ModuleList([
            Interaction(hidden, self.irreps_sh, hidden, radial_in,
                        radial_hidden=max(16, radial_basis * 4), avg_neighbors=avg_neighbors,
                        tp_mode=tp_mode)
            for _ in range(n_layers)
        ])
        self.readout = Linear(hidden, Irreps("1x1o"))

    def forward(self, types: torch.Tensor, edge_index: torch.Tensor,
                edge_vec: torch.Tensor, t_norm: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """Predict CoM-free eps (n_atoms, 3)."""
        edge_len = edge_vec.norm(dim=1)
        edge_sh = self.sh(edge_vec)
        radial = soft_one_hot_linspace(
            edge_len, 0.0, self.cutoff, self.radial_basis, basis="smooth_finite", cutoff=True
        )

        noise = self.noise_embed(t_norm.reshape(()))            # (sigma_embed_dim,)
        edge_scalars = torch.cat(
            [radial, noise.unsqueeze(0).expand(radial.shape[0], -1)], dim=1
        )

        type_feat = self.type_embed(types)                      # (N, sigma_embed_dim)
        node_scalars = torch.cat(
            [type_feat, noise.unsqueeze(0).expand(n_atoms, -1)], dim=1
        )
        node = self.embed_lin(node_scalars)                     # (N, hidden)
        for layer in self.layers:
            node = layer(node, edge_index, edge_sh, edge_scalars)
        eps = self.readout(node)                                # (N, 3)
        return eps - eps.mean(dim=0, keepdim=True)              # enforce CoM-free

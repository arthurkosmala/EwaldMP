"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the DimeNet implementation as part of
pytorch-geometric: https://github.com/rusty1s/pytorch_geometric. License:

---

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math

import torch
from torch import nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    EmbeddingBlock,
    Envelope,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_geometric.nn.resolver import activation_resolver
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_k_index_product_set,
    get_k_voxel_grid,
    get_pbc_distances,
    pos_svd_frame,
    radius_graph_pbc,
    x_to_k_cell,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.models.ewald_block import EwaldBlock

from .gemnet.layers.base_layers import Dense
from .gemnet.layers.embedding_block import AtomEmbedding
from .gemnet.layers.radial_basis import RadialBasis

try:
    import sympy as sym
except ImportError:
    sym = None


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act="silu",
    ):
        act = activation_resolver(act)
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act="silu",
        use_ewald=False,
    ):
        act = activation_resolver(act)
        super(OutputPPBlock, self).__init__()
        self.act = act
        self.use_ewald = use_ewald

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        if self.use_ewald:
            # Combine Ewald-updated atom encodings with the remaining hidden state
            # of the model by the concatenating Ewald and short-range embeddings
            # before the linear upprojection layer of each output block
            self.lin_up = nn.Linear(
                2 * hidden_channels, out_emb_channels, bias=True
            )
        else:
            self.lin_up = nn.Linear(
                hidden_channels, out_emb_channels, bias=True
            )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None, h=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)

        # Combine Ewald part of the model with output
        if self.use_ewald:
            x = torch.cat([x, h], dim=-1)

        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNetPlusPlus(torch.nn.Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`silu`)
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_blocks,
        int_emb_size,
        basis_emb_size,
        out_emb_channels,
        num_spherical,
        num_radial,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act="silu",
        use_ewald=False,
    ):

        act = activation_resolver(act)

        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    use_ewald,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """ """
        raise NotImplementedError


@registry.register_model("dimenetplusplus")
class DimeNetPlusPlusWrap(DimeNetPlusPlus, BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        max_neighbors=50,
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_ewald = ewald_hyperparams is not None
        # whether to use an atom-to-atom message-passing block with
        # position-space cutoff
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None

        # Parse Ewald hyperparams
        if self.use_ewald:
            if self.use_pbc:
                # Integer values to define box of k-lattice indices
                self.num_k_x = ewald_hyperparams["num_k_x"]
                self.num_k_y = ewald_hyperparams["num_k_y"]
                self.num_k_z = ewald_hyperparams["num_k_z"]
                self.delta_k = None
            else:
                self.k_cutoff = ewald_hyperparams["k_cutoff"]
                # Voxel grid resolution
                self.delta_k = ewald_hyperparams["delta_k"]
                # Radial k-filter basis size
                self.num_k_rbf = ewald_hyperparams["num_k_rbf"]
            self.downprojection_size = ewald_hyperparams["downprojection_size"]
            # Number of residuals in update function
            self.num_hidden = ewald_hyperparams["num_hidden"]
            # Whether to detach Ewald blocks from the computational graph
            # (recommended for DimeNet++, as the Ewald embeddings do not
            # mix with the remaining hidden state of the model prior to output
            # and the Ewald blocks barely contribute to the force predictions)
            self.detach_ewald = ewald_hyperparams.get("detach_ewald", True)

        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            use_ewald=self.use_ewald,
        )

        if self.use_ewald:
            # Initialize k-space structure
            if self.use_pbc:
                # Get the reciprocal lattice indices of included k-vectors
                (
                    self.k_index_product_set,
                    self.num_k_degrees_of_freedom,
                ) = get_k_index_product_set(
                    self.num_k_x,
                    self.num_k_y,
                    self.num_k_z,
                )
                self.k_rbf_values = None
                self.delta_k = None

            else:
                # Get the k-space voxel and evaluate Gaussian RBF (can be done at
                # initialization time as voxel grid stays fixed for all structures)
                (
                    self.k_grid,
                    self.k_rbf_values,
                    self.num_k_degrees_of_freedom,
                ) = get_k_voxel_grid(
                    self.k_cutoff,
                    self.delta_k,
                    self.num_k_rbf,
                )

            # Initialize atom embedding block
            self.atom_emb = AtomEmbedding(hidden_channels, num_elements=83)

            # Downprojection layer, weights are shared among all interaction blocks
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                self.downprojection_size,
                activation=None,
                bias=False,
            )

            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        hidden_channels,  # Embedding size of short-range GNN
                        self.downprojection_size,
                        self.num_hidden,  # Number of residuals in update function
                        activation="silu",
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(self.num_blocks)
                ]
            )

        if self.use_atom_to_atom_mp:
            self.atom_emb = AtomEmbedding(hidden_channels, num_elements=83)
            if self.use_pbc:
                # Compute neighbor threshold from cutoff assuming uniform atom density
                self.max_neighbors_at = int(
                    (self.atom_to_atom_cutoff / 6.0) ** 3 * 50
                )
            else:
                self.max_neighbors_at = 100
            # SchNet interactions for atom-to-atom message passing
            self.interactions_at = torch.nn.ModuleList(
                [
                    InteractionBlock(
                        hidden_channels,
                        200,  # num Gaussians
                        256,  # num filters
                        self.atom_to_atom_cutoff,
                    )
                    for i in range(self.num_blocks)
                ]
            )
            self.distance_expansion_at = GaussianSmearing(
                0.0, self.atom_to_atom_cutoff, 200
            )

        self.skip_connection_factor = (
            1.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos = (
            pos_svd_frame(data)
            if (self.use_ewald and not self.use_pbc)
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1

        (
            edge_index,
            dist,
            _,
            cell_offsets,
            offsets,
            neighbors,
        ) = self.generate_graph(data)

        if self.use_atom_to_atom_mp:
            # Use separate graph (larger cutoff) for atom-to-atom long-range block
            (
                edge_index_at,
                edge_weight_at,
                distance_vec_at,
                cell_offsets_at,
                _,  # cell offset distances
                neighbors_at,
            ) = self.generate_graph(
                data,
                cutoff=self.atom_to_atom_cutoff,
                max_neighbors=self.max_neighbors_at,
            )

            edge_attr_at = self.distance_expansion_at(edge_weight_at)

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.neighbors = neighbors
        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data.cell_offsets,
            num_nodes=data.atomic_numbers.size(0),
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        if self.use_ewald:
            if self.use_pbc:
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(data.cell)
                # Translate lattice indices to k-vectors
                k_grid = torch.matmul(
                    self.k_index_product_set.to(batch.device), k_cell
                )
            else:
                k_grid = (
                    self.k_grid.to(batch.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )

        # Embedding block.
        x = self.emb(data.atomic_numbers.long(), rbf, i, j)
        if self.use_ewald:
            # If Ewald MP is used, we have to create atom embeddings borrowing
            # the atomic embedding block from the GemNet architecture
            h = self.atom_emb(data.atomic_numbers.long())
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            pos_detach = pos.detach() if self.detach_ewald else pos
        elif self.use_atom_to_atom_mp:
            h = self.atom_emb(data.atomic_numbers.long())
        else:
            h = None

        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0), h=h)

        # Interaction blocks.
        if self.use_ewald or self.use_atom_to_atom_mp:
            for block_ind in range(self.num_blocks):
                x = self.interaction_blocks[block_ind](
                    x, rbf, sbf, idx_kj, idx_ji
                )

                if self.use_ewald:
                    h_ewald, dot, sinc_damping = self.ewald_blocks[block_ind](
                        h,
                        pos_detach,
                        k_grid,
                        batch_size,
                        batch,
                        dot,
                        sinc_damping,
                    )
                else:
                    h_ewald = 0

                if self.use_atom_to_atom_mp:
                    h_at = self.interactions_at[block_ind](
                        h, edge_index_at, edge_weight_at, edge_attr_at
                    )
                else:
                    h_at = 0

                h = self.skip_connection_factor * (h + h_ewald + h_at)
                P += self.output_blocks[block_ind + 1](
                    x, rbf, i, num_nodes=pos.size(0), h=h
                )

        else:
            for interaction_block, output_block in zip(
                self.interaction_blocks, self.output_blocks[1:]
            ):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                P += output_block(x, rbf, i, num_nodes=pos.size(0))

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

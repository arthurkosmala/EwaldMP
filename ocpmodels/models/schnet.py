"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch
from torch_geometric.nn import SchNet
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_scatter import scatter

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
from ocpmodels.models.gemnet.layers.base_layers import Dense


@registry.register_model("schnet")
class SchNetWrap(SchNet, BaseModel):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_neighbors=50,
        readout="add",
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.reduce = readout
        self.use_ewald = ewald_hyperparams is not None
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
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

        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        # Initialize k-space structure
        if self.use_ewald:
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
                    for i in range(self.num_interactions)
                ]
            )

        if self.use_atom_to_atom_mp:
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
                    for i in range(self.num_interactions)
                ]
            )
            self.distance_expansion_at = GaussianSmearing(
                0.0, self.atom_to_atom_cutoff, 200
            )

        self.skip_connection_factor = (
            2.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = (
            pos_svd_frame(data)
            if (self.use_ewald and not self.use_pbc)
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        # Use separate graph (larger cutoff) for atom-to-atom long-range block
        if self.use_atom_to_atom_mp:
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

        if self.use_pbc or self.use_ewald or self.use_atom_to_atom_mp:

            assert z.dim() == 1 and z.dtype == torch.long

            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)

            if self.use_ewald or self.use_atom_to_atom_mp:
                dot = None  # These will be computed in first Ewald block and then passed
                sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
                for i in range(self.num_interactions):
                    if self.use_ewald:
                        h_ewald, dot, sinc_damping = self.ewald_blocks[i](
                            h,
                            pos,
                            k_grid,
                            batch_size,
                            batch,
                            dot,
                            sinc_damping,
                        )
                    else:
                        h_ewald = 0

                    if self.use_atom_to_atom_mp:
                        h_at = self.interactions_at[i](
                            h, edge_index_at, edge_weight_at, edge_attr_at
                        )
                    else:
                        h_at = 0

                    h = self.skip_connection_factor * (
                        h
                        + self.interactions[i](
                            h, edge_index, edge_weight, edge_attr
                        )
                        + h_ewald
                        + h_at
                    )
            else:
                for interaction in self.interactions:
                    h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.reduce)
        else:
            energy = super(SchNetWrap, self).forward(z, pos, batch)
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

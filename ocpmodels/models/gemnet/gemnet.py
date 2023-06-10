"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_k_index_product_set,
    get_k_voxel_grid,
    get_pbc_distances,
    pos_svd_frame,
    radius_graph_pbc,
    x_to_k_cell,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.scaling.compat import load_scales_compat

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer
from .utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)


@registry.register_model("gemnet_t")
class GemNetT(BaseModel):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        regress_forces: bool = True,
        direct_forces: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        num_elements: int = 83,
        scale_file: Optional[str] = None,
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,
    ):
        super().__init__()
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.cutoff = cutoff
        assert self.cutoff <= 6 or otf_graph

        self.max_neighbors = max_neighbors
        assert self.max_neighbors == 50 or otf_graph

        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None
        if self.use_atom_to_atom_mp:
            if self.use_pbc:
                # Compute neighbor threshold from cutoff assuming uniform atom density
                self.max_neighbors_at = int(
                    (self.atom_to_atom_cutoff / 6.0) ** 3 * 50
                )
            else:
                self.max_neighbors_at = 100
            self.distance_expansion_at = GaussianSmearing(
                0.0, self.atom_to_atom_cutoff, 200
            )
        else:
            self.max_neighbors_at = None
            self.distance_expansion_at = None

        # GemNet variants
        self.direct_forces = direct_forces

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### -------------------------------- Ewald Message Passing ------------------------------ ###

        self.use_ewald = ewald_hyperparams is not None

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
        else:
            self.down = None
            self.downprojection_size = None
            self.delta_k = None
            self.k_rbf_values = None

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                    use_pbc=self.use_pbc,
                    use_ewald=self.use_ewald,
                    ewald_downprojection=self.down,
                    downprojection_size=self.downprojection_size,
                    delta_k=self.delta_k,
                    k_rbf_values=self.k_rbf_values,
                    atom_to_atom_cutoff=self.atom_to_atom_cutoff,
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=direct_forces,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_blocks),
            (self.mlp_cbf3.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]
        if self.use_ewald:
            self.shared_parameters += [
                (self.down.linear.weight, self.num_blocks)
            ]

        load_scales_compat(self, scale_file)

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(self, data):
        num_atoms = data.atomic_numbers.size(0)

        (
            edge_index,
            D_st,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -distance_vec / D_st[:, None]

        # Mask interaction edges if required
        if self.otf_graph or np.isclose(self.cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = self.cutoff
        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = (
            pos_svd_frame(data)
            if (self.use_ewald and not self.use_pbc)
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

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

        else:
            edge_index_at = None
            edge_weight_at = None
            distance_vec_at = None
            cell_offsets_at = None
            neighbors_at = None
            edge_attr_at = None

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

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
        else:
            k_grid = None

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        dot = (
            None  # These will be computed in first Ewald block and then passed
        )
        sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
        for i in range(self.num_blocks):
            # Interaction block
            h, m, dot, sinc_damping = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                pos=pos,
                k_grid=k_grid,
                batch_size=batch_size,
                batch=batch,
                dot=dot,
                sinc_damping=sinc_damping,
                edge_index_at=edge_index_at,
                edge_weight_at=edge_weight_at,
                edge_attr_at=edge_attr_at,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=data.atomic_numbers.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                F_t = F_t.squeeze(1)  # (nAtoms, 3)
            else:
                if self.num_targets > 1:
                    forces = []
                    for i in range(self.num_targets):
                        # maybe this can be solved differently
                        forces += [
                            -torch.autograd.grad(
                                E_t[:, i].sum(), pos, create_graph=True
                            )[0]
                        ]
                    F_t = torch.stack(forces, dim=1)
                    # (nAtoms, num_targets, 3)
                else:
                    F_t = -torch.autograd.grad(
                        E_t.sum(), pos, create_graph=True
                    )[0]
                    # (nAtoms, 3)

            return E_t, F_t  # (nMolecules, num_targets), (nAtoms, 3)
        else:
            return E_t

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

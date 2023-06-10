import torch
from torch_scatter import scatter

from ocpmodels.models.gemnet.layers.base_layers import Dense, ResidualLayer
from ocpmodels.modules.scaling.scale_factor import ScaleFactor


class EwaldBlock(torch.nn.Module):
    """
    Long-range block from the Ewald message passing method

    Parameters
    ----------
        shared_downprojection: Dense,
            Downprojection block in Ewald block update function,
            shared between subsequent Ewald Blocks.
        emb_size_atom: int
            Embedding size of the atoms.
        downprojection_size: int
            Dimension of the downprojection bottleneck
        num_hidden: int
            Number of residual blocks in Ewald block update function.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
        use_pbc: bool
            Set to True if periodic boundary conditions are applied.
        delta_k: float
            Structure factor voxel resolution
            (only relevant if use_pbc == False).
        k_rbf_values: torch.Tensor
            Pre-evaluated values of Fourier space RBF
            (only relevant if use_pbc == False).
        return_k_params: bool = True,
            Whether to return k,x dot product and damping function values.
    """

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        activation=None,
        name=None,  # identifier in case a ScalingFactor is applied to Ewald output
        use_pbc: bool = True,
        delta_k: float = None,
        k_rbf_values: torch.Tensor = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.return_k_params = return_k_params

        self.delta_k = delta_k
        self.k_rbf_values = k_rbf_values

        self.down = shared_downprojection
        self.up = Dense(
            downprojection_size, emb_size_atom, activation=None, bias=False
        )
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.ewald_layers = self.get_mlp(
            emb_size_atom, emb_size_atom, num_hidden, activation
        )
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        k: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        # Dot products k^Tx and damping values: need to be computed only once per structure
        # Ewald block in first interaction block gets None as input, therefore computes these
        # values and then passes them on to Ewald blocks in later interaction blocks
        dot: torch.Tensor = None,
        sinc_damping: torch.Tensor = None,
    ):
        hres = self.pre_residual(h)
        # Compute dot products and damping values if not already done so by an Ewald block
        # in a previous interaction block
        if dot == None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)
        if sinc_damping == None:
            if self.use_pbc == False:
                sinc_damping = (
                    torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
                )
                sinc_damping = sinc_damping.expand(-1, k.shape[-2])
            else:
                sinc_damping = 1

        # Compute Fourier space filter from weights
        if self.use_pbc:
            self.kfilter = (
                torch.matmul(self.up.linear.weight, self.down.linear.weight)
                .T.unsqueeze(0)
                .expand(num_batch, -1, -1)
            )
        else:
            self.k_rbf_values = self.k_rbf_values.to(x.device)
            self.kfilter = (
                self.up(self.down(self.k_rbf_values))
                .unsqueeze(0)
                .expand(num_batch, -1, -1)
            )

        # Compute real and imaginary parts of structure factor
        sf_real = hres.new_zeros(
            num_batch, dot.shape[-1], hres.shape[-1]
        ).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )
        sf_imag = hres.new_zeros(
            num_batch, dot.shape[-1], hres.shape[-1]
        ).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )

        # Apply Fourier space filter; scatter back to position space
        h_update = 0.01 * torch.sum(
            torch.index_select(sf_real * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1])
            + torch.index_select(sf_imag * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
            dim=1,
        )

        if self.ewald_scale_sum is not None:
            h_update = self.ewald_scale_sum(h_update, ref=h)

        # Apply update function
        for layer in self.ewald_layers:
            h_update = layer(h_update)

        if self.return_k_params:
            return h_update, dot, sinc_damping
        else:
            return h_update


# Atom-to-atom continuous-filter convolution
class HadamardBlock(torch.nn.Module):
    """
    Aggregate atom-to-atom messages by Hadamard (i.e., component-wise)
    product of embeddings and radial basis functions

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_bf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "hadamard_atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_bf = Dense(
            emb_size_bf, emb_size_atom, activation=None, bias=False
        )
        self.scale_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum"
        )
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.layers = self.get_mlp(
            emb_size_atom, emb_size_atom, nHidden, activation
        )

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, bf, idx_s, idx_t):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]
        h_res = self.pre_residual(h)

        mlp_bf = self.dense_bf(bf)

        x = torch.index_select(h_res, 0, idx_s) * mlp_bf

        x2 = scatter(x, idx_t, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x = self.scale_sum(h, x2)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x

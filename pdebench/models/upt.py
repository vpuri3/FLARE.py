#
import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F

# supernode
# from torch_scatter import segment_csr

# local
from .perceiver import PerceiverDecoder
from lra.models.backends import SelfAttentionBlock

__all__ = [
    "UPT",
]

#======================================================================#
# Universal Physics Transformer (UPT)
# Based on: https://github.com/BenediktAlkin/upt-tutorial
#======================================================================#
# Supernode Encoder
#======================================================================#
class SupernodeEncoder(nn.Module):
    
    def __init__(
        self,
        space_dim: int,
        fun_dim: int,
        num_supernodes: int,
        channel_dim: int,
    ):
        
        self.num_supernodes = num_supernodes
        
        self.enc_proj = nn.Linear(space_dim + fun_dim, channel_dim)
        
    def forward(self, pos: torch.Tensor, fun: torch.Tensor):

        from torch_geometric.nn.pool import radius_graph

        # for each batch

        supernode_idx = torch.randint(0, self.num_supernodes, (pos.shape[0],))
        
        # get the supernode position
        supernode_pos = pos[supernode_idx]
        
        # get the supernode function
        supernode_fun = fun[supernode_idx]
        
        supernode_edges = radius_graph(
            x=input_pos,
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch=batch_idx,
            loop=True,
            # inverted flow direction is required to have sorted dst_indices
            flow="target_to_source",
        )
        
        return latent


#======================================================================#
# UPT full model
#======================================================================#
class UPT(nn.Module):
    def __init__(
        self,
        space_dim: int,
        fun_dim: int,
        out_dim: int,
        n_encoder_layers: int = 2,
        n_approximator_layers: int = 4,
        n_decoder_layers: int = 2,
        num_supernodes: int = 512,
        channel_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        act: str = "gelu",
    ):
        super().__init__()

        self.encoder = None

        self.approximator = nn.ModuleList([
            SelfAttentionBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act=act,
            )
            for _ in range(n_approximator_layers)
        ])
        
        self.decoder = PerceiverDecoder(
            channel_dim=channel_dim,
            num_heads=num_heads,
        )

        self.ln = nn.LayerNorm(channel_dim)
        self.out_proj = nn.Linear(channel_dim, out_dim)

    def forward(self, x):
        
        pos = x[..., :space_dim]
        fun = x[..., space_dim:] if self.fun_dim > 0 else pos

        latent = self.encoder(pos, fun)

        for block in self.approximator:
            latent = block(latent)

        y = self.decoder(latent, pos)
        y = self.out_proj(self.ln(y))

        return y

#======================================================================#

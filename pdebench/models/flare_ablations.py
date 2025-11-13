#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "BigFLAREModel",
]

from .flare import ResidualMLP, FinalLayer
from .transformer import SelfAttentionBlock

#======================================================================#
# BigFLARE
#======================================================================#
class BigFLARE(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        act: str = None,
        num_layers_kv_proj: int = 3,
        kv_proj_hidden_dim: int = None,
        # ablation parameters
        shared_latents: bool = False,
        num_latent_blocks: int = 0,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

        self.shared_latents = shared_latents
        self.num_latent_blocks = num_latent_blocks

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        ###
        # latent query (shared or not)
        ###
        if self.shared_latents:
            assert self.num_latent_blocks == 0, f"num_latent_blocks must be 0 when shared_latents is True. Got {self.num_latent_blocks}."
            self.latent_q = nn.Parameter(torch.empty(self.head_dim, self.num_latents))
        else:
            self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))

        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [
            ResidualMLP(
                in_dim=self.channel_dim,
                hidden_dim=kv_proj_hidden_dim,
                out_dim=self.channel_dim,
                num_layers=num_layers_kv_proj,
                act=act,
                input_residual=True,
                output_residual=True,
            ) for _ in range(2)
        ]

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

        ###
        # latent blocks
        ###
        if self.num_latent_blocks > 0:
            self.latent_blocks = nn.ModuleList([
                SelfAttentionBlock(
                    channel_dim=self.channel_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=4.0,
                    act=act,
                )
                for _ in range(self.num_latent_blocks)
            ])

    def apply_latent_blocks(self, z):
        if self.num_latent_blocks == 0:
            return z

        z = rearrange(z, 'b h m d -> b m (h d)')
        for block in self.latent_blocks:
            z = block(z) # [B M D]
        z = rearrange(z, 'b m (h d) -> b h m d', h=self.num_heads)

        return z

    def forward(self, x, return_scores: bool = False):

        # x: [B N C]

        if return_scores:
            assert self.num_latent_blocks == 0, f"num_latent_blocks must be 0 when return_scores is True. Got {self.num_latent_blocks}."

        if self.shared_latents:
            q = self.latent_q.view(1, self.num_latents, self.head_dim).expand(self.num_heads, -1, -1)
        else:
            q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]

        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        #--------------------------------------------#
        if not return_scores:
            q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
            z = F.scaled_dot_product_attention(q, k, v, scale=1.0) # [B H M D]
            z = self.apply_latent_blocks(z)
            y = F.scaled_dot_product_attention(k, q, z, scale=1.0)

            scores = None
        else:
            # (1) Compute projection weights
            scores = q @ k.transpose(-2, -1) # [B H M N]
            W_encode = F.softmax(scores, dim=-1)
            W_decode = F.softmax(scores.transpose(-2, -1), dim=-1)

            # (2) Project to latent sequence
            z = W_encode @ v # [B H M D]

            # (3) Project back to input space
            y = W_decode @ z # [B H N D]
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y, scores

#======================================================================#
# FLARE Block
#======================================================================#
class BigFLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        act: str = None,
        num_layers_kv_proj: int = 3,
        num_layers_mlp: int = 3,
        kv_proj_hidden_dim: int = None,
        mlp_hidden_dim: int = None,
        # ablation parameters
        shared_latents: bool = False,
        num_latent_blocks: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = BigFLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            act=act,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_hidden_dim=kv_proj_hidden_dim,
            # ablation parameters
            shared_latents=shared_latents,
            num_latent_blocks=num_latent_blocks,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=channel_dim,
            num_layers=num_layers_mlp,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x, return_scores: bool = False):
        # x: [B, N, C]

        # x = x + att(ln1(x))
        # x = x + mlp(ln2(x))
        # return x

        _x, scores = self.att(self.ln1(x), return_scores=return_scores)
        x = x + _x
        x = x + self.mlp(self.ln2(x))

        return x, scores

#======================================================================#
# MODEL
#======================================================================#
class BigFLAREModel(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_latents: int = None,
        num_heads: int = None,
        act: str = None,
        #
        num_layers_kv_proj: int = 3,
        num_layers_mlp: int = 3,
        num_layers_in_out_proj: int = 2,
        #
        mlp_ratio: float = 1.0,
        kv_proj_ratio: float = 1.0,
        in_out_proj_ratio: float = 1.0,
        #
        out_proj_ln: bool = True,
        # ablation parameters
        shared_latents: bool = False,
        num_latent_blocks: int = 0,
    ):
        super().__init__()

        mlp_hidden_dim = int(channel_dim * mlp_ratio)
        kv_proj_hidden_dim = int(channel_dim * kv_proj_ratio)
        in_out_proj_hidden_dim = int(channel_dim * in_out_proj_ratio)

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=in_out_proj_hidden_dim,
            out_dim=channel_dim,
            num_layers=num_layers_in_out_proj,
            act=act,
            output_residual=True,
        )
        self.out_proj = FinalLayer(
            channel_dim=channel_dim,
            hidden_dim=in_out_proj_hidden_dim,
            out_dim=out_dim,
            act=act,
            num_layers=num_layers_in_out_proj,
            ln=out_proj_ln,
        )

        self.blocks = nn.ModuleList([
            BigFLAREBlock(
                channel_dim=channel_dim,
                num_latents=num_latents,
                num_heads=num_heads,
                act=act,
                num_layers_kv_proj=num_layers_kv_proj,
                num_layers_mlp=num_layers_mlp,
                kv_proj_hidden_dim=kv_proj_hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                # ablation parameters
                shared_latents=shared_latents,
                num_latent_blocks=num_latent_blocks,
            )
            for _ in range(num_blocks)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward(self, x, return_scores: bool = False):
        # x: [B, N, C]

        if return_scores:
            scores = []

        x = self.in_proj(x)
        for block in self.blocks:
            x, score = block(x, return_scores=return_scores)
            if return_scores:
                scores.append(score)

        x = self.out_proj(x)

        return (x, scores) if return_scores else x

#======================================================================#
#
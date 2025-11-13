#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "Transformer",
]

from .flare import ResidualMLP, FinalLayer

#======================================================================#
# Activation Functions
#======================================================================#

ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}

#======================================================================#
# Vanilla Self-Attention Block
#======================================================================#
class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: str = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        if act in ['swiglu', 'geglu']:
            self.fc2 = nn.Linear(hidden_dim // 2, out_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = None):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_heads = channel_dim // 16 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads 
        self.scale = self.head_dim ** -0.5

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.qkv_proj = nn.Linear(self.channel_dim, 3 * self.channel_dim, bias=False)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):

        # x: [B N C]

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            channel_dim: int,
            num_heads: int = None,
            mlp_ratio: float = 4.0,
            act: str = None,
        ):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = MultiHeadedSelfAttention(channel_dim, num_heads)
        self.mlp = MLPBlock( in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act)

    def forward(self, x):
        # x: [B, N, C]

        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

#======================================================================#
# MODEL
#======================================================================#
class Transformer(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = None,
        mlp_ratio: float = 4.0,
        act: str = None,
    ):
        super().__init__()

        # self.in_proj = nn.Linear(in_dim, channel_dim)
        # self.out_proj = nn.Linear(channel_dim, out_dim)
        # self.out_proj = nn.Sequential(nn.LayerNorm(channel_dim), nn.Linear(channel_dim, out_dim))

        self.in_proj = ResidualMLP(
            in_dim=in_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=2, act=act, output_residual=True,
        )
        self.out_proj = FinalLayer(channel_dim, out_dim, act=act, num_layers=2)

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                mlp_ratio=mlp_ratio,
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

    def forward(self, x):
        # x: [B, N, C]

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)

        return x

#======================================================================#
#
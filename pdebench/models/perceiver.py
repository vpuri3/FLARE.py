#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "PerceiverIO",
]

# local
from .flare import ResidualMLP

#======================================================================#
# Local blocks (self-contained Perceiver dependencies)
#======================================================================#
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, act: str = None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
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
        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.qkv_proj = nn.Linear(self.channel_dim, 3 * self.channel_dim, bias=False)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]
        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        return y

class SelfAttentionBlock(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = None, mlp_ratio: float = 4.0, act: str = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = MultiHeadedSelfAttention(channel_dim, num_heads)
        self.mlp = MLPBlock(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * mlp_ratio),
            out_dim=channel_dim,
            act=act,
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class FinalLayer(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        out_dim: int,
        act: str = None,
        num_layers: int = 2,
        hidden_dim: int = None,
        ln: bool = True,
    ):
        if hidden_dim is None:
            hidden_dim = channel_dim
        super().__init__()
        self.ln = nn.LayerNorm(channel_dim) if ln else nn.Identity()
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            act=act,
            input_residual=True,
            output_residual=False,
        )

    def forward(self, x):
        x = self.mlp(self.ln(x))
        return x

#======================================================================#
# Perceiver Encoder/ Decoder
#======================================================================#
class PerceiverEncoder(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8, num_latents: int = 128):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_latents = num_latents
        self.latent_q = nn.Parameter(torch.randn(num_latents, channel_dim))
        nn.init.normal_(self.latent_q, mean=0.0, std=1.0)

        self.ln = nn.LayerNorm(channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        # x: [B, N, C]

        x = self.ln(x)

        q = rearrange(self.latent_q, 'm (h d) -> h m d', h=self.num_heads) # [H, M, D]
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # [B, H, M, D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

class PerceiverDecoder(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)

        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x, y):
        # Args:
        #   x: [B M C]
        #   y: [B N C]
        # Returns:
        #   z: [B N C]

        x = self.ln1(x)
        y = self.ln2(y)

        q = rearrange(self.q_proj(y), 'b m (h d) -> b h m d', h=self.num_heads) # [B H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]

        z = F.scaled_dot_product_attention(q, k, v, scale=self.scale) # [B H N D]

        z = rearrange(z, 'b h n d -> b n (h d)')
        z = self.out_proj(z)

        return z

#======================================================================#
# MODEL
#======================================================================#
class PerceiverIO(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = 8,
        num_latents: int = 128,
        mlp_ratio: float = 4.0,
        act: str = None,
        cross_attn: bool = False,
    ):
        super().__init__()

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=channel_dim,
            out_dim=channel_dim,
            num_layers=2,
            act=act,
            output_residual=True,
        )

        self.encoder = PerceiverEncoder(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
        )
        
        self.cross_attn = cross_attn
        AttnBlock = CrossAttentionBlock if cross_attn else SelfAttentionBlock

        self.blocks = nn.ModuleList([
            AttnBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])

        self.decoder = PerceiverDecoder(
            channel_dim=channel_dim,
            num_heads=num_heads,
        )

        self.out_proj = FinalLayer(
            channel_dim,
            out_dim,
            act=act,
            num_layers=2,
        )

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

        x = self.in_proj(x) # [B N C]
        z = self.encoder(x) # [B M C]

        for block in self.blocks:
            if self.cross_attn:
                z = block(z, x) # [B M C] <- [B M C], [B N C]
            else:
                z = block(z) # [B M C] <- [B M C]

        x = self.decoder(z, x) # [B M C], [B, N, C] -> [B N C]
        x = self.out_proj(x) # [B N C]

        return x

#======================================================================#
# Cross Attention Block
#======================================================================#
class MultiHeadedCrossAttention(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = None):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_heads = channel_dim // 16 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads 
        self.scale = self.head_dim ** -0.5

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.q_proj = nn.Linear(self.channel_dim, self.channel_dim, bias=False)
        self.kv_proj = nn.Linear(self.channel_dim, 2 * self.channel_dim, bias=False)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, z, x):

        # z <- x
        # z: [B M C]
        # x: [B N C]

        q = self.q_proj(z) # [B M C]
        k, v = self.kv_proj(x).chunk(2, dim=-1) # [B N C]
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

class CrossAttentionBlock(nn.Module):
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
        self.ln3 = nn.LayerNorm(channel_dim)
        self.att = MultiHeadedCrossAttention(channel_dim, num_heads)
        self.mlp = MLPBlock( in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act)

    def forward(self, z, x):
        # z <- x
        # z: [B, M, C]
        # x: [B, N, C]

        z = z + self.att(self.ln1(z), self.ln2(x))
        z = z + self.mlp(self.ln3(z))

        return z

#======================================================================#
#

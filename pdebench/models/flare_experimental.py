#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "FLAREExperimentalModel",
]

#======================================================================#
# FLARE
#======================================================================#
class FLAREExperimental(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        attn_scale: float = 1.0,
        act: str = None,
        num_layers_k_proj: int = 3,
        num_layers_v_proj: int = 3,
        k_proj_mlp_ratio: float = 1.0,
        v_proj_mlp_ratio: float = 1.0,
        qk_norm: bool = False,
        rmsnorm: bool = False,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        assert attn_scale > 0.0, f"attn_scale must be greater than 0. Got {attn_scale}."

        self.attn_scale = attn_scale

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.qk_norm = qk_norm
        if self.qk_norm:
            Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
            self.q_norm = Norm(self.head_dim)
            self.k_norm = Norm(self.head_dim)

        self.k_proj = ResidualMLP(
            in_dim=self.channel_dim,
            hidden_dim=int(self.channel_dim * k_proj_mlp_ratio),
            out_dim=self.channel_dim,
            num_layers=num_layers_k_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )

        self.v_proj = ResidualMLP(
            in_dim=self.channel_dim,
            hidden_dim=int(self.channel_dim * v_proj_mlp_ratio),
            out_dim=self.channel_dim,
            num_layers=num_layers_v_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):

        # x: [B N C]

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        #--------------------------------------------#
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
# FLARE Block
#======================================================================#
class FLAREExperimentalBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        attn_scale: float = 1.0,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_k_proj: int = 3,
        num_layers_v_proj: int = 3,
        k_proj_mlp_ratio: float = 1.0,
        v_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 3,
        ffn_mlp_ratio: float = 1.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = FLAREExperimental(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            attn_scale=attn_scale,
            act=act,
            num_layers_k_proj=num_layers_k_proj,
            num_layers_v_proj=num_layers_v_proj,
            k_proj_mlp_ratio=k_proj_mlp_ratio,
            v_proj_mlp_ratio=v_proj_mlp_ratio,
            qk_norm=qk_norm,
            rmsnorm=rmsnorm,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x):
        # x: [B, N, C]

        # x = x + att(norm1(x))
        # x = x + mlp(norm2(x))
        # return x

        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

#======================================================================#
# MODEL
#======================================================================#
class FLAREExperimentalModel(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = None,
        act: str = None,
        rmsnorm: bool = False,
        out_proj_norm: bool = True,
        num_layers_in_out_proj: int = 2,
        #
        attn_scale: float = 1.0,
        num_latents: int = None,
        num_layers_k_proj: int = 3,
        num_layers_v_proj: int = 3,
        k_proj_mlp_ratio: float = 1.0,
        v_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 3,
        ffn_mlp_ratio: float = 1.0,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=channel_dim,
            out_dim=channel_dim,
            num_layers=num_layers_in_out_proj,
            act=act,
            input_residual=False,
            output_residual=True,
        )
        
        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm

        self.out_proj = nn.Sequential(
            Norm(channel_dim) if out_proj_norm else nn.Identity(),
            ResidualMLP(
                in_dim=channel_dim,
                hidden_dim=channel_dim,
                out_dim=out_dim,
                num_layers=num_layers_in_out_proj,
                act=act,
                input_residual=True,
                output_residual=False,
            )
        )

        self.blocks = nn.ModuleList([
            FLAREExperimentalBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                rmsnorm=rmsnorm,
                attn_scale=attn_scale,
                num_latents=num_latents,
                num_layers_k_proj=num_layers_k_proj,
                num_layers_v_proj=num_layers_v_proj,
                k_proj_mlp_ratio=k_proj_mlp_ratio,
                v_proj_mlp_ratio=v_proj_mlp_ratio,
                num_layers_ffn=num_layers_ffn,
                ffn_mlp_ratio=ffn_mlp_ratio,
                qk_norm=qk_norm,
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
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # x: [B, N, C]

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)

        return x

#======================================================================#
# Activation Functions
#======================================================================#
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}

class ResidualMLP(nn.Module):
    def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
            act: str = None, input_residual: bool = False, output_residual: bool = False,
        ):
        super().__init__()

        self.num_layers = num_layers
        assert self.num_layers >= -1, f"num_layers must be at least -1. Got {self.num_layers}."

        # nn.Linear if num_layers == -1
        if self.num_layers == -1:
            self.fc = nn.Linear(in_dim, out_dim)
            self.residual = input_residual and output_residual and (in_dim == out_dim)
            return

        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual  = input_residual  and (in_dim  == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):

        if self.num_layers == -1:
            x = x + self.fc(x) if self.residual else self.fc(x)
            return x

        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for fc in self.fcs:
            x = x + self.act(fc(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)

        return x

#======================================================================#
#
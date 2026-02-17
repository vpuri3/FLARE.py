#
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .flare import ACTIVATIONS, ResidualMLP

__all__ = [
    "UnloopyWrapper",
]

#======================================================================#
# IDEAS
#======================================================================#
# - apply multiplicative gating after attention (https://arxiv.org/pdf/2505.06708).
#   would that nonlinearity allow for an attn only transformer?
# - replace FFN with cross attn where (K, V) are learned parameters.
#   furthermore share FFN bw attn blocks.
# - Understand what is special about the attn arch in Attn only transformer paper
#   https://openreview.net/pdf?id=ITMu1pZTFo.
#

#======================================================================#
# FLARE
#======================================================================#
class FLARE(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        attn_scale: float = 1.0,
        act: str = None,
        num_layers_kv_proj: int = 3,
        kv_proj_mlp_ratio: float = 1.0,
        #
        gating: bool = False,
        num_layers_gating_proj: int = 3,
        gating_proj_mlp_ratio: float = 1.0,
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

        self.k_proj, self.v_proj = [
            ResidualMLP(
                in_dim=self.channel_dim,
                hidden_dim=int(self.channel_dim * kv_proj_mlp_ratio),
                out_dim=self.channel_dim,
                num_layers=num_layers_kv_proj,
                act=act,
                input_residual=True,
                output_residual=True,
            ) for _ in range(2)
        ]

        self.gating = gating
        if gating:
            self.g_proj = ResidualMLP(
                in_dim=self.channel_dim,
                hidden_dim=int(self.channel_dim * gating_proj_mlp_ratio),
                out_dim=self.channel_dim,
                num_layers=num_layers_gating_proj,
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

        #--------------------------------------------#
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')

        if self.gating:
            g = self.g_proj(x).sigmoid()
            y = y * g

        y = self.out_proj(y)

        return y

#======================================================================#
# FLARE Block
#======================================================================#
class FLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        attn_scale: float = 1.0,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = 3,
        num_layers_ffn: int = 3,
        kv_proj_mlp_ratio: float = 1.0,
        ffn_mlp_ratio: float = 1.0,
        #
        shared_ffn: bool = False,
        shared_att: bool = False,
        #
        gating: bool = False,
        num_layers_gating_proj: int = 3,
        gating_proj_mlp_ratio: float = 1.0,
    ):

        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

        self.att = FLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            attn_scale=attn_scale,
            act=act,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_mlp_ratio=kv_proj_mlp_ratio,
        ) if not shared_att else None

        self.ffn = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        ) if not shared_ffn else None

    def forward(self, x, ffn=None, att=None):
        # x: [B, N, C]

        att = att if att is not None else self.att
        ffn = ffn if ffn is not None else self.ffn

        assert att is not None and ffn is not None, "att and ffn must be provided. Got att={att} and ffn={ffn}."

        x = x + att(self.norm1(x))
        x = x + ffn(self.norm2(x))

        return x

#======================================================================#
# MODEL
#======================================================================#
class UnloopyWrapper(nn.Module):
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
        num_layers_kv_proj: int = 3,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 3,
        ffn_mlp_ratio: float = 1.0,
        #
        shared_ffn: bool = False,
        shared_att: bool = False,
        #
        gating: bool = False,
        num_layers_gating_proj: int = 3,
        gating_proj_mlp_ratio: float = 1.0,
    ):
        super().__init__()

        in_out_act = act if act in ['gelu', 'silu'] else 'gelu'

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=channel_dim,
            out_dim=channel_dim,
            num_layers=num_layers_in_out_proj,
            act=in_out_act,
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
                act=in_out_act,
                input_residual=True,
                output_residual=False,
            )
        )

        shared_block = FLAREBlock(
            channel_dim=channel_dim,
            num_heads=num_heads,
            act=act,
            rmsnorm=rmsnorm,
            attn_scale=attn_scale,
            num_latents=num_latents,
            num_layers_kv_proj=num_layers_kv_proj,
            num_layers_ffn=num_layers_ffn,
            kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            ffn_mlp_ratio=ffn_mlp_ratio,
            #
            shared_ffn=False,
            shared_att=False,
            #
            gating=gating,
            num_layers_gating_proj=num_layers_gating_proj,
            gating_proj_mlp_ratio=gating_proj_mlp_ratio,
        ) if shared_ffn or shared_att else None

        self.ffn = shared_block.ffn if shared_ffn else None
        self.att = shared_block.att if shared_att else None

        self.blocks = nn.ModuleList([
            FLAREBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                rmsnorm=rmsnorm,
                attn_scale=attn_scale,
                num_latents=num_latents,
                num_layers_kv_proj=num_layers_kv_proj,
                num_layers_ffn=num_layers_ffn,
                kv_proj_mlp_ratio=kv_proj_mlp_ratio,
                ffn_mlp_ratio=ffn_mlp_ratio,
                #
                shared_ffn=shared_ffn,
                shared_att=shared_att,
                #
                gating=gating,
                num_layers_gating_proj=num_layers_gating_proj,
                gating_proj_mlp_ratio=gating_proj_mlp_ratio,
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
            x = block(x, ffn=self.ffn, att=self.att)
        x = self.out_proj(x)

        return x

#======================================================================#
#
# Encoder-only Luna (Ma et al., NeurIPS 2021): pack then unpack softmax attention
# with a carried packed stream. PDEBench shell follows flare.py (no causal /
# decoder / positional embeddings / context parallel).
#
# Paper: https://arxiv.org/abs/2106.01540
# Encoder layer: https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_layer.py
# Nested attention: https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_attention.py
# Packed-stream init: https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_sentence_encoder.py
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

__all__ = [
    "LunaConfig",
    "LunaEncoderAttention",
    "LunaBlock",
    "LunaModel",
]


@dataclass
class LunaConfig:
    model: str = "luna"
    num_blocks: int = 8
    channel_dim: int = 128
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    num_layers_ffn: int = 0
    ffn_mlp_ratio: float = 2.0
    qk_norm: bool = False
    num_latents: int = 64


#======================================================================#
# Activation Functions
#======================================================================#
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}


#======================================================================#
# Residual MLP Block
#======================================================================#

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


def _make_head_norm(head_dim: int, *, enabled: bool, rmsnorm: bool) -> nn.Module:
    if not enabled:
        return nn.Identity()
    if rmsnorm:
        return nn.RMSNorm(head_dim, eps=1e-6)
    return nn.LayerNorm(head_dim)


#======================================================================#
# Luna encoder attention (pack + unpack)
#======================================================================#
class LunaEncoderAttention(nn.Module):
    """Nested pack/unpack softmax attention (encoder self-attention only).

    Pack:  Y_P = Attn(P, X)   — LunarMultiheadAttention._compute_pcontext
    Unpack: Y_X = Attn(X, Y_P) — LunarMultiheadAttention.forward after pcontext
    """

    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        act: str = None,
        qk_norm: bool = False,
        rmsnorm: bool = False,
    ):
        super().__init__()
        del act

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        # Fairseq LunarMultiheadAttention.scaling / pscaling:
        # https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_attention.py
        self.attn_scale = self.head_dim ** -0.5

        self.q_norm = _make_head_norm(self.head_dim, enabled=qk_norm, rmsnorm=rmsnorm)
        self.k_norm = _make_head_norm(self.head_dim, enabled=qk_norm, rmsnorm=rmsnorm)

        # Pack: pq from P, pk/pv from tokens (untied KV; official default ties them).
        self.pq_proj = nn.Linear(self.channel_dim, self.channel_dim)
        self.pk_proj = nn.Linear(self.channel_dim, self.channel_dim)
        self.pv_proj = nn.Linear(self.channel_dim, self.channel_dim)
        # Unpack: q from tokens, k/v from packed context.
        self.q_proj = nn.Linear(self.channel_dim, self.channel_dim)
        self.k_proj = nn.Linear(self.channel_dim, self.channel_dim)
        self.v_proj = nn.Linear(self.channel_dim, self.channel_dim)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x, p):
        # x: [B N C], p: [B M C]

        heads = self.num_heads

        pq = rearrange(self.pq_proj(p), 'b m (h d) -> b h m d', h=heads)
        pk = rearrange(self.pk_proj(x), 'b n (h d) -> b h n d', h=heads)
        pv = rearrange(self.pv_proj(x), 'b n (h d) -> b h n d', h=heads)
        pq = self.q_norm(pq)
        pk = self.k_norm(pk)
        yp = F.scaled_dot_product_attention(pq, pk, pv, scale=self.attn_scale)

        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=heads)
        k = rearrange(self.k_proj(rearrange(yp, 'b h m d -> b m (h d)')), 'b m (h d) -> b h m d', h=heads)
        v = rearrange(self.v_proj(rearrange(yp, 'b h m d -> b m (h d)')), 'b m (h d) -> b h m d', h=heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        yx = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)

        yx = rearrange(yx, 'b h n d -> b n (h d)')
        yx = self.out_proj(yx)
        yp = rearrange(yp, 'b h m d -> b m (h d)')
        return yx, yp


#======================================================================#
# Luna Block
#======================================================================#
class LunaBlock(nn.Module):
    """Pre-norm FLARE-style block wrapping Luna encoder attention.

    Official LunaEncoderLayer is post-norm and applies FFN only to tokens:
    https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_layer.py
    """

    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 2.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm_p = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = LunaEncoderAttention(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            act=act,
            qk_norm=qk_norm,
            rmsnorm=rmsnorm,
        )
        # Official encoder FFN is fc1 -> activation -> fc2; suite FFN=0 is that stack.
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=False,
            output_residual=False,
        )

    def forward(self, x, p):
        yx, yp = self.att(self.norm1(x), self.norm_p(p))
        x = x + yx
        p = p + yp
        x = x + self.mlp(self.norm2(x))
        return x, p


#======================================================================#
# MODEL
#======================================================================#
class LunaModel(nn.Module):
    def __init__(self, config: LunaConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else metadata
        in_dim = metadata["c_in"]
        out_dim = metadata["c_out"]
        channel_dim = config.channel_dim
        num_blocks = config.num_blocks
        num_heads = config.num_heads
        act = config.act
        rmsnorm = config.rmsnorm
        out_proj_norm = config.out_proj_norm
        num_layers_in_out_proj = config.num_layers_in_out_proj
        num_latents = config.num_latents
        num_layers_ffn = config.num_layers_ffn
        ffn_mlp_ratio = config.ffn_mlp_ratio
        qk_norm = config.qk_norm

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=channel_dim,
            out_dim=channel_dim,
            num_layers=num_layers_in_out_proj,
            act=act,
            input_residual=False,
            output_residual=True,
        )

        self.out_proj = nn.Sequential(
            (nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)) if out_proj_norm else nn.Identity(),
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

        # LunaSentenceEncoder.projected_embeddings: learnable packed queries, no token PE.
        # https://github.com/XuezheMax/fairseq-apollo/blob/master/fairseq/modules/luna_sentence_encoder.py
        self.packed_embed = nn.Parameter(torch.empty(num_latents, channel_dim))
        nn.init.normal_(self.packed_embed, mean=0.0, std=channel_dim ** -0.5)

        self.blocks = nn.ModuleList([
            LunaBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                rmsnorm=rmsnorm,
                num_latents=num_latents,
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
        p = self.packed_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        for block in self.blocks:
            x, p = block(x, p)
        return self.out_proj(x)

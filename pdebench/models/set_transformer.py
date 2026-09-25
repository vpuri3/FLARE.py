#
import math
import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
from typing import Optional

from .flare import ResidualMLP

__all__ = [
    "SetTransformerModel",
]

@dataclass
class SetTransformerConfig:
    model: str = "set_transformer"
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    num_slices: int = 64
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2



ACTIVATION = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
}


class MAB(nn.Module):
    """Multihead attention block used by Set Transformer."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, act: str = 'gelu', ln: bool = True):
        super().__init__()

        if dim_v % num_heads != 0:
            raise ValueError(f"dim_v must be divisible by num_heads. Got dim_v={dim_v}, num_heads={num_heads}.")

        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim = dim_v // num_heads

        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)

        self.ln0 = nn.LayerNorm(dim_v) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_v) if ln else nn.Identity()

        self.fc_o = nn.Linear(dim_v, dim_v)

        act = 'gelu' if act is None else act
        if act not in ACTIVATION:
            raise ValueError(f"Unsupported activation '{act}'. Expected one of {list(ACTIVATION.keys())}.")
        self.act = ACTIVATION[act]()

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, H, N, D]
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, N, D] -> [B, N, C]
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.dim_v)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)

        out = F.scaled_dot_product_attention(qh, kh, vh, scale=1.0 / math.sqrt(self.dim_v))
        out = self._merge_heads(out)

        out = self.ln0(q + out)
        out = self.ln1(out + self.act(self.fc_o(out)))
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block from Set Transformer."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, act: str = 'gelu', ln: bool = True):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_q=dim_out, dim_k=dim_in, dim_v=dim_out, num_heads=num_heads, act=act, ln=ln)
        self.mab1 = MAB(dim_q=dim_in, dim_k=dim_out, dim_v=dim_out, num_heads=num_heads, act=act, ln=ln)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mab0(self.I.expand(x.size(0), -1, -1), x)
        return self.mab1(x, h)


class SetAttentionBlock(nn.Module):
    """Transformer-style block with ISAB attention + FFN."""

    def __init__(self, channel_dim: int, num_heads: int, num_inds: int, mlp_ratio: float = 2.0, act: str = 'gelu', rmsnorm: bool = False):
        super().__init__()
        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.norm1 = Norm(channel_dim)
        self.norm2 = Norm(channel_dim)

        self.attn = ISAB(
            dim_in=channel_dim,
            dim_out=channel_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            act=act,
            ln=True,
        )

        self.ffn = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * mlp_ratio),
            out_dim=channel_dim,
            num_layers=0,
            act=act,
            input_residual=False,
            output_residual=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SetTransformerModel(nn.Module):
    def __init__(self, config: SetTransformerConfig, metadata=None):
        super().__init__()

        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        channel_dim = int(config.channel_dim)
        num_blocks = int(config.num_blocks)
        num_heads = int(config.num_heads)
        num_inds = int(config.num_slices)
        act = "gelu" if config.act is None else config.act
        mlp_ratio = float(config.mlp_ratio)
        rmsnorm = bool(config.rmsnorm)
        out_proj_norm = bool(config.out_proj_norm)
        num_layers_in_out_proj = int(config.num_layers_in_out_proj)

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

        self.blocks = nn.ModuleList([
            SetAttentionBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                num_inds=num_inds,
                mlp_ratio=mlp_ratio,
                act=act,
                rmsnorm=rmsnorm,
            )
            for _ in range(num_blocks)
        ])

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
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        return x

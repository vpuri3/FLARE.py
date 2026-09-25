#
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

__all__ = [
    "TransformerWrapper",
]

@dataclass
class TransformerConfig:
    model: str = "transformer"
    channel_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2

@dataclass
class LinformerConfig:
    model: str = "linformer"
    channel_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    linformer_k: int = 256

@dataclass
class LinearConfig:
    model: str = "linear"
    channel_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    kernel: str = "identity"
    norm_q: bool = True
    norm_k: bool = True

from .flare import ResidualMLP
from lra.models.backends import MODEL_TYPES

#======================================================================#
# MODEL
#======================================================================#
class TransformerWrapper(nn.Module):
    def __init__(self, config: TransformerConfig | LinformerConfig | LinearConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else metadata
        in_dim = metadata["c_in"]
        out_dim = metadata["c_out"]
        if isinstance(config, LinformerConfig):
            backend = "linformer"
            backend_kwargs = dict(mlp_ratio=config.mlp_ratio, seq_len=metadata["max_length"], k=config.linformer_k)
        elif isinstance(config, LinearConfig):
            backend = "linear"
            backend_kwargs = dict(mlp_ratio=config.mlp_ratio, kernel=config.kernel, norm_q=config.norm_q, norm_k=config.norm_k)
        else:
            backend = "transformer"
            backend_kwargs = dict(mlp_ratio=config.mlp_ratio)

        channel_dim = config.channel_dim
        num_blocks = config.num_blocks
        num_heads = config.num_heads
        act = config.act
        rmsnorm = config.rmsnorm
        out_proj_norm = config.out_proj_norm
        num_layers_in_out_proj = config.num_layers_in_out_proj
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

        Block = MODEL_TYPES.get(backend, None)

        if Block is None:
            raise NotImplementedError(f"Backend {backend} not implemented. See pdebench.models.transformer for available backends.")

        self.blocks = nn.ModuleList([
            Block(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                rmsnorm=rmsnorm,
                **backend_kwargs,
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

    def forward(self, x, mask=None, **kwargs):
        # x: [B, N, C]
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x

#======================================================================#

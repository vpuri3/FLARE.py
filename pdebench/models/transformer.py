#
import torch
from torch import nn

__all__ = [
    "TransformerWrapper",
]

from .flare import ResidualMLP
from lra.models.backends import MODEL_TYPES

#======================================================================#
# MODEL
#======================================================================#
class TransformerWrapper(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = None,
        act: str = None,
        rmsnorm: bool = False,
        ###
        out_proj_norm: bool = True,
        num_layers_in_out_proj: int = 2,
        ###
        backend: str = 'transformer',
        **backend_kwargs,
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

    def forward(self, x):
        # x: [B, N, C]

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)

        return x

#======================================================================#
#
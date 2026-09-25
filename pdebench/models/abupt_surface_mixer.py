"""Surface-only AB-UPT mixer for PDEBench pointwise surface data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pdebench.models.abupt_mixer import ABUPTTransformerBlock, _validate_rope_head_dim
from pdebench.models.mixer_backbone import ResidualMLP


@dataclass
class ABUPTSurfaceMixerConfig:
    """Configuration for the surface-only AB-UPT mixer."""

    model: str = "abupt_surface_mixer"
    channel_dim: int = 128
    num_blocks: int = 8
    num_heads: int = 8
    num_layers_in_out_proj: int = 2
    num_layers_ffn: int = 0
    ffn_mlp_ratio: float = 2.0
    out_proj_norm: bool = True
    rmsnorm: bool = True
    coordinate_scale: float = 1000.0
    num_surface_anchors: int | None = 1024

    def __post_init__(self) -> None:
        if self.channel_dim <= 0:
            raise ValueError("channel_dim must be positive")
        if self.num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if self.num_layers_in_out_proj < -1:
            raise ValueError("num_layers_in_out_proj must be at least -1")
        if self.num_layers_ffn < 0:
            raise ValueError("num_layers_ffn must be non-negative")
        if self.ffn_mlp_ratio <= 0:
            raise ValueError("ffn_mlp_ratio must be positive")
        head_dim = _validate_rope_head_dim(self.channel_dim, self.num_heads)
        if head_dim < 6:
            raise ValueError("head dimension must provide RoPE capacity")
        if self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        if self.num_surface_anchors is not None and self.num_surface_anchors <= 0:
            raise ValueError("num_surface_anchors must be positive when provided")


class RopeFrequency(nn.Module):
    """Build float32-safe complex RoPE frequencies from surface coordinates."""

    def __init__(self, dim: int, ndim: int = 3) -> None:
        super().__init__()
        padding_ndim = dim % ndim
        dim_per_ndim = (dim - padding_ndim) // ndim
        sincos_padding = dim_per_ndim % 2
        padding = padding_ndim + sincos_padding * ndim
        effective = (dim - padding) // ndim
        omega = 1.0 / (10000.0 ** (torch.arange(0, effective, 2, dtype=torch.float32) / effective))
        self.register_buffer("omega", omega)
        self.padding = padding

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if not torch.all(coords >= 0):
            raise ValueError("RoPE coordinates must be non-negative")
        with torch.autocast(device_type=str(coords.device).split(":")[0], enabled=False):
            values = coords.float().unsqueeze(-1) @ self.omega.to(coords.device).unsqueeze(0)
        values = values.flatten(start_dim=-2)
        if self.padding:
            values = torch.cat([values, values.new_zeros(*values.shape[:-1], self.padding // 2)], dim=-1)
        return torch.polar(torch.ones_like(values), values.float())


class ABUPTSurfaceMixerModel(nn.Module):
    """AB-UPT surface attention with exactly ``num_blocks`` surface blocks."""

    def __init__(self, config: ABUPTSurfaceMixerConfig, metadata: dict | None = None) -> None:
        super().__init__()
        metadata = {} if metadata is None else metadata
        if metadata and (metadata.get("c_in") != 6 or metadata.get("c_out") != 4):
            raise ValueError("abupt_surface_mixer requires c_in=6 and c_out=4")
        self.config = config
        self.in_proj = ResidualMLP(
            in_dim=6,
            hidden_dim=config.channel_dim,
            out_dim=config.channel_dim,
            num_layers=config.num_layers_in_out_proj,
            input_residual=False,
            output_residual=True,
        )
        self.blocks = nn.ModuleList(
            ABUPTTransformerBlock(
                config.channel_dim,
                config.num_heads,
                rmsnorm=config.rmsnorm,
                num_layers_ffn=config.num_layers_ffn,
                ffn_mlp_ratio=config.ffn_mlp_ratio,
            )
            for _ in range(config.num_blocks)
        )
        self.rope = RopeFrequency(config.channel_dim // config.num_heads)
        self.out_proj = nn.Sequential(
            (nn.RMSNorm(config.channel_dim, eps=1e-6) if config.rmsnorm else nn.LayerNorm(config.channel_dim))
            if config.out_proj_norm
            else nn.Identity(),
            ResidualMLP(
                in_dim=config.channel_dim,
                hidden_dim=config.channel_dim,
                out_dim=4,
                num_layers=config.num_layers_in_out_proj,
                input_residual=True,
                output_residual=False,
            ),
        )
        self.apply(self._init_weights)

    @staticmethod
    def select_surface_permutation(
        num_points: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randperm(num_points, device=device, generator=generator)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 6:
            raise ValueError("abupt_surface_mixer expects [batch, points, 6] inputs")
        generator = None
        if not self.training:
            generator = torch.Generator(device=x.device).manual_seed(0)
        permutation = self.select_surface_permutation(x.shape[1], device=x.device, generator=generator)
        inverse_permutation = torch.argsort(permutation)
        x = x[:, permutation]
        positions = x[..., :3] * self.config.coordinate_scale
        freqs = self.rope(positions)
        tokens = self.in_proj(x)
        num_anchor_tokens = self.config.num_surface_anchors
        if num_anchor_tokens is not None:
            num_anchor_tokens = min(num_anchor_tokens, tokens.shape[1])
        for block in self.blocks:
            tokens = block(tokens, freqs=freqs, num_anchor_tokens=num_anchor_tokens)
        return self.out_proj(tokens)[:, inverse_permutation]


__all__ = ["ABUPTSurfaceMixerConfig", "ABUPTSurfaceMixerModel", "RopeFrequency"]

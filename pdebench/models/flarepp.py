import math
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ..distributed.context_parallel import ContextParallelState
from ..distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive
from ..distributed.utils import gather_sequence_tensor

__all__ = [
    "FLAREPPModel",
    "FlarePPConfig",
    "ResidualMLP",
    "FLAREPPMixer",
]


@dataclass
class FlarePPConfig:
    model: str = "flarepp"
    num_blocks: int = 8
    channel_dim: int = 128
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    num_layers_ffn: int = 0
    ffn_mlp_ratio: float = 2.0
    k_norm: bool = True
    share_k0_v0: bool = True
    num_latents: int = 64
    encoder_cp_backend: str = "flash"
    gate_logit_init: float = 0.25
    q_fixed_norm: bool = True


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


def _init_latent_queries(latent_q: nn.Parameter) -> None:
    """Initialize latent_q with shape [H, M, D] via N(0, 0.02)."""
    nn.init.normal_(latent_q, mean=0.0, std=0.02)


def _make_head_norm(
    head_dim: int,
    *,
    enabled: bool,
    rmsnorm: bool,
    elementwise_affine: bool,
) -> nn.Module:
    if not enabled:
        return nn.Identity()
    if rmsnorm:
        return nn.RMSNorm(head_dim, eps=1e-6, elementwise_affine=elementwise_affine)
    return nn.LayerNorm(head_dim, elementwise_affine=elementwise_affine)


def _make_residual_linear_proj(channel_dim: int) -> nn.Linear:
    proj = nn.Linear(channel_dim, channel_dim, bias=True)
    with torch.no_grad():
        noise = torch.empty_like(proj.weight)
        nn.init.trunc_normal_(noise, mean=0.0, std=0.02, a=-2.0, b=2.0)
        eye = torch.eye(channel_dim, dtype=proj.weight.dtype, device=proj.weight.device)
        proj.weight.copy_(eye + noise)
        proj.bias.zero_()
    proj._skip_backbone_weight_init = True  # type: ignore[attr-defined]
    return proj


#======================================================================#
# FLARE++ Mixer (input-dependent queries)
#======================================================================#
class FLAREPPMixer(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        rmsnorm: bool = False,
        q_fixed_norm: bool = True,
        encoder_cp_backend: str = "flash",
        gate_logit_init: float = 0.25,
    ):
        super().__init__()

        for name, value in (
            ("k_norm", k_norm),
            ("share_k0_v0", share_k0_v0),
            ("q_fixed_norm", q_fixed_norm),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")
        gate_logit_init = float(gate_logit_init)
        if not math.isfinite(gate_logit_init):
            raise ValueError(f"gate_logit_init must be finite, got {gate_logit_init}")

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.share_k0_v0 = share_k0_v0

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.attn_scale = self.head_dim ** -0.5

        self.latent_q0 = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _init_latent_queries(self.latent_q0)

        self.gate_logit = nn.Parameter(torch.full((self.num_heads,), gate_logit_init))
        self.latent_q_fixed = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _init_latent_queries(self.latent_q_fixed)

        # Hop-1 norms always on: q0 keeps affine; k0 matches v0 (no affine).
        self.q0_norm = _make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=True
        )
        self.k0_norm = _make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=False
        )
        self.v0_norm = _make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=False
        )
        self.k_norm = _make_head_norm(
            self.head_dim, enabled=k_norm, rmsnorm=rmsnorm, elementwise_affine=True
        )
        self.q_fixed_norm = _make_head_norm(
            self.head_dim, enabled=q_fixed_norm, rmsnorm=rmsnorm, elementwise_affine=False
        )

        self.k0_proj = _make_residual_linear_proj(self.channel_dim)
        self.v0_proj = (
            _make_residual_linear_proj(self.channel_dim)
            if not self.share_k0_v0
            else None
        )
        self.k_proj = _make_residual_linear_proj(self.channel_dim)
        self.v_proj = _make_residual_linear_proj(self.channel_dim)

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)
        if encoder_cp_backend == "naiive":
            self.encoder_cp = FlareEncoderCPNaiive()
        elif encoder_cp_backend == "flash":
            self.encoder_cp = FlareEncoderCPFlash()
        else:
            raise ValueError(
                f"Invalid encoder_cp_backend: {encoder_cp_backend!r}. Choose from: naiive, flash."
            )
        self.cp_state: Optional[ContextParallelState] = None
        self.cp_debug_gather_outputs: bool = False

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False) -> None:
        self.cp_state = cp_state
        self.cp_debug_gather_outputs = bool(cp_debug_gather_outputs)

    def _use_context_parallel(self) -> bool:
        return (self.cp_state is not None) and (self.cp_state.cp_size > 1)

    def flare_encode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Encode values into the query/latent space (CP-aware)."""
        if self._use_context_parallel():
            return self.encoder_cp(
                q_latent=q,
                k_local=k,
                v_local=v,
                cp_group=self.cp_state.cp_group,
                scale=self.attn_scale,
            )
        return F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)

    def flare_decode(
        self,
        k: torch.Tensor,
        q: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent values back to the token sequence via SDPA."""
        # Flash encode may return low-precision z while norms left q/k in fp32.
        q = q.to(dtype=z.dtype)
        k = k.to(dtype=z.dtype)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
        if self._use_context_parallel() and self.cp_debug_gather_outputs:
            _ = gather_sequence_tensor(y.detach(), self.cp_state, seq_dim=2)
        return y

    def forward(self, x, mask: torch.Tensor = None):
        if mask is not None:
            raise NotImplementedError("FLAREPPMixer does not support mask.")

        batch_size = x.size(0)
        num_heads = self.num_heads

        q0 = self.q0_norm(self.latent_q0.unsqueeze(0).expand(batch_size, -1, -1, -1))
        k0 = self.k0_norm(rearrange(self.k0_proj(x), "b n (h d) -> b h n d", h=num_heads))
        if not self.share_k0_v0:
            v0 = self.v0_norm(rearrange(self.v0_proj(x), "b n (h d) -> b h n d", h=num_heads))
        else:
            v0 = k0

        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=num_heads)

        q_dynamic = self.flare_encode(q0, k0, v0)
        qf = self.latent_q_fixed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        qf = self.q_fixed_norm(qf)
        qf_f = qf.float()
        qd_f = q_dynamic.float()
        g = torch.sigmoid(self.gate_logit).float().view(1, num_heads, 1, 1)
        q = (qf_f + g * qd_f).to(dtype=x.dtype)
        k = self.k_norm(k)

        z = self.flare_encode(q, k, v)
        y = self.flare_decode(k, q, z)

        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.out_proj(y)
        return y


#======================================================================#
# FLARE++ Block
#======================================================================#
class FLAREPPBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        act: str = None,
        rmsnorm: bool = False,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        q_fixed_norm: bool = True,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 2.0,
        encoder_cp_backend: str = "flash",
        gate_logit_init: float = 0.25,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.mixer = FLAREPPMixer(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            k_norm=k_norm,
            share_k0_v0=share_k0_v0,
            rmsnorm=rmsnorm,
            q_fixed_norm=q_fixed_norm,
            encoder_cp_backend=encoder_cp_backend,
            gate_logit_init=gate_logit_init,
        )
        self.ffn = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x, mask: torch.Tensor = None):
        x = x + self.mixer(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False) -> None:
        self.mixer.set_context_parallel(cp_state=cp_state, cp_debug_gather_outputs=cp_debug_gather_outputs)


#======================================================================#
# MODEL
#======================================================================#
class FLAREPPModel(nn.Module):
    def __init__(self, config: FlarePPConfig, metadata=None):
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
        k_norm = config.k_norm
        share_k0_v0 = config.share_k0_v0
        q_fixed_norm = config.q_fixed_norm
        encoder_cp_backend = config.encoder_cp_backend
        gate_logit_init = config.gate_logit_init

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

        self.blocks = nn.ModuleList([
            FLAREPPBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                act=act,
                rmsnorm=rmsnorm,
                k_norm=k_norm,
                share_k0_v0=share_k0_v0,
                q_fixed_norm=q_fixed_norm,
                num_layers_ffn=num_layers_ffn,
                ffn_mlp_ratio=ffn_mlp_ratio,
                encoder_cp_backend=encoder_cp_backend,
                gate_logit_init=gate_logit_init,
            )
            for _ in range(num_blocks)
        ])

        self.cp_state: Optional[ContextParallelState] = None

        self.initialize_weights()

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False) -> None:
        self.cp_state = cp_state
        for block in self.blocks:
            block.set_context_parallel(cp_state=cp_state, cp_debug_gather_outputs=cp_debug_gather_outputs)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if getattr(m, "_skip_backbone_weight_init", False):
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, mask: torch.Tensor = None):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.out_proj(x)

#
import math
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

__all__ = [
    "FLAREPPMixerConfig",
    "FLAREMixerConfig",
    "FLAREPPAblationsMixerConfig",
    "SimplifiedFLAREPPMixerConfig",
    "MHAMixerConfig",
    "MIXER_BY_KIND",
    "MixerBackboneConfig",
    "MixerBackboneModel",
    "MixerConfig",
    "Transolver3MixerConfig",
    "TransolverMixerConfig",
    "TransolverPPMixerConfig",
    "build_mixer",
]


@dataclass
class MHAMixerConfig:
    kind: Literal["mha"] = "mha"
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.kind != "mha":
            raise ValueError(f"MHAMixerConfig kind must be 'mha'; got {self.kind!r}.")


@dataclass
class FLAREMixerConfig:
    kind: Literal["flare"] = "flare"
    num_latents: int = 64
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.kind != "flare":
            raise ValueError(f"FLAREMixerConfig kind must be 'flare'; got {self.kind!r}.")


@dataclass
class SimplifiedFLAREPPMixerConfig:
    kind: Literal["simplifiedflarepp"] = "simplifiedflarepp"
    num_latents: int = 64
    qk_norm: bool = False
    qk0_norm: bool = True
    share_k0_v0: bool = False

    def __post_init__(self) -> None:
        if self.kind != "simplifiedflarepp":
            raise ValueError(f"SimplifiedFLAREPPMixerConfig kind must be 'simplifiedflarepp'; got {self.kind!r}.")


@dataclass
class FLAREPPMixerConfig:
    kind: Literal["flarepp"] = "flarepp"
    num_latents: int = 64
    k_norm: bool = True
    share_k0_v0: bool = True
    gate_logit_init: float = 0.25
    q_fixed_norm: bool = True

    def __post_init__(self) -> None:
        if self.kind != "flarepp":
            raise ValueError(
                f"FLAREPPMixerConfig kind must be 'flarepp'; got {self.kind!r}."
            )


@dataclass
class FLAREPPAblationsMixerConfig:
    kind: Literal["flarepp_ablations"] = "flarepp_ablations"
    num_latents: int = 64
    q0_norm: bool = False
    k0_norm: bool = False
    v0_norm: bool = False
    q_norm: bool = False
    k_norm: bool = False
    q_fixed_norm: bool = True
    q0_elementwise_affine: bool = False
    k0_elementwise_affine: bool = False
    v0_elementwise_affine: bool = False
    q_elementwise_affine: bool = False
    k_elementwise_affine: bool = False
    q_fixed_elementwise_affine: bool = False
    k0_use_bias: bool = False
    v0_use_bias: bool = False
    k_use_bias: bool = False
    v_use_bias: bool = False
    k0_use_residual: bool = False
    v0_use_residual: bool = False
    k_use_residual: bool = False
    v_use_residual: bool = True
    share_k0_v0: bool = False
    use_gate: bool = False
    gate_logit_init: float = 0.25

    def __post_init__(self) -> None:
        if self.kind != "flarepp_ablations":
            raise ValueError(
                f"FLAREPPAblationsMixerConfig kind must be 'flarepp_ablations'; got {self.kind!r}."
            )


@dataclass
class TransolverMixerConfig:
    kind: Literal["transolver"] = "transolver"
    num_latents: int = 64

    def __post_init__(self) -> None:
        if self.kind != "transolver":
            raise ValueError(f"TransolverMixerConfig kind must be 'transolver'; got {self.kind!r}.")


@dataclass
class TransolverPPMixerConfig:
    kind: Literal["transolverpp"] = "transolverpp"
    num_latents: int = 64

    def __post_init__(self) -> None:
        if self.kind != "transolverpp":
            raise ValueError(
                f"TransolverPPMixerConfig kind must be 'transolverpp'; got {self.kind!r}."
            )


@dataclass
class Transolver3MixerConfig:
    kind: Literal["transolver3"] = "transolver3"
    num_latents: int = 64

    def __post_init__(self) -> None:
        if self.kind != "transolver3":
            raise ValueError(
                f"Transolver3MixerConfig kind must be 'transolver3'; got {self.kind!r}."
            )


MixerConfig = (
    MHAMixerConfig
    | FLAREMixerConfig
    | SimplifiedFLAREPPMixerConfig
    | FLAREPPMixerConfig
    | FLAREPPAblationsMixerConfig
    | TransolverMixerConfig
    | TransolverPPMixerConfig
    | Transolver3MixerConfig
)


@dataclass
class MixerBackboneConfig:
    model: str = "mixer_backbone"
    num_blocks: int = 8
    channel_dim: int = 128
    num_heads: Optional[int] = 8
    act: Optional[str] = None
    rmsnorm: Optional[bool] = None
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    num_layers_ffn: int = 0
    mlp_ratio_ffn: float = 2.0
    diagnostics: bool = False
    mixer: MixerConfig = field(default_factory=FLAREMixerConfig)


def _init_latent_queries(latent_q: nn.Parameter) -> None:
    """Initialize latent_q with shape [H, M, D] via N(0, 0.02)."""
    nn.init.normal_(latent_q, mean=0.0, std=0.02)


@torch.compiler.disable
def _rms(t: torch.Tensor) -> float:
    """RMS over all non-batch dims, averaged across the batch dim; returns a Python float."""
    t = t.detach().float()
    flat = t.reshape(t.shape[0], -1)
    return flat.pow(2).mean(dim=-1).sqrt().mean().item()


@torch.compiler.disable
def _flarepp_hop1_diagnostics(
    *,
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q_dynamic: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    gate_logit: Optional[torch.Tensor],
    gate_raw: bool,
) -> dict:
    """Hop-1 synthesis-attention diagnostics for FLAREPPAblations.

    `q0`/`k0` must be the post-norm tensors actually fed into the hop-1 SDPA call; the
    softmax score recompute here is diagnostics-only (SDPA remains the real forward path).
    """
    with torch.no_grad():
        eps = 1e-12
        scores = torch.matmul(q0.float(), k0.float().transpose(-1, -2)) * attn_scale
        p = torch.softmax(scores, dim=-1)
        entropy = -(p * p.clamp_min(eps).log()).sum(dim=-1)
        n_eff = 1.0 / p.pow(2).sum(dim=-1).clamp_min(eps)

        q0n = F.normalize(q0.float(), dim=-1, eps=eps)
        cos = torch.matmul(q0n, q0n.transpose(-1, -2))
        num_latents = cos.shape[-1]
        if num_latents > 1:
            offdiag_sum = cos.sum(dim=(-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            latent_q_offdiag_cos = (offdiag_sum / (num_latents * (num_latents - 1))).mean().item()
        else:
            latent_q_offdiag_cos = float("nan")

        diag = {
            "rms_q_dynamic": _rms(q_dynamic),
            "rms_k0": _rms(k0),
            "rms_v0": _rms(v0),
            "rms_k": _rms(k),
            "rms_v": _rms(v),
            "synth_attn_entropy": entropy.mean().item(),
            "synth_n_eff": n_eff.mean().item(),
            "latent_q_offdiag_cos": latent_q_offdiag_cos,
        }
        if gate_logit is None:
            diag["gate_value"] = float("nan")
        else:
            gate = gate_logit.detach().float()
            diag["gate_value"] = gate.mean().item() if gate_raw else torch.sigmoid(gate).mean().item()
    return diag


@torch.compiler.disable
def _set_flarepp_mixer_diagnostics(
    module: nn.Module,
    *,
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q_dynamic: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    gate_logit: Optional[torch.Tensor],
    gate_raw: bool,
) -> None:
    """Side-effect diagnostics update; kept out of torch.compile graphs."""
    if not getattr(module, "diagnostics", False):
        module.last_diagnostics = None
        return
    module.last_diagnostics = _flarepp_hop1_diagnostics(
        q0=q0, k0=k0, v0=v0, q_dynamic=q_dynamic, k=k, v=v,
        attn_scale=attn_scale, gate_logit=gate_logit, gate_raw=gate_raw,
    )


@torch.compiler.disable
def _set_block_diagnostics(block: nn.Module, normed: torch.Tensor, mixer_out: torch.Tensor) -> None:
    """Side-effect block diagnostics update; kept out of torch.compile graphs."""
    if not getattr(block, "diagnostics", False):
        block.last_diagnostics = None
        return
    residual_stream_rms = _rms(normed)
    mixer_out_rms = _rms(mixer_out)
    diag = {
        "residual_stream_rms": residual_stream_rms,
        "mixer_out_rms": mixer_out_rms,
        "mixer_over_stream": mixer_out_rms / (residual_stream_rms + 1e-8),
    }
    mixer_diag = getattr(block.mixer, "last_diagnostics", None)
    if mixer_diag:
        diag.update(mixer_diag)
    block.last_diagnostics = diag


@torch.compiler.disable
def _mean_across_blocks(block_diagnostics: list) -> dict:
    """Mean of keys common to every block's diagnostics dict (empty list -> {})."""
    if not block_diagnostics:
        return {}
    common_keys = set(block_diagnostics[0])
    for diag in block_diagnostics[1:]:
        common_keys &= set(diag)
    return {
        key: sum(diag[key] for diag in block_diagnostics) / len(block_diagnostics)
        for key in common_keys
    }


@torch.compiler.disable
def _set_model_diagnostics(model: nn.Module, block_diagnostics: list) -> None:
    if not getattr(model, "diagnostics", False):
        model.last_diagnostics = None
        return
    model.last_diagnostics = {
        "blocks": block_diagnostics,
        "mean": _mean_across_blocks(block_diagnostics),
    }


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


def _residual_linear_projection(channel_dim: int) -> ResidualMLP:
    return ResidualMLP(
        in_dim=channel_dim,
        hidden_dim=channel_dim,
        out_dim=channel_dim,
        num_layers=-1,
        input_residual=True,
        output_residual=True,
    )


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


def _make_ablation_proj(channel_dim: int, *, use_bias: bool, use_residual: bool) -> nn.Linear:
    proj = nn.Linear(channel_dim, channel_dim, bias=use_bias)
    with torch.no_grad():
        noise = torch.empty_like(proj.weight)
        nn.init.trunc_normal_(noise, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if use_residual:
            eye = torch.eye(channel_dim, dtype=proj.weight.dtype, device=proj.weight.device)
            proj.weight.copy_(eye + noise)
        else:
            proj.weight.copy_(noise)
        if proj.bias is not None:
            proj.bias.zero_()
    proj._skip_backbone_weight_init = True  # type: ignore[attr-defined]
    return proj


#======================================================================#
# Multi-Head Attention Mixer
#======================================================================#
class MHAMixer(nn.Module):
    def __init__(
        self,
        config: MHAMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        channel_dim = int(backbone_config.channel_dim)
        num_heads = int(backbone_config.num_heads)
        rmsnorm = bool(backbone_config.rmsnorm)
        qk_norm = bool(config.qk_norm)

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.qkv_proj = nn.Linear(self.channel_dim, 3 * self.channel_dim)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

        if rmsnorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        else:
            self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]
        q = self.q_norm(q)
        k = self.k_norm(k)

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        return y

#======================================================================#
# FLARE Mixer
#======================================================================#
class FLAREMixer(nn.Module):
    def __init__(
        self,
        config: FLAREMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        channel_dim = int(backbone_config.channel_dim)
        num_heads = channel_dim // 8 if backbone_config.num_heads is None else int(backbone_config.num_heads)
        rmsnorm = bool(backbone_config.rmsnorm)
        num_latents = int(config.num_latents)
        qk_norm = bool(config.qk_norm)

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.attn_scale = self.head_dim ** -0.5

        self.latent_q = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _init_latent_queries(self.latent_q)

        if rmsnorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        else:
            self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.k_proj = _residual_linear_projection(self.channel_dim)
        self.v_proj = _residual_linear_projection(self.channel_dim)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):
        q = self.latent_q
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        return y

#======================================================================#
# Simplified FLARE++ Mixer (input-dependent queries; no gate; no CP)
#======================================================================#
class SimplifiedFLAREPPMixer(nn.Module):
    def __init__(
        self,
        config: SimplifiedFLAREPPMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        channel_dim = int(backbone_config.channel_dim)
        num_heads = channel_dim // 8 if backbone_config.num_heads is None else int(backbone_config.num_heads)
        rmsnorm = bool(backbone_config.rmsnorm)
        num_latents = int(config.num_latents)
        qk_norm = bool(config.qk_norm)
        qk0_norm = bool(config.qk0_norm)
        share_k0_v0 = config.share_k0_v0

        if not isinstance(share_k0_v0, bool):
            raise TypeError(f"share_k0_v0 must be bool, got {type(share_k0_v0).__name__}")

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.share_k0_v0 = share_k0_v0

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.attn_scale = self.head_dim ** -0.5

        self.latent_q0 = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _init_latent_queries(self.latent_q0)

        if rmsnorm:
            self.k0_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk0_norm else nn.Identity()
            self.q0_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk0_norm else nn.Identity()
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        else:
            self.k0_norm = nn.LayerNorm(self.head_dim) if qk0_norm else nn.Identity()
            self.q0_norm = nn.LayerNorm(self.head_dim) if qk0_norm else nn.Identity()
            self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        # Proj capacity fixed at ResidualMLP(-1) / ratio 1.0 (no MixerBackboneConfig knobs).
        self.k0_proj = _residual_linear_projection(self.channel_dim)
        self.v0_proj = _residual_linear_projection(self.channel_dim) if not self.share_k0_v0 else None
        self.k_proj = _residual_linear_projection(self.channel_dim)
        self.v_proj = _residual_linear_projection(self.channel_dim)

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):
        batch_size = x.size(0)
        num_heads = self.num_heads

        q0 = self.latent_q0.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k0 = rearrange(self.k0_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v0 = rearrange(self.v0_proj(x), "b n (h d) -> b h n d", h=num_heads) if not self.share_k0_v0 else k0

        q0 = self.q0_norm(q0)
        k0 = self.k0_norm(k0)

        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=num_heads)

        q = F.scaled_dot_product_attention(q0, k0, v0, scale=self.attn_scale)

        q = self.q_norm(q)
        k = self.k_norm(k)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)

        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.out_proj(y)
        return y


#======================================================================#
# FLARE++ Mixer (aligned with flarepp.FLAREPPMixer; always-on gate; fixed anchor queries)
#======================================================================#
class FLAREPPMixer(nn.Module):
    def __init__(
        self,
        config: FLAREPPMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        channel_dim = int(backbone_config.channel_dim)
        num_heads = channel_dim // 8 if backbone_config.num_heads is None else int(backbone_config.num_heads)
        rmsnorm = bool(backbone_config.rmsnorm)
        num_latents = int(config.num_latents)
        for config_field in fields(config):
            value = getattr(config, config_field.name)
            if config_field.type is bool and not isinstance(value, bool):
                raise TypeError(f"{config_field.name} must be bool, got {type(value).__name__}")
        gate_logit_init = float(config.gate_logit_init)
        if not math.isfinite(gate_logit_init):
            raise ValueError(f"gate_logit_init must be finite, got {gate_logit_init}")

        k_norm = bool(config.k_norm)
        q_fixed_norm = bool(config.q_fixed_norm)
        share_k0_v0 = config.share_k0_v0

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
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
        self.k_norm = _make_head_norm(
            self.head_dim, enabled=k_norm, rmsnorm=rmsnorm, elementwise_affine=True
        )
        self.v0_norm = _make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=False
        )
        self.q_fixed_norm = _make_head_norm(
            self.head_dim, enabled=q_fixed_norm, rmsnorm=rmsnorm, elementwise_affine=False
        )

        self.k0_proj = _make_ablation_proj(self.channel_dim, use_bias=True, use_residual=True)
        self.v0_proj = (
            _make_ablation_proj(self.channel_dim, use_bias=True, use_residual=True)
            if not self.share_k0_v0
            else None
        )
        self.k_proj = _make_ablation_proj(self.channel_dim, use_bias=True, use_residual=True)
        self.v_proj = _make_ablation_proj(self.channel_dim, use_bias=True, use_residual=True)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):
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

        q_dynamic = F.scaled_dot_product_attention(q0, k0, v0, scale=self.attn_scale)

        qf = self.latent_q_fixed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        qf = self.q_fixed_norm(qf)
        qf_f = qf.float()
        qd_f = q_dynamic.float()
        g = torch.sigmoid(self.gate_logit).float().view(1, num_heads, 1, 1)
        # Anchored mix: fixed query plus gated dynamic residual (non-convex).
        q = (qf_f + g * qd_f).to(dtype=x.dtype)

        k = self.k_norm(k)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.out_proj(y)
        return y


#======================================================================#
# FLARE++ Ablations Mixer (split norm/proj knobs; raw gate logit)
#======================================================================#
class FLAREPPAblations(nn.Module):
    def __init__(
        self,
        config: FLAREPPAblationsMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        channel_dim = int(backbone_config.channel_dim)
        num_heads = channel_dim // 8 if backbone_config.num_heads is None else int(backbone_config.num_heads)
        rmsnorm = bool(backbone_config.rmsnorm)
        diagnostics = bool(backbone_config.diagnostics)
        num_latents = int(config.num_latents)
        for config_field in fields(config):
            value = getattr(config, config_field.name)
            if config_field.type is bool and not isinstance(value, bool):
                raise TypeError(f"{config_field.name} must be bool, got {type(value).__name__}")
        gate_logit_init = float(config.gate_logit_init)
        if not math.isfinite(gate_logit_init):
            raise ValueError(f"gate_logit_init must be finite, got {gate_logit_init}")

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.use_gate = config.use_gate
        self.share_k0_v0 = config.share_k0_v0
        self.diagnostics = bool(diagnostics)
        self.last_diagnostics: Optional[dict] = None

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.attn_scale = self.head_dim ** -0.5

        self.latent_q0 = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _init_latent_queries(self.latent_q0)

        if self.use_gate:
            self.gate_logit = nn.Parameter(torch.full((self.num_heads,), gate_logit_init))
            self.latent_q_fixed = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
            _init_latent_queries(self.latent_q_fixed)

        self.q0_norm = _make_head_norm(
            self.head_dim,
            enabled=config.q0_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.q0_elementwise_affine,
        )
        self.k0_norm = _make_head_norm(
            self.head_dim,
            enabled=config.k0_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.k0_elementwise_affine,
        )
        self.v0_norm = _make_head_norm(
            self.head_dim,
            enabled=config.v0_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.v0_elementwise_affine,
        )
        self.q_norm = _make_head_norm(
            self.head_dim,
            enabled=config.q_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.q_elementwise_affine,
        )
        self.k_norm = _make_head_norm(
            self.head_dim,
            enabled=config.k_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.k_elementwise_affine,
        )
        self.q_fixed_norm = _make_head_norm(
            self.head_dim,
            enabled=config.q_fixed_norm,
            rmsnorm=rmsnorm,
            elementwise_affine=config.q_fixed_elementwise_affine,
        )

        self.k0_proj = _make_ablation_proj(
            self.channel_dim, use_bias=config.k0_use_bias, use_residual=config.k0_use_residual
        )
        self.v0_proj = (
            _make_ablation_proj(
                self.channel_dim, use_bias=config.v0_use_bias, use_residual=config.v0_use_residual
            )
            if not self.share_k0_v0
            else None
        )
        self.k_proj = _make_ablation_proj(
            self.channel_dim, use_bias=config.k_use_bias, use_residual=config.k_use_residual
        )
        self.v_proj = _make_ablation_proj(
            self.channel_dim, use_bias=config.v_use_bias, use_residual=config.v_use_residual
        )

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):
        batch_size = x.size(0)
        num_heads = self.num_heads

        q0 = self.latent_q0.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k0 = rearrange(self.k0_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v0 = rearrange(self.v0_proj(x), "b n (h d) -> b h n d", h=num_heads) if not self.share_k0_v0 else k0

        q0 = self.q0_norm(q0)
        k0 = self.k0_norm(k0)
        v0 = self.v0_norm(v0)

        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=num_heads)

        q = F.scaled_dot_product_attention(q0, k0, v0, scale=self.attn_scale)
        q_dynamic = q

        if self.use_gate:
            qf = self.latent_q_fixed.unsqueeze(0).expand(batch_size, -1, -1, -1)
            qf = self.q_fixed_norm(qf)
            g = self.gate_logit.view(1, num_heads, 1, 1)
            # Accumulate gate mix in fp32 under AMP, then cast back.
            q = (qf.float() + g.float() * q.float()).to(dtype=x.dtype)

        q = self.q_norm(q)
        k = self.k_norm(k)

        _set_flarepp_mixer_diagnostics(
            self,
            q0=q0, k0=k0, v0=v0, q_dynamic=q_dynamic, k=k, v=v,
            attn_scale=self.attn_scale,
            gate_logit=(self.gate_logit if self.use_gate else None),
            gate_raw=True,
        )

        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)

        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.out_proj(y)
        return y

#======================================================================#
# Transolver Mixer
#======================================================================#
class TransolverMixer(nn.Module):
    def __init__(
        self,
        config: TransolverMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        dim = int(backbone_config.channel_dim)
        heads = int(backbone_config.num_heads)
        dim_head = dim // heads
        dropout = 0.0
        slice_num = int(config.num_latents)
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, _C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_logits = self.in_project_slice(x_mid) / temperature
        slice_weights = F.softmax(slice_logits.float(), dim=-1).to(dtype=x_mid.dtype)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots.float(), dim=-1).to(dtype=dots.dtype)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        out_x = self.to_out(out_x)
        return out_x


def _gumbel_softmax(logits, tau):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    return F.softmax((logits + gumbel_noise) / tau, dim=-1)


#======================================================================#
# Transolver++ Mixer
#======================================================================#
class TransolverPPMixer(nn.Module):
    def __init__(
        self,
        config: TransolverPPMixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        dim = int(backbone_config.channel_dim)
        heads = int(backbone_config.num_heads)
        dim_head = dim // heads
        dropout = 0.0
        slice_num = int(config.num_latents)
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        x_mid = self.in_project_x(x).reshape(batch_size, num_tokens, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()
        temperature = torch.clamp(self.proj_temperature(x_mid) + self.bias, min=0.01)
        slice_weights = _gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        out_slice_token = F.scaled_dot_product_attention(
            self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        )
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        return self.to_out(rearrange(out_x, "b h n d -> b n (h d)"))


#======================================================================#
# Transolver-3 Mixer
#======================================================================#
class Transolver3Mixer(nn.Module):
    def __init__(
        self,
        config: Transolver3MixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        del metadata
        dim = int(backbone_config.channel_dim)
        heads = int(backbone_config.num_heads)
        dim_head = dim // heads
        dropout = 0.0
        slice_num = int(config.num_latents)
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        self.in_project = nn.Linear(dim, 2 * heads * dim_head)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_out_linear = nn.Linear(heads * dim_head, dim)

        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

    def _get_fused_weight_slice(self):
        w_in = self.in_project.weight[self.heads * self.dim_head :].view(self.heads, self.dim_head, -1)
        fused_w = torch.matmul(self.in_project_slice.weight, w_in)

        b_in = self.in_project.bias[self.heads * self.dim_head :].view(self.heads, self.dim_head)
        fused_b = torch.matmul(self.in_project_slice.weight, b_in.unsqueeze(-1)).squeeze(-1)
        return fused_w, fused_b + self.in_project_slice.bias

    def _slice_weights(self, x, fused_w, fused_b):
        logits = torch.einsum("bnc, hgc -> bhng", x, fused_w)
        logits = logits + fused_b.view(1, self.heads, 1, self.slice_num)
        return F.softmax(logits / self.temperature, dim=-1)

    def _slice_attend(self, slice_token):
        return F.scaled_dot_product_attention(
            self.to_q(slice_token),
            self.to_k(slice_token),
            self.to_v(slice_token),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

    def _deslice_to_out(self, out_slice_token, slice_weights):
        w_out = self.to_out_linear.weight.view(-1, self.heads, self.dim_head).permute(1, 2, 0)
        projected_slices = torch.einsum("bhgd, hdc -> bhgc", out_slice_token, w_out)
        out_x = torch.einsum("bhng, bhgc -> bnc", slice_weights, projected_slices)
        return self.dropout(out_x + self.to_out_linear.bias)

    def forward(self, x):
        fused_w, fused_b = self._get_fused_weight_slice()
        slice_weights = self._slice_weights(x, fused_w, fused_b)
        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5

        raw_states = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        raw_states = raw_states / slice_norm.transpose(-1, -2)

        w_fx = self.in_project.weight[: self.heads * self.dim_head].view(self.heads, self.dim_head, x.size(-1))
        b_fx = self.in_project.bias[: self.heads * self.dim_head].view(self.heads, self.dim_head)
        slice_token = torch.einsum("bhgc, hdc -> bhgd", raw_states, w_fx) + b_fx.view(1, self.heads, 1, self.dim_head)

        return self._deslice_to_out(self._slice_attend(slice_token), slice_weights)


MIXER_BY_KIND: dict[str, tuple[type, type]] = {
    "mha": (MHAMixerConfig, MHAMixer),
    "flare": (FLAREMixerConfig, FLAREMixer),
    "simplifiedflarepp": (SimplifiedFLAREPPMixerConfig, SimplifiedFLAREPPMixer),
    "flarepp": (FLAREPPMixerConfig, FLAREPPMixer),
    "flarepp_ablations": (FLAREPPAblationsMixerConfig, FLAREPPAblations),
    "transolver": (TransolverMixerConfig, TransolverMixer),
    "transolverpp": (TransolverPPMixerConfig, TransolverPPMixer),
    "transolver3": (Transolver3MixerConfig, Transolver3Mixer),
}
_MIXER_BY_CONFIG_TYPE: dict[type, tuple[str, type]] = {
    config_cls: (kind, module_cls)
    for kind, (config_cls, module_cls) in MIXER_BY_KIND.items()
}


def _resolve_mixer_registry_entry(config: object) -> tuple[str, type]:
    entry = _MIXER_BY_CONFIG_TYPE.get(type(config))
    if entry is not None:
        return entry
    kind = getattr(config, "kind", None)
    if isinstance(kind, str) and kind in MIXER_BY_KIND:
        config_cls, module_cls = MIXER_BY_KIND[kind]
        if isinstance(config, config_cls):
            return kind, module_cls
    raise TypeError(
        f"build_mixer: unsupported mixer config type {type(config)!r} "
        f"(mixer.kind={kind!r}); allowed mixer kinds: {sorted(MIXER_BY_KIND)}."
    )


def build_mixer(
    config: MixerConfig,
    backbone_config: MixerBackboneConfig,
    metadata=None,
) -> nn.Module:
    _, module_cls = _resolve_mixer_registry_entry(config)
    return module_cls(config, backbone_config, metadata)


#======================================================================#
# Mixer Backbone Block
#======================================================================#
class MixerBackboneBlock(nn.Module):
    def __init__(
        self,
        mixer: MixerConfig,
        backbone_config: MixerBackboneConfig,
        metadata=None,
    ):
        super().__init__()
        channel_dim = int(backbone_config.channel_dim)
        rmsnorm = bool(backbone_config.rmsnorm)
        act = backbone_config.act
        num_layers_ffn = int(backbone_config.num_layers_ffn)
        mlp_ratio_ffn = float(backbone_config.mlp_ratio_ffn)
        diagnostics = bool(backbone_config.diagnostics)
        self.diagnostics = bool(diagnostics)
        self.last_diagnostics: Optional[dict] = None
        self.norm1 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.mixer = build_mixer(mixer, backbone_config, metadata)
        self.ffn = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * mlp_ratio_ffn),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=False,
            output_residual=False,
        )

    def forward(self, x):
        normed = self.norm1(x)
        mixer_out = self.mixer(normed)
        _set_block_diagnostics(self, normed, mixer_out)

        x = x + mixer_out
        x = x + self.ffn(self.norm2(x))
        return x

#======================================================================#
# MODEL
#======================================================================#
class MixerBackboneModel(nn.Module):
    def __init__(self, config: MixerBackboneConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else metadata
        in_dim = metadata["c_in"]
        out_dim = metadata["c_out"]
        channel_dim = config.channel_dim
        num_blocks = config.num_blocks
        act = config.act
        rmsnorm = bool(config.rmsnorm)
        out_proj_norm = config.out_proj_norm
        num_layers_in_out_proj = config.num_layers_in_out_proj
        diagnostics = config.diagnostics

        self.diagnostics = bool(diagnostics)
        self.last_diagnostics: Optional[dict] = None

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
            MixerBackboneBlock(
                config.mixer,
                config,
                metadata,
            )
            for _ in range(num_blocks)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if getattr(m, "_skip_backbone_weight_init", False):
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.in_proj(x)
        block_diagnostics = []
        for block in self.blocks:
            x = block(x)
            if self.diagnostics:
                block_diagnostics.append(block.last_diagnostics)
        x = self.out_proj(x)
        _set_model_diagnostics(self, block_diagnostics)
        return x

#======================================================================#
#

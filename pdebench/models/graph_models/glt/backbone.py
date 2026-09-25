"""GLT backbone, attention, and PE-injection wiring.

Layout
------
1. ``GLTConfig`` (backbone knobs + ``pe_inject_mode`` / ``pe_update`` / nested ``pe``)
2. GLT config validation (``_validate_glt_config``)
3. GLT attention (MHA / linear / FLARE) + ``GLTBlock`` + ``GLT``

PE production lives in ``pe_eigen.py`` / ``pe_other.py``; this module builds a
``GraphPE`` from ``config.pe`` and injects its output into attention via
``pe_inject_mode``.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from ..utils import graph_node_input
from .pe_base import GraphPE, TopologyFeatures, _resolve_packed_indices
from .pe_eigen import RawEigenPEConfig
from .registry import GraphPEConfig, build_pe

__all__ = [
    "GLT",
    "GLTBlock",
    "GLTConfig",
    "GLTMHAAttention",
    "PE_INJECT_MODES",
]

PE_INJECT_MODES: tuple[str, ...] = ("concat_input", "concat_qk")


# ---------------------------------------------------------------------------
# GLT model config
# ---------------------------------------------------------------------------

@dataclass
class GLTConfig:
    """GLT model configuration.

    Positional-encoding (PE) production is pluggable: ``pe`` selects a registered
    ``*PEConfig`` (see ``glt.pe_eigen`` / ``glt.pe_other``), e.g. ``RawEigenPEConfig`` for cached Laplacian
    eigenvectors with graph_rms normalization, or ``SpectralFilterPEConfig`` for learnable SPE.

    pe_inject_mode options:
        concat_input — concat PE features into the input stream; Q/K/V from x (no separate c stream)
        concat_qk    — separate x/c streams; Q/K from concat(x, c), V from x

    pe_update:
        When true (only valid with pe_inject_mode="concat_qk"), the c stream is updated by a
        parallel stack of blocks alongside x.
    """

    model: str = "glt"
    channel_dim: int = 128
    num_blocks: int = 4
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: Optional[bool] = None
    mlp_ratio: float = 2.0
    attn_type: str = "mha"
    pe_inject_mode: str = "concat_input"
    pe_update: bool = False
    pe: GraphPEConfig = field(default_factory=RawEigenPEConfig)

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - mixed precision varlen path fails loudly at runtime.
    flash_attn_varlen_func = None


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _norm_cls(rmsnorm: bool) -> type[nn.Module]:
    return nn.RMSNorm if rmsnorm else nn.LayerNorm


def _activation(act: str | None) -> type[nn.Module]:
    return nn.SiLU if act == "silu" else nn.GELU


# ---------------------------------------------------------------------------
# GLT attention helpers
# ---------------------------------------------------------------------------

def _glt_qk_source(
    pe_inject_mode: str, x: torch.Tensor, c: torch.Tensor | None, *, name: str = "GLT"
) -> torch.Tensor:
    if pe_inject_mode == "concat_input":
        return x
    if c is None:
        raise RuntimeError(f"{name} pe_inject_mode='concat_qk' requires c to be provided.")
    return torch.cat([x, c], dim=-1)


def _validate_glt_packed_inputs(
    x: torch.Tensor,
    c: torch.Tensor | None,
    *,
    name: str = "GLT",
    require_flash: bool = False,
    require_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if require_flash and flash_attn_varlen_func is None:
        raise RuntimeError(f"{name} requires flash-attn for packed varlen attention.")
    if x.ndim != 2:
        raise RuntimeError(f"{name} expects packed x [N_tot, C], got {tuple(x.shape)}.")
    if c is not None and c.ndim != 2:
        raise RuntimeError(f"{name} expects packed c [N_tot, C] when provided, got {tuple(c.shape)}.")
    if c is not None and int(x.shape[0]) != int(c.shape[0]):
        raise RuntimeError(f"{name} x/c row mismatch: {tuple(x.shape)} vs {tuple(c.shape)}.")
    if require_fp16:
        if not x.is_cuda:
            raise RuntimeError(f"{name} flash-varlen attention requires CUDA activations.")
        if x.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            x = x.to(target_dtype)
            if c is not None:
                c = c.to(target_dtype)
        if x.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(f"{name} flash-varlen attention requires CUDA fp16/bf16 activations.")
    return x, c


def _make_cu_latents(bsz: int, num_latents: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * num_latents,
        num_latents,
        device=device,
        dtype=torch.int32,
    )


def _parse_glt_attn_type(attn_type: str) -> tuple[str, int | None]:
    attn = str(attn_type).lower()
    if attn in {"mha", "linear"}:
        return attn, None
    if attn.startswith("flare"):
        suffix = attn.removeprefix("flare")
        if not suffix.isdigit() or int(suffix) <= 0:
            raise ValueError("GLT attn_type='flareXXX' requires a positive integer XXX num_latents.")
        return "flare", int(suffix)
    raise ValueError("GLT attn_type must be one of 'mha', 'linear', or 'flareXXX'.")


# ---------------------------------------------------------------------------
# GLT config validation
# ---------------------------------------------------------------------------

def _validate_glt_config(config: GLTConfig, *, pe_out_dim: int) -> None:
    mode = str(config.pe_inject_mode)
    if mode not in PE_INJECT_MODES:
        raise ValueError(f"pe_inject_mode must be one of {PE_INJECT_MODES}; got {mode!r}.")
    if bool(config.pe_update) and mode != "concat_qk":
        raise ValueError("pe_update=True requires pe_inject_mode='concat_qk'.")
    if mode == "concat_qk" and int(pe_out_dim) <= 0:
        raise ValueError("pe_inject_mode='concat_qk' requires a PE config with out_dim > 0.")


# ---------------------------------------------------------------------------
# GLT attention (MHA / linear / FLARE)
# ---------------------------------------------------------------------------

class GLTMHAAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        pe_inject_mode: str,
    ):
        super().__init__()
        if channel_dim % num_heads != 0:
            raise ValueError(f"channel_dim must be divisible by num_heads, got {channel_dim=} and {num_heads=}.")
        if pe_inject_mode not in PE_INJECT_MODES:
            raise ValueError(f"GLT pe_inject_mode must be one of {PE_INJECT_MODES}. Got {pe_inject_mode!r}.")
        self.channel_dim = int(channel_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.channel_dim // self.num_heads
        self.pe_inject_mode = str(pe_inject_mode)
        qk_in_dim = 2 * channel_dim if self.pe_inject_mode == "concat_qk" else channel_dim
        self.q_proj = nn.Linear(qk_in_dim, channel_dim)
        self.k_proj = nn.Linear(qk_in_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out = nn.Linear(channel_dim, channel_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_type = "mha"

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.num_heads, -1)

    def _qkv(self, x: torch.Tensor, c: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qk_source = _glt_qk_source(self.pe_inject_mode, x, c)
        q = self.q_norm(self._reshape_heads(self.q_proj(qk_source)))
        k = self.k_norm(self._reshape_heads(self.k_proj(qk_source)))
        v = self._reshape_heads(self.v_proj(x))
        return q, k, v

    def forward(self, x: torch.Tensor, c: torch.Tensor | None, *, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        x, c = _validate_glt_packed_inputs(x, c, require_flash=True, require_fp16=True)
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        q, k, v = self._qkv(x, c)
        y = flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens,
            cu_seqlens,
            int(max_seqlen),
            int(max_seqlen),
            dropout_p=0.0,
            causal=False,
        )
        return self.out(y.reshape(x.shape[0], -1)).to(dtype=x.dtype)


class GLTLinearAttention(GLTMHAAttention):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        pe_inject_mode: str,
    ):
        super().__init__(
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=pe_inject_mode,
        )
        self.attn_type = "linear"

    def forward(self, x: torch.Tensor, c: torch.Tensor | None, *, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        x, c = _validate_glt_packed_inputs(x, c)
        packed = _resolve_packed_indices(
            cu_seqlens, int(x.shape[0]), x.device, max_seqlen=int(max_seqlen)
        )
        batch_index = packed.batch_index
        lengths = packed.lengths
        q, k, v = self._qkv(x, c)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        num_graphs = int(lengths.numel())
        kv = torch.zeros(
            num_graphs,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=k.dtype,
            device=k.device,
        )
        kv.index_add_(0, batch_index, k.unsqueeze(-1) * v.unsqueeze(-2))

        y = torch.einsum("nhd,nhde->nhe", q, kv[batch_index])
        y = y / lengths[batch_index].to(dtype=y.dtype).view(-1, 1, 1).clamp_min(1) + q
        return self.out(y.reshape(x.shape[0], -1)).to(dtype=x.dtype)


class GLTFlareAttention(GLTMHAAttention):
    """FLARE-style latent gather/scatter attention for GLT.

    Two-stage varlen flash attention (encode latents from nodes, decode nodes from
    latents). See ``topology_flare.FLAREGLTAttention`` for the full tensor-role
    specification and input-conditioned latent routing.
    """

    separate_qk = True
    condition_latents = True

    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        pe_inject_mode: str,
        num_latents: int,
        rmsnorm: bool,
        condition_latents: bool = True,
        latent_conditioning_alpha: float = 0.2,
    ):
        super().__init__(
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=pe_inject_mode,
        )
        self.attn_type = f"flare{num_latents}"
        self.num_latents = int(num_latents)
        self.attn_scale = self.head_dim ** -0.5
        self.condition_latents = bool(condition_latents)
        self.latent_conditioning_alpha = float(latent_conditioning_alpha)
        if self.condition_latents and self.pe_inject_mode != "concat_qk":
            raise ValueError(
                "GLTFlareAttention condition_latents=True requires pe_inject_mode='concat_qk'."
            )

        NormHead = nn.RMSNorm if rmsnorm else nn.LayerNorm
        # Encode / decode latent shifts (q_enc_shift, k_dec_shift).
        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)
        self.latent_k_decode = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_k_decode, mean=0.0, std=0.1)

        self.decode_q_norm = NormHead(self.head_dim, eps=1e-6)
        self.decode_k_norm = NormHead(self.head_dim, eps=1e-6)

        if self.condition_latents:
            self.latent_q_router = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
            nn.init.normal_(self.latent_q_router, mean=0.0, std=0.1)
            self.latent_k_decode_router = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
            nn.init.normal_(self.latent_k_decode_router, mean=0.0, std=0.1)
            self.latent_cond_k_norm = NormHead(self.head_dim, eps=1e-6)
            qk_in_dim = 2 * channel_dim if self.pe_inject_mode == "concat_qk" else channel_dim
            self.latent_cond_k_proj = nn.Linear(qk_in_dim, channel_dim)
            self.latent_cond_v_proj = nn.Linear(qk_in_dim, channel_dim)
            self.alpha_enc = nn.Parameter(torch.full((self.num_heads,), self.latent_conditioning_alpha))
            self.alpha_dec = nn.Parameter(torch.full((self.num_heads,), self.latent_conditioning_alpha))

    def _pack_latent_heads(
        self,
        latent: torch.Tensor,
        *,
        bsz: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``[D, M]`` latent bank -> packed ``[bsz*M, H, d]`` for flash-varlen."""
        q = latent.view(self.num_heads, self.num_latents, self.head_dim)
        q = self.q_norm(q)
        if q.dtype != dtype:
            q = q.to(dtype=dtype)
        return (
            q.unsqueeze(0)
            .expand(bsz, -1, -1, -1)
            .transpose(1, 2)
            .reshape(bsz * self.num_latents, self.num_heads, self.head_dim)
            .contiguous()
        )

    def _pack_decode_latent_heads(
        self,
        latent: torch.Tensor,
        *,
        bsz: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``[D, M]`` decode latent bank -> packed ``[bsz*M, H, d]`` (decode K norm)."""
        k = latent.view(self.num_heads, self.num_latents, self.head_dim)
        k = self.decode_k_norm(k)
        if k.dtype != dtype:
            k = k.to(dtype=dtype)
        return (
            k.unsqueeze(0)
            .expand(bsz, -1, -1, -1)
            .transpose(1, 2)
            .reshape(bsz * self.num_latents, self.num_heads, self.head_dim)
            .contiguous()
        )

    def _condition_latent_queries(
        self,
        latent_router: torch.Tensor,
        latent_shift: torch.Tensor,
        alpha: torch.Tensor,
        latent_cond_k: torch.Tensor,
        latent_cond_v: torch.Tensor,
        *,
        cu_latents: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        renormalize_shift: bool,
    ) -> torch.Tensor:
        """``shift + α * SDPA(router, cond_k, cond_v)`` on the latent axis (FLARE-GLT style)."""
        bsz = int(cu_latents.numel() - 1)
        dtype = latent_cond_k.dtype
        q_router = self._pack_latent_heads(latent_router, bsz=bsz, dtype=dtype)
        q_shift = self._pack_latent_heads(latent_shift, bsz=bsz, dtype=dtype)
        q_delta = flash_attn_varlen_func(
            q_router,
            latent_cond_k,
            latent_cond_v,
            cu_latents,
            cu_seqlens,
            self.num_latents,
            int(max_seqlen),
            dropout_p=0.0,
            softmax_scale=self.attn_scale,
            causal=False,
        )
        q = q_shift + alpha.to(dtype=dtype).view(1, self.num_heads, 1) * q_delta
        if renormalize_shift:
            q = self.q_norm(q.contiguous())
        return q

    def _condition_decode_keys(
        self,
        latent_router: torch.Tensor,
        latent_shift: torch.Tensor,
        alpha: torch.Tensor,
        latent_cond_k: torch.Tensor,
        latent_cond_v: torch.Tensor,
        *,
        cu_latents: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """``k_dec_shift + α * SDPA(k_dec_router, cond_k, cond_v)`` before node decode."""
        bsz = int(cu_latents.numel() - 1)
        dtype = latent_cond_k.dtype
        k_router = self._pack_decode_latent_heads(latent_router, bsz=bsz, dtype=dtype)
        k_shift = self._pack_decode_latent_heads(latent_shift, bsz=bsz, dtype=dtype)
        k_delta = flash_attn_varlen_func(
            k_router,
            latent_cond_k,
            latent_cond_v,
            cu_latents,
            cu_seqlens,
            self.num_latents,
            int(max_seqlen),
            dropout_p=0.0,
            softmax_scale=self.attn_scale,
            causal=False,
        )
        k = k_shift + alpha.to(dtype=dtype).view(1, self.num_heads, 1) * k_delta
        return self.decode_k_norm(k.contiguous())

    def forward(self, x: torch.Tensor, c: torch.Tensor | None, *, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        x, c = _validate_glt_packed_inputs(x, c, name="GLT flare attention", require_flash=True, require_fp16=True)
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        bsz = int(cu_seqlens.numel() - 1)
        cu_latents = _make_cu_latents(bsz, self.num_latents, x.device)

        qk_source = _glt_qk_source(self.pe_inject_mode, x, c, name="GLT flare attention")
        k_enc = self.k_norm(self._reshape_heads(self.k_proj(qk_source))).contiguous()
        v_enc = self._reshape_heads(self.v_proj(x)).contiguous()

        if self.condition_latents:
            latent_cond_k = self.latent_cond_k_norm(
                self._reshape_heads(self.latent_cond_k_proj(qk_source)),
            ).contiguous()
            latent_cond_v = self._reshape_heads(self.latent_cond_v_proj(qk_source)).contiguous()
            q_enc = self._condition_latent_queries(
                self.latent_q_router,
                self.latent_q,
                self.alpha_enc,
                latent_cond_k,
                latent_cond_v,
                cu_latents=cu_latents,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                renormalize_shift=True,
            )
        else:
            q_enc = self._pack_latent_heads(self.latent_q, bsz=bsz, dtype=k_enc.dtype)

        # Encode: z = SDPA(q_enc, k_enc, v_enc)  [M latents per graph]
        z = flash_attn_varlen_func(
            q_enc,
            k_enc,
            v_enc,
            cu_latents,
            cu_seqlens,
            self.num_latents,
            int(max_seqlen),
            dropout_p=0.0,
            softmax_scale=self.attn_scale,
            causal=False,
        )

        q_dec = self.decode_q_norm(
            self.q_norm(self._reshape_heads(self.q_proj(qk_source))).contiguous(),
        )
        if self.condition_latents:
            k_dec = self._condition_decode_keys(
                self.latent_k_decode_router,
                self.latent_k_decode,
                self.alpha_dec,
                latent_cond_k,
                latent_cond_v,
                cu_latents=cu_latents,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            k_dec = self._pack_decode_latent_heads(self.latent_k_decode, bsz=bsz, dtype=z.dtype)

        if q_dec.dtype != z.dtype:
            q_dec = q_dec.to(dtype=z.dtype)

        # Decode: y = SDPA(q_dec, k_dec, z)  [N nodes per graph]
        y = flash_attn_varlen_func(
            q_dec,
            k_dec,
            z,
            cu_seqlens,
            cu_latents,
            int(max_seqlen),
            self.num_latents,
            dropout_p=0.0,
            softmax_scale=self.attn_scale,
            causal=False,
        )
        return self.out(y.reshape(x.shape[0], self.channel_dim)).to(dtype=x.dtype)


def _make_glt_attention(
    *,
    attn_type: str,
    channel_dim: int,
    num_heads: int,
    pe_inject_mode: str,
    rmsnorm: bool,
) -> nn.Module:
    kind, size = _parse_glt_attn_type(attn_type)
    if kind == "mha":
        return GLTMHAAttention(
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=pe_inject_mode,
        )
    if kind == "linear":
        return GLTLinearAttention(
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=pe_inject_mode,
        )
    return GLTFlareAttention(
        channel_dim=channel_dim,
        num_heads=num_heads,
        pe_inject_mode=pe_inject_mode,
        num_latents=int(size),
        rmsnorm=rmsnorm,
        condition_latents=(pe_inject_mode == "concat_qk"),
    )


# ---------------------------------------------------------------------------
# GLT block + model
# ---------------------------------------------------------------------------

class GLTBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        pe_inject_mode: str,
        mlp_ratio: float = 2.0,
        act: str = None,
        rmsnorm: bool = False,
        attn_type: str = "mha",
    ):
        super().__init__()
        Norm = _norm_cls(rmsnorm)
        self.norm1_x = Norm(channel_dim)
        self.norm1_c = Norm(channel_dim)
        self.attn = _make_glt_attention(
            attn_type=attn_type,
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=pe_inject_mode,
            rmsnorm=rmsnorm,
        )
        self.norm2 = Norm(channel_dim)
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, int(channel_dim * mlp_ratio)),
            _activation(act)(),
            nn.Linear(int(channel_dim * mlp_ratio), channel_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if self.attn.pe_inject_mode == "concat_input":
            c_norm = None
        elif c is None:
            raise RuntimeError(f"GLT pe_inject_mode={self.attn.pe_inject_mode!r} requires c to be provided.")
        else:
            c_norm = self.norm1_c(c)
        attn_out = self.attn(self.norm1_x(x), c_norm, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GLT(nn.Module):
    """Topology-conditioned model with separate coordinate/topology streams and configurable Q/K/V conditioning."""

    def __init__(self, config: GLTConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        pos_dim = metadata.get("pos_dim")
        pos_dim = int(pos_dim) if pos_dim is not None else None
        channel_dim = int(config.channel_dim)
        num_blocks = int(config.num_blocks)
        num_heads = int(config.num_heads)
        mlp_ratio = float(config.mlp_ratio)
        act = "gelu" if config.act is None else config.act
        rmsnorm = False if config.rmsnorm is None else bool(config.rmsnorm)
        in_out_act = act if act in {"gelu", "silu"} else "gelu"

        self.in_dim = int(in_dim)
        self.pe_inject_mode = str(config.pe_inject_mode)
        self.pe_update = bool(config.pe_update)
        pe_pos_dim = int(pos_dim) if pos_dim is not None else self.in_dim
        pos_domain = metadata.get("pos_domain")
        if bool(config.pe.to_feature_request().pos_domain) and pos_domain is None:
            raise ValueError(
                "GLT PE requested FeatureRequest.pos_domain=True but metadata['pos_domain'] is missing. "
                "Ensure the train DataLoader PosDomain scan ran before make_model."
            )
        self.pe: GraphPE = build_pe(
            config.pe,
            pos_dim=pe_pos_dim,
            act=in_out_act,
            pos_domain=pos_domain,
        )
        _validate_glt_config(config, pe_out_dim=int(self.pe.out_dim))
        self.channel_dim = channel_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.act = act
        self.rmsnorm = rmsnorm

        block_kwargs = dict(
            channel_dim=channel_dim,
            num_heads=num_heads,
            pe_inject_mode=self.pe_inject_mode,
            mlp_ratio=mlp_ratio,
            act=act,
            rmsnorm=rmsnorm,
            attn_type=str(config.attn_type),
        )
        self.blocks = nn.ModuleList(GLTBlock(**block_kwargs) for _ in range(num_blocks))
        self.out_proj = nn.Linear(channel_dim, out_dim)

        if self.pe_inject_mode == "concat_input":
            self.in_proj = nn.Linear(in_dim + int(self.pe.out_dim), channel_dim)
        else:
            self.x_proj = nn.Linear(self.in_dim, channel_dim)
            self.c_proj = nn.Linear(int(self.pe.out_dim), channel_dim)
            self.blocks_c = (
                nn.ModuleList(GLTBlock(**block_kwargs) for _ in range(num_blocks))
                if self.pe_update
                else nn.ModuleList()
            )

        self.apply(self._init_weights)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        feats: torch.Tensor | None = None,
        use_flash_varlen: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        topology_features: TopologyFeatures = None,
        topology_eigenvalues: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del edge_attr, kwargs
        if not use_flash_varlen:
            raise RuntimeError("GLT only supports packed flash-attn varlen batches.")
        if edge_index is None:
            raise RuntimeError("GLT requires mesh edge_index.")
        if cu_seqlens is None or max_seqlen is None:
            raise RuntimeError("GLT requires cu_seqlens and max_seqlen.")
        if pos.ndim != 2:
            raise RuntimeError(f"GLT expects flat pos [N_tot, C], got {tuple(pos.shape)}.")
        if pos.dtype == torch.float32 and pos.is_cuda and torch.is_autocast_enabled("cuda"):
            pos = pos.to(torch.get_autocast_dtype("cuda"))

        raw_top = self.pe(
            pos=pos,
            edge_index=edge_index,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            num_total_nodes=num_total_nodes,
            topology_features=topology_features,
            topology_eigenvalues=topology_eigenvalues,
            batch_index=batch_index,
        )

        if self.pe_inject_mode == "concat_input":
            x = self.in_proj(torch.cat([graph_node_input(pos, feats), raw_top], dim=-1))
            for block in self.blocks:
                x = block(x, None, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
            return self.out_proj(x)

        x = self.x_proj(graph_node_input(pos, feats))
        c = self.c_proj(raw_top)
        if self.pe_update:
            for block_x, block_c in zip(self.blocks, self.blocks_c, strict=True):
                c = block_c(c, x, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
                x = block_x(x, c, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
        else:
            for block_x in self.blocks:
                x = block_x(x, c, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
        return self.out_proj(x)

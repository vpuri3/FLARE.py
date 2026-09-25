#
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ..distributed.context_parallel import ContextParallelState
from ..distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive
from ..distributed.utils import gather_sequence_tensor

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - mixed precision varlen path fails loudly at runtime.
    flash_attn_varlen_func = None

__all__ = [
    "FLAREModel",
]

@dataclass
class FlareConfig:
    model: str = "flare"
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    out_proj_norm: bool = True
    num_layers_in_out_proj: int = 2
    num_layers_k_proj: int = 3
    num_layers_v_proj: int = 3
    k_proj_mlp_ratio: float = 1.0
    v_proj_mlp_ratio: float = 1.0
    num_layers_ffn: int = 3
    ffn_mlp_ratio: float = 1.0
    qk_norm: bool = False
    attn_scale: str = "one"
    num_latents: int = 64
    encoder_cp_backend: str = "flash"

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
        num_layers_k_proj: int = 3,
        num_layers_v_proj: int = 3,
        k_proj_mlp_ratio: float = 1.0,
        v_proj_mlp_ratio: float = 1.0,
        qk_norm: bool = False,
        rmsnorm: bool = False,
        encoder_cp_backend: str = "flash",
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )
        assert attn_scale > 0.0, f"attn_scale must be greater than 0. Got {attn_scale}."

        self.attn_scale = attn_scale

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        if rmsnorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        else:
            self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

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

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False):
        self.cp_state = cp_state
        self.cp_debug_gather_outputs = bool(cp_debug_gather_outputs)

    def forward(self, x, return_scores: bool = False, mask: torch.Tensor = None):

        # x: [B N C]
        # mask: [B N], True for valid points.

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1)

        #--------------------------------------------#
        mask_enc, mask_dec = self.get_mask(mask)
        use_cp = (self.cp_state is not None) and (self.cp_state.cp_size > 1)
        if use_cp:
            if mask is not None:
                raise NotImplementedError("FLARE context-parallel path does not support padded masks yet.")
            z = self.encoder_cp(q_latent=q, k_local=k, v_local=v, cp_group=self.cp_state.cp_group, scale=self.attn_scale)
            # Flash may return low-precision z while norms left q/k in fp32.
            q = q.to(dtype=z.dtype)
            k = k.to(dtype=z.dtype)
            y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
            if return_scores:
                raise RuntimeError("return_scores is not supported in context-parallel mode.")
            scores = None

            if self.cp_debug_gather_outputs:
                _ = gather_sequence_tensor(y.detach(), self.cp_state, seq_dim=2)
        elif not return_scores:
            z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale, attn_mask=mask_enc)
            y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale, attn_mask=mask_dec)
            scores = None
        else:
            # (1) Compute projection weights
            scores = q @ k.transpose(-2, -1) # [B H M N]
            if mask_enc is not None:
                scores = scores.masked_fill(~mask_enc, torch.finfo(scores.dtype).min)
            W_encode = F.softmax(scores, dim=-1)
            decode_scores = scores.transpose(-2, -1)
            if mask_dec is not None:
                decode_scores = decode_scores.masked_fill(~mask_dec, torch.finfo(decode_scores.dtype).min)
            W_decode = F.softmax(decode_scores, dim=-1)

            # (2) Project to latent sequence
            z = W_encode @ v # [B H M D]

            # (3) Project back to input space
            y = W_decode @ z # [B H N D]
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(dtype=y.dtype)

        return y, scores

    def forward_flash_varlen(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        return_scores: bool = False,
    ):
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "flash-attn is required for mixed-precision FLARE packed varlen attention, but it is not installed."
            )
        if return_scores:
            raise RuntimeError("return_scores is not supported by FLARE packed flash-attn varlen workflow.")
        if self.cp_state is not None and self.cp_state.cp_size > 1:
            raise RuntimeError("FLARE context-parallel path is not supported by packed flash-attn varlen workflow.")
        if not x.is_cuda:
            raise RuntimeError("FLARE packed flash-attn varlen requires CUDA tensors.")
        if x.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        if cu_seqlens is None or max_seqlen is None:
            raise RuntimeError("FLARE packed flash-attn varlen requires cu_seqlens and max_seqlen from the dataloader.")
        if x.ndim != 2:
            raise RuntimeError(f"FLARE packed flash-attn varlen requires flat x [total_tokens, C], got {x.shape}.")

        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        bsz = int(cu_seqlens.numel() - 1)
        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim)
        k = self.k_proj(x).reshape(x.shape[0], self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(x.shape[0], self.num_heads, self.head_dim)
        if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FLARE packed flash-attn varlen requires fp16/bf16 projected activations. "
                f"Got k={k.dtype}, v={v.dtype}."
            )
        k_packed = k.contiguous()
        v_packed = v.contiguous()

        q = self.q_norm(q)
        k_packed = self.k_norm(k_packed)
        if q.dtype != k_packed.dtype:
            q = q.to(dtype=k_packed.dtype)
        q_packed = q.unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2).reshape(
            bsz * self.num_latents,
            self.num_heads,
            self.head_dim,
        ).contiguous()
        cu_latents = torch.arange(
            0,
            (bsz + 1) * self.num_latents,
            self.num_latents,
            device=x.device,
            dtype=torch.int32,
        )

        dropout_p = 0.0
        z_packed = flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_latents,
            cu_seqlens,
            self.num_latents,
            int(max_seqlen),
            dropout_p=dropout_p,
            softmax_scale=self.attn_scale,
            causal=False,
        )
        y_packed = flash_attn_varlen_func(
            k_packed,
            z_packed,
            z_packed,
            cu_seqlens,
            cu_latents,
            int(max_seqlen),
            self.num_latents,
            dropout_p=dropout_p,
            softmax_scale=self.attn_scale,
            causal=False,
        )
        y = y_packed.reshape(y_packed.shape[0], self.channel_dim)
        y = self.out_proj(y)
        return y, None

    def get_mask(self, mask: torch.Tensor = None):
        if mask is None:
            return None, None
        if mask.dim() != 2 or mask.dtype != torch.bool:
            raise ValueError(f"mask must be a boolean tensor with shape [B, N]. Got {mask.shape}, {mask.dtype}.")
        if mask.all():
            return None, None
        # Keep SDPA masks as boolean key masks. The decode pass has valid latent keys,
        # so padded query rows are zeroed after attention instead of using all-false masks.
        mask = mask.contiguous().view(mask.size(0), 1, 1, mask.size(1))
        return mask, None

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
        num_layers_k_proj: int = 3,
        num_layers_v_proj: int = 3,
        k_proj_mlp_ratio: float = 1.0,
        v_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 3,
        ffn_mlp_ratio: float = 1.0,
        qk_norm: bool = False,
        encoder_cp_backend: str = "flash",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = FLARE(
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
            encoder_cp_backend=encoder_cp_backend,
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

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False):
        self.att.set_context_parallel(cp_state=cp_state, cp_debug_gather_outputs=cp_debug_gather_outputs)

    def forward(
        self,
        x,
        return_scores: bool = False,
        mask: torch.Tensor = None,
        use_flash_varlen: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ):
        # x: [B, N, C]

        # x = x + att(norm1(x))
        # x = x + mlp(norm2(x))
        # return x

        if use_flash_varlen:
            _x, scores = self.att.forward_flash_varlen(
                self.norm1(x),
                return_scores=return_scores,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            _x, scores = self.att(self.norm1(x), return_scores=return_scores, mask=mask)
        x = x + _x
        x = x + self.mlp(self.norm2(x))
        if mask is not None:
            if not use_flash_varlen:
                x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x, scores

#======================================================================#
# Final Layer (Perceiver compatibility)
#======================================================================#
class FinalLayer(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        out_dim: int,
        act: str = None,
        num_layers: int = 2,
        hidden_dim: int = None,
        ln: bool = True,
    ):
        if hidden_dim is None:
            hidden_dim = channel_dim
        super().__init__()
        self.ln = nn.LayerNorm(channel_dim) if ln else nn.Identity()
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            act=act,
            input_residual=True,
            output_residual=False,
        )

    def forward(self, x):
        x = self.mlp(self.ln(x))
        return x

#======================================================================#
# MODEL
#======================================================================#
class FLAREModel(nn.Module):
    def __init__(self, config: FlareConfig, metadata=None):
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
        attn_scale = config.attn_scale
        if isinstance(attn_scale, str):
            if attn_scale not in {"sqrt", "one"}:
                raise ValueError(f"Invalid attn_scale: {attn_scale}. Choose from: sqrt, one.")
            head_dim = channel_dim // num_heads
            if head_dim > 16:
                attn_scale = "sqrt"
            attn_scale = (head_dim ** -0.5) if attn_scale == "sqrt" else 1.0
        num_latents = config.num_latents
        num_layers_k_proj = config.num_layers_k_proj
        num_layers_v_proj = config.num_layers_v_proj
        k_proj_mlp_ratio = config.k_proj_mlp_ratio
        v_proj_mlp_ratio = config.v_proj_mlp_ratio
        num_layers_ffn = config.num_layers_ffn
        ffn_mlp_ratio = config.ffn_mlp_ratio
        qk_norm = config.qk_norm
        encoder_cp_backend = config.encoder_cp_backend

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
            FLAREBlock(
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
                encoder_cp_backend=encoder_cp_backend,
            )
            for _ in range(num_blocks)
        ])

        self.initialize_weights()
        self.cp_state: Optional[ContextParallelState] = None

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False):
        self.cp_state = cp_state
        for block in self.blocks:
            block.set_context_parallel(cp_state=cp_state, cp_debug_gather_outputs=cp_debug_gather_outputs)

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

    def forward(
        self,
        x,
        return_scores: bool = False,
        mask: torch.Tensor = None,
        use_flash_varlen: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ):
        # x: [B, N, C]
        # mask: [B, N], True for valid points.

        if return_scores:
            scores = []

        x = self.in_proj(x)
        if mask is not None and not use_flash_varlen:
            if mask.dim() != 2 or mask.dtype != torch.bool:
                raise ValueError(f"mask must be a boolean tensor with shape [B, N]. Got {mask.shape}, {mask.dtype}.")
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        for block in self.blocks:
            x, score = block(
                x,
                return_scores=return_scores,
                mask=mask,
                use_flash_varlen=use_flash_varlen,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_scores:
                scores.append(score)

        x = self.out_proj(x)
        if mask is not None and not use_flash_varlen:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return (x, scores) if return_scores else x

#======================================================================#
#

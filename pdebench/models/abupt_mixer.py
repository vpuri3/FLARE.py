"""Token-mixing primitives used by the surface-only AB-UPT model.

These modules intentionally know nothing about PDEBench batches, geometry
sampling, or model orchestration. They operate on hidden surface tokens and
receive RoPE frequencies explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _validate_rope_head_dim(dim: int, num_heads: int) -> int:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if dim % num_heads:
        raise ValueError("dim must be divisible by num_heads")
    head_dim = dim // num_heads
    if head_dim % 2:
        raise ValueError("head dimension must be even for RoPE")
    return head_dim


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary frequencies to a ``[B, H, N, D]`` tensor."""
    if not torch.is_complex(freqs) or freqs.ndim != 3:
        raise ValueError("freqs must be complex [batch, sequence, head_dim // 2]")
    if not torch.is_floating_point(x) or x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must be floating-point [batch, heads, sequence, head_dim]")
    if freqs.shape[0] != x.shape[0] or freqs.shape[1] != x.shape[2]:
        raise ValueError("freqs batch and sequence dimensions must match x")
    if freqs.shape[2] != x.shape[-1] // 2:
        raise ValueError("freqs head dimension must match x")
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(complex_x * freqs[:, None]).flatten(start_dim=3)
    return rotated.type_as(x)


class ABUPTAnchorAttention(nn.Module):
    """Self-attention for anchors and anchor-only cross-attention for queries."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = _validate_rope_head_dim(dim, num_heads)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs: torch.Tensor,
        num_anchor_tokens: int | None = None,
    ) -> torch.Tensor:
        if num_anchor_tokens is None:
            qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q = rope(q, freqs)
            k = rope(k, freqs)
        else:
            if not 0 < num_anchor_tokens <= x.shape[1]:
                raise ValueError("num_anchor_tokens must be within the sequence")
            anchors = x[:, :num_anchor_tokens]
            queries = x[:, num_anchor_tokens:]
            q_anchor, k, v = self.qkv(anchors).reshape(
                anchors.shape[0], anchors.shape[1], 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            q_bias = self.qkv.bias[: self.dim] if self.qkv.bias is not None else None
            q_query = F.linear(queries, self.qkv.weight[: self.dim], q_bias)
            q_query = q_query.reshape(queries.shape[0], queries.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = torch.cat([q_anchor, q_query], dim=2)
            q = rope(q, freqs)
            k = rope(k, freqs[:, :num_anchor_tokens])
        output = F.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim))


class _Mlp(nn.Module):
    def __init__(self, dim: int, *, num_layers: int = 0, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fcs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        for fc in self.fcs:
            x = self.act(fc(x))
        return self.fc2(x)


class ABUPTTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        kind: str = "s",
        rmsnorm: bool = False,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        norm = nn.RMSNorm(dim, eps=1e-6) if rmsnorm else nn.LayerNorm(dim)
        self.norm1 = norm
        self.attn = ABUPTAnchorAttention(dim, num_heads)
        self.norm2 = type(norm)(dim, eps=1e-6)
        self.mlp = _Mlp(dim, num_layers=num_layers_ffn, mlp_ratio=ffn_mlp_ratio)

    def forward(self, x: torch.Tensor, *, freqs: torch.Tensor, num_anchor_tokens: int | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs=freqs, num_anchor_tokens=num_anchor_tokens)
        return x + self.mlp(self.norm2(x))


__all__ = [
    "ABUPTAnchorAttention",
    "ABUPTTransformerBlock",
    "rope",
]

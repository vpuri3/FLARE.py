#
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'MODEL_TYPES',
]

from .kernels import make_kernel

#======================================================================#
# Vanilla Self-Attention Block
#======================================================================#
class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: str = None,
        drop: float = 0.0,
    ):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU() if act in ['gelu', None] else nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        drop: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % 2 == 0, f"hidden_dim must be even for SwiGLU. Got {hidden_dim}."
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * F.silu(gates)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_heads = channel_dim // 16 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.qkv_proj = nn.Linear(self.channel_dim, 3 * self.channel_dim, bias=True)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):

        B, N, C = x.shape

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # attention_mask: bool [B, 1, N, N]
        attn_mask = (attention_mask.view(B, 1, 1, N) * attention_mask.view(B, 1, N, 1)) if attention_mask is not None else None
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            scale=self.scale,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        y = self.proj_drop(y)

        return y

class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            channel_dim: int,
            num_heads: int = None,
            mlp_ratio: float = 4.0,
            act: str = None,
            rmsnorm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            rope = None,
        ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = MultiHeadedSelfAttention(channel_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
        self.mlp = MLPBlock(in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        # x: [B, N, C]

        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))

        return x

#======================================================================#
# Linformer Attention Block
#======================================================================#
class LinformerAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        seq_len: int,
        k: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        assert channel_dim % num_heads == 0
        self.k = k
        self.max_length = seq_len
        self.rope = rope

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

        # Shared projection over sequence dimension for K and V: [N, k]
        self.E_k = nn.Parameter(torch.randn(seq_len, k) * (self.head_dim ** -0.5))
        self.E_v = nn.Parameter(torch.randn(seq_len, k) * (self.head_dim ** -0.5))

        self.scale = (self.head_dim ** -0.5)

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            raise NotImplementedError("Attention mask is not supported for LinformerAttention.")
        
        B, N, C = x.shape

        q, k, v = rearrange(self.qkv_proj(x), 'b n (h d) -> b h n d', h=self.num_heads).chunk(3, dim=-1)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Project sequence length of K and V: [N,k]
        # If runtime N < max_length, slice; if N > max_length, interpolate by truncation
        E_k = self.E_k[:N]
        E_v = self.E_v[:N]
        # K': [B, H, k, D]
        k_lin = torch.einsum('b h n d, n k -> b h k d', k, E_k)
        v_lin = torch.einsum('b h n d, n k -> b h k d', v, E_v)

        y = F.scaled_dot_product_attention(
            q, k_lin, v_lin, scale=self.scale,
            dropout_p=self.attn_drop_p if self.training else 0.0
        )

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        y = self.proj_drop(y)
        return y

class LinformerBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        seq_len: int,
        k: int = 256,
        mlp_ratio: float = 4.0,
        act: str = None,
        rmsnorm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = LinformerAttention(channel_dim, num_heads, seq_len=seq_len, k=k, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
        self.mlp = MLPBlock(channel_dim, int(channel_dim * mlp_ratio), channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# MEGA: Moving Average Equipped Gated Attention
#======================================================================#
class DampedMultidimEMAConv(nn.Module):
    def __init__(self, channel_dim: int, h: int = 4, kernel_size: int = 2048):
        """
        Multi-dimensional damped EMA per channel (MEGA §3.1), implemented as a
        sum of geometric kernels + grouped Conv1d. Bidirectional (fwd+bwd).
        Input:  x [B, N, C]
        Output: y [B, N, C]
        """
        super().__init__()
        self.h = h
        self.Kmax = kernel_size

        # Parameters per (channel, sub-dimension)
        # alpha, delta, beta in (0,1) via sigmoid; eta is unconstrained
        self.alpha_logits = nn.Parameter(torch.zeros(channel_dim, h))
        self.delta_logits = nn.Parameter(torch.zeros(channel_dim, h))
        self.beta_logits  = nn.Parameter(torch.zeros(channel_dim, h))
        self.eta          = nn.Parameter(torch.randn(channel_dim, h) * 0.02)

        self.proj = nn.Linear(2 * channel_dim, channel_dim)

    def _ema_fwd(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N], w: [C, 1, K] -> y: [B, C, N]
        C, _, K = w.shape
        return F.conv1d(F.pad(x, (K - 1, 0)), w, groups=C)
    
    def _ema_bwd(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self._ema_fwd(x.flip(dims=(-1,)), w).flip(dims=(-1,))
        return x

    def _make_kernel(self, K: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Build per-channel scalar kernel as a sum over h geometric components:
            w_k^(j) = sum_m (eta * alpha * beta) * (1 - alpha*delta)^k
        Returns: w [C, 1, K]
        """
        # Params in [0,1]
        alpha = torch.sigmoid(self.alpha_logits).to(dtype=dtype, device=device)   # [C,h]
        delta = torch.sigmoid(self.delta_logits).to(dtype=dtype, device=device)   # [C,h]
        beta  = torch.sigmoid(self.beta_logits ).to(dtype=dtype, device=device)   # [C,h]
        eta   = self.eta.to(dtype=dtype, device=device)                           # [C,h]

        # geometric ratio r = (1 - alpha * delta) in (0,1)
        r = 1.0 - (alpha * delta) # [C,h]

        k = torch.arange(K, device=device, dtype=dtype) # [K]
        r_pows = r.unsqueeze(-1) ** k                   # [C,h,K]

        # amplitude A = eta * alpha * beta
        A = eta * alpha * beta # [C,h]
        w = A.unsqueeze(-1) * r_pows # [C,h,K]
        # sum over h components
        w = w.sum(dim=1, keepdim=True) # [C,1,K]

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> z: [B, N, 2C]
        """
        dtype = x.dtype
        device = x.device
        B, N, C = x.shape
        K = min(N, self.Kmax)

        w = self._make_kernel(K, dtype=dtype, device=device)   # [C,1,K]

        x = x.mT # [B,C,N]
        x_fwd = self._ema_fwd(x, w)
        x_bwd = self._ema_bwd(x, w)

        x = torch.cat([x_fwd.mT, x_bwd.mT], dim=-1) # [B,N,2C]
        x = self.proj(x) # [B,N,C]

        return x

class DampedMultidimEMACumsum(nn.Module):
    def __init__(self, channel_dim: int, h: int = 4):
        """
        Multi-dimensional damped EMA per channel (MEGA §3.1), implemented as a
        sum of geometric kernels + cumsum. Bidirectional (fwd+bwd).
        Input:  x [B, N, C]
        Output: y [B, N, C]
        """
        super().__init__()
        self.H = h
        self.C = channel_dim

        # Parameters per (channel, sub-dimension)
        # alpha, delta, beta in (0,1) via sigmoid; eta is unconstrained
        self.alpha_logits = nn.Parameter(torch.zeros(self.C, self.H)) # ~0.5
        self.delta_logits = nn.Parameter(torch.zeros(self.C, self.H))
        self.beta_logits  = nn.Parameter(torch.zeros(self.C, self.H))
        self.eta          = nn.Parameter(torch.randn(self.C, self.H) * 0.02)

        self.proj = nn.Linear(2 * self.C, self.C)

    def _ema_bwd(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self._ema_fwd(x.flip(dims=(1,)), *args, **kwargs).flip(dims=(1,))
        return x

    def _ema_fwd(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        eta: torch.Tensor,
        r_pos: torch.Tensor,
        r_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, N, C] -> y: [B, N, C]
        Computes y_t = sum_m eta_m * h_t^{(m)}, with
          h_t^{(m)} = r_m^t * cumsum( A_m * x_t * r_m^{-t}, dim=t )

        """
        # x: [B, N, C], r: [N, C, H], A: [C, H], eta: [C, H]

        x = x.unsqueeze(-1)         # [B, N, C, 1]
        z = A * x * r_neg           # [B, N, C, H]
        h = z.cumsum(dim=1) * r_pos # [B, N, C, H]
        y = (eta * h).sum(dim=-1)   # [B, N, C]

        return y

    def _make_weights(self, N: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        alpha = F.sigmoid(self.alpha_logits).to(dtype=torch.float32, device=device) # [C,H]
        delta = F.sigmoid(self.delta_logits).to(dtype=torch.float32, device=device) # [C,H]
        beta  = F.sigmoid(self.beta_logits ).to(dtype=torch.float32, device=device) # [C,H]
        eta   = self.eta.to(dtype=torch.float32, device=device)                     # [C,H]
        t     = torch.arange(N, dtype=torch.float32, device=device).view(N, 1, 1) # [N,1,1]

        A = alpha * beta          # [C,H]
        r = 1.0 - (alpha * delta) # [C,H]
        r = r.clamp(min=1e-4, max=1-1e-4)
        r = (r.log() * t).clamp(-60, 60).view(N, self.C, self.H) # [N, C, H]
        
        r_pos = torch.exp( r)
        r_neg = torch.exp(-r)

        return A, eta, r_pos, r_neg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> z: [B, N, 2C]
        """
        dtype = x.dtype
        device = x.device
        B, N, C = x.shape

        A, eta, r_pos, r_neg = self._make_weights(N, dtype=dtype, device=device)

        x = x.to(torch.float32)
        x_fwd = self._ema_fwd(x, A, eta, r_pos, r_neg).to(dtype)
        x_bwd = self._ema_bwd(x, A, eta, r_pos, r_neg).to(dtype)

        x = torch.cat([x_fwd, x_bwd], dim=-1) # [B,N,2C]
        x = self.proj(x) # [B,N,C]

        return x

class EMAAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        num_heads = 1

        self.channel_dim = channel_dim
        self.num_heads = channel_dim // 16 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.q_proj = nn.Sequential(nn.Linear(self.channel_dim, self.channel_dim), nn.SiLU())
        self.k_proj = nn.Sequential(nn.Linear(self.channel_dim, self.channel_dim), nn.SiLU())
        self.v_proj = nn.Sequential(nn.Linear(self.channel_dim, self.channel_dim), nn.SiLU())
        # self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

        # self.ema = DampedMultidimEMAConv(self.channel_dim)
        self.ema = DampedMultidimEMACumsum(self.channel_dim)
        self.gate_proj1 = nn.Sequential(nn.Linear(self.channel_dim, self.channel_dim), nn.Sigmoid())
        self.gate_proj2 = nn.Sequential(nn.Linear(self.channel_dim, self.channel_dim), nn.Sigmoid())

        self.Wh = nn.Parameter(torch.randn(channel_dim, channel_dim) * 0.02)
        self.Uh = nn.Parameter(torch.randn(channel_dim, channel_dim) * 0.02)
        self.bh = nn.Parameter(torch.zeros(channel_dim))

        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        x_ema = self.ema(x)    # [B, N, C]
        # x_ema = x
        q = self.q_proj(x_ema)
        k = self.k_proj(x_ema)
        v = self.v_proj(x)

        y_attn = self.mha(q, k, v, attention_mask=attention_mask) # [B, N, C]

        gate1 = self.gate_proj1(x_ema) # [B, N, C]
        gate2 = self.gate_proj2(x_ema) 

        H = F.silu(x_ema @ self.Wh + (y_attn * gate1) @ self.Uh + self.bh)
        y = (H * gate2) + (1 - gate2) * x

        # y = self.out_proj(y)
        # y = self.proj_drop(y)

        return y

    def mha(self, q, k, v, attention_mask=None):
        B, N, C = q.shape
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        # attention_mask: bool [B, 1, N, N]
        attn_mask = (attention_mask.view(B, 1, 1, N) * attention_mask.view(B, 1, N, 1)) if attention_mask is not None else None

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            scale=self.scale,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        y = rearrange(y, 'b h n d -> b n (h d)')

        return y

class EMABlock(nn.Module):
    def __init__(
            self,
            channel_dim: int,
            num_heads: int = None,
            mlp_ratio: float = 4.0,
            act: str = None,
            rmsnorm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            rope = None,
        ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = EMAAttention(channel_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
        self.mlp = MLPBlock(in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        # x: [B, N, C]

        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))

        return x

#======================================================================#
# FLARE
#======================================================================#
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}

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

class FLARE(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        act: str = None,
        attn_scale: float = 1.0,
        num_layers_kv_proj: int = 3,
        kv_proj_hidden_dim: int = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()

        assert attn_scale > 0.0, f"attn_scale must be greater than 0. Got {attn_scale}."

        self.attn_scale = attn_scale
        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads
        self.rope = rope

        self.attn_drop_p = attn_drop
        self.proj_drop_p = proj_drop

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [
            ResidualMLP(
                in_dim=self.channel_dim,
                hidden_dim=kv_proj_hidden_dim,
                out_dim=self.channel_dim,
                num_layers=num_layers_kv_proj,
                act=act,
                input_residual=True,
                output_residual=True,
            ) for _ in range(2)
        ]

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x, attention_mask=None):

        # x: [B N C]

        drop_attn_p = self.attn_drop_p if self.training else 0.0
        drop_proj_p = self.proj_drop_p if self.training else 0.0

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        
        if self.rope is not None:
            k = self.rope(k)

        #--------------------------------------------#
        mask_enc, mask_dec = self.get_mask(attention_mask)
        q = q.unsqueeze(0).expand(k.size(0), -1, -1, -1) # required for fused attention
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_enc, scale=self.attn_scale, dropout_p=drop_attn_p)
        y = F.scaled_dot_product_attention(k, q, z, attn_mask=mask_dec, scale=self.attn_scale, dropout_p=drop_attn_p)
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        y = F.dropout(y, p=drop_proj_p, inplace=True)

        return y

    @staticmethod
    def get_mask(mask: torch.Tensor = None):
        if mask is None:
            return None, None

        B, N = mask.shape
        mask = mask.view(B, 1, 1, N) # broadcastable to [B, H, M, N]
        return mask, mask.mT

class FLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        act: str = None,
        rmsnorm: bool = False,
        attn_scale: float = 1.0,
        num_layers_kv_proj: int = 3,
        num_layers_ffn: int = 3,
        kv_proj_hidden_dim: int = 1.0,
        ffn_hidden_dim: int = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = FLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            act=act,
            attn_scale=attn_scale,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_hidden_dim=kv_proj_hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=ffn_hidden_dim,
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        # x: [B, N, C]

        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))

        return x

#======================================================================#
# Linear Attention (Performer-style approximation)
#
# matrix form
# Y = row_norm(Q * K^T) * V = Q @ (K^T @ V) / Q @ (K^T @ 1)
# vector form (sums are over sequence dimension)
# yi = num / den
# num = Sum_j dot(qi, kj) * vj = dot(Sum_j(vj * kj^T), qi)
# den = Sum_j dot(qi, kj) = dot(Sum_j(kj^T), qi)
#======================================================================#
class LinearAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        kernel: str = 'silu',
        norm_q: bool = False,
        norm_k: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        assert channel_dim % num_heads == 0
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.kernel = make_kernel(kernel, head_dim=self.head_dim)

        self.norm_q = norm_q
        self.norm_k = norm_k

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        q, k, v = rearrange(self.qkv_proj(x), 'b n (h d) -> b h n d', h=self.num_heads).chunk(3, dim=-1)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        q = self.kernel(q)
        k = self.kernel(k)
        
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else q
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else k

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            k = k * mask
            v = v * mask

        #=========================#
        state = k.mT @ v                   # [B, H, D, D]
        k_sum = k.sum(dim=2).unsqueeze(-1) # [B, H, 1, D]

        num = q @ state # [B, H, N, D]
        den = q @ k_sum # [B, H, N, 1]
        out = num / (den + 1e-6)
        #=========================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

class LinearAttentionBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        mlp_ratio: float = 4.0,
        kernel: str = 'silu',
        norm_q: bool = False,
        norm_k: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = LinearAttention(channel_dim, num_heads, kernel=kernel, norm_q=norm_q, norm_k=norm_k, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
        self.mlp = MLPBlock(channel_dim, int(channel_dim * mlp_ratio), channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# Multilinear Attention
#======================================================================#
class MultilinearAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        num_states: int = 2,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):

        # softmax over gates and
        # state = prod([(1 - gate) * state for (gate, state) in zip(gates, states)])

        # softmax over gates for each head?
        # would that make the states go to zero?
        # multiplicative gating with addition of states makes sense.
        # apply phi: R^d -> R^2d
        # layernorm on states before mul with q?

        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(self.head_dim * qk_dim_ratio)
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_states = num_states
        assert num_states > 0, f"num_states must be greater than 0. Got {num_states}."

        res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )

        self.q_proj = ResidualMLP(**res_mlp_kws)
        self.k_projs = nn.ModuleList([ResidualMLP(**res_mlp_kws) for _ in range(num_states)])
        self.v_projs = nn.ModuleList([ResidualMLP(**res_mlp_kws) for _ in range(num_states)])

        self.kernel = make_kernel(kernel, head_dim=self.head_dim, qk_dim=self.qk_dim)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

    def forward(self, x, attention_mask=None):

        dtype = x.dtype
        B, N, C = x.shape
        H = self.num_heads
        K = self.num_states
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q = self.q_proj(x)
        ks = torch.stack([k_proj(x) for k_proj in self.k_projs], dim=0)  # [K, B, N, C]
        vs = torch.stack([v_proj(x) for v_proj in self.v_projs], dim=0)  # [K, B, N, C]

        q = rearrange(q, 'b n (h d) -> b h n d', h=H)
        ks = rearrange(ks, 'k b n (h d) -> k b h n d', h=H)
        vs = rearrange(vs, 'k b n (h d) -> k b h n d', h=H)

        q = self.kernel(q)
        ks = self.kernel(ks)

        # normalize
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else q
        ks = ks / (ks.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else ks

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            q = q * mask
            ks = ks * mask
            vs = vs * mask

        # # apply gates
        # k_gates = self.k_gates(x).sigmoid().view(1, B, N, H, K).permute(4, 1, 3, 2, 0)
        # v_gates = self.v_gates(x).sigmoid().view(1, B, N, H, K).permute(4, 1, 3, 2, 0)

        # ks = ks * k_gates
        # vs = vs * v_gates

        #============================#
        scale_factor = 1.0 / math.sqrt(N) # for stability
        ks = ks.to(torch.float32) * scale_factor
        vs = vs.to(torch.float32) * scale_factor

        states = ks.mT @ vs
        state = states.prod(dim=0)

        out = q.to(torch.float32) @ state # [B, H, N, D]
        out = out.to(dtype)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

class MultilinearBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        num_states: int = 2,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = MultilinearAttention(
            channel_dim, num_heads, act=act, rmsnorm=rmsnorm,
            num_states=num_states, num_layers_kv_proj=num_layers_kv_proj, kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            kernel=kernel, norm_q=norm_q, norm_k=norm_k, qk_dim_ratio=qk_dim_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=int(channel_dim * ffn_mlp_ratio), out_dim=channel_dim,
            num_layers=num_layers_ffn, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# Linearized Strassen Attention
#======================================================================#
class StrassenAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(self.head_dim * qk_dim_ratio)
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )

        self.q_proj  = ResidualMLP(**res_mlp_kws)
        self.k1_proj = ResidualMLP(**res_mlp_kws)
        self.k2_proj = ResidualMLP(**res_mlp_kws)
        self.v1_proj = ResidualMLP(**res_mlp_kws)
        self.v2_proj = ResidualMLP(**res_mlp_kws)

        self.kernel = make_kernel(kernel, head_dim=self.head_dim, qk_dim=self.qk_dim)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

        # gates
        # do softmax over gates for each head?
        self.g1 = nn.Parameter(torch.zeros(num_heads))  # scales y1
        self.g2 = nn.Parameter(torch.zeros(num_heads))  # scales y2
        self.g3 = nn.Parameter(torch.zeros(num_heads))  # scales y3
        self.g4 = nn.Parameter(torch.zeros(num_heads))  # scales y3

    def forward(self, x, attention_mask=None):

        dtype = x.dtype
        B, N, C = x.shape
        H = self.num_heads
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q = self.q_proj(x)
        k1 = self.k1_proj(x)
        k2 = self.k2_proj(x)
        v1 = self.v1_proj(x)
        v2 = self.v2_proj(x)

        q, k1, k2, v1, v2 = [rearrange(z, 'b n (h d) -> b h n d', h=H) for z in [q, k1, k2, v1, v2]]

        # kernel
        q, k1, k2 = [self.kernel(z) for z in [q, k1, k2]]

        # normalize
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else q
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else z for z in [k1, k2]]

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            q = q * mask
            k1 = k1 * mask
            k2 = k2 * mask
            v1 = v1 * mask
            v2 = v2 * mask

        # gates
        g1 = self.g1.view(1, H, 1, 1)
        g2 = self.g2.view(1, H, 1, 1)
        g3 = self.g3.view(1, H, 1, 1)
        g4 = self.g4.view(1, H, 1, 1)

        #============================#
        q, k1, k2, v1, v2 = [k.to(torch.float32) for k in [q, k1, k2, v1, v2]]
        sN = N ** (1/2)
        S1 = (k1.mT / sN) @ (v1 / sN) # [B H D D]
        S2 = (k2.mT / sN) @ (v2 / sN) # [B H D D]

        v1_sum = v1.mean(dim=-2, keepdim=True) # [B H 1 D] == (v1 / N).sum(dim=-2)
        v2_sum = v2.mean(dim=-2, keepdim=True) # [B H 1 D]

        y1 = (q @ S1) * v2_sum                   # [B H N D]
        y2 = (S1 * S2).sum(dim=-2, keepdim=True) # [B H 1 D]
        y3 = (q @ S2) * v1_sum                   # [B H N D]
        y4 = (q @ (S1 * S2))                     # [B H N D] (multiplicative term)

        # out = y1 + y2 + y3 # [B H N D] # OG
        out = y1 * g1 + y2 * g2 + y3 * g3 + y4 * g4 # [B H N D]
        out = out.to(dtype)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

class StrassenBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = StrassenAttention(
            channel_dim, num_heads, act=act, rmsnorm=rmsnorm,
            num_layers_kv_proj=num_layers_kv_proj, kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            kernel=kernel, norm_q=norm_q, norm_k=norm_k, qk_dim_ratio=qk_dim_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=int(channel_dim * ffn_mlp_ratio), out_dim=channel_dim,
            num_layers=num_layers_ffn, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# LinearNO
#======================================================================#
class LinearNO(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)

        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

    def forward(self, x, attention_mask=None):

        B, N, C = x.shape
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."
        assert attention_mask is None, f"Attention mask is not supported by {self.__class__.__name__}."

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        q = q.softmax(dim=-1) # [B, H, N, M]
        k = k.softmax(dim=-2)

        #============================#
        state = k.mT @ v # [B, H, D, D]
        out = q @ state # [B, H, N, D]
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

class LinearNOBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        qk_dim_ratio: float = 1.0,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = LinearNO(
            channel_dim, num_heads, act=act, rmsnorm=rmsnorm,
            num_layers_kv_proj=num_layers_kv_proj, kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            qk_dim_ratio=qk_dim_ratio, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=int(channel_dim * ffn_mlp_ratio), out_dim=channel_dim,
            num_layers=num_layers_ffn, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# TripleAttention
# higher order state (D x D x D)
# might provide better state vs parameter tradeoff.
#======================================================================#
class TripleAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        use_triton: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(self.head_dim * qk_dim_ratio)
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )
        self.q1_proj = ResidualMLP(**res_mlp_kws)
        self.q2_proj = ResidualMLP(**res_mlp_kws)
        self.k1_proj = ResidualMLP(**res_mlp_kws)
        self.k2_proj = ResidualMLP(**res_mlp_kws)
        self.v_proj  = ResidualMLP(**res_mlp_kws)

        self.kernel = make_kernel(kernel, head_dim=self.head_dim, qk_dim=self.qk_dim)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

        from .triton.triple import TripleAttentionFunction
        self.attn = TripleAttentionFunction.apply if use_triton else self.attn_einsum

    def forward(self, x, attention_mask=None):

        B, N, C = x.shape
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q1 = self.q1_proj(x)
        q2 = self.q2_proj(x)
        k1 = self.k1_proj(x)
        k2 = self.k2_proj(x)
        v  = self.v_proj(x)

        q1, q2, k1, k2, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q1, q2, k1, k2, v]]

        # kernel
        q1, q2, k1, k2 = [self.kernel(z) for z in [q1, q2, k1, k2]]

        # normalize
        q1, q2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else z for z in [q1, q2]]
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else z for z in [k1, k2]]

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            q1 = q1 * mask
            q2 = q2 * mask
            k1 = k1 * mask
            k2 = k2 * mask
            v = v * mask

        #============================#
        _, out = self.attn(q1, q2, k1, k2, v)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    def attn_einsum(self, q1, q2, k1, k2, v):

        N = q1.size(-2)
        k1, k2, v = [k.to(torch.float32) / (N ** (1/3)) for k in [k1, k2, v]]
        state = torch.einsum('b h n i, b h n j, b h n k -> b h i j k', k1, v, k2)   # [B H D D D]
        out = torch.einsum('b h n i, b h i j k, b h n k -> b h n j', q1, state, q2) # [B H N D]
        out = out.to(q1.dtype)
        return state, out

class TripleBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        use_triton: bool = False,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = TripleAttention(
            channel_dim, num_heads, act=act, rmsnorm=rmsnorm,
            num_layers_kv_proj=num_layers_kv_proj, kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            kernel=kernel, norm_q=norm_q, norm_k=norm_k, qk_dim_ratio=qk_dim_ratio, use_triton=use_triton,
            attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=int(channel_dim * ffn_mlp_ratio), out_dim=channel_dim,
            num_layers=num_layers_ffn, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class Triple1Attention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        rmsnorm: bool = False,
        qk_dim_ratio: float = 1.0,
        use_triton: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_head_dim = int(self.head_dim * qk_dim_ratio)
        self.qk_channel_dim = int(channel_dim * qk_dim_ratio)
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.q1_proj = SwiGLUFFN(in_dim=channel_dim, hidden_dim=2 * self.qk_channel_dim, out_dim=self.qk_channel_dim)
        # self.q2_proj = SwiGLUFFN(in_dim=channel_dim, hidden_dim=2 * self.qk_channel_dim, out_dim=self.qk_channel_dim)
        # self.k1_proj = SwiGLUFFN(in_dim=channel_dim, hidden_dim=2 * self.qk_channel_dim, out_dim=self.qk_channel_dim)
        # self.k2_proj = SwiGLUFFN(in_dim=channel_dim, hidden_dim=2 * self.qk_channel_dim, out_dim=self.qk_channel_dim)
        # self.v_proj = nn.Linear(channel_dim, channel_dim)

        #======================================================================#
        # IDEAS
        #======================================================================#
        # Use separate kernels per head and for q1/k1, q2/k2
        # Add activation at end of q/k_proj or at start of kernel.
        # Try swiglu with qk_dim_ratio = 2, 3, ...
        # Remove proj_mlp and learn separate swiglu kernels for q, k, v (separate for each head)

        #########

        # res_mlp_kws = dict(
        #     in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
        #     num_layers=-1, act=None, input_residual=True, output_residual=True
        # )
        # self.q1_proj = ResidualMLP(**res_mlp_kws)
        # self.q2_proj = ResidualMLP(**res_mlp_kws)
        # self.k1_proj = ResidualMLP(**res_mlp_kws)
        # self.k2_proj = ResidualMLP(**res_mlp_kws)
        # self.v_proj  = ResidualMLP(**res_mlp_kws)

        # self.kernel = make_kernel(kernel='swiglu', head_dim=self.head_dim, qk_dim=self.qk_head_dim)

        # self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        # self.attn = TripleAttentionFunction.apply if use_triton else self.attn_einsum

        #########

        res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=3, act='gelu', input_residual=True, output_residual=True
        )
        self.q1_proj = ResidualMLP(**res_mlp_kws)
        self.q2_proj = ResidualMLP(**res_mlp_kws)
        self.k1_proj = ResidualMLP(**res_mlp_kws)
        self.k2_proj = ResidualMLP(**res_mlp_kws)
        self.v_proj  = ResidualMLP(**res_mlp_kws)

        self.kernel = make_kernel(kernel='identity', head_dim=self.head_dim, qk_dim=self.qk_head_dim)

        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.attn = TripleAttentionFunction.apply if use_triton else self.attn_einsum

    def forward(self, x, attention_mask=None):

        B, N, C = x.shape
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q1 = self.q1_proj(x)
        q2 = self.q2_proj(x)
        k1 = self.k1_proj(x)
        k2 = self.k2_proj(x)
        v  = self.v_proj(x)

        q1, q2, k1, k2, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q1, q2, k1, k2, v]]

        q1, q2, k1, k2 = [self.kernel(z) for z in [q1, q2, k1, k2]]

        # normalize
        q1, q2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) for z in [q1, q2]]
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) for z in [k1, k2]]

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            q1, q2, k1, k2, v = [z * mask for z in [q1, q2, k1, k2, v]]

        #============================#
        _, out = self.attn(q1, q2, k1, k2, v)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    def attn_einsum(self, q1, q2, k1, k2, v):

        N = q1.size(-2)
        k1, k2, v = [k.to(torch.float32) / (N ** (1/3)) for k in [k1, k2, v]]
        state = torch.einsum('b h n i, b h n j, b h n k -> b h i j k', k1, v, k2)   # [B H D D D]
        out = torch.einsum('b h n i, b h i j k, b h n k -> b h n j', q1, state, q2) # [B H N D]
        out = out.to(q1.dtype)
        return state, out

class Triple1Block(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        mlp_ratio: float = 4.0,
        qk_dim_ratio: float = 1.0,
        use_triton: bool = False,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = Triple1Attention(
            channel_dim, num_heads, rmsnorm=rmsnorm,
            qk_dim_ratio=qk_dim_ratio, use_triton=use_triton,
            attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )

        self.mlp = MLPBlock(
            in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim,
            act=act, drop=proj_drop,
        ) if act not in ['swiglu',] else SwiGLUFFN(
            in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim,
            drop=proj_drop,
        )
        # self.mlp = ResidualMLP(
        #     in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim,
        #     num_layers=0, act=act, input_residual=True, output_residual=True,
        # )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# QuadAttention
# higher order state (D x D x D x D)
# might provide better state vs parameter tradeoff.
#======================================================================#
class QuadAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(self.head_dim * qk_dim_ratio)
        self.rope = rope

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )

        self.q1_proj = ResidualMLP(**res_mlp_kws)
        self.q2_proj = ResidualMLP(**res_mlp_kws)
        self.q3_proj = ResidualMLP(**res_mlp_kws)
        self.k1_proj = ResidualMLP(**res_mlp_kws)
        self.k2_proj = ResidualMLP(**res_mlp_kws)
        self.k3_proj = ResidualMLP(**res_mlp_kws)
        self.v_proj  = ResidualMLP(**res_mlp_kws)

        self.kernel = make_kernel(kernel, head_dim=self.head_dim, qk_dim=self.qk_dim)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)

    def forward(self, x, attention_mask=None):

        dtype = x.dtype
        B, N, C = x.shape
        H = self.num_heads
        assert self.rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q1 = self.q1_proj(x)
        q2 = self.q2_proj(x)
        q3 = self.q3_proj(x)
        k1 = self.k1_proj(x)
        k2 = self.k2_proj(x)
        k3 = self.k3_proj(x)
        v  = self.v_proj(x)

        q1, q2, q3, k1, k2, k3, v = [rearrange(z, 'b n (h d) -> b h n d', h=H) for z in [q1, q2, q3, k1, k2, k3, v]]

        # kernel
        q1, q2, q3, k1, k2, k3 = [self.kernel(z) for z in [q1, q2, q3, k1, k2, k3]]

        # normalize
        q1, q2, q3 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else z for z in [q1, q2, q3]]
        k1, k2, k3 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else z for z in [k1, k2, k3]]

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1)
            q1 = q1 * mask
            q2 = q2 * mask
            q3 = q3 * mask
            k1 = k1 * mask
            k2 = k2 * mask
            k3 = k3 * mask
            v = v * mask

        #============================#
        k1, k2, k3, v = [k.to(torch.float32) / (N ** (1/4)) for k in [k1, k2, k3, v]]
        state = torch.einsum('b h n i, b h n j, b h n k, b h n l -> b h i j k l', k1, v, k2, k3)   # [B H D D D D]
        out = torch.einsum('b h n i, b h i j k l, b h n k, b h n l -> b h n j', q1, state, q2, q3) # [B H N D]
        out = out.to(dtype)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

class QuadBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        #
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        #
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = QuadAttention(
            channel_dim, num_heads, act=act, rmsnorm=rmsnorm,
            num_layers_kv_proj=num_layers_kv_proj, kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            kernel=kernel, norm_q=norm_q, norm_k=norm_k, qk_dim_ratio=qk_dim_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop, rope=rope,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=int(channel_dim * ffn_mlp_ratio), out_dim=channel_dim,
            num_layers=num_layers_ffn, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# StrassenFull Attention
#======================================================================#
class ThirdOrderAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
        third_order_method: str = 'third_order',
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        assert channel_dim % num_heads == 0
        self.rope = rope

        # third order attn method
        self.scale = (self.head_dim ** -0.5)
        self.third_order_method = third_order_method
        self.att_proj = nn.Linear(channel_dim, 5 * channel_dim)
        
        assert third_order_method in ['strassen', 'third_order'], f"Invalid third order method: {third_order_method}. Must be one of: strassen, third_order."

        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def strassen_fwd(self, x, attention_mask=None):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        a, b, c, v1, v2 = rearrange(self.att_proj(x), 'b n (h d) -> b h n d', h=H).chunk(5, dim=-1)

        X = (a @ b.mT) * self.scale # ij
        Y = (b @ c.mT) * self.scale # jk
        Z = (c @ a.mT) * self.scale # ki

        attn_mask = (attention_mask.view(B, 1, 1, N) * attention_mask.view(B, 1, N, 1)) if attention_mask is not None else None
        if attn_mask is not None:
            [X, Y, Z] = [z.masked_fill(~attn_mask, float('-inf')) for z in [X, Y, Z]]

        X = (X - X.amax(dim=-1, keepdim=True)).exp()        # max over j
        Y = (Y - Y.amax(dim=(-1, -2), keepdim=True)).exp()  # max over j, k
        Z = (Z - Z.amax(dim=-2, keepdim=True)).exp()        # max over k

        [X, Y, Z] = [self.attn_drop(z) for z in [X, Y, Z]]

        V = v1.view(B, H, N, 1, D) * v2.view(B, H, 1, N, D) # [B, H, N, N, D]

        T = torch.einsum("b h i j, b h j k, b h j k d -> b h i k d", X, Y, V) # [B,H,N,N,Dh]
        up = torch.einsum("b h i k d, b h k i -> b h i d", T, Z)              # [B,H,N,Dh]
        D  = torch.einsum("b h i j, b h j k -> b h i k", X, Y)                # [B,H,N,N]
        down = torch.einsum("b h i k, b h k i -> b h i", D, Z) + 1e-6         # [B,H,N]
        y = up / down.unsqueeze(-1)                                           # [B,H,N,Dh]

        return y

    def third_order_fwd(self, x, attention_mask=None):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        N2 = N * N

        qi, kj, kk, vj, vk = rearrange(self.att_proj(x), 'b n (h d) -> b h n d', h=H).chunk(5, dim=-1)
        scores = torch.einsum("b h i d, b h j d, b h k d -> b h i j k", qi, kj, kk) * self.scale # [B, H, N, N, N]
        weights = scores.flatten(-2).softmax(dim=-1).reshape_as(scores)
        y = torch.einsum("b h i j k, b h j d, b h k d -> b h i d", weights, vj, vk) # [B, H, N, D]
        return y

    def forward(self, x, attention_mask=None):

        if self.rope is not None:
            raise NotImplementedError("Rope is not supported by ThirdOrderAttention.")

        if self.third_order_method == 'strassen':
            y = self.strassen_fwd(x, attention_mask=attention_mask)
        elif self.third_order_method == 'third_order':
            y = self.third_order_fwd(x, attention_mask=attention_mask)
        elif self.third_order_method == 'triangle':
            y = self.triangle_fwd(x, attention_mask=attention_mask)
        else:
            raise NotImplementedError(f"Third order method {self.third_order_method} not implemented.")

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        y = self.proj_drop(y)

        return y

class ThirdOrderAttentionBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        act: str = None,
        rmsnorm: bool = False,
        third_order_method: str = 'third_order',
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = ThirdOrderAttention(channel_dim, num_heads, third_order_method=third_order_method, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
        self.mlp = MLPBlock(channel_dim, int(channel_dim * mlp_ratio), channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

#======================================================================#
# Performer Attention
#======================================================================#

def _draw_gaussian_projection_matrix(num_heads: int, nb_features: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates a Gaussian random projection matrix for FAVOR+."""
    matrix = torch.randn(num_heads, nb_features, head_dim, device=device, dtype=dtype)
    # Normalize rows for numerical stability.
    matrix = torch.nn.functional.normalize(matrix, dim=-1)
    return matrix

class PerformerAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        nb_features: int = 256,
        redraw_interval: int = 0,
        normalize_inputs: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.nb_features = nb_features
        self.redraw_interval = redraw_interval
        self.normalize_inputs = normalize_inputs
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.rope = rope

        self.eps = 1e-6
        self.data_normalizer = (self.head_dim ** -0.25) if self.normalize_inputs else 1.0

        proj = _draw_gaussian_projection_matrix(num_heads, nb_features, self.head_dim, device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('proj_matrix', proj, persistent=False)
        self.register_buffer('_feature_redraw_counter', torch.zeros(1, dtype=torch.long), persistent=False)

    def _maybe_redraw_features(self):
        if self.redraw_interval <= 0 or not self.training:
            return
        self._feature_redraw_counter += 1
        if self._feature_redraw_counter.item() % self.redraw_interval == 0:
            with torch.no_grad():
                new_proj = _draw_gaussian_projection_matrix(
                    self.num_heads, self.nb_features, self.head_dim,
                    device=self.proj_matrix.device, dtype=self.proj_matrix.dtype
                )
                self.proj_matrix.copy_(new_proj)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D]
        proj = self.proj_matrix.to(device=x.device)
        x = x.to(torch.float32)
        proj = proj.to(torch.float32)
        x = x * self.data_normalizer
        x_proj = torch.einsum('b h n d, h m d -> b h n m', x, proj)
        # Stabilize exponentials
        squared_norms = (x.pow(2).sum(dim=-1, keepdim=True)) / 2.0
        x_proj = x_proj - squared_norms
        max_val, _ = torch.max(x_proj, dim=-1, keepdim=True)
        x_proj = x_proj - max_val
        features = torch.exp(x_proj) + self.eps
        features = features / math.sqrt(self.nb_features)
        return features

    def forward(self, x, attention_mask=None):
        # x: [B, N, C]
        self._maybe_redraw_features()

        B, N, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        out_dtype = q.dtype
        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)
        v = v.to(torch.float32)

        # Apply attention mask if provided (after feature map transformation)
        if attention_mask is not None:
            mask = attention_mask.view(B, 1, N, 1).to(torch.float32)
            # Mask k_prime [B, H, N, M] and v [B, H, N, D] to exclude masked positions
            k_prime = k_prime * mask
            v = v * mask

        k_sum = k_prime.sum(dim=2)  # [B, H, M]
        kv = torch.einsum('b h n m, b h n d -> b h m d', k_prime, v)
        numerator = torch.einsum('b h n m, b h m d -> b h n d', q_prime, kv)
        denominator = torch.einsum('b h n m, b h m -> b h n', q_prime, k_sum) + self.eps
        out = numerator / denominator.unsqueeze(-1)
        out = self.attn_drop(out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out.to(out_dtype)

class PerformerBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        nb_features: int = 256,
        redraw_interval: int = 0,
        normalize_inputs: bool = True,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = PerformerAttention(
            channel_dim=channel_dim,
            num_heads=num_heads,
            nb_features=nb_features,
            redraw_interval=redraw_interval,
            normalize_inputs=normalize_inputs,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.mlp = MLPBlock(in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act, drop=proj_drop)

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


#======================================================================#
MODEL_TYPES = {
    'transformer': SelfAttentionBlock,
    'flare': FLAREBlock,
    'linformer': LinformerBlock,
    'linear': LinearAttentionBlock,
    'multilinear': MultilinearBlock,
    'triple': TripleBlock,
    'triple1': Triple1Block,
    'quad': QuadBlock,
    'strassen': StrassenBlock,
    'ema': EMABlock,
    'third_order': ThirdOrderAttentionBlock,
    'performer': PerformerBlock

}

#======================================================================#
#

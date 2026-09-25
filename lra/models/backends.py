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
from .transolver import TransolverBlock

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
        share_kv: bool = False,
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
        self.share_kv = share_kv
        self.rope = rope

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

        self.E_k = nn.Parameter(torch.randn(seq_len, k) * (self.head_dim ** -0.5))
        self.E_v = self.E_k if self.share_kv else nn.Parameter(torch.randn(seq_len, k) * (self.head_dim ** -0.5))

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
        share_kv: bool = False,
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
        self.att = LinformerAttention(
            channel_dim,
            num_heads,
            seq_len=seq_len,
            k=k,
            share_kv=share_kv,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
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
        q_norm: bool = False,
        k_norm: bool = False,
        num_layers_kv_proj: int = 3,
        kv_proj_hidden_dim: int = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
        rmsnorm: bool = False,
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

        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.q_norm = Norm(self.head_dim) if q_norm else nn.Identity()
        self.k_norm = Norm(self.head_dim) if k_norm else nn.Identity()

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

        q = self.q_norm(q)
        k = self.k_norm(k)

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
        q_norm: bool = False,
        k_norm: bool = False,
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
            q_norm=q_norm,
            k_norm=k_norm,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_hidden_dim=kv_proj_hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
            rmsnorm=rmsnorm,
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
# FLARE++ mixer helpers (match pdebench.models.flarepp, no CP)
#======================================================================#
def _flarepp_init_latent_queries(latent_q: nn.Parameter) -> None:
    """Initialize latent_q with shape [H, M, D] via N(0, 0.02)."""
    nn.init.normal_(latent_q, mean=0.0, std=0.02)


def _flarepp_make_head_norm(
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


def _flarepp_make_residual_linear_proj(channel_dim: int) -> nn.Linear:
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
# FLARE++ (input-dependent queries)
#======================================================================#
class FLAREPP(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        rmsnorm: bool = False,
        q_fixed_norm: bool = True,
        gate_logit_init: float = 0.25,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
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
        self.rope = rope
        self.attn_drop_p = attn_drop
        self.proj_drop_p = proj_drop

        assert self.channel_dim % self.num_heads == 0, (
            f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        )

        self.attn_scale = self.head_dim ** -0.5

        self.latent_q0 = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _flarepp_init_latent_queries(self.latent_q0)

        self.gate_logit = nn.Parameter(torch.full((self.num_heads,), gate_logit_init))
        self.latent_q_fixed = nn.Parameter(torch.empty(self.num_heads, self.num_latents, self.head_dim))
        _flarepp_init_latent_queries(self.latent_q_fixed)

        # Hop-1 norms always on: q0 keeps affine; k0 matches v0 (no affine).
        self.q0_norm = _flarepp_make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=True
        )
        self.k0_norm = _flarepp_make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=False
        )
        self.v0_norm = _flarepp_make_head_norm(
            self.head_dim, enabled=True, rmsnorm=rmsnorm, elementwise_affine=False
        )
        self.k_norm = _flarepp_make_head_norm(
            self.head_dim, enabled=k_norm, rmsnorm=rmsnorm, elementwise_affine=True
        )
        self.q_fixed_norm = _flarepp_make_head_norm(
            self.head_dim, enabled=q_fixed_norm, rmsnorm=rmsnorm, elementwise_affine=False
        )

        self.k0_proj = _flarepp_make_residual_linear_proj(self.channel_dim)
        self.v0_proj = (
            _flarepp_make_residual_linear_proj(self.channel_dim)
            if not self.share_k0_v0
            else None
        )
        self.k_proj = _flarepp_make_residual_linear_proj(self.channel_dim)
        self.v_proj = _flarepp_make_residual_linear_proj(self.channel_dim)

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def flare_encode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.attn_scale, dropout_p=dropout_p,
        )

    def flare_decode(
        self,
        k: torch.Tensor,
        q: torch.Tensor,
        z: torch.Tensor,
        attn_mask: torch.Tensor = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        q = q.to(dtype=z.dtype)
        k = k.to(dtype=z.dtype)
        return F.scaled_dot_product_attention(
            k, q, z, attn_mask=attn_mask, scale=self.attn_scale, dropout_p=dropout_p,
        )

    def forward(self, x, attention_mask=None):
        drop_attn_p = self.attn_drop_p if self.training else 0.0
        drop_proj_p = self.proj_drop_p if self.training else 0.0

        batch_size = x.size(0)
        num_heads = self.num_heads
        mask_enc, mask_dec = FLARE.get_mask(attention_mask)

        q0 = self.q0_norm(self.latent_q0.unsqueeze(0).expand(batch_size, -1, -1, -1))
        k0 = self.k0_norm(rearrange(self.k0_proj(x), "b n (h d) -> b h n d", h=num_heads))
        if not self.share_k0_v0:
            v0 = self.v0_norm(rearrange(self.v0_proj(x), "b n (h d) -> b h n d", h=num_heads))
        else:
            v0 = k0

        if self.rope is not None:
            k0 = self.rope(k0)

        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=num_heads)

        q_dynamic = self.flare_encode(q0, k0, v0, attn_mask=mask_enc, dropout_p=drop_attn_p)
        qf = self.latent_q_fixed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        qf = self.q_fixed_norm(qf)
        qf_f = qf.float()
        qd_f = q_dynamic.float()
        g = torch.sigmoid(self.gate_logit).float().view(1, num_heads, 1, 1)
        q = (qf_f + g * qd_f).to(dtype=x.dtype)
        k = self.k_norm(k)
        if self.rope is not None:
            k = self.rope(k)

        z = self.flare_encode(q, k, v, attn_mask=mask_enc, dropout_p=drop_attn_p)
        y = self.flare_decode(k, q, z, attn_mask=mask_dec, dropout_p=drop_attn_p)

        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.out_proj(y)
        y = F.dropout(y, p=drop_proj_p, inplace=True)
        return y

class FLAREPPBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_ffn: int = 3,
        ffn_hidden_dim: int = 1.0,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        q_fixed_norm: bool = True,
        gate_logit_init: float = 0.25,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim, eps=1e-6) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = FLAREPP(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            k_norm=k_norm,
            share_k0_v0=share_k0_v0,
            rmsnorm=rmsnorm,
            q_fixed_norm=q_fixed_norm,
            gate_logit_init=gate_logit_init,
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
        q_norm: bool = False,
        k_norm: bool = False,
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

        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        q, k, v = rearrange(self.qkv_proj(x), 'b n (h d) -> b h n d', h=self.num_heads).chunk(3, dim=-1)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        q = self.kernel(q)
        k = self.kernel(k)
        
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.q_norm else q
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6) if self.k_norm else k

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
        q_norm: bool = False,
        k_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = LinearAttention(channel_dim, num_heads, kernel=kernel, q_norm=q_norm, k_norm=k_norm, attn_drop=attn_drop, proj_drop=proj_drop, rope=rope)
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
        q_norm: bool = False,
        k_norm: bool = False,
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

        self.q_norm = q_norm
        self.k_norm = k_norm
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
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.q_norm else q
        ks = ks / (ks.norm(dim=-1, keepdim=True) + 1e-6) if self.k_norm else ks

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
        q_norm: bool = False,
        k_norm: bool = False,
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
            kernel=kernel, q_norm=q_norm, k_norm=k_norm, qk_dim_ratio=qk_dim_ratio,
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


class NormAttentionBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        mlp_ratio: float = 4.0,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        num_layers_ffn: int = 0,
        ffn_mlp_ratio: float = 4.0,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope = None,
    ):
        super().__init__()
        del mlp_ratio
        self.block = MultilinearBlock(
            channel_dim=channel_dim,
            num_heads=num_heads,
            act=act,
            rmsnorm=rmsnorm,
            num_states=1,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_mlp_ratio=kv_proj_mlp_ratio,
            num_layers_ffn=num_layers_ffn,
            ffn_mlp_ratio=ffn_mlp_ratio,
            kernel='identity',
            q_norm=True,
            k_norm=True,
            qk_dim_ratio=qk_dim_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )

    def forward(self, x, attention_mask=None):
        return self.block(x, attention_mask=attention_mask)

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
        q_norm: bool = False,
        k_norm: bool = False,
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

        self.q_norm = q_norm
        self.k_norm = k_norm
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
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) if self.q_norm else q
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.k_norm else z for z in [k1, k2]]

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
        q_norm: bool = False,
        k_norm: bool = False,
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
            kernel=kernel, q_norm=q_norm, k_norm=k_norm, qk_dim_ratio=qk_dim_ratio,
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
        q_norm: bool = False,
        k_norm: bool = False,
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

        self.q_norm = q_norm
        self.k_norm = k_norm
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
        q1, q2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.q_norm else z for z in [q1, q2]]
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.k_norm else z for z in [k1, k2]]

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
        q_norm: bool = False,
        k_norm: bool = False,
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
            kernel=kernel, q_norm=q_norm, k_norm=k_norm, qk_dim_ratio=qk_dim_ratio, use_triton=use_triton,
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
        q_norm: bool = False,
        k_norm: bool = False,
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

        self.q_norm = q_norm
        self.k_norm = k_norm
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
        q1, q2, q3 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.q_norm else z for z in [q1, q2, q3]]
        k1, k2, k3 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.k_norm else z for z in [k1, k2, k3]]

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
        q_norm: bool = False,
        k_norm: bool = False,
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
            kernel=kernel, q_norm=q_norm, k_norm=k_norm, qk_dim_ratio=qk_dim_ratio,
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

def _draw_orthogonal_projection_matrix(num_heads: int, nb_features: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates orthogonal random features for FAVOR-style attention."""
    blocks = []
    for _ in range(num_heads):
        rows = []
        remaining = nb_features
        while remaining > 0:
            block = torch.randn(head_dim, head_dim, device=device, dtype=dtype)
            q, _ = torch.linalg.qr(block, mode="reduced")
            rows.append(q.mT[: min(head_dim, remaining)])
            remaining -= head_dim
        blocks.append(torch.cat(rows, dim=0))
    return torch.stack(blocks, dim=0)

class PerformerAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        nb_features: int = 256,
        feature_map: str = "favor_plus",
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
        self.feature_map = feature_map
        self.redraw_interval = redraw_interval
        self.normalize_inputs = normalize_inputs
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.rope = rope

        self.eps = 1e-6
        self.data_normalizer = (self.head_dim ** -0.25) if self.normalize_inputs else 1.0

        if self.feature_map not in ["favor_plus", "favor_pp"]:
            raise ValueError(f"Unsupported performer feature_map: {self.feature_map}.")

        proj = _draw_orthogonal_projection_matrix(num_heads, self.nb_features, self.head_dim, device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('proj_matrix', proj)
        self.register_buffer('_feature_redraw_counter', torch.zeros(1, dtype=torch.long), persistent=False)

    def _maybe_redraw_features(self):
        if self.redraw_interval <= 0 or not self.training:
            return
        self._feature_redraw_counter += 1
        if self._feature_redraw_counter.item() % self.redraw_interval == 0:
            with torch.no_grad():
                new_proj = _draw_orthogonal_projection_matrix(
                    self.num_heads, self.proj_matrix.size(1), self.head_dim,
                    device=self.proj_matrix.device, dtype=self.proj_matrix.dtype
                )
                self.proj_matrix.copy_(new_proj)

    def _compute_oprf_params(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the OPRF/FAVOR++ heuristic from CRT Theorem 3.3 using
        # average ||x_i + y_j||^2 over the current attention batch.
        if attention_mask is None:
            valid = torch.ones(q.size(0), 1, q.size(2), device=q.device, dtype=q.dtype)
        else:
            valid = attention_mask.view(q.size(0), 1, q.size(2)).to(dtype=q.dtype, device=q.device)

        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B, 1, 1]
        q_sq = (q.pow(2).sum(dim=-1) * valid).sum(dim=-1, keepdim=True) / count  # [B, H, 1]
        k_sq = (k.pow(2).sum(dim=-1) * valid).sum(dim=-1, keepdim=True) / count  # [B, H, 1]

        q_sum = (q * valid.unsqueeze(-1)).sum(dim=2)  # [B, H, D]
        k_sum = (k * valid.unsqueeze(-1)).sum(dim=2)  # [B, H, D]
        pair_dot = (q_sum * k_sum).sum(dim=-1, keepdim=True) / (count * count)  # [B, H, 1]
        t = (q_sq + k_sq + 2.0 * pair_dot).clamp_min(0.0)  # [B, H, 1]

        d = torch.tensor(float(self.head_dim), device=q.device, dtype=q.dtype)
        rho = torch.ones_like(t)
        nonzero = t > self.eps
        numer = torch.sqrt((2.0 * t + d).pow(2) + 8.0 * d * t) - 2.0 * t - d
        rho_est = numer / (4.0 * t.clamp_min(self.eps))
        rho = torch.where(nonzero, rho_est.clamp_min(self.eps).clamp_max(1.0), rho)

        a_star = (1.0 - rho.reciprocal()) / 8.0  # [B, H, 1]
        b_star = torch.sqrt((1.0 - 4.0 * a_star).clamp_min(self.eps))
        log_d_star = 0.25 * d * torch.log((1.0 - 4.0 * a_star).clamp_min(self.eps))
        return a_star.unsqueeze(-1), b_star.unsqueeze(-1), log_d_star.unsqueeze(-1)

    def _feature_map(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, N, D]
        proj = self.proj_matrix.to(device=q.device, dtype=torch.float32)
        q = q.to(torch.float32) * self.data_normalizer
        k = k.to(torch.float32) * self.data_normalizer

        q_proj = torch.einsum('b h n d, h m d -> b h n m', q, proj)
        k_proj = torch.einsum('b h n d, h m d -> b h n m', k, proj)
        q_sq = q.pow(2).sum(dim=-1, keepdim=True) / 2.0
        k_sq = k.pow(2).sum(dim=-1, keepdim=True) / 2.0

        if self.feature_map == "favor_plus":
            q_logits = q_proj - q_sq
            k_logits = k_proj - k_sq
        else:
            a_star, b_star, log_d_star = self._compute_oprf_params(q, k, attention_mask)
            proj_norm_sq = proj.pow(2).sum(dim=-1).view(1, self.num_heads, 1, self.nb_features)
            q_logits = log_d_star + a_star * proj_norm_sq + b_star * q_proj - q_sq
            k_logits = log_d_star + a_star * proj_norm_sq + b_star * k_proj - k_sq

        q_logits = q_logits - q_logits.max(dim=-1, keepdim=True).values
        k_logits = k_logits - k_logits.max(dim=-1, keepdim=True).values
        q_features = (torch.exp(q_logits) + self.eps) / math.sqrt(self.nb_features)
        k_features = (torch.exp(k_logits) + self.eps) / math.sqrt(self.nb_features)
        return q_features, k_features

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
        q_prime, k_prime = self._feature_map(q, k, attention_mask)
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
        feature_map: str = "favor_plus",
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
            feature_map=feature_map,
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
# Cosformer Attention
#======================================================================#
class CosformerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        dropout_rate: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        has_outproj: bool = True,
        act_fun: str = "relu",
        rope = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.rope = rope

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_index(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        index = (math.pi / 2) * torch.arange(1, seq_len + 1, device=device, dtype=dtype)
        return index.view(1, 1, seq_len, 1)

    def get_act_fun(self, act_fun: str):
        if act_fun == "relu":
            return F.relu
        if act_fun == "elu":
            return lambda x: 1 + F.elu(x)
        raise ValueError(f"Unsupported cosformer activation: {act_fun}.")

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

    def _feature_map(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_len = q.size(2)
        src_len = k.size(2)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m, device=q.device, dtype=torch.float32)
        q_angle = weight_index[:, :, :tgt_len] / m
        k_angle = weight_index[:, :, :src_len] / m
        q_ = torch.cat([q * torch.sin(q_angle), q * torch.cos(q_angle)], dim=-1)
        k_ = torch.cat([k * torch.sin(k_angle), k * torch.cos(k_angle)], dim=-1)
        return q_, k_

    def _mask_tensors(
        self,
        q_: torch.Tensor,
        k_: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if attention_mask is None:
            return q_, k_, v, None

        mask = attention_mask[:, None, :, None].to(dtype=q_.dtype, device=q_.device)
        return q_ * mask, k_ * mask, v * mask, mask

    def _project(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = query if key is None else key
        value = query if value is None else value

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        q = self.act_fun(q)
        k = self.act_fun(k)
        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        out_dtype = query.dtype
        q, k, v = self._project(query, key=key, value=value)
        q_, k_ = self._feature_map(q, k)
        q_, k_, v, mask = self._mask_tensors(q_, k_, v, attention_mask)

        if self.causal:
            kv = torch.einsum("b h n m, b h n d -> b h n m d", k_, v)
            kv_cum = torch.cumsum(kv, dim=2)
            numerator = torch.einsum("b h n m, b h n m d -> b h n d", q_, kv_cum)
            k_cum = torch.cumsum(k_, dim=2)
            denominator = torch.clamp_min(torch.einsum("b h n m, b h n m -> b h n", q_, k_cum), eps)
        else:
            kv = torch.einsum("b h n m, b h n d -> b h m d", k_, v)
            k_sum = k_.sum(dim=2)
            numerator = torch.einsum("b h n m, b h m d -> b h n d", q_, kv)
            denominator = torch.clamp_min(torch.einsum("b h n m, b h m -> b h n", q_, k_sum), eps)

        attn_output = numerator / denominator.unsqueeze(-1)
        if mask is not None:
            attn_output = attn_output * mask
        attn_output = self.attn_drop(attn_output)
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)").to(out_dtype)

        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output

    def left_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        out_dtype = query.dtype
        q, k, v = self._project(query, key=key, value=value)
        q_, k_ = self._feature_map(q, k)
        q_, k_, v, mask = self._mask_tensors(q_, k_, v, attention_mask)

        weights = torch.einsum("b h l d, b h s d -> b h l s", q_, k_)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    weights.size(-2),
                    weights.size(-1),
                    device=weights.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            weights = weights.masked_fill(causal_mask.view(1, 1, weights.size(-2), weights.size(-1)), 0.0)
        if mask is not None:
            weights = weights * mask.squeeze(-1).unsqueeze(2)

        denominator = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        attn_weights = weights / denominator
        attn_output = torch.einsum("b h l s, b h s d -> b h l d", attn_weights, v)
        if mask is not None:
            attn_output = attn_output * mask
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)").to(out_dtype)

        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class CosformerBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        rmsnorm: bool = False,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        has_outproj: bool = True,
        act_fun: str = "relu",
        rope = None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = CosformerAttention(
            embed_dim=channel_dim,
            num_heads=num_heads,
            dropout_rate=attn_drop,
            proj_drop=proj_drop,
            causal=causal,
            has_outproj=has_outproj,
            act_fun=act_fun,
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
    'transolver': TransolverBlock,
    'flare': FLAREBlock,
    'flarepp': FLAREPPBlock,
    'cosformer': CosformerBlock,
    'linformer': LinformerBlock,
    'linear': LinearAttentionBlock,
    'multilinear': MultilinearBlock,
    'normattention': NormAttentionBlock,
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

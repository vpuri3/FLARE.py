#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, einsum

__all__ = [
    "TS1Uncond",
]

#---------------#
# TODO:
# - X/Y positional embedding
# - nGPT: https://github.com/NVIDIA/ngpt/blob/main/model.py
# - Differential Transformer: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
# - Dynamic Tanh: https://arxiv.org/pdf/2503.10622
#---------------#

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()
        
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        return u * self.silu(v)

# the incremental speedup isn't worth dealing with versioning hell
FastGELU = nn.GELU
# FastGELU = lambda: nn.GELU(approximate='tanh')

ACTIVATIONS = {
    'gelu': FastGELU(),
    'silu': nn.SiLU(),
    'swiglu': SwiGLU(),
}

def mha(q, k, v):
    scale = q.shape[-1] ** -0.5
    dots = einsum(q, k, 'b h q d, b h k d -> b h q k') * scale
    attn = F.softmax(dots, dim=-1)
    out = einsum(attn, v, 'b h q k, b h k d -> b h q d')
    return out, attn


def normalize(x):
    return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-5)

def gumbel_noise(shape, device=None):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + 1e-10) + 1e-10)

def topk_sparse(weights, bias, k_val=None):
    if k_val is None:
        return weights
    B, H, M, N = weights.shape
    _, topk_indices = (weights + bias).topk(k_val, dim=-2)
    values = weights.gather(-2, topk_indices).reshape(-1)

    batch_indices = torch.arange(B, device=weights.device).repeat_interleave(H * k_val * N)
    head_indices = torch.arange(H, device=weights.device).repeat_interleave(k_val * N).repeat(B)
    query_indices = torch.arange(k_val, device=weights.device).repeat_interleave(N).repeat(B * H)
    slice_indices = topk_indices.reshape(-1)  # [b * h * k_val * k]
    indices = torch.stack([batch_indices, head_indices, query_indices, slice_indices], dim=0)

    sparse_weights = torch.sparse_coo_tensor(
        indices, values, size=(B, H, M, N), device=weights.device
    )
    return sparse_weights

def topk_dense(weights, bias, k_val=None):
    if k_val is None:
        return weights
    _, topk_indices = (weights + bias).topk(k_val, dim=-2)
    sparse_weights = torch.full_like(weights, 0.)
    sparse_weights.scatter_(-2, topk_indices, weights.gather(-2, topk_indices))
    return sparse_weights

#======================================================================#
# SLICE ATTENTION
#======================================================================#
class SliceAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads=8,
            dropout=0.,
            num_slices=32,
            qk_norm=False,
            k_val=None,
        ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_val = k_val
        self.qk_norm = qk_norm

        ### (1) Get slice weights
        self.wt_kv_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)

        self.wtq = nn.Parameter(torch.empty(self.num_heads, self.num_slices, self.head_dim))
        trunc_normal_(self.wtq, std=0.02)
        self.register_buffer('wtq_bias', torch.zeros(self.num_heads, self.num_slices))

        ### (2) Attention among slice tokens
        self.qkv_proj = nn.Parameter(torch.empty(self.num_heads, self.head_dim, 3 * self.head_dim))
        trunc_normal_(self.qkv_proj, std=0.02)

        ### (3) Deslice and return
        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, gamma:float=None, noise=None):

        # x: [B, N, C]
        
        ### (1) Slicing

        xq = self.wtq
        xk, xv = self.wt_kv_proj(x).chunk(2, dim=-1)
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)
        
        if self.qk_norm:
            xq = F.normalize(xq, dim=-1)
            xk = F.normalize(xk, dim=-1)
        
        slice_scores = einsum(xq, xk, 'h m d, b h n d -> b h m n') # [B H M N]
        slice_weights = F.softmax(slice_scores, dim=-2)
        
        # #-------#
        # # Dynamic bias load balancing (like deepseek v3) + TopK (can be made much more efficient)
        # #-------#

        bias = self.wtq_bias.unsqueeze(-1)
        # slice_weights = topk_dense(slice_weights, bias, self.k_val)
        # slice_weights = slice_weights / slice_weights.sum(dim=-2, keepdim=True)

        # if self.training:
        #     if gamma is not None:
        #         with torch.no_grad():
        #             slice_usage = slice_weights.mean(dim=-1)
        #             target_usage = 1 / self.num_slices
        #             bias_update = target_usage - slice_usage.mean(dim=0)
        #             self.wtq_bias.data += gamma * bias_update

        #-------#
        #-------#
        
        slice_token = einsum(slice_weights, xv, 'b h m n, b h n d -> b h m d') # [B H M D]

        slice_norm = slice_weights.sum(dim=-1, keepdim=True) # [B H M]
        slice_token = slice_token / (slice_norm + 1e-5)
        
        ### (2) Attention among slice tokens

        qkv_token = einsum(slice_token, self.qkv_proj, 'b h m d, h d e -> b h m e')
        q_token, k_token, v_token = qkv_token.chunk(3, dim=-1)
        out_token, attn_weights = mha(q_token, k_token, v_token)

        ### (3) Deslice

        out = einsum(out_token, slice_weights, 'b h m d, b h m n -> b h n d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        temperature = torch.tensor(1., device=x.device)

        return out, slice_weights, temperature, bias, attn_weights

#======================================================================#
# BLOCK
#======================================================================#

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        if act == 'swiglu':
            self.fc2 = nn.Linear(hidden_features // 2, out_features)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            mlp_ratio=2,
            num_slices=32,
            qk_norm=False,
            act=None,
            k_val=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.att = SliceAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_slices=num_slices,
            qk_norm=qk_norm,
            k_val=k_val,
        )
        
        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act=act,
        )
        
    def forward(self, x, gamma:float=None, noise=None):
        # x: [B, N, C]
       
        _x, *stats = self.att(self.ln1(x), gamma=gamma, noise=noise)
        x = x + _x
        x = x + self.mlp(self.ln2(x))
        return x, *stats

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.mlp(self.ln(x))
        return x

#======================================================================#
# MODEL
#======================================================================#
class TS1Uncond(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        num_blocks=5,
        hidden_dim=128,
        dropout=0,
        num_heads=8,
        mlp_ratio=1,
        num_slices=32,
        act=None,
        qk_norm=False,
        k_val=None,
    ):
        super().__init__()
        
        self.k_val = k_val
        self.gamma = 0.0

        self.num_blocks = num_blocks
        self.num_slices = num_slices
        self.num_heads = num_heads

        self.x_embedding = MLP(
            in_features=in_dim,
            hidden_features=hidden_dim * 2,
            out_features=hidden_dim,
            act=act,
        )
        
        self.blocks = nn.ModuleList([
            Block(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                num_slices=num_slices,
                act=act,
                qk_norm=qk_norm,
                k_val=k_val,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(hidden_dim, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def normalize(self):
        for block in self.blocks:
            att = block.att
            mlp = block.mlp
            # att.wtq.data.copy_(normalize(att.wtq.data))

        return

    def forward(self, x, gamma:float=None, noise=None, return_stats: bool=False):
        # x: [B, N, C]
        x = self.x_embedding(x) # [B, N, C]
        
        if (gamma is not None) and self.training:
            self.gamma = gamma

        if return_stats:
            slice_weights = []
            temperature = []
            bias = []
            attn_weights = []

        for block in self.blocks:
            x, *stats = block(x, gamma=gamma, noise=noise)
            if return_stats:
                slice_weights.append(stats[0])
                temperature.append(stats[1])
                bias.append(stats[2])
                attn_weights.append(stats[3])

        x = self.final_layer(x)

        if return_stats:
            return x, slice_weights, temperature, bias, attn_weights
        else:
            return x

#======================================================================#
#
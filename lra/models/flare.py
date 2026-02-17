#
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
]

class SelfAttention(nn.Module):
    def __init__(self, channel_dim: int):
        super().__init__()

        self.channel_dim = channel_dim
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.scale = channel_dim ** -0.5

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1) # [B N C]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        return y

class LatentHeadMixer(nn.Module):
    def __init__(self, num_heads:int, num_latents: int):
        super().__init__()
        
        # In Scores [B H M N], mix [B (H M) N]

        # IDEAS:
        # - Low-rank decomposition: W = A @ B.T with A, B ∈ ℝ^[num_heads*M, k]
        # - Group convolutions across heads.
        # - Treat [num_heads, M] as sequence of num_heads tokens of size M or M tokens of size num_heads.
        #   and do attention. Makes mixing weights dynamic.

        H = num_heads
        M = num_latents
        HM = H * M

        ###
        # full rank weights
        ###
        # self.weights = nn.Parameter(torch.empty([HM, HM]))
        # k = 1 / math.sqrt(HM)
        # nn.init.uniform_(self.weights, -k, k)

        ###
        # low rank weights
        ###
        K = HM // 8
        k = 1 / math.sqrt(K)
        self.A = nn.Parameter(torch.empty([HM, K]))
        self.B = nn.Parameter(torch.empty([K, HM]))
        nn.init.uniform_(self.A, -k, k)
        nn.init.uniform_(self.B, -k, k)

        # ###
        # # attention mixer
        # ###
        # self.ln  = nn.LayerNorm(M)
        # self.att = SelfAttention(M)

    def forward(self, x):

        x_in = x
        B, H, M, N = x.shape
        
        HM = H * M
        # weights = self.weights.view(HM, HM, 1)
        weights = (self.A @ self.B).view(HM, HM, 1)

        x = x.view(B, HM, N)
        x = F.conv1d(x, weights)
        x = x.view(B, H, M, N)

        # x = rearrange(x, 'b h m n -> (b n) h m')
        # x = self.att(self.ln(x))
        # x = rearrange(x, '(b n) h m -> b h m n', n=N)

        x = x + x_in

        return x

#======================================================================#
#
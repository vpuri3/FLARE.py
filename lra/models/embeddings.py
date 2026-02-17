#
import math
import torch
from torch import nn

__all__ = [
    'TokenEmb',
    'PosEmb',
    'RotaryPositionalEmbeddings',
]

#======================================================================#
# Token Embeddings
#======================================================================#
class TokenEmb(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channel_dim: int,
        drop: float = 0.0,
        padding_idx: int = None,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, channel_dim, padding_idx=padding_idx)
        self.emb_drop = nn.Dropout(drop)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: [B, N] token indices
        Returns:
            x: [B, N, C] token embeddings
        """
        x = self.token_embed(input_ids)  # [B, N, C]
        x = self.emb_drop(x)
        return x

#======================================================================#
# Positional Embeddings
#======================================================================#
class PosEmb(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        max_length: int,
        pos_embed: str = 'abs',  # 'sin', 'abs', 'rope'
    ):
        super().__init__()
        assert pos_embed in {'sin','abs','rope'}
        if pos_embed == 'sin':
            assert channel_dim % 2 == 0, "sincos needs even channel_dim"

        self.pos_embed = pos_embed
        self.max_length = max_length

        if self.pos_embed == 'abs':
            self.pe = nn.Embedding(max_length, channel_dim)
        elif self.pos_embed == 'sin':
            pe = self._build_sinusoidal_pe(max_length, channel_dim)
            self.register_buffer("pe", pe, persistent=False)   # fp32 table
            # pos_scale = 1.0
            pos_scale = channel_dim ** -0.5
            self.pos_scale = nn.Parameter(torch.tensor(pos_scale, dtype=torch.float32))
        else:
            self.pe = None
            self.pos_scale = None
            
    @staticmethod
    def _build_sinusoidal_pe(max_length: int, d: int) -> torch.Tensor:
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1) # [N,1]
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_length, d, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, B: int, N: int, device: torch.device):
        """
        Args:
            B: batch size
            N: sequence length
            device: device
        Returns:
            pos: [B, N, C] or [N, C] positional embeddings
        """
        assert N <= self.max_length, f"Sequence length {N} > max_length={self.max_length}"

        if self.pos_embed == 'abs':
            position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            pos = self.pe(position_ids)  # [B, N, C]
        elif self.pos_embed == 'sin':
            pos = self.pe[:N]  # [N, C] - will be broadcast to [B, N, C] when added
            pos_scale = self.pos_scale
            pos = pos * pos_scale
        else:
            pos = 0

        return pos

#======================================================================#
# Rotary Positional Embeddings (RoPE)
#======================================================================#
class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, D: int, max_length: int, base: int = 10_000):
        super().__init__()

        assert D % 2 == 0
        self.D = D
        self.N = max_length
        self.base = float(base)

        # frequencies for half-dim, then repeat for pairwise layout
        half = D // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))  # [D/2]
        inv_freq = inv_freq.repeat_interleave(2)  # [D]

        # positions start at 0
        pos = torch.arange(self.N, dtype=torch.float32) # [N]
        freqs = pos.view(-1, 1) * inv_freq.view(1, -1)  # [N, D]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # register as buffers (not params)
        self.register_buffer("C", cos, persistent=False)  # [N, D]
        self.register_buffer("S", sin, persistent=False)  # [N, D]

    def forward(self, x: torch.Tensor):

        dtype = x.dtype
        B, H, N, D = x.shape
        assert D == self.D
        assert N <= self.N

        S = self.S[:N].to(dtype=dtype) # [N, D]
        C = self.C[:N].to(dtype=dtype)

        x_rot = x.view(B, H, N, D//2, 2)

        x_odd = x_rot[..., 1]
        x_evn = x_rot[..., 0]

        x_rot = torch.cat([-x_odd, x_evn], dim=-1)

        return x * C + x_rot * S

#======================================================================#
#
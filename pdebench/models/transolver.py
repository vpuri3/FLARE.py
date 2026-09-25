#
# https://github.com/thuml/Transolver/blob/main/Car-Design-ShapeNetCar/models/Transolver.py
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

from ..distributed.context_parallel import ContextParallelState

__all__ = [
    "Transolver",
    "Transolver_Structured_Mesh_2D",
]

@dataclass
class TransolverConfig:
    """Transolver configuration.

    ``conv2d`` is not read inside ``Transolver`` / ``Transolver_Structured_Mesh_2D``;
    ``model_factory`` uses it only to dispatch the structured-mesh 2D variant.
    """

    model: str = "transolver"
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    num_slices: int = 64
    conv2d: bool = False
    unified_pos: bool = False


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

#======================================================================#
# Physics Attention (general)
#======================================================================#
class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
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
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.cp_state: ContextParallelState | None = None

    def set_context_parallel(self, cp_state: ContextParallelState | None):
        self.cp_state = cp_state

    def forward(self, x, mask: torch.Tensor = None):
        B, N, C = x.shape
        if mask is not None:
            if mask.shape != (B, N) or mask.dtype != torch.bool:
                raise ValueError(f"mask must be a boolean tensor with shape [B, N]. Got {mask.shape}, {mask.dtype}.")
            valid = mask[:, None, :, None].to(dtype=x.dtype)
        else:
            valid = None

        ### (1) Sliceing (value, key) [B H N C]
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_logits = self.in_project_slice(x_mid) / temperature
        slice_weights = F.softmax(slice_logits.float(), dim=-1).to(dtype=x_mid.dtype)  # B H N G
        if valid is not None:
            slice_weights = slice_weights * valid
            fx_mid = fx_mid * valid
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        if self.cp_state is not None and self.cp_state.cp_size > 1:
            if self.cp_state.cp_group is None:
                raise RuntimeError("Context parallel group is not initialized.")
            slice_norm = dist_nn.all_reduce(slice_norm, op=dist.ReduceOp.SUM, group=self.cp_state.cp_group)
            slice_token = dist_nn.all_reduce(slice_token, op=dist.ReduceOp.SUM, group=self.cp_state.cp_group)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots.float(), dim=-1).to(dtype=dots.dtype)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        out_x = self.to_out(out_x)
        if mask is not None:
            out_x = out_x * mask.unsqueeze(-1).to(dtype=out_x.dtype)
        return out_x

#======================================================================#
# MLP
#======================================================================#

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output,
                 n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            rmsnorm: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.ln_1 = Norm(hidden_dim)
        self.Attn = PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = Norm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = Norm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def set_context_parallel(self, cp_state: ContextParallelState | None):
        self.Attn.set_context_parallel(cp_state)

    def forward(self, fx, mask: torch.Tensor = None):
        fx = self.Attn(self.ln_1(fx), mask=mask) + fx
        if mask is not None:
            fx = fx * mask.unsqueeze(-1).to(dtype=fx.dtype)
        fx = self.mlp(self.ln_2(fx)) + fx
        if mask is not None:
            fx = fx * mask.unsqueeze(-1).to(dtype=fx.dtype)
        if self.last_layer:
            out = self.mlp2(self.ln_3(fx))
            if mask is not None:
                out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
            return out
        else:
            return fx

class Transolver(nn.Module):
    def __init__(self, config: TransolverConfig, metadata=None):
        super(Transolver, self).__init__()
        metadata = {} if metadata is None else metadata
        space_dim = metadata.get("space_dim", metadata["c_in"])
        n_layers = config.num_blocks
        n_hidden = config.channel_dim
        dropout = 0.0
        n_head = config.num_heads
        act = config.act or "gelu"
        mlp_ratio = config.mlp_ratio
        fun_dim = metadata.get("fun_dim", 1)
        out_dim = metadata["c_out"]
        slice_num = config.num_slices
        rmsnorm = config.rmsnorm
        self.__name__ = 'Transolver'
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([
            Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                             dropout=dropout,
                             act=act,
                             mlp_ratio=mlp_ratio,
                             out_dim=out_dim,
                             slice_num=slice_num,
                             rmsnorm=rmsnorm,
                             last_layer=(_ == n_layers - 1))
            for _ in range(n_layers)
        ])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.cp_state: ContextParallelState | None = None

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def set_context_parallel(self, cp_state: ContextParallelState | None, cp_debug_gather_outputs: bool = False):
        del cp_debug_gather_outputs
        self.cp_state = cp_state
        for block in self.blocks:
            block.set_context_parallel(cp_state)

    def forward(self, x, f=None, mask: torch.Tensor = None):
        if mask is not None:
            if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"mask must be a boolean tensor with shape [B, N]. Got {mask.shape}, {mask.dtype}.")

        if f is not None:
            f = torch.cat((x, f), -1)
            f = self.preprocess(f)
        else:
            f = self.preprocess(x)
            f = f + self.placeholder[None, None, :]
        if mask is not None:
            f = f * mask.unsqueeze(-1).to(dtype=f.dtype)

        for block in self.blocks:
            f = block(f, mask=mask)

        return f

#======================================================================#
# Physics Attention Structured 2D Mesh
#======================================================================#
class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_logits = self.in_project_slice(x_mid) / temperature
        slice_weights = F.softmax(slice_logits.float(), dim=-1).to(dtype=x_mid.dtype)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots.float(), dim=-1).to(dtype=dots.dtype)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class Transolver_block_Structured_Mesh_2D(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85,
            rmsnorm: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        Norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.ln_1 = Norm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W)

        self.ln_2 = Norm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = Norm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx
        
class Transolver_Structured_Mesh_2D(nn.Module):
    def __init__(self, config: TransolverConfig, metadata=None):
        super(Transolver_Structured_Mesh_2D, self).__init__()
        metadata = {} if metadata is None else metadata
        space_dim = metadata.get("space_dim", metadata["c_in"])
        n_layers = config.num_blocks
        n_hidden = config.channel_dim
        dropout = 0.0
        n_head = config.num_heads
        Time_Input = metadata.get("dataset") == "plasticity"
        act = config.act or "gelu"
        mlp_ratio = config.mlp_ratio
        fun_dim = metadata.get("fun_dim", 1)
        out_dim = metadata["c_out"]
        slice_num = config.num_slices
        ref = 8
        unified_pos = config.unified_pos
        H = metadata["H"]
        W = metadata["W"]
        rmsnorm = config.rmsnorm
        self.__name__ = 'Transolver_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block_Structured_Mesh_2D(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      rmsnorm=rmsnorm,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx=None, T=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
    
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

#======================================================================#
#

#
# Adapted from:
# https://github.com/thuml/Transolver_plus/blob/main/models/Transolver_plus.py
#
# This repo keeps the upstream architecture but intentionally uses the local
# PDEBench single-tensor `forward(x)` interface instead of the upstream
# `(x, pos, condition)` API.
import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import Optional

from dataclasses import dataclass

from ..distributed.context_parallel import ContextParallelState

__all__ = [
    "TransolverPlusPlus",
]

@dataclass
class TransolverPlusPlusConfig:
    model: str = "transolver++"
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    num_slices: int = 64


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

#======================================================================#
# Physics Attention
#======================================================================#
def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    # Keep U strictly inside (0, 1) so nested logs stay finite in low precision.
    finfo = torch.finfo(u.dtype)
    u = torch.clamp(u, min=finfo.tiny, max=1.0 - finfo.eps)
    gumbel_noise = -torch.log(-torch.log(u))

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
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
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
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
        self.cp_state: Optional[ContextParallelState] = None

    def set_context_parallel(self, cp_state: Optional[ContextParallelState]):
        self.cp_state = cp_state
    
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2, keepdim=False).unsqueeze(-1)  # B H G 1
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        if self.cp_state is not None and self.cp_state.cp_size > 1:
            slice_norm = dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM, group=self.cp_state.cp_group)
            slice_token = dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM, group=self.cp_state.cp_group)
        slice_token = slice_token / (slice_norm + 1e-5)

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

#======================================================================#
# MLP
#======================================================================#
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
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


class Transolver_plus_block(nn.Module):
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
        self.Attn = Physics_Attention_1D_Eidetic(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = Norm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = Norm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def set_context_parallel(self, cp_state: Optional[ContextParallelState]):
        self.Attn.set_context_parallel(cp_state)

    def forward(self, fx):
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=False) + fx
        else:
            fx = fx + self.Attn(self.ln_1(fx))
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=False) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

#======================================================================#
# Transolver_plus
#======================================================================#
class TransolverPlusPlus(nn.Module):
    def __init__(self, config: TransolverPlusPlusConfig, metadata=None):
        super(TransolverPlusPlus, self).__init__()
        metadata = {} if metadata is None else dict(metadata)
        space_dim = int(metadata.get("space_dim", metadata.get("c_in", 1)))
        fun_dim = int(metadata.get("fun_dim", 1))
        out_dim = int(metadata.get("c_out", 1))
        n_layers = int(config.num_blocks)
        n_hidden = int(config.channel_dim)
        dropout = float(getattr(config, "dropout", 0.0))
        n_head = int(config.num_heads)
        act = "gelu" if config.act is None else config.act
        mlp_ratio = float(config.mlp_ratio)
        slice_num = int(config.num_slices)
        ref = int(getattr(config, "ref", 8))
        unified_pos = bool(getattr(config, "unified_pos", False))
        rmsnorm = bool(config.rmsnorm)
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(3, n_hidden)
        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      rmsnorm=rmsnorm,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.cp_state: Optional[ContextParallelState] = None

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_context_parallel(self, cp_state: Optional[ContextParallelState], cp_debug_gather_outputs: bool = False):
        del cp_debug_gather_outputs
        self.cp_state = cp_state
        for block in self.blocks:
            block.set_context_parallel(cp_state)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, x):

        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx

#======================================================================#
#

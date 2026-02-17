"""
This script measures the time and memory usage for different attention models.
"""

import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triton.testing
from contextlib import contextmanager

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUT_PNG_FP16 = os.path.join(PROJDIR, 'figs', 'time_memory_bwd_fp16.png')
OUT_PDF_FP16 = os.path.join(PROJDIR, 'figs', 'time_memory_bwd_fp16.pdf')
OUT_CSV_FP16 = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_bwd_fp16.csv')

OUT_PNG_FP32 = os.path.join(PROJDIR, 'figs', 'time_memory_bwd_fp32.png')
OUT_PDF_FP32 = os.path.join(PROJDIR, 'figs', 'time_memory_bwd_fp32.pdf')
OUT_CSV_FP32 = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_bwd_fp32.csv')

SEQ_LENGTHS = [
            1_000,
          100_000,
          200_000,
          300_000,
          400_000,
          500_000,
          600_000,
          700_000,
          800_000,
          900_000,
        1_000_000,
    ]

NUM_LATENTS = [128, 512, 2048]  # [128, 512, 2048] (FLARE)
NUM_STATES = [1, 2, 4] # (Multilinear)

#======================================================================#
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#======================================================================#
class MultiHeadedSelfAttentionUnfused(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()

        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim, bias=False)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        score = (q @ k.transpose(-1, -2)) * self.scale
        attn = F.softmax(score, dim=-1)
        y = attn @ v

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()

        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim, bias=False)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

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

        self.dropout_p = dropout

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

        self.channel_dim = dim

    def forward(self, x):
        B, N, C = x.shape

        ### (1) Sliceing (value, key) [B H N C]
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        # dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        # attn = self.softmax(dots)
        # attn = self.dropout(attn)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token, scale=self.scale, dropout_p=self.dropout_p)
        # out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)

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
        num_layers_kv_proj: int = 3,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [ResidualMLP(
            in_dim=self.channel_dim, hidden_dim=self.channel_dim, out_dim=self.channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True,
        ) for _ in range(2)]

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        #--------------------------------------------#
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
        z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        y = F.scaled_dot_product_attention(k, q, z, scale=1.0)
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
def make_kernel(kernel: str):
    if kernel == 'silu':
        kernel = lambda x: F.silu(x) + 0.5
    elif kernel == 'elu':
        kernel = lambda x: F.elu(x) + 1.0
    elif kernel == 'sigmoid':
        kernel = nn.Sigmoid()
    elif kernel == 'identity':
        kernel = nn.Identity()
    elif kernel == 'relu':
        kernel = nn.ReLU()
    else:
        raise NotImplementedError(f"Kernel {kernel} not implemented. Choose from: silu, elu, sigmoid, identity.")

    return kernel

class TripleAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(channel_dim * qk_dim_ratio)

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        qk_res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=self.qk_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )
        v_res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )

        self.q1_proj = ResidualMLP(**qk_res_mlp_kws)
        self.q2_proj = ResidualMLP(**qk_res_mlp_kws)
        self.k1_proj = ResidualMLP(**qk_res_mlp_kws)
        self.k2_proj = ResidualMLP(**qk_res_mlp_kws)
        self.v_proj  = ResidualMLP(**v_res_mlp_kws)

        self.kernel = make_kernel(kernel)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.ln = nn.LayerNorm(self.channel_dim)

        from lra.models.triton.triple import TripleAttentionFunction
        self.attn = TripleAttentionFunction.apply

    def forward(self, x, rope=None):

        dtype = x.dtype
        B, N, C = x.shape
        H = self.num_heads
        assert rope is None, f"Rope is not supported by {self.__class__.__name__}."

        q1 = self.q1_proj(x)
        q2 = self.q2_proj(x)
        k1 = self.k1_proj(x)
        k2 = self.k2_proj(x)
        v  = self.v_proj(x)

        q1, q2, k1, k2, v = [rearrange(z, 'b n (h d) -> b h n d', h=H) for z in [q1, q2, k1, k2, v]]

        # kernel
        q1, q2, k1, k2 = [self.kernel(z) for z in [q1, q2, k1, k2]]

        # normalize
        q1, q2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_q else z for z in [q1, q2]]
        k1, k2 = [z / (z.norm(dim=-1, keepdim=True) + 1e-6) if self.norm_k else z for z in [k1, k2]]

        #============================#
        _, out = self.attn(q1, q2, k1, k2, v)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.ln(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

#======================================================================#
class MultilinearAttention(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str = None,
        num_states: int = 2,
        num_layers_kv_proj: int = -1,
        kv_proj_mlp_ratio: float = 1.0,
        kernel: str = 'identity',
        norm_q: bool = False,
        norm_k: bool = False,
        qk_dim_ratio: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):

        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.qk_dim = int(channel_dim * qk_dim_ratio)

        assert channel_dim % num_heads == 0
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_states = num_states
        assert num_states > 0, f"num_states must be greater than 0. Got {num_states}."

        qk_res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=self.qk_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )
        v_res_mlp_kws = dict(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True
        )

        self.q_proj = ResidualMLP(**qk_res_mlp_kws)
        self.k_projs = nn.ModuleList([ResidualMLP(**qk_res_mlp_kws) for _ in range(num_states)])
        self.v_projs = nn.ModuleList([ResidualMLP(**v_res_mlp_kws) for _ in range(num_states)])

        self.kernel = make_kernel(kernel)

        self.norm_q = norm_q
        self.norm_k = norm_k
        self.ln = nn.LayerNorm(self.channel_dim)

    def forward(self, x, rope=None):

        dtype = x.dtype
        B, N, C = x.shape
        H = self.num_heads
        K = self.num_states
        assert rope is None, f"Rope is not supported by {self.__class__.__name__}."

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

        #============================#
        scale_factor = N ** (-1/2)
        ks = ks.to(torch.float32) * scale_factor
        vs = vs.to(torch.float32) * scale_factor

        states = ks.mT @ vs
        state = states.prod(dim=0)

        out = q.to(torch.float32) @ state # [B, H, N, D]
        out = out.to(dtype)
        #============================#

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.ln(out)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

#======================================================================#
@contextmanager
def cuda_memory_manager(model):
    """Context manager to ensure proper GPU memory cleanup"""
    try:
        model.zero_grad()
        torch.cuda.empty_cache()
        yield
    finally:
        model.zero_grad()
        torch.cuda.empty_cache()

def benchmark_model(model, x, target, num_runs=10, warmup_runs=5, autocast_enabled=True):
    def forward_backward():
        model.zero_grad()
        with torch.autocast(device_type='cuda', enabled=autocast_enabled):
            output = model(x)
            loss = F.mse_loss(output, target)
        loss.backward()
        return loss.item()

    # Benchmark timing.
    times = triton.testing.do_bench(forward_backward, warmup=warmup_runs, rep=num_runs, return_mode='median')

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    forward_backward()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return times, peak_memory

#======================================================================#
def run_analysis(fp16=True):

    #---------------------------------------#
    device = torch.device(0)

    # Check for thermal throttling
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Initial GPU memory: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

        # Suggest memory optimization
        print("Tip: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation")

        # Wait for GPU to cool down if it was recently used
        import time
        print("Waiting for GPU to stabilize...")
        time.sleep(5)

        # Check initial memory usage
        initial_memory = torch.cuda.memory_allocated(device) / (1024**3)
        if initial_memory > 10:  # More than 10GB already used
            print(f"Warning: GPU already using {initial_memory:.1f}GB - consider restarting process")
            print("Clearing cache and continuing...")
            torch.cuda.empty_cache()
    #---------------------------------------#

    num_heads = 8
    channel_dim = 128

    assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

    num_layers_kv_proj = 3

    models = []
    model_names = []

    # Softmax Attention
    MSHA = MultiHeadedSelfAttention(channel_dim=channel_dim, num_heads=num_heads)
    MSHA_name = 'Softmax Attn'
    models = models + [MSHA]
    model_names = model_names + [MSHA_name]

    # # Softmax Attention Unfused
    # MSHA_unfused_name = 'Softmax Attn Unfused'
    # MSHA_unfused = MultiHeadedSelfAttentionUnfused(channel_dim=channel_dim, num_heads=num_heads)
    # models = models + [MSHA_unfused]
    # model_names = model_names + [MSHA_unfused_name]

    # FLARE
    FLARE_names = [f'FLARE (M={NUM_LATENTS[i]})' for i in range(len(NUM_LATENTS))]
    FLAREs = [FLARE(channel_dim=channel_dim, num_heads=num_heads, num_latents=NUM_LATENTS[i], num_layers_kv_proj=num_layers_kv_proj) for i in range(len(NUM_LATENTS))]
    models = models + FLAREs
    model_names = model_names + FLARE_names

    # Physics Attention
    PhA_names = [f'Phys Attn (M={NUM_LATENTS[i]})' for i in range(len(NUM_LATENTS))]
    PhAs = [PhysicsAttention(dim=channel_dim, heads=num_heads, dim_head=channel_dim // num_heads, slice_num=NUM_LATENTS[i]) for i in range(len(NUM_LATENTS))]
    models = models + PhAs
    model_names = model_names + PhA_names

    # # Triple Attention
    # TRIA_name = 'Triple Attn'
    # TRIA = TripleAttention(channel_dim=channel_dim, num_heads=num_heads, num_layers_kv_proj=num_layers_kv_proj)
    # models = models + [TRIA]
    # model_names = model_names + [TRIA_name]

    # # Multilinear Attention
    # MLA_names = [f'Multilinear (L={k})' for k in NUM_STATES]
    # MLAs = [MultilinearAttention(channel_dim=channel_dim, num_heads=num_heads, num_states=l, num_layers_kv_proj=num_layers_kv_proj) for l in NUM_STATES]
    # models = models + MLAs
    # model_names = model_names + MLA_names

    for model in models:
        model.to(device)
        model.train()  # Set to training mode for backward pass

        # ENABLE gradients for all parameters (needed for backward pass)
        for param in model.parameters():
            param.requires_grad_(True)

    # # Compile models for H100 optimization
    # print("Compiling models for H100 optimization...")
    # models = compile_models(models)

    # Set output files based on fp16 flag
    OUT_CSV = OUT_CSV_FP16 if fp16 else OUT_CSV_FP32
    OUT_PDF = OUT_PDF_FP16 if fp16 else OUT_PDF_FP32
    OUT_PNG = OUT_PNG_FP16 if fp16 else OUT_PNG_FP32

    # Set autocast based on fp16 flag
    autocast_enabled = fp16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    data = run_loop(models, model_names, device, SEQ_LENGTHS, autocast_enabled=autocast_enabled)

    df = pd.DataFrame(data)
    df.to_csv(OUT_CSV, index=False)

    return df

def run_loop(models, model_names, device, seq_lengths, autocast_enabled=True):
    for model in models:
        model.train()

    data = []

    print("Starting timing measurements...")

    for N in seq_lengths:
        print('='*80)
        for model, model_name in zip(models, model_names):
            print(f"N={N:<7}: {model_name:<20}:", end=' ', flush=True)

            channel_dim = model.channel_dim
            
            # Create input and target once
            torch.cuda.empty_cache()
            x = torch.randn(1, N, channel_dim, device=device, requires_grad=True)
            target = torch.randn(1, N, channel_dim, device=device)
            
            # Initialize to NaN in case of failure
            time_median = np.nan
            peak_memory = np.nan
            
            try:
                with cuda_memory_manager(model):
                    time_median, peak_memory = benchmark_model(model, x, target, autocast_enabled=autocast_enabled)
                    print(f"Time: {time_median:.3g}ms, Memory: {peak_memory:.3g}GB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM error: {e}")
                    time_median = np.nan
                    peak_memory = np.nan
                else:
                    print(f"Runtime error: {e}")
                    time_median = np.nan
                    peak_memory = np.nan
            except Exception as e:
                print(f"Unexpected error: {e}")
                time_median = np.nan
                peak_memory = np.nan
            finally:
                # Clean up
                del x, target
                torch.cuda.empty_cache()

            data.append({
                'model_name': model_name,
                'N': N,
                'time': time_median,
                'memory': peak_memory,
                'num_valid_runs': 10 
            })

            # Force memory cleanup between models
            torch.cuda.empty_cache()

    return data

#======================================================================#
def plot_analysis(fp16=True):
    # Set input/output files based on fp16 flag
    OUT_CSV = OUT_CSV_FP16 if fp16 else OUT_CSV_FP32
    OUT_PDF = OUT_PDF_FP16 if fp16 else OUT_PDF_FP32
    OUT_PNG = OUT_PNG_FP16 if fp16 else OUT_PNG_FP32
    
    df = pd.read_csv(OUT_CSV)

    # Set matplotlib to use LaTeX fonts if available, otherwise use default
    try:
        import subprocess
        subprocess.run(['latex', '--version'], capture_output=True, check=True)
        # LaTeX is available, use it
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        print("Using LaTeX for plot rendering")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # LaTeX not available, use default matplotlib fonts
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]
        })
        print("LaTeX not available, using default matplotlib fonts")

    fontsize = 20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    for ax in [ax1, ax2]:
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(r'Sequence Length', fontsize=fontsize)

    ax1.set_yscale('log', base=10)
    ax2.set_yscale('linear')
    ax2.set_ylim(0, 85)

    ax1.set_ylabel(r'Time (s)', fontsize=fontsize)
    ax2.set_ylabel(r'Memory (GB)', fontsize=fontsize)

    # Increase tick label size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Define markers for different num_latents
    latent_map = {
        f"{NUM_LATENTS[0]}": 's',
        f"{NUM_LATENTS[1]}": 'v',
        f"{NUM_LATENTS[2]}": 'D',
    }

    # Define markers for different num_states (Multilinear)
    states_map = {
        '1': 'o',
        '2': 'v',
        '4': 'D',
    }

    # Set custom x-ticks for sequence lengths
    x_ticks = [1000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    x_tick_labels = ['1k', '', '200k', '', '400k', '', '600k', '', '800k', '', '1m']

    for ax in [ax1, ax2]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    # Add horizontal dashed line at Memory = 80 GB
    ax2.axhline(y=80, color='black', linestyle='--', linewidth=3.0)

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        model_data = model_data.sort_values(by='N')

        if 'Softmax' in model_name:
            marker = 'o'
            color = 'black'
            linestyle = '-'
            label = r'Softmax Attention'
        elif 'Phys' in model_name:
            slice_num = model_name.split('=')[1].strip(')')
            marker = latent_map[slice_num]
            linestyle = '--'
            color = 'blue'
            label = r'PhysAttention ($%s$ slices)' % slice_num
        elif 'FLARE' in model_name:
            latent_size = model_name.split('=')[1].strip(')')
            marker = latent_map[latent_size]
            linestyle = '--'
            color = 'red'
            label = r'FLARE ($%s$ latents) (ours)' % latent_size
        # elif 'Triple' in model_name:
        #     marker = '^'
        #     color = 'green'
        #     linestyle = '-.'
        #     label = r'Triple Attention'
        # elif 'Multilinear' in model_name:
        #     num_states = model_name.split('=')[1].strip(')')
        #     marker = states_map[num_states]
        #     linestyle = ':'
        #     color = 'blue'
        #     label = r'Multilinear ($L=%s$)' % num_states

        marker_size = 8

        ax1.plot(model_data['N'], model_data['time'] / 1e3, label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
        ax2.plot(model_data['N'], model_data['memory'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)

    # Add legend to bottom of the figure with 4 columns, 2 rows
    handles, labels = ax1.get_legend_handles_labels()

    # Organize legend: FLARE variants in row 1 cols 1-3, Softmax in row 1 col 4, PhysAttention variants in row 2 cols 1-3
    flare_items = [(h, l) for h, l in zip(handles, labels) if 'FLARE' in l]
    physics_items = [(h, l) for h, l in zip(handles, labels) if 'PhysAttention' in l]
    softmax_items = [(h, l) for h, l in zip(handles, labels) if 'Softmax' in l]
    
    # Sort FLARE items by number of latents (128, 512, 2048)
    def extract_num(s):
        import re
        match = re.search(r'\((\d+)\)', s)
        return int(match.group(1)) if match else 0
    flare_items.sort(key=lambda x: extract_num(x[1]))
    physics_items.sort(key=lambda x: extract_num(x[1]))
    
    # Create ordered lists for multi-row layout: 2 rows, 4 columns
    # With ncol=4, matplotlib fills row by row:
    # Position 0: Row 1, Col 1
    # Position 1: Row 1, Col 2
    # Position 2: Row 1, Col 3
    # Position 3: Row 1, Col 4
    # Position 4: Row 2, Col 1
    # Position 5: Row 2, Col 2
    # Position 6: Row 2, Col 3
    # Position 7: Row 2, Col 4
    ordered_handles = []
    ordered_labels = []

    # Order: [*FLARE[1:3], *PhysAttn[1:3], SoftmaxAttn] in row-first layout
    # Row 1: FLARE variants (positions 0, 1, 2) + PhysAttention (128) (position 3)
    # Row 2: PhysAttention (512, 2048) (positions 4, 5) + Softmax (position 6)
    
    # All FLARE variants first
    for i in range(len(flare_items)):
        ordered_handles.append(flare_items[i][0])
        ordered_labels.append(flare_items[i][1])
    
    # All PhysAttention variants next
    for i in range(len(physics_items)):
        ordered_handles.append(physics_items[i][0])
        ordered_labels.append(physics_items[i][1])

    # Softmax last
    if len(softmax_items) > 0:
        ordered_handles.append(softmax_items[0][0])
        ordered_labels.append(softmax_items[0][1])

    # Place legend below the subplots with 4 columns
    legend = fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=4, 
              frameon=True, fancybox=False, shadow=False, fontsize=fontsize, 
              bbox_to_anchor=(0.5, 0.00), columnspacing=0.5, handletextpad=0.2,
              bbox_transform=fig.transFigure, handlelength=1.5, markerscale=1.5)

    # Add title with larger font
    ax1.set_title(r'Execution Time (Forward + Backward)', fontsize=fontsize)
    ax2.set_title(r'Peak Memory Usage (Forward + Backward)', fontsize=fontsize)

    # Adjust layout with extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.26)

    # Save the figure with both plots
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    return

#======================================================================#
def optimize_for_h100():
    """Apply H100-specific optimizations for maximum performance"""

    import torch._dynamo.config
    torch._dynamo.config.recompile_limit = 1000

    # Environment variables for H100 optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # H100-specific PyTorch backend optimizations
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # FP8 support (only available in newer PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_fp8_e4m3fn = True
        torch.backends.cuda.matmul.allow_fp8_e5m2 = True
    except AttributeError:
        print("FP8 support not available in this PyTorch version")

    # Performance vs reproducibility settings (choose one)
    torch.backends.cudnn.benchmark = True  # Enable for better performance
    torch.backends.cudnn.deterministic = False  # Disable for better performance

    # Memory management optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        torch.cuda.empty_cache()

    print("H100 optimizations applied successfully!")

def compile_models(models):
    """Compile models for H100 optimization using torch.compile"""
    compiled_models = []

    for i, model in enumerate(models):
        try:
            # Use default mode for FWD+BWD compatibility (max-autotune uses CUDA graphs which conflict with gradients)
            compiled_model = torch.compile(model, mode='default')
            compiled_models.append(compiled_model)
            print(f"Model {i} compiled successfully")
        except Exception as e:
            print(f"Failed to compile model {i}: {e}")
            compiled_models.append(model)  # Fallback to original model

    return compiled_models

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward + Backward Pass Timing Analysis for Attention Models')

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--run', type=str_to_bool, default=False, help='Run forward+backward timing analysis')
    parser.add_argument('--plot', type=str_to_bool, default=False, help='Plot forward+backward timing results')
    parser.add_argument('--clean', type=str_to_bool, default=False, help='Clean forward+backward timing results')
    parser.add_argument('--fp16', type=str_to_bool, default=True, help='Use FP16 precision (default: True)')

    args = parser.parse_args()

    if args.clean:
        # Set output files based on fp16 flag
        OUT_CSV = OUT_CSV_FP16 if args.fp16 else OUT_CSV_FP32
        OUT_PDF = OUT_PDF_FP16 if args.fp16 else OUT_PDF_FP32
        OUT_PNG = OUT_PNG_FP16 if args.fp16 else OUT_PNG_FP32

        if os.path.exists(OUT_CSV):
            print(f"Removing {OUT_CSV}")
            os.remove(OUT_CSV)
        if os.path.exists(OUT_PNG):
            print(f"Removing {OUT_PNG}")
            os.remove(OUT_PNG)
        if os.path.exists(OUT_PDF):
            print(f"Removing {OUT_PDF}")
            os.remove(OUT_PDF)
    if args.run:
        print(f"Running analysis with FP16={args.fp16}")
        optimize_for_h100()
        run_analysis(fp16=args.fp16)
    if args.plot:
        print(f"Plotting analysis with FP16={args.fp16}")
        plot_analysis(fp16=args.fp16)

    if not args.run and not args.plot and not args.clean:
        print("No action specified. Please specify either --run or --plot or --clean.")

#======================================================================#
#
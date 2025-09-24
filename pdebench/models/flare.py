#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "FLAREModel",
]

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
        act: str = None,
        num_layers_kv_proj: int = 3,
        kv_proj_hidden_dim: int = None,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

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

    def forward(self, x, return_scores: bool = False):

        # x: [B N C]

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        #--------------------------------------------#
        if not return_scores:
            q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
            z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            y = F.scaled_dot_product_attention(k, q, z, scale=1.0)
            scores = None
        else:
            # (1) Compute projection weights
            scores = q @ k.transpose(-2, -1) # [B H M N]
            W_encode = F.softmax(scores, dim=-1)
            W_decode = F.softmax(scores.transpose(-2, -1), dim=-1)

            # (2) Project to latent sequence
            z = W_encode @ v # [B H M D]

            # (3) Project back to input space
            y = W_decode @ z # [B H N D]
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y, scores

#======================================================================#
# FLARE Block
#======================================================================#
class FLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        act: str = None,
        num_layers_kv_proj: int = 3,
        num_layers_mlp: int = 3,
        kv_proj_hidden_dim: int = None,
        mlp_hidden_dim: int = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = FLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            act=act,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_hidden_dim=kv_proj_hidden_dim,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=channel_dim,
            num_layers=num_layers_mlp,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x, return_scores: bool = False):
        # x: [B, N, C]

        # x = x + att(ln1(x))
        # x = x + mlp(ln2(x))
        # return x

        _x, scores = self.att(self.ln1(x), return_scores=return_scores)
        x = x + _x
        x = x + self.mlp(self.ln2(x))

        return x, scores

#======================================================================#
# Final Layer
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
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_latents: int = None,
        num_heads: int = None,
        act: str = None,
        #
        num_layers_kv_proj: int = 3,
        num_layers_mlp: int = 3,
        num_layers_in_out_proj: int = 2,
        #
        mlp_ratio: float = 1.0,
        kv_proj_ratio: float = 1.0,
        in_out_proj_ratio: float = 1.0,
        #
        out_proj_ln: bool = True,
    ):
        super().__init__()
        
        mlp_hidden_dim = int(channel_dim * mlp_ratio)
        kv_proj_hidden_dim = int(channel_dim * kv_proj_ratio)
        in_out_proj_hidden_dim = int(channel_dim * in_out_proj_ratio)

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=in_out_proj_hidden_dim,
            out_dim=channel_dim,
            num_layers=num_layers_in_out_proj,
            act=act,
            output_residual=True,
        )
        self.out_proj = FinalLayer(
            channel_dim=channel_dim,
            hidden_dim=in_out_proj_hidden_dim,
            out_dim=out_dim,
            act=act,
            num_layers=num_layers_in_out_proj,
            ln=out_proj_ln,
        )

        self.blocks = nn.ModuleList([
            FLAREBlock(
                channel_dim=channel_dim,
                num_latents=num_latents,
                num_heads=num_heads,
                act=act,
                num_layers_kv_proj=num_layers_kv_proj,
                num_layers_mlp=num_layers_mlp,
                kv_proj_hidden_dim=kv_proj_hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim,
            )
            for i in range(num_blocks)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward(self, x, return_scores: bool = False):
        # x: [B, N, C]

        if return_scores:
            scores = []

        x = self.in_proj(x)
        for block in self.blocks:
            x, score = block(x, return_scores=return_scores)
            if return_scores:
                scores.append(score)

        x = self.out_proj(x)

        return (x, scores) if return_scores else x

#======================================================================#
# TESTING
#======================================================================#
def flare_vanilla(q, k, v):
    """
    Inputs:
    q: [H, M, D]
    k: [B, H, N, D]
    v: [B, H, N, D]
    Outputs:
    y: [B, H, N, D]
    """

    S = q @ k.transpose(-2, -1) # [B H M N]
    We = F.softmax(S, dim=-1) # sum over N
    Wd = F.softmax(S.transpose(-2, -1), dim=-1) # sum over M
    z = We @ v # [B H M D]
    y = Wd @ z # [B H N D]

    return y

def flare_fused(q, k, v):
    """
    Inputs:
    q: [H, M, D]
    k: [B, H, N, D]
    v: [B, H, N, D]
    Outputs:
    y: [B, H, N, D]
    """
    
    q = q.unsqueeze(0).expand(k.size(0), -1, -1, -1)
    z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
    y = F.scaled_dot_product_attention(k, q, z, scale=1.0)

    return y

def benchmark(B, H, M, N, D, device, dtype, verbose=False):

    q_vanilla = torch.randn(H, N, D, device=device, dtype=dtype, requires_grad=True)
    k_vanilla = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v_vanilla = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    q_fused = q_vanilla.clone().detach().requires_grad_(True)
    k_fused = k_vanilla.clone().detach().requires_grad_(True)
    v_fused = v_vanilla.clone().detach().requires_grad_(True)

    #--------------------------------------------#
    # Measure vanilla implementation - forward pass
    #--------------------------------------------#

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    y_vanilla = flare_vanilla(q_vanilla, k_vanilla, v_vanilla)
    end_time.record()
    torch.cuda.synchronize()

    vanilla_time_fwd = start_time.elapsed_time(end_time)
    vanilla_memory_fwd = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    #--------------------------------------------#
    # Measure fused implementation - forward pass
    #--------------------------------------------#

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    y_fused = flare_fused(q_fused, k_fused, v_fused)
    end_time.record()
    torch.cuda.synchronize()

    fused_time_fwd = start_time.elapsed_time(end_time)
    fused_memory_fwd = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    #--------------------------------------------#
    # Compute forward pass differences
    #--------------------------------------------#

    time_speedup_fwd = vanilla_time_fwd / fused_time_fwd
    mem_reduction_fwd = fused_memory_fwd / vanilla_memory_fwd
    value_diff_fwd = torch.abs(y_vanilla - y_fused).mean().item()

    #--------------------------------------------#
    # Create dummy gradients for backward pass
    #--------------------------------------------#

    grad_output = torch.randn_like(y_vanilla)
    grad_output_fused = grad_output.clone()

    #--------------------------------------------#
    # Measure vanilla implementation - backward pass
    #--------------------------------------------#

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    y_vanilla.backward(grad_output, retain_graph=True)
    end_time.record()
    torch.cuda.synchronize()

    vanilla_time_bwd = start_time.elapsed_time(end_time)
    vanilla_memory_bwd = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    #--------------------------------------------#
    # Measure fused implementation - backward pass
    #--------------------------------------------#

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    y_fused.backward(grad_output_fused, retain_graph=True)
    end_time.record()
    torch.cuda.synchronize()

    fused_time_bwd = start_time.elapsed_time(end_time)
    fused_memory_bwd = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    #--------------------------------------------#
    # Compute backward pass differences
    #--------------------------------------------#

    time_speedup_bwd = vanilla_time_bwd / fused_time_bwd
    mem_reduction_bwd = fused_memory_bwd / vanilla_memory_bwd

    # Compare gradients
    grad_diff_q = torch.abs(q_vanilla.grad - q_fused.grad).mean().item()
    grad_diff_k = torch.abs(k_vanilla.grad - k_fused.grad).mean().item()
    grad_diff_v = torch.abs(v_vanilla.grad - v_fused.grad).mean().item()

    #--------------------------------------------#
    # Print results
    #--------------------------------------------#

    if verbose:
        print("=" * 80)
        print(f"{'':^15}|{'Vanilla':^20}|{'Fused':^20}|{'Comparison':^20}")
        print("=" * 80)
        print(f"{'Forward Pass':^15}|{'':<20}|{'':<20}|{'':<20}")
        print(f"{'Time (ms)':^15}|{vanilla_time_fwd:^20.2f}|{fused_time_fwd:^20.2f}|{f'Speedup: {time_speedup_fwd:.2f}x':^20}")
        print(f"{'Memory (MB)':^15}|{vanilla_memory_fwd:^20.2f}|{fused_memory_fwd:^20.2f}|{f'Ratio: {mem_reduction_fwd:.2f}':^20}")
        print(f"{'Value diff':^15}|{'':<20}|{'':<20}|{value_diff_fwd:^20.6f}")
        print("-" * 80)
        print(f"{'Backward Pass':^15}|{'':<20}|{'':<20}|{'':<20}")
        print(f"{'Time (ms)':^15}|{vanilla_time_bwd:^20.2f}|{fused_time_bwd:^20.2f}|{f'Speedup: {time_speedup_bwd:.2f}x':^20}")
        print(f"{'Memory (MB)':^15}|{vanilla_memory_bwd:^20.2f}|{fused_memory_bwd:^20.2f}|{f'Ratio: {mem_reduction_bwd:.2f}':^20}")
        print(f"{'Gradient diff':^15}|{'':<20}|{'':<20}|{f'q={grad_diff_q:.6f}':^20}")
        print(f"{'':<15}|{'':<20}|{'':<20}|{f'k={grad_diff_k:.6f}':^20}")
        print(f"{'':<15}|{'':<20}|{'':<20}|{f'v={grad_diff_v:.6f}':^20}")
        print("=" * 80)

    return

def cublas_warmup(device):
    a = torch.randn(2, 2, device=device, requires_grad=True)
    b = torch.randn(2, 2, device=device, requires_grad=True)
    c = a @ b
    c.sum().backward()  # Force backward cuBLAS initialization
    del a, b, c  # Clean up

if __name__ == "__main__":
    B, H, M, N, D = 2, 8, 64, 2048, 8

    device = torch.device(0)
    dtype = torch.float32

    dtype = torch.float16
    dtype = torch.bfloat16

    cublas_warmup(device)
    benchmark(B, H, M, N, D, device, dtype, verbose=False)
    benchmark(B, H, M, N, D, device, dtype, verbose=True)

    exit()
#
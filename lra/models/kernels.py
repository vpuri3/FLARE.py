#
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

__all__ = [
    'make_kernel',
]

#======================================================================#
# Non-learnable kernels
#======================================================================#
def _identity(x: torch.Tensor) -> torch.Tensor:
    return x

def _silu_shift(x: torch.Tensor) -> torch.Tensor:
    # strictly positive-ish map; keeps small baseline to avoid dead regions
    return F.silu(x) + 0.5

def _elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0

def _elu_norm(x: torch.Tensor) -> torch.Tensor:
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.elu(x) + 1.0

def _relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)

def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    r = F.relu(x)
    return r * r

def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)

def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def _softplus(x: torch.Tensor) -> torch.Tensor:
    # softplus is smoother than ReLU; shift keeps it > 0
    return F.softplus(x) + 1e-6

def _softsign(x: torch.Tensor) -> torch.Tensor:
    return F.softsign(x)

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def _cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)

def _sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def _exp(x: torch.Tensor) -> torch.Tensor:
    # Numerically safer than raw exp for mixed precision / large magnitudes
    return torch.exp(torch.clamp(x, min=-30.0, max=30.0))

#======================================================================#
# Learnable kernels
#======================================================================#
class HadamardFeatureMap(nn.Module):
    def __init__(self, head_dim: int, qk_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(head_dim, qk_dim)
        self.layer2 = nn.Linear(head_dim, qk_dim)

    def forward(self, x: torch.Tensor):
        return self.layer1(x) * self.layer2(x)

class SwiGLUFeatureMap(nn.Module):
    def __init__(self, head_dim: int, qk_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(head_dim, qk_dim * 2)
        self.layer3 = nn.Linear(qk_dim, qk_dim)

    def forward(self, x: torch.Tensor):
        x1, x2 = self.layer1(x).chunk(2, dim=-1)
        return self.layer3(x1 * F.silu(x2))

class HedgehogFeatureMap(nn.Module):
    r"""
    Hedgehog feature map as introduced in
    `The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry <https://arxiv.org/abs/2402.04347>`_
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize traiable map as identity"""
        with torch.no_grad():
            identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
            self.layer.weight.copy_(identity.to(self.layer.weight))
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x)  # shape b, h, l, d
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)

# class DPFPFeatureMap(nn.Module):

#     r"""
#     Deterministic Parameter-Free Projection (DPFP) feature map in
#     `Linear Transformers Are Secretly Fast Weight Programmers <https://arxiv.org/abs/2102.11174>`_
#     """

#     def __init__(
#         self,
#         head_dim: int,
#         nu: int = 4
#     ):
#         super().__init__()
#         self.nu = nu

#     def forward(self, x: torch.Tensor):
#         x = torch.cat([x.relu(), -x.relu()], dim=-1)
#         x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
#         x_repeat = torch.cat([x] * self.nu, dim=-1)
#         return x_repeat * x_rolled

def flatten_diag_outer_product(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    return z[..., indicies[0], indicies[1]]


class LearnableOuterProductFeatureMap(nn.Module):
    def __init__(self, head_dim: int, feature_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(head_dim, feature_dim, bias=False)
        self.layer2 = nn.Linear(head_dim, feature_dim, bias=False)
        self.normalizer = feature_dim ** -0.5

    def forward(self, x: torch.Tensor):
        return flatten_diag_outer_product(self.layer1(x), self.layer2(x))

# class LearnablePolySketchNonNegativeFeatureMap(nn.Module):

#     def __init__(
#         self,
#         head_dim: int,
#         sketch_size: Optional[int] = None,
#         degree: Optional[int] = 2
#     ):
#         super().__init__()

#         assert is_power_of_2(degree) and degree >= 2, f"The degree {degree} must be a power of 2"

#         self.head_dim = head_dim
#         self.sketch_size = sketch_size if sketch_size is not None else head_dim
#         self.degree = degree

#         self.gamma = nn.Parameter(torch.ones(head_dim))
#         self.beta = nn.Parameter(torch.zeros(head_dim))
#         # NOTE: the sketch layers defined here are quite different from the original paper
#         # currently we simply use linear layers without any non-linear activations
#         self.sketches1 = nn.ModuleList([
#             nn.Linear(head_dim, sketch_size, bias=False),
#             *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
#         ])
#         self.sketches2 = nn.ModuleList([
#             nn.Linear(head_dim, sketch_size, bias=False),
#             *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
#         ])

#     def forward(self, x: torch.Tensor):
#         # Section 2.1
#         x = layer_norm(x, self.gamma, self.beta)
#         # first map the input to sketch size with learnable parameters
#         x = self.sketches1[0](x) * self.sketches2[0](x) * self.head_dim ** -0.5
#         for i in range(1, int(math.log2(self.degree)) - 1):
#             x = self.sketches1[i](x) * self.sketches2[i](x) * self.head_dim ** -0.5
#         # do sketch mapping for log2(p) - 1 times in total
#         # do p=2 mapping to ensure non-negativity
#         return flatten_diag_outer_product(x, x)

def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]


class TaylorFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)


#======================================================================#
NON_LEARNABLE_KERNEL_REGISTRY = {
    'identity': _identity,
    'relu': _relu,
    'relu_squared': _relu_squared,
    'silu': _silu_shift,
    'elu': _elu,
    'elu_norm': _elu_norm,
    'gelu': _gelu,
    'tanh': _tanh,
    'softplus': _softplus,
    'softsign': _softsign,
    'sigmoid': _sigmoid,
    'exp': _exp,
    'cos': _cos,
    'sin': _sin
}

LEARNABLE_KERNEL_REGISTRY = {
    'swiglu': SwiGLUFeatureMap,
    'hadamard': HadamardFeatureMap,
    'hedgehog': HedgehogFeatureMap,
}

def make_kernel(kernel: str, head_dim: int, qk_dim: int = None):
    """
    Return a callable kernel map f(x) compatible with torch.compile.
    All kernels below keep the same shape as x and are elementwise or per-row stable ops.
    They play nicely with torch.compile (no Python-side dynamic shapes or side effects).

    All kernels preserve input shape. Choose from:
      identity, relu, squared_relu, silu, elu, gelu, tanh, softplus, softsign, sigmoid,
      cos, sin, exp, exp_clamped, l2unit, l2unit_elu, zmu_unitvar
    """
    qk_dim = qk_dim if qk_dim is not None else head_dim

    if kernel in NON_LEARNABLE_KERNEL_REGISTRY:
        return NON_LEARNABLE_KERNEL_REGISTRY[kernel]
    elif kernel in LEARNABLE_KERNEL_REGISTRY:
        return LEARNABLE_KERNEL_REGISTRY[kernel](head_dim=head_dim, qk_dim=qk_dim)
    else:
        raise NotImplementedError(f"Kernel '{kernel}' not implemented. Available: {sorted(NON_LEARNABLE_KERNEL_REGISTRY.keys())} + {sorted(LEARNABLE_KERNEL_REGISTRY.keys())}")

#======================================================================#
#
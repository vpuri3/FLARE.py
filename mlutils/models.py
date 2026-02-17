#
import torch
from torch import nn
from torch.nn import functional as F

import math
from mlutils.utils import check_package_version_lteq

__all__ = [
    'SwiGLU',
    'GEGLU',
    'ACTIVATIONS',
    'MLP',
    'Sine',
    'SDFClamp',
]

#======================================================================#
# activation functions
#======================================================================#
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class GEGLU(nn.Module):
    def forward(self, x):
        if check_package_version_lteq('torch', '2.4.0'):
            kw = {}
        else:
            kw = {'approximate': 'tanh'}
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates, **kw)

if check_package_version_lteq('torch', '2.4.0'):
    FastGELU = nn.GELU
else:
    FastGELU = lambda: nn.GELU(approximate='tanh')

ACTIVATIONS = {
    'gelu': FastGELU(),
    'silu': nn.SiLU(),
    'swiglu': SwiGLU(),
    'geglu': GEGLU(),
}

#======================================================================#
# MLP
#======================================================================#
def MLP(
    in_dim, out_dim, width, hidden_layers, act=None,
    siren=False, w0=10.0,
):
    if siren:
        act = Sine()
    if act is None:
        act = nn.Tanh()

    layers = []
    layers.extend([nn.Linear(in_dim, width), act])
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(width, width), act])
    layers.extend([nn.Linear(width, out_dim)])

    if siren:
        for (i, layer) in enumerate(layers):
            w = w0 if i == 0 else 1.0
            if isinstance(layer, nn.Linear):
                siren_init_(layer, w)

    return nn.Sequential(*layers)

#======================================================================#
# SIREN
# initialization modified from https://github.com/dalmia/siren/
#======================================================================#
class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.sin(x)

@torch.no_grad()
def siren_init_(layer: nn.Linear, w):
    fan = nn.init._calculate_correct_fan(layer.weight, "fan_in")
    bound = math.sqrt(6 / fan)

    layer.bias.uniform_(-math.pi, math.pi)
    layer.weight.uniform_(-bound, bound)
    layer.weight.mul_(w)

    return

#======================================================================#
# SDF Clamp
#======================================================================#
class SDFClamp(nn.Module):
    def __init__(self, eps, act = nn.Tanh()):
        super().__init__()
        self.eps = eps
        self.act = act
    def forward(self, x):
        return self.eps * self.act(x)

#======================================================================#
#
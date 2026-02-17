#
import torch
import einops
from torch import nn
import torch.nn.functional as F

__all__ = [
    'make_optimizer_adamw',
    'make_optimizer_lion',
    'make_optimizer_muon',
    #
    'darcy_deriv_loss',
    #
    'RelL1Loss',
    'RelL2Loss',
    #
    'IdentityNormalizer',
    'UnitCubeNormalizer',
    'UnitGaussianNormalizer',
]

#======================================================================#
NO_DECAY_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.RMSNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

LATENT_TYPES = (
    nn.Parameter,
)

def _collect_param_ids_by_module_types(model, types):
    ids = set()
    for module in model.modules():
        if isinstance(module, types):
            for p in module.parameters(recurse=False):
                ids.add(id(p))
    return ids

#======================================================================#
def split_params_adamw(model, no_decay_types=NO_DECAY_TYPES, latent_types=LATENT_TYPES):
    decay_params = []
    no_decay_params = []
    latent_params = []

    latent_param_ids = _collect_param_ids_by_module_types(model, latent_types)
    no_decay_param_ids = _collect_param_ids_by_module_types(model, no_decay_types)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        if (
            (no_decay_param_ids is not None and id(param) in no_decay_param_ids) or 
            name.endswith("bias") or 
            "LayerNorm" in name or "layernorm" in name or 
            "RMSNorm" in name or "rmsnorm" in name or
            "embed" in name.lower() or
            "cls_token" in name
        ):
            no_decay_params.append(param)
        elif (
            "latent" in name and
            (latent_param_ids is not None and id(param) in latent_param_ids)
        ):
            latent_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params, latent_params

#======================================================================#
def make_optimizer_adamw(model, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
    decay_params, no_decay_params, latent_params = split_params_adamw(model, NO_DECAY_TYPES, LATENT_TYPES)

    if isinstance(lr, float) or isinstance(lr, int):
        lr = [lr] * 3
    if isinstance(weight_decay, float) or isinstance(weight_decay, int):
        weight_decay = [weight_decay, 0.0, 0.0]
    if isinstance(beta1, float) or isinstance(beta1, int):
        beta1 = [beta1] * 3
    if isinstance(beta2, float) or isinstance(beta2, int):
        beta2 = [beta2] * 3
    if isinstance(eps, float) or isinstance(eps, int):
        eps = [eps] * 3

    assert len(lr) == 3, f"lr must be a list of 3 elements, got {lr} with {len(lr)} elements"
    assert len(weight_decay) == 3, f"weight_decay must be a list of 3 elements, got {weight_decay} with {len(weight_decay)} elements"
    assert len(beta1) == 3, f"beta1 must be a list of 3 elements, got {beta1} with {len(beta1)} elements"
    assert len(beta2) == 3, f"beta2 must be a list of 3 elements, got {beta2} with {len(beta2)} elements"
    assert len(eps) == 3, f"eps must be a list of 3 elements, got {eps} with {len(eps)} elements"

    decay_param_group = {
        'params': decay_params,
        'weight_decay': weight_decay[0],
        'lr': lr[0],
        'betas': (beta1[0], beta2[0]),
        'eps': eps[0]
    }
    no_decay_param_group = {
        'params': no_decay_params,
        'weight_decay': weight_decay[1],
        'lr': lr[1],
        'betas': (beta1[1], beta2[1]),
        'eps': eps[1]
    }
    latent_param_group = {
        'params': latent_params,
        'weight_decay': weight_decay[2],
        'lr': lr[2],
        'betas': (beta1[2], beta2[2]),
        'eps': eps[2]
    }

    param_groups = [decay_param_group, no_decay_param_group, latent_param_group]
    optimizer = torch.optim.AdamW(param_groups)

    return optimizer

def make_optimizer_lion(model, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
    decay_params, no_decay_params, latent_params = split_params_adamw(model, NO_DECAY_TYPES)
    
    if isinstance(lr, float) or isinstance(lr, int):
        lr = [lr] * 3
    if isinstance(weight_decay, float) or isinstance(weight_decay, int):
        weight_decay = [weight_decay, 0.0, 0.0]
    if isinstance(beta1, float) or isinstance(beta1, int):
        beta1 = [beta1] * 3
    if isinstance(beta2, float) or isinstance(beta2, int):
        beta2 = [beta2] * 3
    if isinstance(eps, float) or isinstance(eps, int):
        eps = [eps] * 3

    assert len(lr) == 3, f"lr must be a list of 3 elements, got {lr} with {len(lr)} elements"
    assert len(weight_decay) == 3, f"weight_decay must be a list of 3 elements, got {weight_decay} with {len(weight_decay)} elements"
    assert len(beta1) == 3, f"beta1 must be a list of 3 elements, got {beta1} with {len(beta1)} elements"
    assert len(beta2) == 3, f"beta2 must be a list of 3 elements, got {beta2} with {len(beta2)} elements"
    assert len(eps) == 3, f"eps must be a list of 3 elements, got {eps} with {len(eps)} elements"

    decay_param_group = {
        'params': decay_params,
        'lr': lr[0],
        'weight_decay': weight_decay[0],
        'betas': (beta1[0], beta2[0]),
        'eps': eps[0]
    }
    no_decay_param_group = {
        'params': no_decay_params,
        'lr': lr[1],
        'weight_decay': weight_decay[1],
        'betas': (beta1[1], beta2[1]),
        'eps': eps[1]
    }
    latent_param_group = {
        'params': latent_params,
        'lr': lr[2],
        'weight_decay': weight_decay[2],
        'betas': (beta1[2], beta2[2]),
        'eps': eps[2]
    }
    
    param_groups = [decay_param_group, no_decay_param_group, latent_param_group]
    optimizer = Lion(param_groups)

    return optimizer

def make_optimizer_muon(model, lr, weight_decay=0.0, betas=None, eps=None, **kwargs):
    betas = betas if betas is not None else (0.9, 0.999)
    eps = eps if eps is not None else 1e-8

    import torch.distributed as dist
    try:
        is_distributed = dist.is_available() and dist.is_initialized()
    except:
        is_distributed = False

    if is_distributed:
        from .muon import MuonWithAuxAdam
    else:
        from .muon import SingleDeviceMuonWithAuxAdam

    adamw_params_decay = []
    adamw_params_no_decay = []
    adamw_params_latent = []
    muon_params_decay = []
    no_decay_param_ids = _collect_param_ids_by_module_types(model, NO_DECAY_TYPES)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights

        if "block" in name and param.ndim >= 2 and "latent" not in name:
            muon_params_decay.append(param)
        else:
            if (
                (no_decay_param_ids is not None and id(param) in no_decay_param_ids) or 
                name.endswith(".bias") or 
                "LayerNorm" in name or "layernorm" in name or 
                "RMSNorm" in name or "rmsnorm" in name or
                "pos_embed" in name or
                "embedding" in name.lower() or
                "cls_token" in name
            ):
                adamw_params_no_decay.append(param)
            elif (
                "latent" in name
            ):
                adamw_params_latent.append(param)
            else:
                adamw_params_decay.append(param)

    # assemble param groups
    if isinstance(lr, float) or isinstance(lr, int):
        lr = [lr] * 4
    if isinstance(weight_decay, float) or isinstance(weight_decay, int):
        weight_decay = [weight_decay, 0.0, 0.0, 0.0, weight_decay]
    if isinstance(beta1, float) or isinstance(beta1, int):
        beta1 = [beta1] * 4
    if isinstance(beta2, float) or isinstance(beta2, int):
        beta2 = [beta2] * 4
    if isinstance(eps, float) or isinstance(eps, int):
        eps = [eps] * 4

    assert len(lr) == 4, f"lr must be a list of 4 elements, got {lr} with {len(lr)} elements"
    assert len(weight_decay) == 4, f"weight_decay must be a list of 4 elements, got {weight_decay} with {len(weight_decay)} elements"
    assert len(beta1) == 4, f"beta1 must be a list of 4 elements, got {beta1} with {len(beta1)} elements"
    assert len(beta2) == 4, f"beta2 must be a list of 4 elements, got {beta2} with {len(beta2)} elements"
    assert len(eps) == 4, f"eps must be a list of 4 elements, got {eps} with {len(eps)} elements"

    adamw_decay_group = {
        'params': adamw_params_decay,
        'lr': lr[0],
        'weight_decay': weight_decay[0],
        'betas': (beta1[0], beta2[0]),
        'eps': eps[0],
        'use_muon': False,
    }
    adamw_no_decay_group = {
        'params': adamw_params_no_decay,
        'lr': lr[1],
        'weight_decay': weight_decay[1],
        'betas': (beta1[1], beta2[1]),
        'eps': eps[1],
        'use_muon': False,
    }
    adamw_latent_group = {
        'params': adamw_params_latent,
        'lr': lr[2],
        'weight_decay': weight_decay[2],
        'betas': (beta1[2], beta2[2]),
        'eps': eps[2],
        'use_muon': False,
    }
    muon_group = {
        'params': muon_params_decay,
        'lr': lr[3],
        'weight_decay': weight_decay[3],
        'momentum': beta1[3],
        'use_muon': True,
    }

    param_groups = [adamw_decay_group, adamw_no_decay_group, adamw_latent_group, muon_group]

    if is_distributed:
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    return optimizer

#======================================================================#
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    r"""
    Lion Optimizer (Chen et al., 2023):
    https://arxiv.org/abs/2302.06675

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        w_{t+1} = w_t - lr * sign(m_t)

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        betas (Tuple[float, float]): momentum coefficients (beta1, beta2). Note that beta2 is not used.
        weight_decay (float): optional weight decay (L2 penalty)
        eps (float): optional epsilon. Note that eps is not used.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay directly to weights
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # State (momentum) initialization
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Parameter update (sign of momentum)
                p.add_(exp_avg.sign(), alpha=-lr)

        return loss

#======================================================================#
def central_diff(x: torch.Tensor, h: float, resolution: int):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = einops.rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y

def darcy_deriv_loss(yh, y, s, dx):
    yh = einops.rearrange(yh, 'b (h w) c -> b c h w', h=s)
    yh = yh[..., 1:-1, 1:-1].contiguous()
    yh = F.pad(yh, (1, 1, 1, 1), "constant", 0)
    yh = einops.rearrange(yh, 'b c h w -> b (h w) c')

    gt_grad_x, gt_grad_y = central_diff(y, dx, s)
    pred_grad_x, pred_grad_y = central_diff(yh, dx, s)

    return (gt_grad_x, gt_grad_y), (pred_grad_x, pred_grad_y)

#======================================================================#
class IdentityNormalizer():
    def __init__(self):
        pass
    
    def to(self, device):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

#======================================================================#
class UnitCubeNormalizer():
    def __init__(self, X):
        xmin = X[:,:,0].min().item()
        ymin = X[:,:,1].min().item()

        xmax = X[:,:,0].max().item()
        ymax = X[:,:,1].max().item()

        self.min = torch.tensor([xmin, ymin])
        self.max = torch.tensor([xmax, ymax])

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)

        return self

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        return x * (self.max - self.min) + self.min

#======================================================================#
class UnitGaussianNormalizer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

#======================================================================#
class RelL2Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dim = tuple(range(1, pred.ndim))

        error = torch.sum((pred - target) ** 2, dim=dim).sqrt()
        target = torch.sum(target ** 2, dim=dim).sqrt()

        loss = torch.mean(error / target)
        return loss

class RelL1Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dim = tuple(range(1, pred.ndim))

        error = torch.sum(torch.abs(pred - target), dim=dim)
        target = torch.sum(torch.abs(target), dim=dim)

        loss = torch.mean(error / target)
        return loss

#======================================================================#
#
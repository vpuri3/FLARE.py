#
import torch
from torch import nn
from mlutils.utils import get_module

__all__ = [
    'EMA',
    'copy_model_state',
    'load_model_state',
]

#=======================================================================#
def copy_model_state(model):
    module = get_module(model)
    return {name: param.clone().detach()
        for name, param in module.named_parameters() if param.requires_grad
    }
    
def load_model_state(model, state_dict):
    module = get_module(model)
    for name, param in module.named_parameters():
        assert name in state_dict, f"Parameter {name} not found in state_dict"
        if not param.requires_grad:
            continue
        param.data.copy_(state_dict[name])

#=======================================================================#
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        module = get_module(model)

        self.shadow = { name: p.clone().detach()
            for name, p in module.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        module = get_module(model)
        for name, param in module.named_parameters():
            assert name in self.shadow, f"Parameter {name} not found in shadow"
            if not param.requires_grad:
                continue
            self.shadow[name].data = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].data

    @torch.no_grad()
    def load_ema_weights(self, model):
        """Copy EMA weights into target model (e.g., for evaluation)."""
        module = get_module(model)
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

#=======================================================================#
#
#
import math
import torch

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "LNO",
]

@dataclass
class LNOConfig:
    model: str = "lno"
    num_blocks: int = 8
    channel_dim: int = 64
    num_heads: int = 8
    num_modes: int = 256
    num_layers_kv_proj: int = 3
    act: Optional[str] = None


# "model": {
#     "name": "LNO",
#     "n_block": 4,
#     "n_mode": 256,
#     "n_dim" : 192,
#     "n_head" : 8,
#     "n_layer": 3,
#     "attn": "Attention_Vanilla",
#     "act": "GELU"
# },

ACTIVATION = {
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(),
    "LeakyReLU": torch.nn.LeakyReLU(0.1),
    "ELU": torch.nn.ELU(),
    "GELU": torch.nn.GELU()
}

#======================================================================#
def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = act
        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r
        
#======================================================================#
class SelfAttention(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, attn):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
        self.attn = attn
        self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
    
    def forward(self, x):
        B, N, D = x.size()
        q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        r = self.proj(r)
        return r

class AttentionBlock(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, act):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.act = act
        
        self.self_attn = SelfAttention(self.n_mode, self.n_dim, self.n_head, Attention_Vanilla)
        
        self.ln1 = torch.nn.LayerNorm(self.n_dim)
        self.ln2 = torch.nn.LayerNorm(self.n_dim)
        self.drop = torch.nn.Dropout(0.0)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim*2),
            self.act,
            torch.nn.Linear(self.n_dim*2, self.n_dim),
        )

    def forward(self, y):   
        y = y + self.drop(self.self_attn(self.ln1(y)))
        y = y + self.mlp(self.ln2(y))
        return y

#======================================================================#
    
class LNO(torch.nn.Module):
    def __init__(self, config: LNOConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else metadata
        dataset = metadata.get("dataset")
        n_block = config.num_blocks
        n_mode = config.num_modes
        n_dim = config.channel_dim
        n_head = config.num_heads
        n_layer = config.num_layers_kv_proj
        c_in = metadata["c_in"]
        c_out = metadata["c_out"]
        x_dim = c_in
        y1_dim = c_in
        y2_dim = c_out
        act = config.act or "GELU"
        model_attr = {"time": False if dataset in {"navier_stokes", "plasticity"} else metadata.get("time_cond")}

        if dataset in {"navier_stokes", "plasticity"}:
            x_dim = int(metadata["space_dim"])
            y1_dim = int(metadata.get("fun_dim", c_in - int(metadata["space_dim"])))
            if dataset == "plasticity":
                x_dim = int(metadata["space_dim"]) + 1
        if dataset == "plasticity":
            if model_attr["time"]:
                y2_dim = 1
            y1_dim = int(metadata.get("fun_dim", y1_dim))

        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]

        self.x_dim = x_dim
        self.y1_dim = y1_dim
        self.y2_dim = 1 if model_attr["time"] else y2_dim

        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)])

    def forward(self, x, y=None):
        if y is None:
            y = x
        x = self.trunk_projector(x)
        y = self.branch_projector(y)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        return r

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

#======================================================================#
#

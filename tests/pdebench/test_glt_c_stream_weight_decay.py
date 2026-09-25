"""Tests for GLT dual-stream (c_proj + blocks_c) optimizer weight decay groups."""

from torch import nn

from pdebench.utils import is_glt_c_stream_param, make_optimizer_adamw, split_params_adamw


class _DualStreamStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = nn.Linear(8, 8)
        self.x_proj = nn.Linear(8, 8)
        self.blocks_c = nn.ModuleList([nn.Linear(8, 8)])
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])


def test_is_glt_c_stream_param_matches_c_proj_and_blocks_c() -> None:
    assert is_glt_c_stream_param("glt.c_proj.weight")
    assert is_glt_c_stream_param("module.blocks_c.2.attn.q_proj.weight")
    assert is_glt_c_stream_param("blocks_c.0.weight")
    assert not is_glt_c_stream_param("glt.x_proj.weight")
    assert not is_glt_c_stream_param("glt.blocks.0.attn.q_proj.weight")


def test_make_optimizer_adamw_splits_c_stream_group() -> None:
    model = _DualStreamStub()
    opt = make_optimizer_adamw(model, lr=1e-3, weight_decay=1e-3, c_stream_weight_decay=5e-2)
    assert len(opt.param_groups) == 4
    assert opt.param_groups[0]["weight_decay"] == 1e-3
    assert opt.param_groups[1]["weight_decay"] == 5e-2
    assert opt.param_groups[2]["weight_decay"] == 0.0
    assert opt.param_groups[3]["weight_decay"] == 0.0

    c_stream_ids = {id(p) for p in opt.param_groups[1]["params"]}
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert c_stream_ids
    for name, param in model.named_parameters():
        if not param.requires_grad or name.endswith(".bias"):
            continue
        if is_glt_c_stream_param(name):
            assert id(param) in c_stream_ids, name
        elif "latent" not in name:
            assert id(param) in decay_ids, name


def test_split_params_without_c_stream_returns_three_groups() -> None:
    model = nn.Linear(4, 2)
    decay, no_decay, latent = split_params_adamw(model)
    assert len(decay) + len(no_decay) + len(latent) == sum(1 for p in model.parameters() if p.requires_grad)

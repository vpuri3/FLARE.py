from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import pdebench
from pdebench.models.transolver import Transolver


@dataclass
class Case:
    name: str
    ctor: Callable[[], torch.nn.Module]
    make_input: Callable[[int, int], tuple[tuple, dict]]
    expected_shape: Callable[[int, int], tuple[int, ...]]


CASES: list[Case] = [
    Case(
        name="FLAREModel",
        ctor=lambda: pdebench.FLAREModel(
            in_dim=4,
            out_dim=3,
            channel_dim=32,
            num_blocks=2,
            num_heads=4,
            num_latents=16,
            out_proj_norm=True,
            num_layers_in_out_proj=2,
            attn_scale=1.0,
            num_layers_k_proj=2,
            num_layers_v_proj=2,
            num_layers_ffn=2,
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="TransformerWrapper",
        ctor=lambda: pdebench.TransformerWrapper(
            in_dim=4,
            out_dim=3,
            channel_dim=32,
            num_blocks=2,
            num_heads=4,
            backend="transformer",
            out_proj_norm=True,
            num_layers_in_out_proj=2,
            mlp_ratio=2.0,
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="PerceiverIO",
        ctor=lambda: pdebench.PerceiverIO(
            in_dim=4,
            out_dim=3,
            channel_dim=32,
            num_blocks=2,
            num_heads=4,
            num_latents=16,
            cross_attn=False,
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="Transolver",
        ctor=lambda: Transolver(
            space_dim=2,
            fun_dim=0,
            out_dim=3,
            n_layers=2,
            n_hidden=32,
            n_head=4,
            slice_num=8,
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 2),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
]


def _run(case: Case, batch: int, seq: int) -> torch.Tensor:
    model = case.ctor().eval()
    args, kwargs = case.make_input(batch, seq)
    with torch.no_grad():
        out = model(*args, **kwargs)
    assert isinstance(out, torch.Tensor), f"{case.name} did not return a Tensor"
    return out


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_cpu_smoke(case: Case) -> None:
    out = _run(case, batch=2, seq=32)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_determinism_eval(case: Case) -> None:
    torch.manual_seed(0)
    model = case.ctor().eval()
    args, kwargs = case.make_input(2, 32)
    with torch.no_grad():
        y1 = model(*args, **kwargs)
        y2 = model(*args, **kwargs)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_state_dict_roundtrip(case: Case) -> None:
    torch.manual_seed(0)
    model_a = case.ctor().eval()
    model_b = case.ctor().eval()
    model_b.load_state_dict(model_a.state_dict(), strict=True)

    args, kwargs = case.make_input(2, 32)
    with torch.no_grad():
        y_a = model_a(*args, **kwargs)
        y_b = model_b(*args, **kwargs)
    assert torch.allclose(y_a, y_b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
@pytest.mark.parametrize("batch,seq", [(1, 16), (2, 32)])
def test_shape_matrix(case: Case, batch: int, seq: int) -> None:
    out = _run(case, batch=batch, seq=seq)
    assert tuple(out.shape) == case.expected_shape(batch, seq)

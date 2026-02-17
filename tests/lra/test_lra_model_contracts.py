from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from lra.models.trm import TRMWrapper
from lra.models.wrapper import ModelWrapper


@dataclass
class Case:
    name: str
    ctor: Callable[[], torch.nn.Module]
    make_input: Callable[[int, int], tuple[tuple, dict]]
    expected_shape: Callable[[int, int], tuple[int, ...]]


def _make_model_wrapper(backend: str, **kwargs) -> ModelWrapper:
    return ModelWrapper(
        task="text",
        vocab_size=128,
        num_labels=10,
        max_length=64,
        backend=backend,
        channel_dim=64,
        num_blocks=1,
        num_heads=4,
        **kwargs,
    )


CASES: list[Case] = [
    Case(
        name="ModelWrapper[transformer]",
        ctor=lambda: _make_model_wrapper("transformer", mlp_ratio=2.0),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[linear]",
        ctor=lambda: _make_model_wrapper(
            "linear",
            kernel="identity",
            norm_q=True,
            norm_k=True,
            mlp_ratio=2.0,
        ),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[flare]",
        ctor=lambda: _make_model_wrapper(
            "flare",
            num_latents=16,
            attn_scale=1.0,
            num_layers_kv_proj=2,
            kv_proj_hidden_dim=64,
            num_layers_ffn=2,
            ffn_hidden_dim=64,
        ),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="TRMWrapper[transformer]",
        ctor=lambda: TRMWrapper(
            task="text",
            vocab_size=128,
            num_labels=10,
            max_length=64,
            backend="transformer",
            channel_dim=64,
            num_blocks=1,
            num_heads=4,
            trm_N_steps=1,
            trm_n=1,
            trm_T=1,
        ),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
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
@pytest.mark.parametrize("batch,seq", [(1, 16), (2, 64)])
def test_shape_matrix(case: Case, batch: int, seq: int) -> None:
    out = _run(case, batch=batch, seq=seq)
    assert tuple(out.shape) == case.expected_shape(batch, seq)

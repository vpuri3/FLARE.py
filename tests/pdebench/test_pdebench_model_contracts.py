from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import pdebench
from pdebench.config import FlareConfig, PerceiverIOConfig, TransolverConfig, TransformerConfig
from pdebench.models import flare as flare_module


@dataclass
class Case:
    name: str
    ctor: Callable[[], torch.nn.Module]
    make_input: Callable[[int, int], tuple[tuple, dict]]
    expected_shape: Callable[[int, int], tuple[int, ...]]


def _make_perceiver() -> torch.nn.Module:
    config = PerceiverIOConfig(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_latents=16,
    )
    return pdebench.PerceiverIO(config, metadata={"c_in": 4, "c_out": 3, "dataset": "elasticity"})


CASES: list[Case] = [
    Case(
        name="FLAREModel",
        ctor=lambda: pdebench.FLAREModel(
            FlareConfig(
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
            metadata={"c_in": 4, "c_out": 3, "dataset": "elasticity"},
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="TransformerWrapper",
        ctor=lambda: pdebench.TransformerWrapper(
            TransformerConfig(
                channel_dim=32,
                num_blocks=2,
                num_heads=4,
                mlp_ratio=2.0,
                out_proj_norm=True,
                num_layers_in_out_proj=2,
            ),
            metadata={"c_in": 4, "c_out": 3, "dataset": "elasticity"},
        ),
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="PerceiverIO",
        ctor=_make_perceiver,
        make_input=lambda b, n: ((torch.randn(b, n, 4),), {}),
        expected_shape=lambda b, n: (b, n, 3),
    ),
    Case(
        name="Transolver",
        ctor=lambda: pdebench.Transolver(
            TransolverConfig(
                num_blocks=2,
                channel_dim=32,
                num_heads=4,
                num_slices=8,
            ),
            metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLARE packed flash-varlen parity requires CUDA.")
def test_flare_flash_varlen_preserves_sequence_boundaries() -> None:
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    model = flare_module.FLARE(
        channel_dim=16,
        num_heads=4,
        num_latents=4,
        rmsnorm=True,
    ).cuda().to(dtype=dtype).eval()
    x_a = torch.randn(5, 16, device="cuda", dtype=dtype)
    x_b = torch.randn(8, 16, device="cuda", dtype=dtype)
    flat = torch.cat([x_a, x_b], dim=0)
    cu_ab = torch.tensor([0, 5, 13], device="cuda", dtype=torch.int32)
    cu_a = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    cu_b = torch.tensor([0, 8], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        flat_out, _ = model.forward_flash_varlen(flat, cu_seqlens=cu_ab, max_seqlen=8)
        out_a, _ = model.forward_flash_varlen(x_a, cu_seqlens=cu_a, max_seqlen=5)
        out_b, _ = model.forward_flash_varlen(x_b, cu_seqlens=cu_b, max_seqlen=8)

    expected = torch.cat([out_a, out_b], dim=0)
    assert torch.allclose(flat_out, expected, atol=0.0, rtol=0.0)


def test_flare_sdpa_masks_only_keys_for_flash_compatibility(monkeypatch) -> None:
    calls = []
    original_sdpa = flare_module.F.scaled_dot_product_attention

    def _wrapped_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(flare_module.F, "scaled_dot_product_attention", _wrapped_sdpa)

    model = flare_module.FLARE(channel_dim=16, num_heads=4, num_latents=3, rmsnorm=True).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ]
    )

    with torch.no_grad():
        out, scores = model(x, mask=mask)

    assert out.shape == x.shape
    assert scores is None
    assert len(calls) == 2
    encode_mask = calls[0][1]["attn_mask"]
    decode_mask = calls[1][1]["attn_mask"]
    assert encode_mask.dtype == torch.bool
    assert encode_mask.shape == (2, 1, 1, 5)
    assert decode_mask is None


def test_flare_all_valid_mask_drops_sdpa_masks(monkeypatch) -> None:
    calls = []
    original_sdpa = flare_module.F.scaled_dot_product_attention

    def _wrapped_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(flare_module.F, "scaled_dot_product_attention", _wrapped_sdpa)

    model = flare_module.FLARE(channel_dim=16, num_heads=4, num_latents=3, rmsnorm=True).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        out, _ = model(x, mask=mask)

    assert out.shape == x.shape
    assert len(calls) == 2
    assert calls[0][1]["attn_mask"] is None
    assert calls[1][1]["attn_mask"] is None


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

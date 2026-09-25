from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from lra.models.backends import CosformerAttention
from lra.models.trm import TRMWrapper
from lra.models.backends import LinearAttentionBlock
from lra.models.external import ExternalModelWrapper
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


def _make_external_wrapper(backend: str, **kwargs) -> ExternalModelWrapper:
    return ExternalModelWrapper(
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


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


CASES: list[Case] = [
    Case(
        name="ModelWrapper[transformer]",
        ctor=lambda: _make_model_wrapper("transformer", mlp_ratio=2.0),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[transolver]",
        ctor=lambda: _make_model_wrapper("transolver", mlp_ratio=2.0, num_slices=8),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[cosformer]",
        ctor=lambda: _make_model_wrapper("cosformer", mlp_ratio=2.0),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[linformer+shared_kv]",
        ctor=lambda: _make_model_wrapper("linformer", seq_len=64, k=16, share_kv=True, mlp_ratio=2.0),
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
        name="ModelWrapper[flarepp]",
        ctor=lambda: _make_model_wrapper(
            "flarepp",
            num_latents=16,
            k_norm=True,
            share_k0_v0=True,
            q_fixed_norm=True,
            gate_logit_init=0.25,
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
    Case(
        name="ModelWrapper[normattention]",
        ctor=lambda: _make_model_wrapper(
            "normattention",
            num_layers_kv_proj=-1,
            kv_proj_mlp_ratio=1.0,
            num_layers_ffn=0,
            ffn_mlp_ratio=2.0,
            qk_dim_ratio=1.0,
        ),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
    Case(
        name="ModelWrapper[performer+favor_pp]",
        ctor=lambda: _make_model_wrapper(
            "performer",
            nb_features=32,
            feature_map="favor_pp",
            redraw_interval=0,
            normalize_inputs=True,
            mlp_ratio=2.0,
        ),
        make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
        expected_shape=lambda b, n: (b, 10),
    ),
]


if "attention_mask" in inspect.signature(LinearAttentionBlock.forward).parameters:
    CASES.insert(
        1,
        Case(
            name="ModelWrapper[linear]",
            ctor=lambda: _make_model_wrapper(
                "linear",
                kernel="identity",
                q_norm=True,
                k_norm=True,
                mlp_ratio=2.0,
            ),
            make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
            expected_shape=lambda b, n: (b, 10),
        ),
    )
    CASES.insert(
        2,
        Case(
            name="ModelWrapper[linear+hedgehog]",
            ctor=lambda: _make_model_wrapper(
                "linear",
                kernel="hedgehog",
                q_norm=True,
                k_norm=True,
                mlp_ratio=2.0,
            ),
            make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
            expected_shape=lambda b, n: (b, 10),
        ),
    )


if HAS_TRANSFORMERS:
    CASES.extend(
        [
            Case(
                name="ExternalModelWrapper[funnel_hf]",
                ctor=lambda: _make_external_wrapper("funnel_hf", mlp_ratio=2.0),
                make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
                expected_shape=lambda b, n: (b, 10),
            ),
            Case(
                name="ExternalModelWrapper[reformer_hf]",
                ctor=lambda: _make_external_wrapper("reformer_hf", mlp_ratio=2.0),
                make_input=lambda b, n: ((torch.randint(0, 128, (b, n)),), {}),
                expected_shape=lambda b, n: (b, 10),
            ),
        ]
    )


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


@pytest.mark.parametrize("causal", [False, True])
def test_cosformer_left_product_matches_forward(causal: bool) -> None:
    torch.manual_seed(0)
    model = CosformerAttention(embed_dim=64, num_heads=4, causal=causal).eval()
    x = torch.randn(2, 16, 64)
    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    with torch.no_grad():
        fast = model(x, attention_mask=attention_mask)
        dense = model.left_product(x, attention_mask=attention_mask)

    assert torch.allclose(fast, dense, atol=1e-5, rtol=1e-4)

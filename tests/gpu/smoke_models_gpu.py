#!/usr/bin/env python3
"""
GPU smoke test for lra/pdebench model modules.

What it checks:
1) Source compiles via py_compile.
2) Model class can be instantiated.
3) Model state_dict can be loaded into a fresh instance (strict=True).
4) One CUDA forward pass runs successfully.

Usage:
  source .venv/bin/activate
  python tests/gpu/smoke_models_gpu.py
"""

from __future__ import annotations

import glob
import importlib.util
import inspect
import json
import os
import py_compile
import sys
from dataclasses import dataclass
from typing import Callable

import torch

from pdebench.config import (
    FlareAblationsConfig,
    FlareConfig,
    FlareExperimentalConfig,
    GNOTConfig,
    LaMOConfig,
    LNOConfig,
    PerceiverIOConfig,
    TransolverConfig,
    TransolverPlusPlusConfig,
    TransformerConfig,
)


@dataclass
class Case:
    name: str
    ctor: Callable[[], torch.nn.Module]
    make_input: Callable[[torch.device], tuple]
    expected_fail: bool = False
    reason: str = ""


def _compile_sources(repo_root: str) -> list[str]:
    patterns = [
        os.path.join(repo_root, "lra", "models", "**", "*.py"),
        os.path.join(repo_root, "pdebench", "models", "**", "*.py"),
        os.path.join(repo_root, "lra", "__main__.py"),
        os.path.join(repo_root, "pdebench", "__main__.py"),
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    errors: list[str] = []
    for path in sorted(set(files)):
        try:
            py_compile.compile(path, doraise=True)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def _run_case(case: Case, device: torch.device) -> dict:
    rec = {
        "name": case.name,
        "expected_fail": case.expected_fail,
        "ok": False,
    }
    try:
        model = case.ctor().to(device).eval()
        rec["num_params"] = sum(p.numel() for p in model.parameters())

        # strict load check on fresh instance
        model2 = case.ctor().to(device).eval()
        model2.load_state_dict(model.state_dict(), strict=True)

        args, kwargs = case.make_input(device)
        with torch.no_grad():
            out = model(*args, **kwargs)

        if isinstance(out, tuple):
            out_shape = tuple(out[0].shape) if hasattr(out[0], "shape") else str(type(out[0]))
        else:
            out_shape = tuple(out.shape) if hasattr(out, "shape") else str(type(out))
        rec["out_shape"] = out_shape
        rec["ok"] = True
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    compile_errors = _compile_sources(repo_root)

    if not torch.cuda.is_available():
        print("CUDA not available; GPU smoke test requires CUDA.")
        return 2
    device = torch.device("cuda")

    # imports after compile check
    import pdebench
    import lra

    from lra.models.backends import MODEL_TYPES
    from lra.models.backends import LinearAttentionBlock
    from lra.models.external import ExternalModelWrapper
    from lra.models.trm import TRMWrapper
    from lra.models.wrapper import ModelWrapper

    try:
        from pdebench.models.flare_ablations import BigFLAREModel
        has_flare_ablations = True
    except ModuleNotFoundError:
        has_flare_ablations = False
    try:
        from pdebench.models.flare_experimental import FLAREExperimentalModel
        has_flare_experimental = True
    except ModuleNotFoundError:
        has_flare_experimental = False
    from pdebench.models.loopy import LoopyWrapper
    from pdebench.models.unloopy import UnloopyWrapper
    from pdebench.models.upt import UPT
    try:
        import pdebench.models.lamo  # noqa: F401
        has_lamo = True
    except ModuleNotFoundError:
        has_lamo = False
    has_transformers = importlib.util.find_spec("transformers") is not None

    cases: list[Case] = [
        # pdebench wrappers/models
        Case(
            "pdebench.FLAREModel",
            lambda: pdebench.FLAREModel(
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
            lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
        ),
        Case(
            "pdebench.TransformerWrapper",
            lambda: pdebench.TransformerWrapper(
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
            lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
        ),
        Case(
            "pdebench.PerceiverIO",
            lambda: pdebench.PerceiverIO(
                PerceiverIOConfig(channel_dim=32, num_blocks=2, num_heads=4, num_latents=16),
                metadata={"c_in": 4, "c_out": 3, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
        ),
        Case(
            "pdebench.LoopyWrapper",
            lambda: LoopyWrapper(
                in_dim=4,
                out_dim=3,
                channel_dim=32,
                num_blocks=2,
                num_heads=4,
                num_latents=16,
                num_passes=1,
            ),
            lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
        ),
        Case(
            "pdebench.UnloopyWrapper",
            lambda: UnloopyWrapper(
                in_dim=4,
                out_dim=3,
                channel_dim=32,
                num_blocks=2,
                num_heads=4,
                num_latents=16,
                shared_ffn=False,
                shared_att=False,
                gating=False,
            ),
            lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
        ),
        Case(
            "pdebench.Transolver",
            lambda: pdebench.Transolver(
                TransolverConfig(channel_dim=32, num_blocks=2, num_heads=4, num_slices=8),
                metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
        ),
        Case(
            "pdebench.Transolver_Structured_Mesh_2D",
            lambda: pdebench.Transolver_Structured_Mesh_2D(
                TransolverConfig(channel_dim=32, num_blocks=2, num_heads=4, num_slices=8, conv2d=True, unified_pos=False),
                metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "H": 8, "W": 8, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
        ),
        Case(
            "pdebench.TransolverPlusPlus",
            lambda: pdebench.TransolverPlusPlus(
                TransolverPlusPlusConfig(channel_dim=32, num_blocks=2, num_heads=4, num_slices=8),
                metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
        ),
        Case(
            "pdebench.LNO",
            lambda: pdebench.LNO(
                LNOConfig(num_blocks=2, channel_dim=32, num_heads=4, num_modes=16, num_layers_kv_proj=1, act="GELU"),
                metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
        ),
        Case(
            "pdebench.GNOT",
            lambda: pdebench.GNOT(
                GNOTConfig(num_blocks=2, channel_dim=32, num_heads=4, num_experts=2, mlp_ratio=2.0),
                metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
            ),
            lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
        ),
        Case(
            "pdebench.UPT",
            lambda: UPT(
                space_dim=2,
                fun_dim=2,
                out_dim=3,
                n_encoder_layers=1,
                n_approximator_layers=1,
                n_decoder_layers=1,
                num_supernodes=16,
                channel_dim=32,
                num_heads=4,
            ),
            lambda d: ((torch.randn(2, 64, 4, device=d),), {}),
            expected_fail=True,
            reason="Known incomplete implementation in repository.",
        ),
        # lra wrappers
        Case(
            "lra.ModelWrapper[transformer]",
            lambda: ModelWrapper(
                task="text",
                vocab_size=128,
                num_labels=10,
                max_length=64,
                backend="transformer",
                channel_dim=64,
                num_blocks=2,
                num_heads=4,
            ),
            lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
        ),
        Case(
            "lra.TRMWrapper[transformer]",
            lambda: TRMWrapper(
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
            lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
        ),
    ]

    if has_flare_experimental:
        cases.append(
            Case(
                "pdebench.FLAREExperimentalModel",
                lambda: pdebench.FLAREExperimentalModel(
                    FlareExperimentalConfig(
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
                lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
            )
        )
    else:
        print("Skipping FLAREExperimentalModel GPU smoke case: module unavailable.")

    if has_flare_ablations:
        cases.append(
            Case(
                "pdebench.BigFLAREModel",
                lambda: pdebench.BigFLAREModel(
                    FlareAblationsConfig(channel_dim=32, num_blocks=2, num_heads=4, num_latents=16),
                    metadata={"c_in": 4, "c_out": 3, "dataset": "elasticity"},
                ),
                lambda d: ((torch.randn(2, 32, 4, device=d),), {}),
            )
        )
    else:
        print("Skipping BigFLAREModel GPU smoke case: module unavailable.")

    if has_lamo:
        cases.extend(
            [
                Case(
                    "pdebench.LaMO",
                    lambda: pdebench.LaMO(
                        LaMOConfig(num_blocks=2, channel_dim=128, num_heads=4, num_slices=8),
                        metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
                    ),
                    lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
                ),
                Case(
                    "pdebench.LaMO_Structured_Mesh_2D",
                    lambda: pdebench.LaMO_Structured_Mesh_2D(
                        LaMOConfig(num_blocks=2, channel_dim=32, num_heads=4, num_slices=8, conv2d=True, unified_pos=False),
                        metadata={"c_in": 2, "c_out": 3, "space_dim": 2, "fun_dim": 0, "H": 8, "W": 8, "dataset": "elasticity"},
                    ),
                    lambda d: ((torch.randn(2, 64, 2, device=d),), {}),
                ),
            ]
        )
    else:
        print("Skipping LaMO GPU smoke cases: optional mamba_ssm dependencies unavailable.")

    # lra backend sweep via ModelWrapper
    backend_kwargs = {
        "linformer": {"seq_len": 64, "k": 16, "share_kv": False},
        "linear": {"kernel": "identity", "q_norm": True, "k_norm": True, "mlp_ratio": 2.0},
        "multilinear": {
            "num_states": 2,
            "num_layers_kv_proj": -1,
            "kv_proj_mlp_ratio": 1.0,
            "num_layers_ffn": 0,
            "ffn_mlp_ratio": 2.0,
            "kernel": "identity",
            "q_norm": False,
            "k_norm": False,
            "qk_dim_ratio": 1.0,
        },
        "triple": {
            "num_layers_kv_proj": -1,
            "kv_proj_mlp_ratio": 1.0,
            "num_layers_ffn": 0,
            "ffn_mlp_ratio": 2.0,
            "kernel": "identity",
            "q_norm": False,
            "k_norm": False,
            "qk_dim_ratio": 1.0,
            "use_triton": False,
        },
        "triple1": {"qk_dim_ratio": 1.0, "use_triton": False, "mlp_ratio": 2.0},
        "quad": {
            "num_layers_kv_proj": -1,
            "kv_proj_mlp_ratio": 1.0,
            "num_layers_ffn": 0,
            "ffn_mlp_ratio": 2.0,
            "kernel": "identity",
            "q_norm": False,
            "k_norm": False,
            "qk_dim_ratio": 1.0,
        },
        "strassen": {
            "num_layers_kv_proj": -1,
            "kv_proj_mlp_ratio": 1.0,
            "num_layers_ffn": 0,
            "ffn_mlp_ratio": 2.0,
            "kernel": "identity",
            "q_norm": False,
            "k_norm": False,
            "qk_dim_ratio": 1.0,
        },
        "ema": {"mlp_ratio": 2.0},
        "third_order": {"mlp_ratio": 2.0},
        "performer": {"mlp_ratio": 2.0, "nb_features": 64, "feature_map": "favor_plus", "redraw_interval": 0},
        "flare": {
            "num_latents": 16,
            "attn_scale": 1.0,
            "num_layers_kv_proj": 2,
            "kv_proj_hidden_dim": 64,
            "num_layers_ffn": 2,
            "ffn_hidden_dim": 64,
        },
    }

    for backend in sorted(MODEL_TYPES.keys()):
        if (
            backend == "linear"
            and "attention_mask" not in inspect.signature(LinearAttentionBlock.forward).parameters
        ):
            print("Skipping lra.ModelWrapper[linear] GPU smoke case: backend forward lacks attention_mask.")
            continue
        kw = backend_kwargs.get(backend, {"mlp_ratio": 2.0})
        cases.append(
            Case(
                name=f"lra.ModelWrapper[{backend}]",
                ctor=lambda b=backend, k=kw: ModelWrapper(
                    task="text",
                    vocab_size=128,
                    num_labels=10,
                    max_length=64,
                    backend=b,
                    channel_dim=64,
                    num_blocks=1,
                    num_heads=4,
                    **k,
                ),
                make_input=lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
            )
        )

    if "attention_mask" in inspect.signature(LinearAttentionBlock.forward).parameters:
        cases.append(
            Case(
                "lra.ModelWrapper[linear+hedgehog]",
                lambda: ModelWrapper(
                    task="text",
                    vocab_size=128,
                    num_labels=10,
                    max_length=64,
                    backend="linear",
                    channel_dim=64,
                    num_blocks=1,
                    num_heads=4,
                    kernel="hedgehog",
                    q_norm=True,
                    k_norm=True,
                    mlp_ratio=2.0,
                ),
                lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
            )
        )

    cases.extend(
        [
            Case(
                "lra.ModelWrapper[linformer+shared_kv]",
                lambda: ModelWrapper(
                    task="text",
                    vocab_size=128,
                    num_labels=10,
                    max_length=64,
                    backend="linformer",
                    channel_dim=64,
                    num_blocks=1,
                    num_heads=4,
                    seq_len=64,
                    k=16,
                    share_kv=True,
                    mlp_ratio=2.0,
                ),
                lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
            ),
            Case(
                "lra.ModelWrapper[performer+favor_pp]",
                lambda: ModelWrapper(
                    task="text",
                    vocab_size=128,
                    num_labels=10,
                    max_length=64,
                    backend="performer",
                    channel_dim=64,
                    num_blocks=1,
                    num_heads=4,
                    nb_features=32,
                    feature_map="favor_pp",
                    redraw_interval=0,
                    mlp_ratio=2.0,
                ),
                lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
            ),
        ]
    )

    if has_transformers:
        cases.extend(
            [
                Case(
                    "lra.ExternalModelWrapper[funnel_hf]",
                    lambda: ExternalModelWrapper(
                        task="text",
                        vocab_size=128,
                        num_labels=10,
                        max_length=64,
                        backend="funnel_hf",
                        channel_dim=64,
                        num_blocks=1,
                        num_heads=4,
                        mlp_ratio=2.0,
                    ),
                    lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
                ),
                Case(
                    "lra.ExternalModelWrapper[reformer_hf]",
                    lambda: ExternalModelWrapper(
                        task="text",
                        vocab_size=128,
                        num_labels=10,
                        max_length=64,
                        backend="reformer_hf",
                        channel_dim=64,
                        num_blocks=1,
                        num_heads=4,
                        mlp_ratio=2.0,
                    ),
                    lambda d: ((torch.randint(0, 128, (2, 64), device=d),), {}),
                ),
            ]
        )

    records = []
    for case in cases:
        rec = _run_case(case, device)
        records.append(rec)
        print(json.dumps(rec, sort_keys=True))

    failures = [r for r in records if not r.get("ok", False) and not r.get("expected_fail", False)]
    expected_failures = [r for r in records if not r.get("ok", False) and r.get("expected_fail", False)]

    print("\n=== Smoke Summary ===")
    print(f"compile_errors={len(compile_errors)}")
    print(f"total_cases={len(records)}")
    print(f"passed={len(records) - len(failures) - len(expected_failures)}")
    print(f"expected_failures={len(expected_failures)}")
    print(f"unexpected_failures={len(failures)}")

    if compile_errors:
        print("\nCompile errors:")
        for e in compile_errors:
            print(f"- {e}")

    if expected_failures:
        print("\nExpected failures:")
        for r in expected_failures:
            msg = r.get("error", "unknown")
            print(f"- {r['name']}: {msg}")

    if failures:
        print("\nUnexpected failures:")
        for r in failures:
            msg = r.get("error", "unknown")
            print(f"- {r['name']}: {msg}")

    if compile_errors or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

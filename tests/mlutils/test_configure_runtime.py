from __future__ import annotations

import os

import pytest
import torch

import mlutils


@pytest.fixture
def restore_runtime_backends():
    """Isolate backend + env mutations from other tests."""
    prev = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_det": torch.backends.cudnn.deterministic,
        "cudnn_bench": torch.backends.cudnn.benchmark,
        "precision": torch.get_float32_matmul_precision(),
        "det_algs": torch.are_deterministic_algorithms_enabled(),
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = prev["cudnn_tf32"]
    torch.backends.cudnn.deterministic = prev["cudnn_det"]
    torch.backends.cudnn.benchmark = prev["cudnn_bench"]
    torch.set_float32_matmul_precision(prev["precision"])
    torch.use_deterministic_algorithms(prev["det_algs"])
    if prev["cublas"] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev["cublas"]


def test_fidelity_profile_disables_tf32(restore_runtime_backends) -> None:
    status = mlutils.configure_runtime(0, mixed_precision=False, deterministic=False, compile_model=False)
    assert status["profile"] == "fidelity"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.are_deterministic_algorithms_enabled() is False
    assert status["tf32"] is False
    assert status["cudnn_benchmark"] is False
    assert status["deterministic_algorithms"] is False


def test_speed_profile_enables_tf32(restore_runtime_backends) -> None:
    status = mlutils.configure_runtime(0, mixed_precision=True, deterministic=False, compile_model=False)
    assert status["profile"] == "speed"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.are_deterministic_algorithms_enabled() is False
    assert status["tf32"] is True


def test_strict_profile_enables_deterministic_algorithms(restore_runtime_backends) -> None:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    status = mlutils.configure_runtime(0, mixed_precision=False, deterministic=True, compile_model=False)
    assert status["profile"] == "strict"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert status["deterministic_algorithms"] is True


def test_strict_refuses_mixed_precision(restore_runtime_backends) -> None:
    with pytest.raises(ValueError, match="deterministic"):
        mlutils.configure_runtime(0, mixed_precision=True, deterministic=True, compile_model=False)


def test_strict_refuses_compile_model(restore_runtime_backends) -> None:
    with pytest.raises(ValueError, match="compile"):
        mlutils.configure_runtime(0, mixed_precision=False, deterministic=True, compile_model=True)


def test_bare_set_seed_uses_speed_profile(restore_runtime_backends) -> None:
    status = mlutils.set_seed(0)
    assert status["profile"] == "speed"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True

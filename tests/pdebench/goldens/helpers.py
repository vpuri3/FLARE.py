from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from pdebench.dataset.sample import Sample

_GOLDENS_DIR = Path(__file__).resolve().parent


def load_manifest() -> dict[str, Any]:
    return json.loads((_GOLDENS_DIR / "manifest.json").read_text())


def assert_tensor_equal(a: torch.Tensor | None, b: torch.Tensor | None, *, name: str) -> None:
    if a is None and b is None:
        return
    assert a is not None and b is not None, f"{name}: None mismatch"
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
    assert torch.equal(a.cpu(), b.cpu()), f"{name}: values differ"


def assert_sample_roundtrip(
    sample: Sample,
    *,
    to_legacy: Callable[[Sample], Any],
    from_legacy: Callable[[Any], Sample],
    fields: Iterable[str] = (
        "pos",
        "y",
        "edge_index",
        "edge_attr",
        "feats",
        "laplacian_eig",
        "laplacian_eigvals",
        "sample_id",
        "kind",
    ),
) -> Sample:
    legacy = to_legacy(sample)
    out = from_legacy(legacy)
    for name in fields:
        left = getattr(sample, name)
        right = getattr(out, name)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            assert_tensor_equal(left, right, name=name)
        else:
            assert left == right, f"{name}: {left!r} != {right!r}"
    return out

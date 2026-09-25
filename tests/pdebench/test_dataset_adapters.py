from __future__ import annotations

import inspect

from pdebench.dataset.registry import CANONICAL, resolve_dataset_name

_REMOVED_DUAL_FEATURE_KWARGS = (
    "ginot_include_edges",
    "ginot_include_padded_boundary",
    "ginot_laplacian_eig_dim",
    "ginot_laplacian_spec",
    "plaid_laplacian_eig_dim",
    "plaid_laplacian_spec",
)


def test_load_dataset_signature_has_no_dual_feature_kwargs() -> None:
    """C5: features come only from ``feature_request``; dual-prefixed feature
    kwargs must not be part of the public ``load_dataset`` surface."""
    from pdebench.dataset.utils import load_dataset

    params = inspect.signature(load_dataset).parameters
    for name in _REMOVED_DUAL_FEATURE_KWARGS:
        assert name not in params, f"load_dataset must not accept {name!r}"
    assert "feature_request" in params


def test_every_canonical_has_adapter() -> None:
    from pdebench.dataset.adapters import get_adapter

    for name in sorted(CANONICAL):
        assert get_adapter(name).name == name


def test_get_adapter_resolves_alias() -> None:
    from pdebench.dataset.adapters import get_adapter

    assert get_adapter("tensile2d").name == "plaid_tensile2d"


def test_load_dataset_routes_through_adapter(monkeypatch, tmp_path) -> None:
    import pdebench.dataset.utils as utils
    from pdebench.dataset.adapters import get_adapter

    adapter = get_adapter("poisson_unstructured")
    calls: list[str] = []

    def wrapped(data_root: str, **kwargs):
        calls.append(adapter.name)
        return ("train", "test", {"dataset": adapter.name})

    monkeypatch.setattr(adapter, "load", wrapped)
    # Ensure get_adapter returns same instance
    monkeypatch.setattr(
        "pdebench.dataset.adapters.get_adapter",
        lambda name, _orig=get_adapter: (
            adapter if resolve_dataset_name(str(name).lower()) == adapter.name else _orig(name)
        ),
    )

    train, test, meta = utils.load_dataset(
        "poisson_unstructured",
        str(tmp_path),
        str(tmp_path),
    )
    assert calls == ["poisson_unstructured"]
    assert meta["dataset"] == "poisson_unstructured"

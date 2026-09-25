from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from pdebench.dataset.sample import FeatureRequest, SampleKind
from pdebench.dataset.sample_bridge import pyg_data_to_sample, sample_to_pyg_data
from tests.pdebench.goldens.helpers import assert_sample_roundtrip


def _tiny_pyg() -> Data:
    return Data(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        x=torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        y=torch.tensor([[0.5], [1.5]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        sample_id=torch.tensor([7], dtype=torch.long),
    )


def _tiny_pyg_with_context() -> Data:
    """elpl dynamics graphs carry ``input_scalars`` / ``context`` on the PyG ``Data``."""
    data = _tiny_pyg()
    data.input_scalars = torch.tensor([[0.25]], dtype=torch.float32)
    data.context = data.input_scalars
    return data


def test_pyg_roundtrip_preserves_core_fields() -> None:
    data = _tiny_pyg()
    sample = pyg_data_to_sample(data, kind=SampleKind.STATIC)
    assert sample.sample_id == "7"
    assert sample.kind == SampleKind.STATIC
    assert_sample_roundtrip(
        sample,
        to_legacy=lambda s: sample_to_pyg_data(s),
        from_legacy=lambda d: pyg_data_to_sample(d, kind=SampleKind.STATIC),
        fields=("pos", "y", "edge_index", "edge_attr", "feats", "sample_id", "kind"),
    )


def test_pyg_roundtrip_preserves_context_and_input_scalars() -> None:
    data = _tiny_pyg_with_context()
    sample = pyg_data_to_sample(data, kind=SampleKind.STATIC)
    assert torch.equal(sample.context, data.context)
    assert torch.equal(sample.extras["input_scalars"], data.input_scalars)
    restored = sample_to_pyg_data(sample)
    assert torch.equal(restored.context, data.context)
    assert torch.equal(restored.input_scalars, data.input_scalars)


def test_pyg_data_to_sample_missing_sample_id_defaults_to_zero() -> None:
    data = Data(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        y=torch.tensor([[0.5], [1.5]], dtype=torch.float32),
    )
    sample = pyg_data_to_sample(data, kind=SampleKind.STATIC)
    assert sample.sample_id == "0"


def test_wrapper_getitem_roundtrips(monkeypatch, tmp_path) -> None:
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample_wrappers import RoundTripPygDataset

    base = GraphListDataset([_tiny_pyg()])
    wrapped = RoundTripPygDataset(base, kind=SampleKind.STATIC)
    out = wrapped[0]
    assert torch.equal(out.pos, base[0].pos)
    assert torch.equal(out.y, base[0].y)


def test_plaid_canonical_adapter_wraps_tensile2d_and_hyperelasticity(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample_wrappers import RoundTripPygDataset

    for name in ("plaid_tensile2d", "plaid_hyperelasticity"):
        train_ds = GraphListDataset([_tiny_pyg()])
        test_ds = GraphListDataset([_tiny_pyg()])
        mesh_test_ds = GraphListDataset([_tiny_pyg()])

        def _fake_mesh_loader(**kwargs):
            return train_ds, test_ds, {"meta": 1, "mesh_test_data": mesh_test_ds}

        monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert isinstance(train, RoundTripPygDataset)
        assert isinstance(test, RoundTripPygDataset)
        assert isinstance(meta["mesh_test_data"], RoundTripPygDataset)
        assert meta["sample_bridge"] == "c2a_pyg"
        assert torch.equal(train[0].pos, train_ds[0].pos)
        assert torch.equal(test[0].pos, test_ds[0].pos)


def test_pyg_roundtrip_preserves_terminal_kind() -> None:
    data = _tiny_pyg()
    sample = pyg_data_to_sample(data, kind=SampleKind.TERMINAL)
    assert sample.kind == SampleKind.TERMINAL
    assert_sample_roundtrip(
        sample,
        to_legacy=lambda s: sample_to_pyg_data(s),
        from_legacy=lambda d: pyg_data_to_sample(d, kind=SampleKind.TERMINAL),
        fields=("pos", "y", "edge_index", "edge_attr", "feats", "sample_id", "kind"),
    )


def test_wrapper_getitem_roundtrips_terminal_kind() -> None:
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample_wrappers import RoundTripPygDataset

    base = GraphListDataset([_tiny_pyg()])
    wrapped = RoundTripPygDataset(base, kind=SampleKind.TERMINAL)
    out = wrapped[0]
    assert torch.equal(out.pos, base[0].pos)
    assert torch.equal(out.y, base[0].y)


def test_plaid_canonical_adapter_wraps_elpl_dynamics_and_terminal(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample_wrappers import RoundTripPygDataset

    cases = (
        ("plaid_el_pl_dynamics", SampleKind.STATIC),
        ("plaid_elpl_terminal", SampleKind.TERMINAL),
    )
    for name, expected_kind in cases:
        train_ds = GraphListDataset([_tiny_pyg()])
        test_ds = GraphListDataset([_tiny_pyg()])
        mesh_test_ds = GraphListDataset([_tiny_pyg()])

        def _fake_mesh_loader(**kwargs):
            return train_ds, test_ds, {"meta": 1, "mesh_test_data": mesh_test_ds}

        monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert isinstance(train, RoundTripPygDataset)
        assert isinstance(test, RoundTripPygDataset)
        assert isinstance(meta["mesh_test_data"], RoundTripPygDataset)
        assert train.kind == expected_kind
        assert test.kind == expected_kind
        assert meta["mesh_test_data"].kind == expected_kind
        assert meta["sample_bridge"] == "c2a_pyg"
        assert torch.equal(train[0].pos, train_ds[0].pos)
        assert torch.equal(test[0].pos, test_ds[0].pos)


def test_plaid_canonical_adapter_skips_wrap_for_non_dataset_sentinels(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter

    def _fake_mesh_loader(**kwargs):
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    for name in ("plaid_el_pl_dynamics", "plaid_elpl_terminal"):
        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert (train, test) == ("train", "test")
        assert meta == {"meta": 1, "sample_bridge": "c2a_pyg"}


def test_wrapper_yield_sample_mode_returns_sample_not_data() -> None:
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample import Sample
    from pdebench.dataset.sample_wrappers import RoundTripPygDataset

    base = GraphListDataset([_tiny_pyg()])
    wrapped = RoundTripPygDataset(base, kind=SampleKind.STATIC, yield_sample=True)
    out = wrapped[0]
    assert isinstance(out, Sample)
    assert torch.equal(out.pos, base[0].pos)
    assert torch.equal(out.y, base[0].y)


_PLAID_SAMPLE_COLLATE_NAMES = (
    "plaid_tensile2d",
    "plaid_hyperelasticity",
    "plaid_el_pl_dynamics",
    "plaid_elpl_terminal",
)


def test_plaid_canonical_adapter_sample_collate_default_off(monkeypatch) -> None:
    """PDEBENCH_SAMPLE_COLLATE unset: dataset still yields Data (C2a only, no C4)."""
    from torch_geometric.data import Data as PygData

    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset

    monkeypatch.delenv("PDEBENCH_SAMPLE_COLLATE", raising=False)

    train_ds = GraphListDataset([_tiny_pyg()])
    test_ds = GraphListDataset([_tiny_pyg()])

    def _fake_mesh_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    for name in _PLAID_SAMPLE_COLLATE_NAMES:
        adapter = get_adapter(name)
        train, _test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert "sample_collate" not in meta
        assert isinstance(train[0], PygData)


@pytest.mark.parametrize("env_value", ["plaid_static", "1"])
def test_plaid_canonical_adapter_sample_collate_enabled_via_env(monkeypatch, env_value) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample import Sample
    from pdebench.dataset.sample_collate import collate_plaid_static

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", env_value)

    for name in _PLAID_SAMPLE_COLLATE_NAMES:
        train_ds = GraphListDataset([_tiny_pyg(), _tiny_pyg()])
        test_ds = GraphListDataset([_tiny_pyg()])

        def _fake_mesh_loader(**kwargs):
            return train_ds, test_ds, {"meta": 1}

        monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert meta["sample_collate"] is True
        assert meta["collate_fn_name"] == "plaid_static_sample"
        assert meta["train_collate_fn"] is collate_plaid_static
        assert meta["eval_collate_fn"] is collate_plaid_static
        assert isinstance(train[0], Sample)
        assert isinstance(test[0], Sample)

        batch = meta["train_collate_fn"]([train[0], train[1]])
        expected = Batch.from_data_list([_tiny_pyg(), _tiny_pyg()])
        for field in ("x", "y", "pos", "edge_index", "batch"):
            assert torch.equal(getattr(batch, field), getattr(expected, field)), field


@pytest.mark.parametrize("name", ("plaid_el_pl_dynamics", "plaid_elpl_terminal"))
def test_plaid_elpl_sample_collate_enabled_via_explicit_env(monkeypatch, name) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample import Sample
    from pdebench.dataset.sample_collate import collate_plaid_static

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", name)

    graph = _tiny_pyg_with_context() if name == "plaid_el_pl_dynamics" else _tiny_pyg()
    train_ds = GraphListDataset([graph, graph])
    test_ds = GraphListDataset([graph])

    def _fake_mesh_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    adapter = get_adapter(name)
    train, test, meta = adapter.load(
        "unused",
        mesh_split_seed=0,
        mesh_graph_backend="pyg",
        plaid_use_sdf_features=True,
        plaid_max_samples=0,
        plaid_load_public_test=False,
        plaid_terminal_y_norm="asinh_iqr",
        plaid_terminal_target_fields=None,
        feature_request=FeatureRequest(),
    )

    assert meta["sample_collate"] is True
    assert meta["train_collate_fn"] is collate_plaid_static
    assert isinstance(train[0], Sample)

    batch = meta["train_collate_fn"]([train[0], train[1]])
    expected = Batch.from_data_list([graph, graph])
    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(batch, field), getattr(expected, field)), field
    if name == "plaid_el_pl_dynamics":
        assert torch.equal(batch.context, expected.context)
        assert torch.equal(batch.input_scalars, expected.input_scalars)


def test_plaid_canonical_adapter_sample_collate_partial_explicit_list(monkeypatch) -> None:
    """Comma-separated env enables only the listed PLAID families."""
    from torch_geometric.data import Data as PygData

    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset
    from pdebench.dataset.sample import Sample

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", "plaid_tensile2d,plaid_hyperelasticity")

    train_ds = GraphListDataset([_tiny_pyg()])
    test_ds = GraphListDataset([_tiny_pyg()])

    def _fake_mesh_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    for name, should_enable in (
        ("plaid_tensile2d", True),
        ("plaid_hyperelasticity", True),
        ("plaid_el_pl_dynamics", False),
        ("plaid_elpl_terminal", False),
    ):
        adapter = get_adapter(name)
        train, _test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        if should_enable:
            assert meta["sample_collate"] is True
            assert isinstance(train[0], Sample)
        else:
            assert "sample_collate" not in meta
            assert isinstance(train[0], PygData)


def test_plaid_canonical_adapter_sample_collate_scoped_to_plaid_families_only(monkeypatch) -> None:
    """PDEBENCH_SAMPLE_COLLATE=ginot must not affect PLAID families."""
    from torch_geometric.data import Data as PygData

    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.plaid_datasets import GraphListDataset

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", "ginot")

    train_ds = GraphListDataset([_tiny_pyg()])
    test_ds = GraphListDataset([_tiny_pyg()])

    def _fake_mesh_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    for name in _PLAID_SAMPLE_COLLATE_NAMES:
        adapter = get_adapter(name)
        train, _test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            mesh_graph_backend="pyg",
            plaid_use_sdf_features=True,
            plaid_max_samples=0,
            plaid_load_public_test=False,
            plaid_terminal_y_norm="asinh_iqr",
            plaid_terminal_target_fields=None,
            feature_request=FeatureRequest(),
        )

        assert "sample_collate" not in meta
        assert isinstance(train[0], PygData)

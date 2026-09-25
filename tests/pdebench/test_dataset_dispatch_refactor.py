from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest
import torch

import pdebench
import pdebench.dataset.utils as dataset_utils
from pdebench.config import FlareConfig, TransolverConfig
from pdebench.dataset.ginot import (
    GinotRawDataset,
    ginot_collate_fn,
    ginot_model_forward,
    ginot_per_graph_channel_rel_l2,
    load_ginot_dataset,
)
from pdebench.dataset.loss import compute_packed_loss, packed_per_graph_channel_mean_rel_l2
from pdebench.dataset.mesh_runtime import (
    make_mesh_static_statsfun,
    mesh_model_forward,
    mesh_sequence_collate_fn,
)
from pdebench.dataset.plaid_core import PlaidParsedSample, build_plaid_benchmark_node_features
from pdebench.dataset.plaid_datasets import (
    NodeFeatureNormalizer,
    _apply_graph_normalizers_once,
    _split_labeled_train_test,
)
from pdebench.dataset.registry import RegistryError
from pdebench.dataset.sample import FeatureRequest, LossSpec
from pdebench.models.flare import FLAREModel
from pdebench.models.transolver import Transolver


def test_load_dataset_dispatches_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "TENSILE2D",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
    )

    # C2a: plaid_tensile2d is wrapped with the Sample round-trip bridge; train/test
    # are plain strings here (not Dataset instances) so they pass through unwrapped,
    # but metadata gains the sample_bridge marker.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls == {
        "dataset_name": "plaid_tensile2d",
        "data_root": str(tmp_path),
        "split_seed": 11,
        "graph_backend": "pyg",
        "use_sdf_features": True,
        "max_samples": 0,
        "load_public_test": False,
        "laplacian_eig_dim": 0,
        "laplacian_spec": "graph",
    }


def test_load_dataset_alias_hyperelasticity_to_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "hyperelasticity",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
    )

    # C2a: plaid_hyperelasticity is wrapped with the Sample round-trip bridge; see
    # test_load_dataset_dispatches_mesh_static_backend for the tensile2d rationale.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls["dataset_name"] == "plaid_hyperelasticity"
    assert calls["split_seed"] == 11


def test_load_dataset_dispatches_plaid_hyperelasticity_to_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "PLAID_HYPERELASTICITY",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
        plaid_use_sdf_features=False,
    )

    # C2a: plaid_hyperelasticity is wrapped with the Sample round-trip bridge.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls == {
        "dataset_name": "plaid_hyperelasticity",
        "data_root": str(tmp_path),
        "split_seed": 11,
        "graph_backend": "pyg",
        "use_sdf_features": False,
        "max_samples": 0,
        "load_public_test": False,
        "laplacian_eig_dim": 0,
        "laplacian_spec": "graph",
    }


def test_load_dataset_dispatches_plaid_el_pl_dynamics_to_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "PLAID_EL_PL_DYNAMICS",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
        plaid_use_sdf_features=False,
    )

    # C2a: plaid_el_pl_dynamics is wrapped with the Sample round-trip bridge.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls == {
        "dataset_name": "plaid_el_pl_dynamics",
        "data_root": str(tmp_path),
        "split_seed": 11,
        "graph_backend": "pyg",
        "use_sdf_features": False,
        "max_samples": 0,
        "load_public_test": False,
        "laplacian_eig_dim": 0,
        "laplacian_spec": "graph",
    }


def test_load_dataset_dispatches_plaid_elpl_terminal_to_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "PLAID_ELPL_TERMINAL",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
        plaid_use_sdf_features=False,
    )

    # C2a: plaid_elpl_terminal is wrapped with the Sample round-trip bridge.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls == {
        "dataset_name": "plaid_elpl_terminal",
        "data_root": str(tmp_path),
        "split_seed": 11,
        "graph_backend": "pyg",
        "use_sdf_features": False,
        "max_samples": 0,
        "load_public_test": False,
        "laplacian_eig_dim": 0,
        "laplacian_spec": "graph",
        "y_norm_mode": "asinh_iqr",
        "terminal_target_fields": ["U_x"],
    }


def test_uses_ginot_pipeline_false_for_static_plaid_el_pl_dynamics() -> None:
    from pdebench.dataset.ginot.types import GINOT_DATASETS
    from pdebench.dataset.utils import uses_ginot_pipeline

    assert "plaid_el_pl_dynamics" not in GINOT_DATASETS
    assert uses_ginot_pipeline("plaid_el_pl_dynamics", model_type="glt") is False


def test_load_dataset_canonical_plaid_tensile2d_to_mesh_static_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "plaid_tensile2d",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
        mesh_graph_backend="pyg",
    )

    # C2a: plaid_tensile2d is wrapped with the Sample round-trip bridge.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_pyg"}
    assert calls["dataset_name"] == "plaid_tensile2d"


def test_plaid_benchmark_node_features_can_disable_sdf_features() -> None:
    parsed = PlaidParsedSample(
        pos=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        targets=None,
        target_scalars=None,
        boundary_pos=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        scalar_values=np.arange(10, dtype=np.float32),
        boundary_tag_ids={"Holes": np.array([0, 1], dtype=np.int64)},
        boundary_ids=np.array([0, 1], dtype=np.int64),
        boundary_tags=("Holes",),
        node_type_ids=np.zeros(3, dtype=np.int64),
        boundary_distance=np.zeros(3, dtype=np.float32),
        target_fields=(),
        space_dim=2,
    )

    with_sdf = build_plaid_benchmark_node_features(parsed, use_sdf_features=True)
    without_sdf = build_plaid_benchmark_node_features(parsed, use_sdf_features=False)

    assert with_sdf.shape == (3, 15)
    assert without_sdf.shape == (3, 12)
    np.testing.assert_allclose(without_sdf[:, :2], parsed.pos)
    np.testing.assert_allclose(without_sdf[:, 2:], np.repeat(parsed.scalar_values[None, :], 3, axis=0))


def test_load_dataset_dispatches_ginot_backend(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_ginot_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "POISSON_UNSTRUCTURED",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
    )

    # poisson_unstructured is C2a Sample-bridge wrapped: sentinel strings from the
    # monkeypatched loader are not Dataset instances, so they pass through unwrapped,
    # but metadata gains the sample_bridge marker.
    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_ginot"}
    assert calls == {
        "dataset_name": "poisson_unstructured",
        "data_root": str(tmp_path),
        "split_seed": 11,
        "max_samples": 0,
        "include_edges": False,
        "use_flash_varlen": False,
        "laplacian_eig_dim": 0,
        "laplacian_spec": "graph",
        "include_padded_boundary": False,
    }


@pytest.mark.parametrize(
    "dataset_name",
    ["bracket_lug", "micro_puc", "micro_puc_fixed", "deform_plate"],
)
def test_load_dataset_dispatches_remaining_ginot_families_with_sample_bridge(
    monkeypatch, tmp_path, dataset_name: str
) -> None:
    calls = {}

    def _fake_ginot_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    train, test, metadata = dataset_utils.load_dataset(
        dataset_name.upper(),
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=11,
    )

    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_ginot"}
    assert calls["dataset_name"] == dataset_name
    assert calls["split_seed"] == 11


def test_load_dataset_dispatches_geo_fno_loader(monkeypatch, tmp_path) -> None:
    marker = object()

    def _fake_loader(data_root):
        assert data_root == str(tmp_path)
        return marker, marker, marker

    monkeypatch.setattr(dataset_utils, "load_elasticity_dataset", _fake_loader)

    out = dataset_utils.load_dataset("elasticity", DATADIR_BASE=str(tmp_path), PROJDIR=str(tmp_path))
    assert out == (marker, marker, marker)


def test_load_dataset_dispatches_drivaerml_loader(monkeypatch, tmp_path) -> None:
    marker = object()
    calls = {}

    def _fake_loader(dataset_name, data_root):
        calls["dataset_name"] = dataset_name
        calls["data_root"] = data_root
        return marker, marker, marker

    monkeypatch.setattr(dataset_utils, "load_drivaerml_dataset", _fake_loader)
    out = dataset_utils.load_dataset("drivaerml_40k", DATADIR_BASE=str(tmp_path), PROJDIR=str(tmp_path))
    assert out == (marker, marker, marker)
    assert calls == {"dataset_name": "drivaerml_40k", "data_root": str(tmp_path)}


def test_load_dataset_dispatches_lpbf_to_lpbf_py(monkeypatch, tmp_path) -> None:
    marker = object()
    called = {"count": 0}

    def _fake_loader(**_kwargs):
        called["count"] += 1
        return marker, marker, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_loader)
    train, test, meta = dataset_utils.load_dataset("lpbf", DATADIR_BASE=str(tmp_path), PROJDIR=str(tmp_path))
    assert (train, test) == (marker, marker)
    assert meta == {"meta": 1}
    assert called["count"] == 1


def test_load_dataset_dispatches_lpbf_ginot_path_with_sample_bridge(monkeypatch, tmp_path) -> None:
    marker = object()
    called = {"count": 0, "kwargs": {}}

    def _fake_loader(**kwargs):
        called["count"] += 1
        called["kwargs"] = kwargs
        return marker, marker, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_loader)

    fr = FeatureRequest(edges=True, boundary=False, laplacian_k=0, laplacian_spec="graph")
    train, test, metadata = dataset_utils.load_dataset(
        "lpbf",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        feature_request=fr,
    )

    assert (train, test) == (marker, marker)
    assert metadata == {"meta": 1, "sample_bridge": "c2a_ginot"}
    assert called["count"] == 1
    assert called["kwargs"]["include_edges"] is True


def test_load_dataset_dispatches_lpbf_for_all_model_types(monkeypatch, tmp_path) -> None:
    marker = object()
    called = {"count": 0}

    def _fake_loader(**_kwargs):
        called["count"] += 1
        return marker, marker, marker

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_loader)
    for model_type in (None, "flare", "glt", "gito"):
        called["count"] = 0
        train, test, meta = dataset_utils.load_dataset(
            "lpbf",
            DATADIR_BASE=str(tmp_path),
            PROJDIR=str(tmp_path),
            model_type=model_type,
        )
        assert (train, test) == (marker, marker)
        assert meta == marker
        assert called["count"] == 1


def test_uses_ginot_pipeline_lpbf_dual_path() -> None:
    from pdebench.dataset.utils import uses_ginot_pipeline

    assert uses_ginot_pipeline("lpbf") is False
    assert uses_ginot_pipeline("lpbf", model_type="flare") is False
    assert uses_ginot_pipeline("lpbf", model_type="glt") is False


def test_load_dataset_feature_request_maps_to_ginot_kwargs(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_ginot_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    fr = FeatureRequest(edges=True, boundary=False, laplacian_k=16, laplacian_spec="fem-l")
    dataset_utils.load_dataset(
        "poisson_unstructured",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        feature_request=fr,
    )

    assert calls["include_edges"] is True
    assert calls["include_padded_boundary"] is False
    assert calls["laplacian_eig_dim"] == 16
    assert calls["laplacian_spec"] == "fem-l"


def test_load_dataset_feature_request_maps_to_plaid_kwargs(monkeypatch, tmp_path) -> None:
    calls = {}

    def _fake_mesh_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_mesh_static_dataset", _fake_mesh_loader)

    fr = FeatureRequest(edges=True, boundary=True, laplacian_k=8, laplacian_spec="edge")
    dataset_utils.load_dataset(
        "plaid_hyperelasticity",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        feature_request=fr,
    )

    assert calls["laplacian_eig_dim"] == 8
    assert calls["laplacian_spec"] == "edge"


def test_load_dataset_unknown_name_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="Dataset does_not_exist not found"):
        dataset_utils.load_dataset("does_not_exist", DATADIR_BASE=str(tmp_path), PROJDIR=str(tmp_path))


def test_load_dataset_deleted_name_raises(tmp_path) -> None:
    with pytest.raises(RegistryError, match="jeb was removed"):
        dataset_utils.load_dataset("jeb", DATADIR_BASE=str(tmp_path), PROJDIR=str(tmp_path))


def test_load_dataset_poisson_structured_is_not_deleted(tmp_path, monkeypatch) -> None:
    from pdebench.dataset.registry import resolve_dataset_name

    assert resolve_dataset_name("poisson_structured") == "poisson_structured"


def test_mesh_sequence_collate_fn_rejects_variable_lengths() -> None:
    a = types.SimpleNamespace(x=torch.zeros(4, 3), y=torch.zeros(4, 2))
    b = types.SimpleNamespace(x=torch.zeros(5, 3), y=torch.zeros(5, 2))

    with pytest.raises(NotImplementedError, match="masking support"):
        mesh_sequence_collate_fn([a, b])


def test_mesh_model_forward_graph_path_handles_pyg_like_batch() -> None:
    class _FakeBatch:
        def __init__(self):
            self.y = torch.tensor([[1.0], [2.0], [3.0]])
            self.batch = torch.tensor([0, 0, 1], dtype=torch.long)
            self.num_graphs = 2

    class _FakeModel(torch.nn.Module):
        def forward(self, batch):
            del batch
            return torch.tensor([[10.0], [20.0], [30.0]])

    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="tensile2d"),
        model=types.SimpleNamespace(model="meshgraphnet"),
    )
    yh, y, batch_index, num_graphs = mesh_model_forward(cfg, _FakeModel(), _FakeBatch())

    assert num_graphs == 2
    assert torch.equal(batch_index, torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(yh, torch.tensor([[10.0], [20.0], [30.0]]))
    assert torch.equal(y, torch.tensor([[1.0], [2.0], [3.0]]))


def test_mesh_model_forward_sequence_path_flattens_batch() -> None:
    class _SeqModel(torch.nn.Module):
        def forward(self, x):
            return x[..., :1]

    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="tensile2d"),
        model=types.SimpleNamespace(model="transolver"),
    )
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 1)

    yh_flat, y_flat, batch_index, num_graphs = mesh_model_forward(cfg, _SeqModel(), [x, y])

    assert num_graphs == 2
    assert yh_flat.shape == (6, 1)
    assert y_flat.shape == (6, 1)
    assert torch.equal(batch_index, torch.tensor([0, 0, 0, 1, 1, 1]))


def test_mesh_statsfun_returns_nan_for_unlabeled_batches() -> None:
    class _FakeUnlabeledBatch:
        def __init__(self):
            self.y = None
            self.batch = torch.zeros(2, dtype=torch.long)
            self.num_graphs = 1

    class _FakeModel(torch.nn.Module):
        def forward(self, batch):
            del batch
            return torch.zeros(2, 1)

    trainer = types.SimpleNamespace(
        verbose=False,
        GLOBAL_RANK=0,
        print_iterator=False,
        move_to_device=lambda b: b,
        auto_cast=contextlib.nullcontext(),
        model=_FakeModel(),
        DDP=False,
        device=torch.device("cpu"),
    )
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="tensile2d"),
        model=types.SimpleNamespace(model="meshgraphnet"),
    )
    metadata = {"y_normalizer": pdebench.IdentityNormalizer()}

    loss, stats = make_mesh_static_statsfun(cfg, metadata)(trainer, [_FakeUnlabeledBatch()], split="test")
    assert torch.isnan(torch.as_tensor(loss))
    assert torch.isnan(torch.as_tensor(stats["plaid_loss"]))
    assert torch.isnan(torch.as_tensor(stats["plaid_rrmse"]))
    assert torch.isnan(torch.as_tensor(stats["total_error"]))


def test_ginot_per_graph_channel_rel_l2_averages_features_independently() -> None:
    yh = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 2.0],
        ]
    )
    y = torch.tensor(
        [
            [100.0, 1.0],
            [0.0, 1.0],
            [1.0, 4.0],
        ]
    )
    batch_index = torch.tensor([0, 0, 1])

    rel_l2 = ginot_per_graph_channel_rel_l2(yh, y, batch_index=batch_index, num_graphs=2)

    expected_graph0 = (torch.tensor(1.0) + torch.tensor(1.0 / 2.0**0.5)) / 2.0
    expected_graph1 = (torch.tensor(0.0) + torch.tensor(0.5)) / 2.0
    torch.testing.assert_close(rel_l2, torch.stack([expected_graph0, expected_graph1]))


def test_compute_packed_loss_averages_channels_independently_for_plaid() -> None:
    y_normalizer = NodeFeatureNormalizer(
        mean=torch.zeros(1, 2),
        std=torch.ones(1, 2),
    )
    yh = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 2.0]])
    y = torch.tensor([[100.0, 1.0], [0.0, 1.0], [1.0, 4.0]])
    batch_index = torch.tensor([0, 0, 1])

    loss = compute_packed_loss(
        yh,
        y,
        y_normalizer,
        LossSpec(),
        batch_index=batch_index,
        num_graphs=2,
    )
    expected = ginot_per_graph_channel_rel_l2(yh, y, batch_index=batch_index, num_graphs=2)
    per_graph = packed_per_graph_channel_mean_rel_l2(yh, y, batch_index=batch_index, num_graphs=2)
    torch.testing.assert_close(per_graph, expected)
    torch.testing.assert_close(loss, expected.mean())


def test_apply_graph_normalizers_once_skips_duplicate_graph_objects() -> None:
    graph = types.SimpleNamespace(
        x=torch.tensor([[3.0, 5.0]], dtype=torch.float32),
        y=torch.tensor([[2.0]], dtype=torch.float32),
    )
    x_normalizer = NodeFeatureNormalizer(
        mean=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        std=torch.tensor([[2.0, 4.0]], dtype=torch.float32),
    )
    y_normalizer = NodeFeatureNormalizer(
        mean=torch.tensor([[1.0]], dtype=torch.float32),
        std=torch.tensor([[2.0]], dtype=torch.float32),
    )

    _apply_graph_normalizers_once(
        [[graph], [graph]],
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )

    assert torch.allclose(graph.x, torch.tensor([[1.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(graph.y, torch.tensor([[0.5]], dtype=torch.float32))


def test_split_labeled_train_test_is_deterministic_80_20() -> None:
    indices = list(range(500))

    train_a, test_a = _split_labeled_train_test(indices, test_ratio=0.2, seed=13)
    train_b, test_b = _split_labeled_train_test(indices, test_ratio=0.2, seed=13)

    assert len(train_a) == 400
    assert len(test_a) == 100
    assert train_a == train_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(test_a)
    assert set(train_a) | set(test_a) == set(indices)


def test_ginot_poisson_unstructured_uses_80_20_split_and_per_channel_target_normalization(monkeypatch, tmp_path) -> None:
    num_samples = 1200
    query_points = []
    point_clouds = []
    targets = []
    for idx in range(num_samples):
        query_points.append(
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ).numpy()
        )
        point_clouds.append(
            torch.tensor(
                [[0.0, 0.0], [float(idx % 7), 1.0], [1.0, 0.0]],
                dtype=torch.float32,
            ).numpy()
        )
        targets.append(
            torch.tensor(
                [
                    [float(idx), float(idx) * 10.0 + 1.0],
                    [float(idx) + 1.0, float(idx) * 10.0 + 2.0],
                    [float(idx) + 2.0, float(idx) * 10.0 + 3.0],
                ],
                dtype=torch.float32,
            ).numpy()
        )

    raw = GinotRawDataset(
        query_points=query_points,
        point_clouds=point_clouds,
        targets=targets,
        cells=torch.tensor([[0, 1, 2]], dtype=torch.long).numpy(),
        input_params=None,
        target_fields=("a", "b"),
        space_dim=2,
    )
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", lambda dataset_name, data_root: raw)

    train, test, metadata = load_ginot_dataset(
        "poisson_unstructured",
        data_root=str(tmp_path),
        split_seed=5,
    )

    assert len(train) == 960
    assert len(test) == 240
    assert set(train.indices).isdisjoint(test.indices)
    assert metadata["target_fields"] == ["a", "b"]
    assert metadata["c_out"] == 2
    assert metadata["c_in"] == 2
    assert metadata["fun_dim"] == 0
    assert callable(metadata["train_collate_fn"])

    sample = train[0]
    assert set(["pos", "edge_index", "edge_attr", "boundary_pos", "y", "sample_id"]).issubset(sample)
    assert "x" not in sample

    y_train = torch.cat([train[i]["y"] for i in range(len(train))], dim=0)
    assert torch.allclose(y_train.mean(dim=0), torch.zeros(2), atol=2e-6)
    assert torch.allclose(y_train.std(dim=0, unbiased=False), torch.ones(2), atol=2e-6)

    batch = ginot_collate_fn([train[0], train[1]])
    assert batch["pos"].shape == (2, 3, 2)
    assert batch["y"].shape == (2, 3, 2)
    assert batch["boundary_pos"].shape == (2, 3, 2)
    assert torch.equal(batch["mask"], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(batch["boundary_mask"], torch.ones(2, 3, dtype=torch.bool))
    assert batch["flat_y"].shape == (6, 2)
    assert batch["edge_index"].shape[0] == 2
    assert torch.equal(batch["batch_index"], torch.tensor([0, 0, 0, 1, 1, 1]))


def _mock_micro_puc_raw(num_samples: int) -> GinotRawDataset:
    mesh_idx = np.concatenate(
        [
            np.arange(min(num_samples, 10_000), dtype=np.int32),
            (np.arange(max(num_samples - 10_000, 0), dtype=np.int32) % 10_000),
        ]
    )
    cells = [np.array([[1, 2]], dtype=np.int64) for _ in range(10_000)]
    return GinotRawDataset(
        query_points=[
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(num_samples)
        ],
        point_clouds=[
            torch.tensor([[0.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(num_samples)
        ],
        targets=[
            torch.tensor([[float(idx)], [float(idx) + 1.0]], dtype=torch.float32).numpy()
            for idx in range(num_samples)
        ],
        cells=cells,
        input_params=None,
        target_fields=("u",),
        space_dim=2,
        micro_puc_mesh_idx=mesh_idx,
    )


def test_ginot_max_samples_caps_each_split_head(monkeypatch, tmp_path) -> None:
    num_samples = 1000
    raw = GinotRawDataset(
        query_points=[
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(num_samples)
        ],
        point_clouds=[
            torch.tensor([[0.5, 0.5]], dtype=torch.float32).numpy() for _ in range(num_samples)
        ],
        targets=[
            torch.tensor([[float(idx)]], dtype=torch.float32).numpy() for idx in range(num_samples)
        ],
        cells=torch.tensor([[0, 1]], dtype=torch.long).numpy(),
        input_params=None,
        target_fields=("u",),
        space_dim=2,
    )
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", lambda dataset_name, data_root: raw)
    monkeypatch.setattr(
        "pdebench.dataset.ginot.loader.expected_micro_puc_fixed_samples",
        lambda: num_samples,
    )

    full_train, full_test, _ = load_ginot_dataset(
        "micro_puc_fixed",
        data_root=str(tmp_path),
        split_seed=0,
        max_samples=0,
    )
    capped_train, capped_test, metadata = load_ginot_dataset(
        "micro_puc_fixed",
        data_root=str(tmp_path),
        split_seed=0,
        max_samples=128,
    )

    assert len(full_train) == 800
    assert len(full_test) == 200
    assert capped_train.indices == full_train.indices[:128]
    assert capped_test.indices == full_test.indices[:128]
    assert metadata["ginot_max_samples"] == 128
    assert metadata["ginot_split_train_size"] == 800
    assert metadata["ginot_split_test_size"] == 200
    assert metadata["mesh_split_sizes"] == dict(train=128, val=128, test=128)


@pytest.mark.parametrize("dataset_name", ["poisson_unstructured", "micro_puc", "bracket_lug"])
def test_ginot_max_samples_rejects_non_micro_puc_fixed(dataset_name: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="only supported for micro_puc_fixed"):
        load_ginot_dataset(dataset_name, data_root=str(tmp_path), max_samples=128)


def test_ginot_bracket_lug_uses_full_dataset_80_20_split(monkeypatch, tmp_path) -> None:
    num_samples = 3000
    raw = GinotRawDataset(
        query_points=[
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(num_samples)
        ],
        point_clouds=[
            torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32).numpy() for _ in range(num_samples)
        ],
        targets=[
            torch.tensor([[float(idx), float(idx) + 1.0]], dtype=torch.float32).numpy()
            for idx in range(num_samples)
        ],
        cells=torch.tensor([[0, 1]], dtype=torch.long).numpy(),
        input_params=None,
        target_fields=("a", "b"),
        space_dim=3,
    )
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", lambda dataset_name, data_root: raw)

    train, test, metadata = load_ginot_dataset(
        "bracket_lug",
        data_root=str(tmp_path),
        split_seed=0,
    )

    assert len(train) == 2400
    assert len(test) == 600
    assert max(train.indices + test.indices) == num_samples - 1
    assert set(train.indices).isdisjoint(test.indices)
    assert len(train.indices) + len(test.indices) == num_samples


def test_ginot_micro_puc_uses_full_dataset_80_20_split(monkeypatch, tmp_path) -> None:
    num_samples = 12_000
    raw = _mock_micro_puc_raw(num_samples)
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", lambda dataset_name, data_root: raw)

    train, test, metadata = load_ginot_dataset(
        "micro_puc",
        data_root=str(tmp_path),
        split_seed=5,
    )

    assert len(train) == 9_600
    assert len(test) == 2_400
    assert max(train.indices + test.indices) == num_samples - 1
    assert metadata["ginot_micro_puc_mesh_samples"] == 10_000
    assert metadata["ginot_micro_puc_num_rows"] == num_samples
    assert metadata["ginot_micro_puc_split"] == "full73879_80_20"
    assert set(train.indices).isdisjoint(test.indices)


@pytest.mark.parametrize("dataset_name", ["poisson_unstructured", "bracket_lug"])
def test_ginot_minmax_position_datasets_map_train_coords_to_unit_box(dataset_name: str) -> None:
    from pdebench.dataset.ginot.loader import _build_normalizers
    from pdebench.dataset.ginot.sample import encode_raw_sample

    raw = GinotRawDataset(
        query_points=[
            np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 2.0], [3.0, 5.0, 3.0]], dtype=np.float32),
        ],
        point_clouds=[
            np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            np.array([[1.5, 2.5, 2.5]], dtype=np.float32),
        ],
        targets=[
            np.array([[1.0], [3.0]], dtype=np.float32),
            np.array([[0.0], [2.0]], dtype=np.float32),
        ],
        cells=None,
        input_params=None,
        target_fields=("y",),
        space_dim=3 if dataset_name == "bracket_lug" else 2,
    )
    train_ids = [0, 1]
    pos_n, bnd_n, y_n, feats_n = _build_normalizers(dataset_name, raw, train_ids)
    assert feats_n is None
    assert torch.allclose(pos_n.mean, bnd_n.mean)
    assert torch.allclose(pos_n.std, bnd_n.std)

    pos, bpos, _ = encode_raw_sample(raw, 0, pos_n, bnd_n, y_n)
    assert float(pos.amin()) >= -1e-5
    assert float(pos.amax()) <= 1.0 + 1e-5
    assert float(bpos.amin()) >= -1e-5
    assert float(bpos.amax()) <= 1.0 + 1e-5

    assert float(y_n.mean) == pytest.approx(1.5)
    assert float(y_n.std) == pytest.approx(1.118033988749895)
    encoded_targets = [
        encode_raw_sample(raw, sample_id, pos_n, bnd_n, y_n)[2]
        for sample_id in train_ids
    ]
    assert float(torch.cat(encoded_targets, dim=0).mean()) == pytest.approx(0.0, abs=1e-5)


def test_ginot_collate_fn_pads_variable_node_counts_with_masks() -> None:
    samples = [
        {
            "pos": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            "boundary_pos": torch.tensor([[3.0, 0.0]]),
            "y": torch.tensor([[10.0], [20.0]]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
            "sample_id": torch.tensor(0),
        },
        {
            "pos": torch.tensor([[4.0, 0.0]]),
            "boundary_pos": torch.tensor([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]),
            "y": torch.tensor([[30.0]]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
            "sample_id": torch.tensor(1),
        },
    ]
    batch = ginot_collate_fn(samples)

    assert torch.equal(batch["mask"], torch.tensor([[True, True], [True, False]]))
    assert torch.equal(batch["boundary_mask"], torch.tensor([[True, False, False], [True, True, True]]))
    assert torch.equal(batch["pos"][1, 1], torch.zeros(2))
    assert torch.equal(batch["y"][1, 1], torch.zeros(1))
    assert torch.equal(batch["boundary_pos"][0, 1:], torch.zeros(2, 2))
    assert torch.equal(batch["flat_y"], torch.tensor([[10.0], [20.0], [30.0]]))


def test_ginot_collate_fn_emits_flash_varlen_metadata() -> None:
    samples = [
        {
            "pos": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            "boundary_pos": torch.tensor([[3.0, 0.0]]),
            "y": torch.tensor([[10.0], [20.0]]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
            "sample_id": torch.tensor(0),
        },
        {
            "pos": torch.tensor([[4.0, 0.0]]),
            "boundary_pos": torch.tensor([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]),
            "y": torch.tensor([[30.0]]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
            "sample_id": torch.tensor(1),
        },
    ]
    batch = ginot_collate_fn(samples, use_flash_varlen=True)

    assert batch["use_flash_varlen"] is True
    assert torch.equal(batch["node_lengths"], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(batch["boundary_lengths"], torch.tensor([1, 3], dtype=torch.int32))
    assert torch.equal(batch["cu_seqlens"], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.equal(batch["boundary_cu_seqlens"], torch.tensor([0, 1, 4], dtype=torch.int32))
    assert batch["max_seqlen"] == 2
    assert batch["boundary_max_seqlen"] == 3


def test_ginot_loader_skips_edges_by_default_and_can_opt_in(monkeypatch, tmp_path) -> None:
    raw = GinotRawDataset(
        query_points=[
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(1200)
        ],
        point_clouds=[
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32).numpy()
            for _ in range(1200)
        ],
        targets=[
            torch.tensor([[0.0], [1.0]], dtype=torch.float32).numpy()
            for _ in range(1200)
        ],
        cells=torch.tensor([[0, 1]], dtype=torch.long).numpy(),
        input_params=None,
        target_fields=("u",),
        space_dim=2,
        dataset_dir=str(tmp_path),
    )
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", lambda dataset_name, data_root: raw)

    train, _, metadata = load_ginot_dataset(
        "poisson_unstructured",
        data_root=str(tmp_path),
    )

    sample = train[0]
    assert metadata["ginot_include_edges"] is False
    assert sample["edge_index"].shape == (2, 0)
    assert sample["edge_attr"].shape == (0, 3)

    train_edges, _, metadata_edges = load_ginot_dataset(
        "poisson_unstructured",
        data_root=str(tmp_path),
        include_edges=True,
    )

    sample_edges = train_edges[0]
    assert metadata_edges["ginot_include_edges"] is True
    assert sample_edges["edge_index"].numel() > 0
    assert sample_edges["edge_attr"].shape[-1] == 3


def test_ginot_model_forward_passes_supported_optional_inputs() -> None:
    batch = {
        "pos": torch.randn(3, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.randn(2, 3),
        "boundary_pos": torch.randn(4, 2),
        "y": torch.randn(3, 1),
        "batch_index": torch.zeros(3, dtype=torch.long),
        "boundary_batch_index": torch.zeros(4, dtype=torch.long),
        "ptr": torch.tensor([0, 3], dtype=torch.long),
        "boundary_ptr": torch.tensor([0, 4], dtype=torch.long),
        "num_graphs": 1,
    }

    class _EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, pos, feats=None, edge_index=None, edge_attr=None, **kwargs):
            del feats, kwargs
            self.seen["edge_index"] = edge_index
            self.seen["edge_attr"] = edge_attr
            return pos[:, :1]

    model = _EdgeModel()
    model.requires_edge_info = True
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="micro_puc"),
        model=types.SimpleNamespace(model="custom"),
    )
    yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)

    assert torch.equal(yh, batch["pos"][:, :1])
    assert torch.equal(y, batch["y"])
    assert torch.equal(batch_index, batch["batch_index"])
    assert num_graphs == 1
    assert model.seen["edge_index"] is batch["edge_index"]
    assert model.seen["edge_attr"] is batch["edge_attr"]


def test_ginot_model_forward_supports_meshgraphnet_tensor_adapter() -> None:
    samples = [
        {
            "pos": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            "boundary_pos": torch.tensor([[0.0, 0.0]]),
            "y": torch.tensor([[1.0], [2.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_attr": torch.randn(2, 3),
            "sample_id": torch.tensor(0),
            "edge_cache_key": "shared:2:2",
        },
        {
            "pos": torch.tensor([[2.0, 0.0]]),
            "boundary_pos": torch.tensor([[2.0, 0.0]]),
            "y": torch.tensor([[3.0]]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
            "sample_id": torch.tensor(1),
            "edge_cache_key": "shared:1:0",
        },
    ]
    batch = ginot_collate_fn(samples)

    class _MeshGraphNetLikeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, pos, edge_index, edge_attr, graph_cache_key=None, **kwargs):
            self.seen = {
                "pos": pos,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "graph_cache_key": graph_cache_key,
                "kwargs": kwargs,
            }
            return pos[:, :1]

    model = _MeshGraphNetLikeModel()
    model.requires_edge_info = True
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="micro_puc"),
        model=types.SimpleNamespace(model="meshgraphnet"),
    )
    yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)

    assert torch.equal(model.seen["pos"], batch["flat_pos"])
    assert model.seen["edge_index"] is batch["edge_index"]
    assert model.seen["edge_attr"] is batch["edge_attr"]
    assert torch.equal(yh, batch["flat_pos"][:, :1])
    assert torch.equal(y, batch["flat_y"])
    assert torch.equal(batch_index, batch["batch_index"])
    assert num_graphs == 2


def test_ginot_model_forward_uses_orig_mod_signature_for_compiled_wrappers() -> None:
    batch = {
        "pos": torch.randn(3, 2),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_attr": torch.empty((0, 3)),
        "boundary_pos": torch.randn(4, 2),
        "y": torch.randn(3, 1),
        "batch_index": torch.zeros(3, dtype=torch.long),
        "boundary_batch_index": torch.zeros(4, dtype=torch.long),
        "ptr": torch.tensor([0, 3], dtype=torch.long),
        "boundary_ptr": torch.tensor([0, 4], dtype=torch.long),
        "num_graphs": 1,
    }

    class _CoordinateOnlyModel(torch.nn.Module):
        def forward(self, x, mask=None):
            if mask is not None:
                x = x * mask.unsqueeze(-1)
            return x[..., :1]

    class _CompiledLikeWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module
            self.kwargs_seen = None

        def forward(self, *args, **kwargs):
            self.kwargs_seen = kwargs
            return self._orig_mod(*args, **kwargs)

    model = _CompiledLikeWrapper(_CoordinateOnlyModel())
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="micro_puc"),
        model=types.SimpleNamespace(model="transolver"),
    )
    yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)

    assert model.kwargs_seen == {}
    assert torch.equal(yh, batch["pos"][:, :1])
    assert torch.equal(y, batch["y"])
    assert torch.equal(batch_index, batch["batch_index"])
    assert num_graphs == 1


def test_ginot_model_forward_reads_flags_from_compiled_orig_mod() -> None:
    batch = {
        "pos": torch.randn(3, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.randn(2, 3),
        "boundary_pos": torch.randn(4, 2),
        "y": torch.randn(3, 1),
        "batch_index": torch.zeros(3, dtype=torch.long),
        "boundary_batch_index": torch.zeros(4, dtype=torch.long),
        "ptr": torch.tensor([0, 3], dtype=torch.long),
        "boundary_ptr": torch.tensor([0, 4], dtype=torch.long),
        "num_graphs": 1,
    }

    class _FlaggedModel(torch.nn.Module):
        requires_edge_info = True

        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, pos, feats=None, edge_index=None, edge_attr=None, **kwargs):
            del feats, kwargs
            self.seen = {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }
            return pos[:, :1]

    class _CompiledLikeWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    inner = _FlaggedModel()
    model = _CompiledLikeWrapper(inner)
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="micro_puc"),
        model=types.SimpleNamespace(model="custom"),
    )
    yh, _, _, _ = ginot_model_forward(cfg, model, batch)

    assert torch.equal(yh, batch["pos"][:, :1])
    assert inner.seen["edge_index"] is batch["edge_index"]
    assert inner.seen["edge_attr"] is batch["edge_attr"]


def test_ginot_model_forward_geo_transolver_without_edges_when_not_required() -> None:
    batch = {
        "pos": torch.randn(4, 2),
        "feats": torch.randn(4, 1),
        "y": torch.randn(4, 1),
        "batch_index": torch.zeros(4, dtype=torch.long),
        "num_graphs": 1,
    }

    class _GeoLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, pos=None, feats=None, edge_index=None, edge_attr=None, **kwargs):
            self.seen = {
                "pos": pos,
                "feats": feats,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "kwargs": kwargs,
            }
            return pos[:, :1]

    model = _GeoLike()
    model.requires_edge_info = False
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="micro_puc"),
        model=types.SimpleNamespace(model="geo_transolver"),
    )
    yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)

    assert torch.equal(yh, batch["pos"][:, :1])
    assert torch.equal(y, batch["y"])
    assert torch.equal(batch_index, batch["batch_index"])
    assert num_graphs == 1
    assert model.seen["edge_index"] is None
    assert model.seen["edge_attr"] is None
    assert torch.equal(model.seen["pos"], batch["pos"])
    assert torch.equal(model.seen["feats"], batch["feats"])


def test_ginot_model_forward_transolver_passes_pos_and_feats() -> None:
    """Bumper/GINOT builds Transolver with c_in=space+fun; forward must pass feats."""
    batch = {
        "pos": torch.randn(4, 3),
        "feats": torch.randn(4, 4),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "y": torch.randn(1, 4, 2),
        "batch_index": torch.zeros(4, dtype=torch.long),
        "num_graphs": 1,
    }

    class _TransolverLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, x, f=None, mask=None):
            self.seen = {"x": x, "f": f, "mask": mask}
            return x[:, :2]

    model = _TransolverLike()
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(dataset="bumper_beam"),
        model=types.SimpleNamespace(model="transolver"),
    )
    yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)

    assert torch.equal(model.seen["x"], batch["pos"])
    assert torch.equal(model.seen["f"], batch["feats"])
    assert torch.equal(model.seen["mask"], batch["mask"])
    assert yh.shape[-1] == 2
    assert num_graphs == 1
    assert torch.equal(batch_index, batch["batch_index"])


def test_flare_masked_batch_matches_unpadded_sample_and_zeroes_invalid_tokens() -> None:
    torch.manual_seed(0)
    model = FLAREModel(
        FlareConfig(
            channel_dim=16,
            num_blocks=1,
            num_heads=4,
            num_latents=4,
            num_layers_in_out_proj=0,
            num_layers_k_proj=0,
            num_layers_v_proj=0,
            num_layers_ffn=0,
        ),
        metadata={"c_in": 2, "c_out": 1, "dataset": "elasticity"},
    )
    model.eval()
    sample = torch.randn(1, 3, 2)
    padded = torch.cat([sample, torch.zeros(1, 2, 2)], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        expected = model(sample)
        actual = model(padded, mask=mask)

    assert torch.allclose(actual[:, :3], expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual[:, 3:], torch.zeros_like(actual[:, 3:]), atol=1e-7, rtol=0.0)


def test_transolver_masked_batch_matches_unpadded_sample_and_zeroes_invalid_tokens() -> None:
    torch.manual_seed(0)
    model = Transolver(
        TransolverConfig(channel_dim=16, num_blocks=2, num_heads=4, num_slices=4),
        metadata={"c_in": 2, "c_out": 1, "space_dim": 2, "fun_dim": 0, "dataset": "elasticity"},
    )
    model.eval()
    sample = torch.randn(1, 3, 2)
    padded = torch.cat([sample, torch.zeros(1, 2, 2)], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        expected = model(sample)
        actual = model(padded, mask=mask)

    assert torch.allclose(actual[:, :3], expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual[:, 3:], torch.zeros_like(actual[:, 3:]), atol=1e-7, rtol=0.0)

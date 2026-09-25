from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import pdebench.dataset.ginot.visualize as ginot_visualize
import pdebench.dataset.plaid_visualize as plaid_visualize
import pdebench.dataset.visualize as unified_visualize
from pdebench.dataset.ginot import GINOT_DATASETS
from pdebench.dataset.lpbf import LPBF_DATASETS

PLAID_MESH_STATIC_DATASETS = (
    "plaid_tensile2d",
    "plaid_hyperelasticity",
    "plaid_el_pl_dynamics",
)


@pytest.mark.parametrize("dataset", PLAID_MESH_STATIC_DATASETS)
def test_canonical_plaid_dataset_uses_mesh_static_visualizer(monkeypatch, tmp_path: Path, dataset: str) -> None:
    """ginot.visualize.load_dataset_samples must route to the injected PLAID loader
    exactly as pdebench.dataset.plaid_visualize.main() configures it (MESH_STATIC_PLAID_VIZ_DATASETS
    membership + load_mesh_static_plaid_dataset_samples attribute), and never touch the GINOT loader.
    """
    calls: list[str] = []
    sentinel = ({"train": []}, [], {}, {})

    def fake_mesh_static_loader(dataset_name, *args, **kwargs):
        del args, kwargs
        calls.append(dataset_name)
        return sentinel

    def fail_ginot_loader(*args, **kwargs):
        del args, kwargs
        raise AssertionError("canonical PLAID dataset was routed through GINOT")

    monkeypatch.setattr(ginot_visualize, "MESH_STATIC_PLAID_VIZ_DATASETS", frozenset(PLAID_MESH_STATIC_DATASETS))
    monkeypatch.setattr(ginot_visualize, "load_mesh_static_plaid_dataset_samples", fake_mesh_static_loader, raising=False)
    monkeypatch.setattr(ginot_visualize, "_load_ginot_split_datasets", fail_ginot_loader)

    assert ginot_visualize.load_dataset_samples(dataset, tmp_path, 0, 1, ["train"]) == sentinel
    assert calls == [dataset]


def test_plaid_sample_loader_maps_metadata_ids_to_graph_list_indices(monkeypatch, tmp_path: Path) -> None:
    from pdebench.dataset.plaid_datasets import GraphListDataset

    train_ds = GraphListDataset(["train-101", "train-202"])
    test_ds = GraphListDataset(["test-303"])
    metadata = {
        "mesh_train_ids": [101, 202],
        "mesh_val_ids": [303],
        "target_fields": ["U_x"],
        "y_normalizer": object(),
    }

    monkeypatch.setattr(
        "pdebench.dataset.plaid_datasets.load_mesh_static_dataset",
        lambda **kwargs: (train_ds, test_ds, metadata),
    )
    monkeypatch.setattr(
        ginot_visualize,
        "_pyg_graph_to_viz_sample",
        lambda graph, _metadata: {"graph": graph},
    )
    monkeypatch.setattr(
        ginot_visualize,
        "random_sample_ids",
        lambda pool, **kwargs: list(pool),
    )

    samples, fields, context, datasets = plaid_visualize.load_mesh_static_plaid_dataset_samples(
        "plaid_tensile2d",
        tmp_path,
        seed=0,
        max_samples=10,
        splits=["train", "test"],
    )

    assert samples == {
        "train": [{"graph": "train-101"}, {"graph": "train-202"}],
        "test": [{"graph": "test-303"}],
    }
    assert fields == ["U_x"]
    assert context["index_maps"] == {"train": {101: 0, 202: 1}, "test": {303: 0}}
    assert datasets == {"train": train_ds, "test": test_ds}


def test_plaid_dynamics_sample_loader_uses_last_manifest_transition_per_sim(monkeypatch, tmp_path: Path) -> None:
    class ManifestDataset:
        def __init__(self, sim_ids: list[int], graphs: list[str]) -> None:
            self.manifest = pd.DataFrame({"sim_id": sim_ids})
            self.graphs = graphs

        def __len__(self) -> int:
            return len(self.graphs)

        def __getitem__(self, index: int) -> str:
            return self.graphs[index]

    train_ds = ManifestDataset(
        [101, 101, 202, 202],
        ["train-101-step0", "train-101-step1", "train-202-step0", "train-202-step1"],
    )
    test_ds = ManifestDataset([303, 303], ["test-303-step0", "test-303-step1"])
    metadata = {
        "mesh_train_ids": [101, 202],
        "mesh_val_ids": [303],
        "target_fields": ["U_x"],
        "y_normalizer": object(),
    }

    monkeypatch.setattr(
        "pdebench.dataset.plaid_datasets.load_mesh_static_dataset",
        lambda **kwargs: (train_ds, test_ds, metadata),
    )
    monkeypatch.setattr(
        ginot_visualize,
        "_pyg_graph_to_viz_sample",
        lambda graph, _metadata: {"graph": graph},
    )
    monkeypatch.setattr(ginot_visualize, "random_sample_ids", lambda pool, **kwargs: list(pool))

    samples, _fields, context, _datasets = plaid_visualize.load_mesh_static_plaid_dataset_samples(
        "plaid_el_pl_dynamics",
        tmp_path,
        seed=0,
        max_samples=10,
        splits=["train", "test"],
    )

    assert samples == {
        "train": [{"graph": "train-101-step1"}, {"graph": "train-202-step1"}],
        "test": [{"graph": "test-303-step1"}],
    }
    assert context["index_maps"] == {"train": {101: 1, 202: 3}, "test": {303: 1}}


def test_pyg_viz_adapter_flattens_single_graph_eigenvalues() -> None:
    graph = SimpleNamespace(
        y=torch.tensor([[1.0]]),
        pos=torch.tensor([[0.0, 0.0]]),
        x=torch.tensor([[0.0, 0.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        laplacian_eigvals=torch.tensor([[0.0, 1.5, 2.5]]),
        metadata={"sample_index": 7},
    )

    sample = ginot_visualize._pyg_graph_to_viz_sample(
        graph,
        {"y_normalizer": _RecordingNormalizer(offset=0.0)},
    )

    assert sample["laplacian_eigvals"].shape == (3,)
    assert ginot_visualize.spectral_mode_title(1, sample["laplacian_eigvals"].numpy()).endswith(
        r"$\lambda$=1.50e+00"
    )


def test_ginot_visualize_datasets_excludes_plaid() -> None:
    assert set(ginot_visualize.DATASETS).isdisjoint(plaid_visualize.DATASETS)
    assert ginot_visualize.MESH_STATIC_PLAID_VIZ_DATASETS == frozenset()


def test_lpbf_is_visualization_supported_without_becoming_a_ginot_training_dataset() -> None:
    assert LPBF_DATASETS <= set(ginot_visualize.DATASETS)
    assert LPBF_DATASETS.isdisjoint(GINOT_DATASETS)


def test_unified_visualize_routes_lpbf_to_ginot_module(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(ginot_visualize, "main", lambda: calls.append(tuple(sys.argv)))
    monkeypatch.setattr(plaid_visualize, "main", lambda: pytest.fail("LPBF must use the GINOT visualizer"))
    monkeypatch.setattr(sys, "argv", ["visualize", "--dataset", "lpbf"])

    unified_visualize.main()

    assert calls == [("visualize", "--dataset", "lpbf", "--outdir", "out/pdebench/dataset_viz")]


class _RecordingNormalizer:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.decode_calls = 0

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        return value + self.offset


def _sample_to_viz(*, normalize_targets: bool) -> tuple[dict, _RecordingNormalizer]:
    y_normalizer = _RecordingNormalizer(offset=10.0)
    raw = SimpleNamespace(
        query_points=[np.array([[1.0, 2.0, 3.0]], dtype=np.float32)],
        targets=[np.array([[99.0]], dtype=np.float32)],
        cells=None,
        space_dim=3,
        normalize_targets=normalize_targets,
    )
    sample = ginot_visualize._ginot_sample_to_viz(
        {"pos": torch.tensor([[1.0, 2.0, 3.0]]), "y": torch.tensor([[4.0]])},
        raw,
        0,
        {"pos_normalizer": _RecordingNormalizer(offset=0.0), "y_normalizer": y_normalizer},
        perimeter_edges=False,
    )
    return sample, y_normalizer


def test_ginot_sample_to_viz_preserves_raw_targets() -> None:
    sample, y_normalizer = _sample_to_viz(normalize_targets=False)

    np.testing.assert_array_equal(sample["y"], np.array([[4.0]], dtype=np.float32))
    assert y_normalizer.decode_calls == 0


def test_ginot_sample_to_viz_decodes_normalized_targets() -> None:
    sample, y_normalizer = _sample_to_viz(normalize_targets=True)

    np.testing.assert_array_equal(sample["y"], np.array([[14.0]], dtype=np.float32))
    assert y_normalizer.decode_calls == 1


def test_plaid_visualize_main_configures_and_restores_ginot_module(monkeypatch) -> None:
    calls: list[tuple] = []
    original_datasets = ginot_visualize.DATASETS
    original_plaid_set = ginot_visualize.MESH_STATIC_PLAID_VIZ_DATASETS

    def fake_ginot_main() -> None:
        calls.append(
            (
                tuple(ginot_visualize.DATASETS),
                ginot_visualize.MESH_STATIC_PLAID_VIZ_DATASETS,
                ginot_visualize.load_mesh_static_plaid_dataset_samples,
            )
        )

    monkeypatch.setattr(ginot_visualize, "main", fake_ginot_main)

    plaid_visualize.main()

    assert len(calls) == 1
    seen_datasets, seen_plaid_set, seen_loader = calls[0]
    assert set(seen_datasets) == set(plaid_visualize.DATASETS)
    assert seen_plaid_set == plaid_visualize.DATASETS
    assert seen_loader is plaid_visualize.load_mesh_static_plaid_dataset_samples
    # Module state must be restored after main() returns, so a later plain
    # ginot_visualize.main() call never treats PLAID names as valid.
    assert ginot_visualize.DATASETS == original_datasets
    assert ginot_visualize.MESH_STATIC_PLAID_VIZ_DATASETS == original_plaid_set == frozenset()


def test_unified_visualizer_rewrites_alias_to_supported_canonical_name(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize", "--dataset", "hyperelasticity", "--no-spectral"],
    )

    unified_visualize._rewrite_argv_datasets({"plaid_hyperelasticity"})

    assert sys.argv == ["visualize", "--dataset", "plaid_hyperelasticity", "--no-spectral"]


def test_unified_visualizer_rejects_unsupported_canonical_dataset(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["visualize", "--dataset", "lpbf"])

    with pytest.raises(SystemExit, match="Unsupported visualization dataset 'lpbf'"):
        unified_visualize._rewrite_argv_datasets({"poisson_unstructured"})


def test_unified_visualize_main_smoke_no_display(monkeypatch, tmp_path: Path) -> None:
    """CLI entry resolves argv and delegates PLAID datasets to plaid_visualize, never to ginot."""
    calls: list[tuple] = []

    def fake_main() -> None:
        calls.append(tuple(sys.argv))

    def fail_ginot_main() -> None:
        raise AssertionError("PLAID viz must not call ginot.visualize.main")

    monkeypatch.setattr(plaid_visualize, "DATASETS", frozenset({"plaid_tensile2d"}))
    monkeypatch.setattr(plaid_visualize, "main", fake_main)
    monkeypatch.setattr(ginot_visualize, "DATASETS", ["poisson_unstructured"])
    monkeypatch.setattr(ginot_visualize, "main", fail_ginot_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--dataset",
            "tensile2d",
            "--mode",
            "raw",
            "--outdir",
            str(tmp_path / "viz"),
        ],
    )

    unified_visualize.main()

    assert len(calls) == 1
    assert "--dataset" in calls[0]
    ds_idx = calls[0].index("--dataset")
    assert calls[0][ds_idx + 1] == "plaid_tensile2d"
    assert "--outdir" in calls[0]


def test_unified_visualize_routes_ginot_to_ginot_module(monkeypatch, tmp_path: Path) -> None:
    """CLI entry delegates GINOT datasets to ginot_visualize, never to plaid_visualize."""
    calls: list[tuple] = []

    def fake_main() -> None:
        calls.append(tuple(sys.argv))

    def fail_plaid_main() -> None:
        raise AssertionError("GINOT viz must not call plaid_visualize.main")

    monkeypatch.setattr(ginot_visualize, "DATASETS", ["poisson_unstructured"])
    monkeypatch.setattr(ginot_visualize, "main", fake_main)
    monkeypatch.setattr(plaid_visualize, "DATASETS", frozenset({"plaid_tensile2d"}))
    monkeypatch.setattr(plaid_visualize, "main", fail_plaid_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize", "--dataset", "poisson_unstructured", "--outdir", str(tmp_path / "viz")],
    )

    unified_visualize.main()

    assert len(calls) == 1
    assert "--dataset" in calls[0]
    ds_idx = calls[0].index("--dataset")
    assert calls[0][ds_idx + 1] == "poisson_unstructured"


def test_unified_visualize_rejects_mixed_plaid_and_ginot_datasets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ginot_visualize, "DATASETS", ["poisson_unstructured"])
    monkeypatch.setattr(plaid_visualize, "DATASETS", frozenset({"plaid_tensile2d"}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--dataset",
            "plaid_tensile2d",
            "poisson_unstructured",
            "--outdir",
            str(tmp_path / "viz"),
        ],
    )

    with pytest.raises(SystemExit, match="must be all-PLAID or all-GINOT"):
        unified_visualize.main()


def test_bumper_beam_deformed_positions_stack_xyz_channels() -> None:
    from pdebench.dataset.ginot.bumper_beam import TARGET_FIELDS

    n = 4
    y = np.zeros((n, len(TARGET_FIELDS)), dtype=np.float32)
    t = 20
    ix = TARGET_FIELDS.index(f"position_x_t{t}")
    iy = TARGET_FIELDS.index(f"position_y_t{t}")
    iz = TARGET_FIELDS.index(f"position_z_t{t}")
    y[:, ix] = 1.0
    y[:, iy] = 2.0
    y[:, iz] = 3.0
    pos = ginot_visualize._bumper_beam_deformed_positions(y, list(TARGET_FIELDS), t)
    assert pos.shape == (n, 3)
    np.testing.assert_allclose(pos, np.tile([1.0, 2.0, 3.0], (n, 1)))


def test_visualize_bumper_beam_sample_uses_deformed_mesh_for_fields(tmp_path: Path, monkeypatch) -> None:
    from pdebench.dataset.ginot.bumper_beam import TARGET_FIELDS, TARGET_TIMES

    n = 5
    pos0 = np.zeros((n, 3), dtype=np.float32)
    y = np.zeros((n, len(TARGET_FIELDS)), dtype=np.float32)
    for time in TARGET_TIMES:
        scale = float(time) / 100.0
        y[:, TARGET_FIELDS.index(f"position_x_t{time}")] = scale
        y[:, TARGET_FIELDS.index(f"position_y_t{time}")] = 2.0 * scale
        y[:, TARGET_FIELDS.index(f"position_z_t{time}")] = 3.0 * scale
        y[:, TARGET_FIELDS.index(f"effective_plastic_strain_t{time}")] = 0.1 * scale
        y[:, TARGET_FIELDS.index(f"stress_vm_t{time}")] = 10.0 * scale
    sample = {
        "sample_id": 7,
        "pos": pos0,
        "y": y,
        "cells": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64),
    }
    args = SimpleNamespace(max_points=100000, point_size=2.0, dpi=80)
    geometry_calls: list[tuple[str, np.ndarray]] = []
    field_calls: list[tuple[str, np.ndarray]] = []

    def fake_geometry(ax, pos, cells, title, point_size):
        del ax, cells, point_size
        geometry_calls.append((title.split("\n", 1)[0], np.asarray(pos).copy()))

    def fake_field(ax, pos, values, title, point_size):
        del ax, values, point_size
        field_calls.append((title, np.asarray(pos).copy()))

    monkeypatch.setattr(ginot_visualize, "add_3d_geometry", fake_geometry)
    monkeypatch.setattr(ginot_visualize, "add_3d_field", fake_field)
    monkeypatch.setattr(ginot_visualize, "set_3d_equalish", lambda *a, **k: None)

    out_path = tmp_path / "sample_00007.png"
    ginot_visualize.visualize_bumper_beam_sample(sample, list(TARGET_FIELDS), "train", out_path, args)

    assert out_path.is_file()
    assert geometry_calls[0][0] == "query mesh (t=0)"
    np.testing.assert_allclose(geometry_calls[0][1], pos0)
    assert len(geometry_calls) == 1 + len(TARGET_TIMES)
    assert len(field_calls) == 2 * len(TARGET_TIMES)
    for time, (title, pos) in zip(TARGET_TIMES, geometry_calls[1:]):
        assert title == f"mesh t={time}"
        scale = float(time) / 100.0
        np.testing.assert_allclose(pos, np.tile([scale, 2.0 * scale, 3.0 * scale], (n, 1)))
    for time in TARGET_TIMES:
        scale = float(time) / 100.0
        expected = np.tile([scale, 2.0 * scale, 3.0 * scale], (n, 1))
        strain = next(pos for name, pos in field_calls if name == f"effective_plastic_strain_t{time}")
        stress = next(pos for name, pos in field_calls if name == f"stress_vm_t{time}")
        np.testing.assert_allclose(strain, expected)
        np.testing.assert_allclose(stress, expected)


def test_main_dispatches_bumper_beam_to_specialized_visualizer(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_load(*args, **kwargs):
        del args, kwargs
        sample = {"sample_id": 1, "pos": np.zeros((2, 3)), "y": np.zeros((2, 1)), "cells": np.zeros((0, 0))}
        meta = {
            "num_samples": 1,
            "ids_by_split": {"train": [1]},
            "y_normalizer": None,
            "index_maps": {},
            "dataset": "bumper_beam",
            "data_root": str(tmp_path),
            "seed": 0,
            "max_samples": 1,
        }
        return {"train": [sample]}, ["dummy"], meta, {"train": SimpleNamespace(raw=None)}

    def fake_bumper(sample, target_fields, split, out_path, args):
        del sample, target_fields, args
        calls.append(f"{split}:{out_path.name}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ok")

    def fail_generic(*args, **kwargs):
        del args, kwargs
        raise AssertionError("bumper_beam must not use generic visualize_sample")

    monkeypatch.setattr(ginot_visualize, "load_dataset_samples", fake_load)
    monkeypatch.setattr(ginot_visualize, "_purge_dataset_viz_outputs", lambda *a, **k: None)
    monkeypatch.setattr(ginot_visualize, "visualize_bumper_beam_sample", fake_bumper)
    monkeypatch.setattr(ginot_visualize, "visualize_sample", fail_generic)
    monkeypatch.setattr(ginot_visualize, "plot_spectral_features", lambda *a, **k: None)
    monkeypatch.setattr(
        ginot_visualize,
        "parse_args",
        lambda: SimpleNamespace(
            dataset=["bumper_beam"],
            data_root=tmp_path,
            outdir=tmp_path / "viz",
            split_seed=0,
            max_samples=1,
            splits=["train"],
            spectral_only=False,
            overwrite=True,
            no_spectral=True,
            laplacian_specs="graph:32",
            spectral_modes=32,
            spectral_operators=None,
            use_sdf_features=False,
            micro_puc_fixed_resolved_samples=[],
        ),
    )

    ginot_visualize.main()
    assert calls == ["train:sample_00001.png"]

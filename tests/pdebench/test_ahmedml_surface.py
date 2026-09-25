import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pdebench
import pdebench.callbacks as callbacks
from pdebench.dataset import ahmedml

SPLIT = Path("pdebench/dataset/splits/ahmedml.json")
_AHMEDML_FULL_ROOTS = (
    Path("data/AhmedML/surface_full"),
    Path("/project/community/vedantpu/FLARE-dev.py/data/AhmedML/surface_full"),
)


def _ahmedml_full_root() -> Path | None:
    for root in _AHMEDML_FULL_ROOTS:
        if (root / "run_1").is_dir():
            return root
    return None


def test_ahmedml_split_counts_and_disjoint() -> None:
    data = json.loads(SPLIT.read_text())
    assert data["seed"] == 42
    train, val, test = set(data["train"]), set(data["val"]), set(data["test"])
    assert len(train) == 400 and len(val) == 50 and len(test) == 50
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == {f"run_{i}" for i in range(1, 501)}
    # AB-UPT / Noether pin: first few IDs from seed-42 create_split
    assert data["train"][:5] == ["run_1", "run_2", "run_3", "run_5", "run_6"]
    assert data["test"][:5] == ["run_4", "run_11", "run_12", "run_19", "run_20"]


def _write_full_run(root: Path, run_id: str, *, n: int = 30) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    indices = np.arange(n, dtype=np.float32)
    points = np.stack((indices / n, indices / n, indices / n), axis=-1)
    normals = np.tile(np.array([3.0, 4.0, 0.0], dtype=np.float32), (n, 1))
    pressure = indices.copy()
    tau = np.stack((indices, indices + 1, indices + 2), axis=-1)
    prefix = run_dir / f"boundary_{run_id.split('_')[1]}"
    np.save(f"{prefix}_points.npy", points)
    np.save(f"{prefix}_normals.npy", normals)
    np.save(f"{prefix}_p.npy", pressure)
    np.save(f"{prefix}_tau.npy", tau)


def _write_split(root: Path) -> None:
    split_path = root / "splits" / "ahmedml.json"
    split_path.parent.mkdir()
    split_path.write_text(json.dumps({"seed": 0, "train": ["run_1"], "val": [], "test": ["run_2"]}))


@pytest.mark.skipif(_ahmedml_full_root() is None, reason="full-mesh AhmedML data missing")
def test_real_ahmedml_shapes() -> None:
    data_root = _ahmedml_full_root()
    assert data_root is not None
    train, test, meta = ahmedml.load_ahmedml_surface_dataset(str(data_root))
    assert len(train) == 400
    assert len(test) == 50
    assert len(meta["ahmedml_train_run_data"]) == 400
    x, y = train[0]
    assert x.shape[-1] == 6 and y.shape[-1] == 4
    assert x.shape[0] == ahmedml.SUBSET_SIZE or x.shape[0] < ahmedml.SUBSET_SIZE
    assert meta["c_in"] == 6 and meta["c_out"] == 4
    assert meta["ahmedml_subset_size"] == ahmedml.SUBSET_SIZE


def test_amortized_train_uniform_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", True)
    monkeypatch.setattr(ahmedml, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_STD", np.ones(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "SUBSET_SIZE", 7)
    _write_full_run(tmp_path, "run_1", n=20)
    _write_full_run(tmp_path, "run_2", n=20)
    _write_split(tmp_path)
    monkeypatch.setattr(ahmedml, "SPLIT_PATH", tmp_path / "splits" / "ahmedml.json")

    forced = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.int64)

    class FixedRng:
        def choice(self, n, size=None, replace=True):
            assert n == 20
            assert size == 7
            assert replace is False
            return forced.copy()

    monkeypatch.setattr(
        ahmedml.AhmedMLSurfaceAmortizedTrainDataset,
        "_rng",
        lambda self, idx: FixedRng(),
    )

    train, test, meta = ahmedml.load_ahmedml_surface_dataset(str(tmp_path))
    assert isinstance(train, ahmedml.AhmedMLSurfaceAmortizedTrainDataset)
    assert len(train) == 1
    assert len(test) == 1

    x, y = train[0]
    assert x.shape == (7, 6)
    assert y.shape == (7, 4)
    full_x, full_y = meta["ahmedml_train_run_data"][0]
    assert torch.equal(x, full_x[forced])
    assert torch.equal(y, full_y[forced])
    assert meta["max_length"] == 7


def test_sample_indices_without_replacement_is_uniform_distinct() -> None:
    rng = np.random.default_rng(0)
    counts = np.zeros(50, dtype=np.int64)
    for _ in range(2000):
        idx = ahmedml.sample_indices_without_replacement(50, 10, rng=rng)
        assert len(idx) == 10
        assert len(set(idx.tolist())) == 10
        counts[idx] += 1
    # Each index should appear roughly equally often.
    assert counts.min() > 200
    assert counts.max() < 600


def test_load_ahmedml_surface_dataset_honors_subset_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", True)
    monkeypatch.setattr(ahmedml, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=20)
    _write_full_run(tmp_path, "run_2", n=20)
    _write_split(tmp_path)
    monkeypatch.setattr(ahmedml, "SPLIT_PATH", tmp_path / "splits" / "ahmedml.json")

    train, test, meta = ahmedml.load_ahmedml_surface_dataset(str(tmp_path), subset_size=7)
    assert train.subset_size == 7
    assert test.subset_size == 7
    assert meta["ahmedml_subset_size"] == 7
    assert meta["max_length"] == 7
    assert meta["ahmedml_iid_samples"] is True
    assert "ahmedml_num_parts" not in meta
    x, y = train[0]
    assert x.shape[0] == 7 and y.shape[0] == 7
    # Full-mesh eval path unchanged.
    full_x, full_y = meta["ahmedml_train_run_data"][0]
    assert full_x.shape[0] == 20 and full_y.shape[0] == 20


def test_load_ahmedml_surface_dataset_iid_samples_false_uses_strided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", True)
    monkeypatch.setattr(ahmedml, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=50)
    _write_full_run(tmp_path, "run_2", n=50)
    _write_split(tmp_path)
    monkeypatch.setattr(ahmedml, "SPLIT_PATH", tmp_path / "splits" / "ahmedml.json")

    train, test, meta = ahmedml.load_ahmedml_surface_dataset(
        str(tmp_path), subset_size=7, iid_samples=False
    )
    assert train.sampling == "strided_parts"
    assert test.sampling == "strided_parts"
    assert train.num_parts == ahmedml.NUM_PARTS
    assert meta["ahmedml_iid_samples"] is False
    assert meta["ahmedml_num_parts"] == ahmedml.NUM_PARTS
    assert meta["max_length"] == 5  # ceil(50/10)
    x, _ = train[0]
    assert x.shape[0] == 5


def test_sample_indices_strided_part_covers_all_residues() -> None:
    class FixedRng:
        def __init__(self, k: int) -> None:
            self.k = k

        def integers(self, low: int, high: int) -> int:
            assert (low, high) == (0, 10)
            return self.k

    idx = ahmedml.sample_indices_strided_part(30, num_parts=10, rng=FixedRng(3))
    assert idx.tolist() == list(range(3, 30, 10))


def test_load_ahmedml_surface_dataset_default_subset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", True)
    monkeypatch.setattr(ahmedml, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=20)
    _write_full_run(tmp_path, "run_2", n=20)
    _write_split(tmp_path)
    monkeypatch.setattr(ahmedml, "SPLIT_PATH", tmp_path / "splits" / "ahmedml.json")

    train, test, meta = ahmedml.load_ahmedml_surface_dataset(str(tmp_path))
    assert train.subset_size == ahmedml.SUBSET_SIZE
    assert test.subset_size == ahmedml.SUBSET_SIZE
    assert meta["ahmedml_subset_size"] == ahmedml.SUBSET_SIZE
    assert meta["ahmedml_iid_samples"] is True
    assert meta["max_length"] == min(ahmedml.SUBSET_SIZE, 20)


def test_loader_raises_until_hardcoded_stats_are_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", False)

    with pytest.raises(RuntimeError, match="surface prep.*paste stats"):
        ahmedml.load_ahmedml_surface_dataset(str(tmp_path))


def test_ahmedml_adapter_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import _AhmedMLSurfaceAdapter, get_adapter

    def fake(data_root: str, *, subset_size: int = 100_000, iid_samples: bool = True):
        assert data_root == str(tmp_path)
        assert subset_size == 100_000
        assert iid_samples is True
        return "train", "test", {"c_in": 6}

    monkeypatch.setattr(dataset_utils, "load_ahmedml_surface_dataset", fake)
    adapter = get_adapter("ahmedml_surface")
    assert isinstance(adapter, _AhmedMLSurfaceAdapter)
    train, test, meta = adapter.load(str(tmp_path))

    assert (train, test, meta["c_in"]) == ("train", "test", 6)


def test_ahmedml_adapter_forwards_subset_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter

    seen: dict[str, object] = {}

    def fake(data_root: str, *, subset_size: int = 100_000, iid_samples: bool = True):
        seen["data_root"] = data_root
        seen["subset_size"] = subset_size
        seen["iid_samples"] = iid_samples
        return "train", "test", {"c_in": 6}

    monkeypatch.setattr(dataset_utils, "load_ahmedml_surface_dataset", fake)
    get_adapter("ahmedml_surface").load(str(tmp_path), subset_size=1234, iid_samples=False)

    assert seen == {"data_root": str(tmp_path), "subset_size": 1234, "iid_samples": False}


def test_dataset_config_subset_size_default() -> None:
    from pdebench.config import DatasetConfig

    assert DatasetConfig().subset_size == 100_000
    assert DatasetConfig().iid_samples is True
    assert DatasetConfig().rel_l2_loss is False


def test_mse_datasets_include_ahmedml() -> None:
    from pdebench.__main__ import MSE_NORMALIZED_DATASETS

    assert "ahmedml_surface" in MSE_NORMALIZED_DATASETS
    assert "nasa_crm" in MSE_NORMALIZED_DATASETS


def test_callback_writes_full_mesh_stats_without_model_call(tmp_path: Path) -> None:
    class FailOnCall(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("callback must consume fullbatch stats without calling the model")

    trainer = SimpleNamespace(model=FailOnCall(), GLOBAL_RANK=0)
    stat_vals = {
        "train_stats": {
            "full_rel_l2": 0.11,
            "pressure_rel_l2": 0.12,
            "wall_shear_rel_l2": 0.13,
        },
        "test_stats": {
            "full_rel_l2": 0.21,
            "pressure_rel_l2": 0.22,
            "wall_shear_rel_l2": 0.23,
        },
        "train_stats_ema": {
            "full_rel_l2": 0.31,
            "pressure_rel_l2": 0.32,
            "wall_shear_rel_l2": 0.33,
        },
        "test_stats_ema": {
            "full_rel_l2": 0.41,
            "pressure_rel_l2": 0.42,
            "wall_shear_rel_l2": 0.43,
        },
    }
    callback = callbacks.AhmedMLSurfaceRelL2Callback(
        case_dir=str(tmp_path), dataset="ahmedml_surface", x_normalizer=None, y_normalizer=None
    )
    (tmp_path / "checkpoint").mkdir()

    callback.evaluate(trainer, str(tmp_path / "checkpoint"), stat_vals)

    expected = {
        "train_pressure_rel_l2": 0.12,
        "test_pressure_rel_l2": 0.22,
        "train_wall_shear_rel_l2": 0.13,
        "test_wall_shear_rel_l2": 0.23,
        "train_full_rel_l2": 0.11,
        "test_full_rel_l2": 0.21,
        "train_pressure_rel_l2_ema": 0.32,
        "test_pressure_rel_l2_ema": 0.42,
        "train_wall_shear_rel_l2_ema": 0.33,
        "test_wall_shear_rel_l2_ema": 0.43,
        "train_full_rel_l2_ema": 0.31,
        "test_full_rel_l2_ema": 0.41,
    }
    assert json.loads((tmp_path / "checkpoint" / "rel_error.json").read_text()) == expected
    assert json.loads((tmp_path / "rel_error.json").read_text()) == expected


def test_callback_skips_when_fullbatch_stats_empty(tmp_path: Path) -> None:
    trainer = SimpleNamespace(model=torch.nn.Identity(), GLOBAL_RANK=0)
    callback = callbacks.AhmedMLSurfaceRelL2Callback(
        case_dir=str(tmp_path), dataset="ahmedml_surface", x_normalizer=None, y_normalizer=None
    )
    (tmp_path / "checkpoint").mkdir()

    callback.evaluate(
        trainer,
        str(tmp_path / "checkpoint"),
        {
            "train_stats": {},
            "test_stats": {},
            "train_stats_ema": {},
            "test_stats_ema": {},
        },
    )

    assert not (tmp_path / "checkpoint" / "rel_error.json").exists()
    assert not (tmp_path / "rel_error.json").exists()


def test_getitem_shapes_unit_normals_and_run_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ahmedml, "STATS_READY", True)
    monkeypatch.setattr(ahmedml, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "XYZ_MAX", np.full(3, 2.0, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_MEAN", np.ones(4, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "Y_STD", np.full(4, 2.0, dtype=np.float32))
    monkeypatch.setattr(ahmedml, "SUBSET_SIZE", 100_000)
    _write_full_run(tmp_path, "run_1", n=30)
    _write_full_run(tmp_path, "run_2", n=30)
    _write_split(tmp_path)
    monkeypatch.setattr(ahmedml, "SPLIT_PATH", tmp_path / "splits" / "ahmedml.json")

    train, test, meta = ahmedml.load_ahmedml_surface_dataset(str(tmp_path))

    x, y = train[0]
    assert len(train) == 1
    assert len(test) == 1
    assert train.source_run_ids == ("run_1",)
    assert meta["ahmedml_test_run_data"].source_run_ids == ("run_2",)
    assert x.shape == (30, 6)  # N < SUBSET_SIZE → use all cells
    assert y.shape == (30, 4)
    norms = torch.linalg.vector_norm(x[:, 3:6], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert meta["c_in"] == 6 and meta["c_out"] == 4
    assert meta["max_length"] == 30
    assert meta["y_field_metrics"]["tau"]["kind"] == "vector_norm"


def test_run_dataset_returns_full_mesh(tmp_path: Path) -> None:
    _write_full_run(tmp_path, "run_1", n=7)
    runs = ahmedml.AhmedMLSurfaceRunDataset(
        tmp_path,
        ["run_1"],
        xyz_min=np.zeros(3, dtype=np.float32),
        xyz_max=np.ones(3, dtype=np.float32),
        y_mean=np.zeros(4, dtype=np.float32),
        y_std=np.ones(4, dtype=np.float32),
    )
    x, y = runs[0]
    assert x.shape == (7, 6)
    assert y.shape == (7, 4)


def test_run_dataset_reports_missing_field(tmp_path: Path) -> None:
    _write_full_run(tmp_path, "run_7", n=7)
    (tmp_path / "run_7" / "boundary_7_points.npy").unlink()
    runs = ahmedml.AhmedMLSurfaceRunDataset(tmp_path, ["run_7"])
    with pytest.raises(ValueError, match=r"run_7"):
        _ = runs[0]


def test_run_dataset_reports_malformed_channels(tmp_path: Path) -> None:
    _write_full_run(tmp_path, "run_8", n=7)
    np.save(tmp_path / "run_8" / "boundary_8_tau.npy", np.zeros((7, 2), dtype=np.float32))
    runs = ahmedml.AhmedMLSurfaceRunDataset(tmp_path, ["run_8"])
    with pytest.raises(ValueError, match=r"run_8"):
        _ = runs[0]


def test_run_dataset_rejects_duplicate_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate source run"):
        ahmedml.AhmedMLSurfaceRunDataset(tmp_path, ["run_4", "run_4"])


def test_ahmedml_metric_sums_match_physical_full_tensor_relative_l2() -> None:
    target = torch.tensor([[[1.0, 3.0, 4.0, 0.0], [2.0, 0.0, 0.0, 5.0], [4.0, 0.0, 0.0, 0.0]]])
    pred = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 10.0], [4.0, 3.0, 4.0, 0.0]]])

    sums = callbacks.ahmedml_surface_metric_sums(pred, target)

    assert tuple(float(v) for v in sums["full_rel_l2"]) == pytest.approx((80.0, 71.0))
    assert tuple(float(v) for v in sums["pressure_rel_l2"]) == pytest.approx((5.0, 21.0))
    assert tuple(float(v) for v in sums["wall_shear_rel_l2"]) == pytest.approx((75.0, 50.0))


@pytest.mark.parametrize("parts", [2, 4])
def test_ahmedml_metric_sums_reduce_uneven_shards_to_full_mesh(parts: int) -> None:
    torch.manual_seed(7)
    target = torch.randn(1, 11, 4)
    pred = target + torch.randn_like(target) * 0.2
    full = callbacks.ahmedml_surface_metric_sums(pred, target)
    shards = [
        callbacks.ahmedml_surface_metric_sums(p, t)
        for p, t in zip(torch.tensor_split(pred, parts, 1), torch.tensor_split(target, parts, 1))
    ]

    for name, (numerator, denominator) in full.items():
        assert torch.allclose(sum(shard[name][0] for shard in shards), numerator)
        assert torch.allclose(sum(shard[name][1] for shard in shards), denominator)


def test_ahmedml_full_mesh_stats_match_normalized_mse_and_report_physical_rel_l2() -> None:
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return x[..., :4]

    normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(2, 1, 4))
    normalizer.mean = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    normalizer.std = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]])
    pred = target + torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    model = CountingModel()
    trainer = SimpleNamespace(
        model=model,
        DDP=False,
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        verbose=False,
        GLOBAL_RANK=0,
        print_iterator=False,
        move_to_device=lambda batch: batch,
        preprocess_fn_=lambda batch: batch,
    )
    statsfun = callbacks.make_ahmedml_surface_statsfun({"y_normalizer": normalizer})

    loss, stats = statsfun(trainer, [(pred, target), (pred, target)], split="test")
    expected_mse = float(torch.nn.functional.mse_loss(pred, target))
    physical_pred, physical_target = normalizer.decode(pred), normalizer.decode(target)
    expected_rel = {
        name: float(torch.sqrt(num / den))
        for name, (num, den) in callbacks.ahmedml_surface_metric_sums(physical_pred, physical_target).items()
    }
    # Primary fullbatch loss matches training: normalized MSE (no decode).
    assert loss == pytest.approx(expected_mse)
    assert stats["mse"] == pytest.approx(expected_mse)
    # Physical Rel-L2 retained as diagnostic paper columns.
    for name, value in expected_rel.items():
        assert stats[name] == pytest.approx(value)
    assert not any(key.startswith("ts3_") for key in stats)
    # One full-mesh forward per run (no ts3 part loop).
    assert model.calls == 2


def test_ahmedml_full_mesh_stats_rel_l2_loss_uses_pressure_wall_shear_blend() -> None:
    class SliceModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[..., :4]

    normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(2, 1, 4))
    normalizer.mean = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    normalizer.std = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]])
    pred = target + torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    model = SliceModel()
    trainer = SimpleNamespace(
        model=model,
        DDP=False,
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        verbose=False,
        GLOBAL_RANK=0,
        print_iterator=False,
        move_to_device=lambda batch: batch,
        preprocess_fn_=lambda batch: batch,
    )
    statsfun = callbacks.make_ahmedml_surface_statsfun(
        {"y_normalizer": normalizer, "rel_l2_loss": True}
    )

    loss, stats = statsfun(trainer, [(pred, target)], split="test")
    expected_mse = float(torch.nn.functional.mse_loss(pred, target))
    expected_batch = float(
        callbacks.surface_batch_loss(
            pred, target, rel_l2_loss=True, y_normalizer=normalizer
        ).item()
    )
    assert loss == pytest.approx(expected_batch)
    assert stats["mse"] == pytest.approx(expected_mse)
    assert loss == pytest.approx(0.5 * (stats["pressure_rel_l2"] + stats["wall_shear_rel_l2"]))


def test_surface_batch_loss_mse_and_rel_l2_modes() -> None:
    normalizer = pdebench.IdentityNormalizer()
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]]])
    pred = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])

    mse = callbacks.surface_batch_loss(pred, target, rel_l2_loss=False, y_normalizer=normalizer)
    assert float(mse.item()) == pytest.approx(float(torch.nn.functional.mse_loss(pred, target)))

    blend = callbacks.surface_batch_loss(pred, target, rel_l2_loss=True, y_normalizer=normalizer)
    sums = callbacks.ahmedml_surface_metric_sums(pred, target)
    p = float(torch.sqrt(sums["pressure_rel_l2"][0] / sums["pressure_rel_l2"][1]))
    t = float(torch.sqrt(sums["wall_shear_rel_l2"][0] / sums["wall_shear_rel_l2"][1]))
    assert float(blend.item()) == pytest.approx(0.5 * (p + t))


def _ahmedml_stats_trainer(model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
        preprocess_fn_=lambda batch: batch,
    )


def test_ahmedml_full_mesh_stats_restore_training_mode_after_success() -> None:
    model = torch.nn.Identity().train()
    statsfun = callbacks.make_ahmedml_surface_statsfun({"y_normalizer": pdebench.IdentityNormalizer()})
    batch = (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    statsfun(_ahmedml_stats_trainer(model), [batch], split="test")

    assert model.training


def test_ahmedml_full_mesh_stats_restore_training_mode_after_exception() -> None:
    class FailingModel(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("forward failed")

    model = FailingModel().train()
    statsfun = callbacks.make_ahmedml_surface_statsfun({"y_normalizer": pdebench.IdentityNormalizer()})
    batch = (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    with pytest.raises(RuntimeError, match="forward failed"):
        statsfun(_ahmedml_stats_trainer(model), [batch], split="test")

    assert model.training

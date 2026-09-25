import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pdebench
from pdebench import callbacks
from pdebench.dataset import drivaerml_surface

SPLIT = Path("pdebench/dataset/splits/drivaerml.json")
_DRIVAERML_FULL_ROOTS = (
    Path("data/DrivAerML/surface_full"),
    Path("/project/community/sbandred/FLARE-dev.py/data/DrivAerML/surface_full"),
    Path("/project/community/vedantpu/FLARE-dev.py/data/DrivAerML/surface_full"),
)


def _drivaerml_full_root() -> Path | None:
    for root in _DRIVAERML_FULL_ROOTS:
        if (root / "run_1").is_dir():
            return root
    return None


def test_drivaerml_split_abupt_counts_and_pin() -> None:
    data = json.loads(SPLIT.read_text())
    assert data["seed"] == 42
    train, val, test = set(data["train"]), set(data["val"]), set(data["test"])
    hidden = set(data["hidden_test"])
    # Train matches Transolver-3 / drivaerml_split.json (399); run_45 is excluded from all splits.
    assert len(train) == 399 and len(val) == 34 and len(test) == 50 and len(hidden) == 16
    assert "run_45" not in train | val | test | hidden
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert (train | val | test).isdisjoint(hidden)
    universe = {f"run_{i}" for i in range(1, 501)}
    assert train | val | test | hidden == universe - {"run_45"}
    assert data["train"][:5] == ["run_1", "run_2", "run_3", "run_5", "run_6"]
    assert data["val"][:5] == ["run_4", "run_22", "run_56", "run_109", "run_150"]
    assert data["test"][:5] == ["run_11", "run_12", "run_19", "run_20", "run_24"]
    assert data["hidden_test"][:5] == ["run_167", "run_211", "run_218", "run_221", "run_248"]


def test_drivaerml_stride_part_lengths_balanced_and_cover_mesh() -> None:
    lengths = drivaerml_surface.stride_part_lengths(8_000_017, 80)
    assert sum(lengths) == 8_000_017
    assert max(lengths) - min(lengths) <= 1
    assert min(lengths) > 0
    # Ideal mesh for K=80 → 100k/part (TARGET_PART_SIZE).
    assert drivaerml_surface.surprising_stride_parts(8_000_000, 80) == []
    # Too-small mesh → parts outside 100k±10%.
    surprises = drivaerml_surface.surprising_stride_parts(2_000_000, 80)
    assert len(surprises) == 80
    assert all(length == 25_000 for _part_id, length in surprises)


def test_drivaerml_loader_uses_train_and_test_only_not_val(tmp_path: Path, monkeypatch) -> None:
    """Harness must never wire val into train/test datasets or fullbatch stats metadata."""
    root = tmp_path / "data"
    for run_id, n in (("run_1", 30), ("run_2", 30), ("run_3", 30)):
        _write_full_run(root, run_id, n=n)
    split_path = root / "splits" / "drivaerml.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "train": ["run_1"],
                "val": ["run_2"],
                "test": ["run_3"],
                "hidden_test": [],
            }
        )
    )
    monkeypatch.setattr(drivaerml_surface, "STATS_READY", True)
    monkeypatch.setattr(drivaerml_surface, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_STD", np.ones(4, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "SPLIT_PATH", split_path)

    train, test, meta = drivaerml_surface.load_drivaerml_surface_dataset(str(root))
    assert list(train.source_run_ids) == ["run_1"]
    assert list(test.source_run_ids) == ["run_3"]
    assert list(meta["drivaerml_train_run_data"].source_run_ids) == ["run_1"]
    assert list(meta["drivaerml_test_run_data"].source_run_ids) == ["run_3"]
    assert "drivaerml_val_run_data" not in meta
    assert "run_2" not in set(train.source_run_ids) | set(test.source_run_ids)


@pytest.mark.skipif(_drivaerml_full_root() is None, reason="full-mesh DrivAerML data missing")
def test_real_drivaerml_shapes() -> None:
    data_root = _drivaerml_full_root()
    assert data_root is not None
    train, test, meta = drivaerml_surface.load_drivaerml_surface_dataset(
        str(data_root), iid_samples=False
    )
    assert len(train) == 399
    assert len(test) == 50
    assert len(meta["drivaerml_train_run_data"]) == 399
    assert "run_45" not in train.source_run_ids
    x, y = train[0]
    assert x.shape[-1] == 6 and y.shape[-1] == 4
    assert x.shape[0] > 0
    assert train.sampling == "strided_parts"
    n = meta["drivaerml_train_run_data"]._mesh_length(train.source_run_ids[0])
    n_subset = drivaerml_surface.SUBSET_SIZE
    k = drivaerml_surface.NUM_PARTS
    expected_idx = drivaerml_surface.sample_train_indices(
        n, subset_size=n_subset, num_parts=k, rng=train._rng(0)
    )
    assert x.shape[0] == len(expected_idx)
    assert x.shape[0] <= max((n + k - 1) // k, min(n_subset, n))
    x_te, _ = test[0]
    n_te = meta["drivaerml_test_run_data"]._mesh_length(test.source_run_ids[0])
    assert x_te.shape[0] == min(drivaerml_surface.SUBSET_SIZE, n_te)
    assert meta["c_in"] == 6 and meta["c_out"] == 4
    assert meta["drivaerml_subset_size"] == drivaerml_surface.SUBSET_SIZE
    assert meta["drivaerml_iid_samples"] is False
    assert drivaerml_surface.STATS_READY is True


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


def _write_split(root: Path) -> Path:
    split_path = root / "splits" / "drivaerml.json"
    split_path.parent.mkdir()
    split_path.write_text(
        json.dumps({"seed": 42, "train": ["run_1"], "val": [], "test": ["run_2"], "hidden_test": []})
    )
    return split_path


def test_train_pad_short_part_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """n=30 with K=80: strided part is shorter than N, so pad returns all mesh cells."""
    monkeypatch.setattr(drivaerml_surface, "STATS_READY", True)
    monkeypatch.setattr(drivaerml_surface, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=30)
    _write_full_run(tmp_path, "run_2", n=30)
    monkeypatch.setattr(drivaerml_surface, "SPLIT_PATH", _write_split(tmp_path))

    train, _, _ = drivaerml_surface.load_drivaerml_surface_dataset(
        str(tmp_path), subset_size=100_000, iid_samples=False
    )
    assert train.num_parts == 80
    assert train.sampling == "strided_parts"
    x, _ = train[0]
    assert x.shape[0] == 30


def test_load_honors_subset_size_and_derived_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drivaerml_surface, "STATS_READY", True)
    monkeypatch.setattr(drivaerml_surface, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=30)
    _write_full_run(tmp_path, "run_2", n=30)
    monkeypatch.setattr(drivaerml_surface, "SPLIT_PATH", _write_split(tmp_path))

    train, test, meta = drivaerml_surface.load_drivaerml_surface_dataset(
        str(tmp_path), subset_size=100_000, iid_samples=False
    )
    assert train.sampling == "strided_parts"
    assert train.num_parts == 80
    assert train.subset_size == 100_000
    assert test.sampling == "iid"
    assert test.subset_size == 100_000
    assert meta["drivaerml_subset_size"] == 100_000
    assert meta["drivaerml_num_parts"] == 80
    assert meta["drivaerml_iid_samples"] is False
    assert "drivaerml_part_stride" not in meta

    train2, test2, meta2 = drivaerml_surface.load_drivaerml_surface_dataset(
        str(tmp_path), subset_size=200_000, iid_samples=False
    )
    assert train2.num_parts == 40
    assert meta2["drivaerml_num_parts"] == 40


def test_load_iid_samples_true_uses_sorted_iid_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drivaerml_surface, "STATS_READY", True)
    monkeypatch.setattr(drivaerml_surface, "XYZ_MIN", np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "XYZ_MAX", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_MEAN", np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(drivaerml_surface, "Y_STD", np.ones(4, dtype=np.float32))
    _write_full_run(tmp_path, "run_1", n=50)
    _write_full_run(tmp_path, "run_2", n=50)
    monkeypatch.setattr(drivaerml_surface, "SPLIT_PATH", _write_split(tmp_path))

    train, test, meta = drivaerml_surface.load_drivaerml_surface_dataset(
        str(tmp_path), subset_size=10, iid_samples=True
    )
    assert train.sampling == "iid"
    assert test.sampling == "iid"
    assert train.subset_size == 10
    assert meta["drivaerml_iid_samples"] is True
    assert meta["drivaerml_subset_size"] == 10
    assert "drivaerml_num_parts" not in meta
    assert meta["max_length"] == 10

    train._seed = 0
    x, _ = train[0]
    assert x.shape[0] == 10


def test_sample_indices_without_replacement_is_uniform_distinct() -> None:
    rng = np.random.default_rng(0)
    counts = np.zeros(50, dtype=np.int64)
    for _ in range(2000):
        idx = drivaerml_surface.sample_indices_without_replacement(50, 10, rng=rng)
        assert len(idx) == len(set(idx.tolist())) == 10
        counts[idx] += 1
    assert counts.min() > 200
    assert counts.max() < 600


def test_num_parts_from_subset_size_table() -> None:
    assert drivaerml_surface.num_parts_from_subset_size(100_000) == 80
    assert drivaerml_surface.num_parts_from_subset_size(200_000) == 40
    assert drivaerml_surface.num_parts_from_subset_size(400_000) == 20
    assert drivaerml_surface.num_parts_from_subset_size(800_000) == 10


def test_num_parts_from_subset_size_rejects_bad() -> None:
    with pytest.raises(ValueError):
        drivaerml_surface.num_parts_from_subset_size(0)
    with pytest.raises(ValueError):
        drivaerml_surface.num_parts_from_subset_size(150_000)  # not multiple of 100k
    with pytest.raises(ValueError):
        drivaerml_surface.num_parts_from_subset_size(300_000)  # multiple of 100k but K not int


def test_drivaerml_adapter_forwards_subset_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter

    seen: dict[str, object] = {}

    def fake(data_root: str, *, subset_size: int = 100_000, iid_samples: bool = True):
        seen["data_root"] = data_root
        seen["subset_size"] = subset_size
        seen["iid_samples"] = iid_samples
        return object(), object(), {}

    monkeypatch.setattr(dataset_utils, "load_drivaerml_surface_dataset", fake)
    get_adapter("drivaerml_surface").load(str(tmp_path), subset_size=200_000, iid_samples=True)

    assert seen == {"data_root": str(tmp_path), "subset_size": 200_000, "iid_samples": True}


def test_dataset_config_iid_samples_default() -> None:
    from pdebench.config import DatasetConfig

    assert DatasetConfig().iid_samples is True
    assert DatasetConfig().subset_size == 100_000
    assert DatasetConfig().rel_l2_loss is False


def test_sample_indices_strided_part_rejects_bad_args() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        drivaerml_surface.sample_indices_strided_part(0, rng=rng)
    with pytest.raises(ValueError):
        drivaerml_surface.sample_indices_strided_part(10, num_parts=0, rng=rng)


def test_sample_indices_strided_part_fixed_k() -> None:
    class FixedRng:
        def __init__(self, k: int) -> None:
            self._k = k

        def integers(self, low, high=None, **kwargs):
            if high is None:
                low, high = 0, low
            return self._k

    n, k, K = 100, 3, 10
    idx = drivaerml_surface.sample_indices_strided_part(n, num_parts=K, rng=FixedRng(k))
    np.testing.assert_array_equal(idx, np.arange(k, n, K, dtype=np.int64))


def test_sample_train_indices_pads_when_part_short() -> None:
    class FixedRng:
        def __init__(self, k: int, pad: np.ndarray) -> None:
            self._k = k
            self._pad = pad
            self._choice_calls = 0

        def integers(self, low, high=None, **kwargs):
            if high is None:
                low, high = 0, low
            return self._k

        def choice(self, population, size=None, replace=True):
            self._choice_calls += 1
            assert replace is False
            assert size == 3
            return self._pad.copy()

    # n=10, K=8, k=0 → part [0,8] (M=2); N=5 → pad 3 from complement
    rng = FixedRng(k=0, pad=np.array([1, 2, 3], dtype=np.int64))
    idx = drivaerml_surface.sample_train_indices(10, subset_size=5, num_parts=8, rng=rng)
    assert rng._choice_calls == 1
    np.testing.assert_array_equal(idx, np.array([0, 1, 2, 3, 8], dtype=np.int64))  # sorted


def test_sample_train_indices_keeps_all_when_part_long() -> None:
    class FixedRng:
        def integers(self, low, high=None, **kwargs):
            return 0

        def choice(self, *args, **kwargs):
            raise AssertionError("must not IID-pad when M >= N")

    # n=20, K=4, k=0 → part length 5; N=3 → keep all 5
    idx = drivaerml_surface.sample_train_indices(20, subset_size=3, num_parts=4, rng=FixedRng())
    np.testing.assert_array_equal(idx, np.arange(0, 20, 4, dtype=np.int64))


def test_unit_normals_preserves_zeros() -> None:
    normals = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)

    actual = drivaerml_surface.unit_normals(normals)

    np.testing.assert_array_equal(actual[0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(actual[1], np.array([0.6, 0.8, 0.0], dtype=np.float32))


def test_drivaerml_callback_symbols_are_exported() -> None:
    assert {
        "drivaerml_surface_metric_sums",
        "_reduce_drivaerml_surface_metric_sums",
        "make_drivaerml_surface_statsfun",
        "DrivAerMLSurfaceRelL2Callback",
    } <= set(callbacks.__all__)


def test_drivaerml_callback_writes_full_mesh_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trainer = SimpleNamespace(GLOBAL_RANK=0)
    stat_vals = {
        "train_stats": {"full_rel_l2": 0.11, "pressure_rel_l2": 0.12, "wall_shear_rel_l2": 0.13},
        "test_stats": {"full_rel_l2": 0.21, "pressure_rel_l2": 0.22, "wall_shear_rel_l2": 0.23},
        "train_stats_ema": {"full_rel_l2": 0.31, "pressure_rel_l2": 0.32, "wall_shear_rel_l2": 0.33},
        "test_stats_ema": {"full_rel_l2": 0.41, "pressure_rel_l2": 0.42, "wall_shear_rel_l2": 0.43},
    }
    callback = callbacks.DrivAerMLSurfaceRelL2Callback(
        case_dir=str(tmp_path), dataset="drivaerml_surface", x_normalizer=None, y_normalizer=None
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
    assert "DrivAerML pressure Rel-L2" in capsys.readouterr().out


def test_drivaerml_callback_skips_when_fullbatch_stats_empty(tmp_path: Path) -> None:
    trainer = SimpleNamespace(GLOBAL_RANK=0)
    callback = callbacks.DrivAerMLSurfaceRelL2Callback(
        case_dir=str(tmp_path), dataset="drivaerml_surface", x_normalizer=None, y_normalizer=None
    )
    (tmp_path / "checkpoint").mkdir()

    callback.evaluate(
        trainer,
        str(tmp_path / "checkpoint"),
        {"train_stats": {}, "test_stats": {}, "train_stats_ema": {}, "test_stats_ema": {}},
    )

    assert not (tmp_path / "checkpoint" / "rel_error.json").exists()
    assert not (tmp_path / "rel_error.json").exists()


def test_drivaerml_metric_sums_match_physical_full_tensor_relative_l2() -> None:
    target = torch.tensor([[[1.0, 3.0, 4.0, 0.0], [2.0, 0.0, 0.0, 5.0], [4.0, 0.0, 0.0, 0.0]]])
    pred = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 10.0], [4.0, 3.0, 4.0, 0.0]]])

    sums = callbacks.drivaerml_surface_metric_sums(pred, target)

    assert tuple(float(v) for v in sums["full_rel_l2"]) == pytest.approx((80.0, 71.0))
    assert tuple(float(v) for v in sums["pressure_rel_l2"]) == pytest.approx((5.0, 21.0))
    assert tuple(float(v) for v in sums["wall_shear_rel_l2"]) == pytest.approx((75.0, 50.0))


@pytest.mark.parametrize("parts", [2, 4])
def test_drivaerml_metric_sums_reduce_uneven_shards_to_full_mesh(parts: int) -> None:
    torch.manual_seed(7)
    target = torch.randn(1, 11, 4)
    pred = target + torch.randn_like(target) * 0.2
    full = callbacks.drivaerml_surface_metric_sums(pred, target)
    shards = [
        callbacks.drivaerml_surface_metric_sums(p, t)
        for p, t in zip(torch.tensor_split(pred, parts, 1), torch.tensor_split(target, parts, 1))
    ]

    for name, (numerator, denominator) in full.items():
        assert torch.allclose(sum(shard[name][0] for shard in shards), numerator)
        assert torch.allclose(sum(shard[name][1] for shard in shards), denominator)


def test_drivaerml_full_mesh_stats_match_normalized_mse_and_report_physical_rel_l2() -> None:
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
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
        preprocess_fn_=lambda batch: batch,
    )
    # RunDataset-style unbatched [N, C] items; DataLoader adds the batch dim.
    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {
            "y_normalizer": normalizer,
            "drivaerml_test_run_data": [(pred.squeeze(0), target.squeeze(0))],
        }
    )

    loss, stats = statsfun(trainer, [], split="test")
    expected_mse = float(torch.nn.functional.mse_loss(pred, target))
    physical_pred, physical_target = normalizer.decode(pred), normalizer.decode(target)
    expected_rel = {
        name: float(torch.sqrt(num / den))
        for name, (num, den) in callbacks.drivaerml_surface_metric_sums(physical_pred, physical_target).items()
    }
    assert loss == pytest.approx(expected_mse)
    assert stats["mse"] == pytest.approx(expected_mse)
    for name, value in expected_rel.items():
        assert stats[name] == pytest.approx(value)
    assert model.calls == 1


def _drivaerml_stats_trainer(model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
        preprocess_fn_=lambda batch: batch,
    )


def test_drivaerml_statsfun_reuses_trainer_num_workers(monkeypatch) -> None:
    captured: dict = {}

    class _FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured.update(kwargs)
            self._dataset = dataset

        def __iter__(self):
            return iter(self._dataset)

    monkeypatch.setattr(callbacks.torch.utils.data, "DataLoader", _FakeLoader)
    batch = (torch.ones(1, 2, 4), torch.ones(1, 2, 4))
    trainer = _drivaerml_stats_trainer(torch.nn.Identity())
    trainer.num_workers = 8
    trainer.prefetch_factor = 4
    trainer.is_cuda = False
    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {
            "y_normalizer": pdebench.IdentityNormalizer(),
            "drivaerml_test_run_data": [batch],
        }
    )

    statsfun(trainer, [], split="test")

    assert captured["num_workers"] == 8
    assert captured["prefetch_factor"] == 4
    assert captured["persistent_workers"] is True


def test_drivaerml_statsfun_maps_trainer_val_split_to_test_run_data(monkeypatch) -> None:
    """Trainer fullbatch uses split='val'; metadata stores *_test_run_data."""
    captured: dict = {}

    class _FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            self._dataset = dataset

        def __iter__(self):
            return iter(self._dataset)

    class _SliceModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[..., :4]

    monkeypatch.setattr(callbacks.torch.utils.data, "DataLoader", _FakeLoader)
    # Already-batched items so FakeLoader (no collate) matches statsfun expectations.
    full_mesh_runs = [(torch.ones(1, 4, 6), torch.ones(1, 4, 4))]
    trainer = _drivaerml_stats_trainer(_SliceModel())
    trainer.num_workers = 0
    trainer.is_cuda = False
    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {
            "y_normalizer": pdebench.IdentityNormalizer(),
            "drivaerml_test_run_data": full_mesh_runs,
        }
    )

    # Empty fallback: if val is not remapped to test_run_data, no DataLoader is built.
    loss, stats = statsfun(trainer, [], split="val")

    assert captured.get("dataset") is full_mesh_runs
    assert loss == pytest.approx(0.0)
    assert stats["mse"] == pytest.approx(0.0)
    assert stats["full_rel_l2"] == pytest.approx(0.0)


def test_drivaerml_full_mesh_stats_restore_training_mode_after_success() -> None:
    model = torch.nn.Identity().train()
    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()}
    )
    batch = (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    statsfun(_drivaerml_stats_trainer(model), [batch], split="test")

    assert model.training


def test_drivaerml_full_mesh_stats_restore_training_mode_after_exception() -> None:
    class FailingModel(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("forward failed")

    model = FailingModel().train()
    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()}
    )
    batch = (torch.ones(1, 2, 4), torch.ones(1, 2, 4))

    with pytest.raises(RuntimeError, match="forward failed"):
        statsfun(_drivaerml_stats_trainer(model), [batch], split="test")

    assert model.training

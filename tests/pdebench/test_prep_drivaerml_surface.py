import numpy as np
import pytest

from scripts import prep_drivaerml_surface as prep_module
from scripts.prep_drivaerml_surface import compute_train_stats, finalize_run_outputs


def _write_full_run(out_root, run_name: str, *, n: int = 4) -> None:
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True)
    prefix = f"boundary_{run_name.removeprefix('run_')}"
    np.save(run_dir / f"{prefix}_points.npy", np.full((n, 3), 1.0, dtype=np.float32))
    np.save(run_dir / f"{prefix}_normals.npy", np.full((n, 3), 1.0, dtype=np.float32))
    np.save(run_dir / f"{prefix}_p.npy", np.full(n, 2.0, dtype=np.float32))
    np.save(run_dir / f"{prefix}_tau.npy", np.full((n, 3), 3.0, dtype=np.float32))


def test_compute_train_stats_rejects_missing_train_run(tmp_path):
    _write_full_run(tmp_path, "run_2")
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        compute_train_stats(tmp_path, ["run_2", "run_3"])


def test_compute_train_stats_on_full_meshes(tmp_path):
    _write_full_run(tmp_path, "run_1", n=2)
    _write_full_run(tmp_path, "run_2", n=2)
    stats = compute_train_stats(tmp_path, ["run_1", "run_2"])
    assert stats["xyz_min"] == [1.0, 1.0, 1.0]
    assert stats["y_mean"] == [2.0, 3.0, 3.0, 3.0]
    assert stats["y_std"] == [0.0, 0.0, 0.0, 0.0]


def test_compute_train_stats_float64_moments_match_reference(tmp_path):
    """Regression: mean/std must match float64 population moments, not float32 concat."""
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    prefix = "boundary_1"
    p = np.array([-230.0, -200.0, -260.0, -220.0], dtype=np.float32)
    tau = np.array(
        [
            [-1.2, 0.0, -0.07],
            [-1.0, 0.1, -0.05],
            [-1.4, -0.1, -0.09],
            [-1.1, 0.05, -0.06],
        ],
        dtype=np.float32,
    )
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(run_dir / f"{prefix}_points.npy", xyz)
    np.save(run_dir / f"{prefix}_normals.npy", np.ones((4, 3), dtype=np.float32))
    np.save(run_dir / f"{prefix}_p.npy", p)
    np.save(run_dir / f"{prefix}_tau.npy", tau)

    y64 = np.concatenate([p.astype(np.float64).reshape(-1, 1), tau.astype(np.float64)], axis=1)
    expected_mean = y64.mean(axis=0)
    expected_std = y64.std(axis=0)

    # Force multiple chunks so the incremental path is exercised.
    stats = compute_train_stats(tmp_path, ["run_1"], chunk_cells=2)
    np.testing.assert_allclose(stats["y_mean"], expected_mean, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(stats["y_std"], expected_std, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(stats["xyz_min"], xyz.min(axis=0).astype(np.float64))
    np.testing.assert_allclose(stats["xyz_max"], xyz.max(axis=0).astype(np.float64))


def test_compute_train_stats_chunked_matches_unchunked(tmp_path):
    rng = np.random.default_rng(0)
    n = 5_000
    run_dir = tmp_path / "run_7"
    run_dir.mkdir()
    prefix = "boundary_7"
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    p = rng.normal(loc=-230.0, scale=270.0, size=n).astype(np.float32)
    tau = rng.normal(size=(n, 3)).astype(np.float32)
    np.save(run_dir / f"{prefix}_points.npy", xyz)
    np.save(run_dir / f"{prefix}_normals.npy", np.ones((n, 3), dtype=np.float32))
    np.save(run_dir / f"{prefix}_p.npy", p)
    np.save(run_dir / f"{prefix}_tau.npy", tau)

    full = compute_train_stats(tmp_path, ["run_7"], chunk_cells=n)
    chunked = compute_train_stats(tmp_path, ["run_7"], chunk_cells=1_024)
    np.testing.assert_allclose(chunked["y_mean"], full["y_mean"], rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(chunked["y_std"], full["y_std"], rtol=1e-14, atol=1e-11)
    np.testing.assert_allclose(chunked["xyz_min"], full["xyz_min"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(chunked["xyz_max"], full["xyz_max"], rtol=0.0, atol=0.0)

def test_finalize_deletes_vtp_when_enabled(tmp_path):
    vtp = tmp_path / "boundary_1.vtp"
    vtp.write_bytes(b"fake")
    out_dir = tmp_path / "run_1"
    out_dir.mkdir()
    for field in ("points", "normals", "p", "tau"):
        np.save(out_dir / f"boundary_1_{field}.npy", np.zeros(1))
    finalize_run_outputs(vtp_path=vtp, out_dir=out_dir, prefix="boundary_1", delete_vtp=True)
    assert not vtp.exists()


def test_finalize_keeps_vtp_when_disabled(tmp_path):
    vtp = tmp_path / "boundary_1.vtp"
    vtp.write_bytes(b"fake")
    out_dir = tmp_path / "run_1"
    out_dir.mkdir()
    for field in ("points", "normals", "p", "tau"):
        np.save(out_dir / f"boundary_1_{field}.npy", np.zeros(1))
    finalize_run_outputs(vtp_path=vtp, out_dir=out_dir, prefix="boundary_1", delete_vtp=False)
    assert vtp.exists()


def test_prep_resumes_after_completed_run_vtp_was_deleted(tmp_path, monkeypatch):
    data_root = tmp_path / "raw"
    out_root = tmp_path / "surface_full"
    completed_raw = data_root / "run_1"
    completed_raw.mkdir(parents=True)
    _write_full_run(out_root, "run_1")

    pending_raw = data_root / "run_2"
    pending_raw.mkdir()
    pending_vtp = pending_raw / "boundary_2.vtp"
    pending_vtp.write_bytes(b"fake")

    processed = []
    monkeypatch.setattr(prep_module, "probe_field_names", lambda path: (prep_module.ARRAY_P, prep_module.ARRAY_TAU))

    def fake_worker(args):
        vtp_path, _, _, _, _ = args
        processed.append(vtp_path)
        return "run_2", 4, "boundary_2"

    monkeypatch.setattr(prep_module, "_prep_vtp_worker", fake_worker)

    manifest = prep_module.prep_drivaerml_surface(data_root, out_root, workers=1)

    assert processed == [pending_vtp]
    assert set(manifest["runs"]) == {"run_1", "run_2"}

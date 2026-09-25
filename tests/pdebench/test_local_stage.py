import json
from pathlib import Path

import pytest

from pdebench.dataset import local_stage


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fixed_source(root: Path, *, k: int = 64) -> Path:
    _write(root / "PeriodUnitCell_fixed" / "manifest.json", b"{}")
    _write(
        root
        / "PeriodUnitCell_fixed"
        / "static_cache"
        / "graph_lmdb"
        / "cache"
        / "train"
        / "shards"
        / "shard_000000"
        / "data.mdb"
    )
    _write(
        root
        / "PeriodUnitCell_fixed"
        / "static_cache"
        / "laplacian_lmdb"
        / "graph64"
        / f"K{k}"
        / "cache"
        / "train"
        / "shards"
        / "shard_000000"
        / "data.mdb"
    )
    _write(root / "PeriodUnitCell" / "sample_ids.npy")
    return root


def test_stage_fixed_dataset_and_reuse_valid_marker(tmp_path, monkeypatch):
    source = _fixed_source(tmp_path / "source")
    destination = tmp_path / "local-data"
    calls = 0
    real_copy = local_stage.copy_tree

    def counted_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(local_stage, "copy_tree", counted_copy)

    assert local_stage.stage_dataset("micro_puc_fixed", source, destination, laplacian_k=64) == destination
    assert (destination / "PeriodUnitCell_fixed" / "manifest.json").is_file()
    assert (destination / "PeriodUnitCell" / "sample_ids.npy").is_file()
    marker = json.loads((destination / ".pdebench-stage.json").read_text())
    assert marker["dataset"] == "micro_puc_fixed"
    assert marker["laplacian_k"] == 64
    first_calls = calls

    assert local_stage.stage_dataset("micro_puc_fixed", source, destination, laplacian_k=64) == destination
    assert calls == first_calls


def test_stage_rejects_missing_requested_laplacian_cache(tmp_path):
    source = _fixed_source(tmp_path / "source", k=32)

    with pytest.raises(FileNotFoundError, match="K64 Laplacian"):
        local_stage.stage_dataset("micro_puc_fixed", source, tmp_path / "local-data", laplacian_k=64)


def test_stage_rejects_missing_fixed_geometry_mapping(tmp_path):
    source = _fixed_source(tmp_path / "source")
    (source / "PeriodUnitCell" / "sample_ids.npy").unlink()

    with pytest.raises(FileNotFoundError, match="sample_ids.npy"):
        local_stage.stage_dataset("micro_puc_fixed", source, tmp_path / "local-data", laplacian_k=64)


def test_marker_for_different_k_is_not_reused(tmp_path, monkeypatch):
    source = _fixed_source(tmp_path / "source", k=32)
    _fixed_source(source, k=64)
    destination = tmp_path / "local-data"
    local_stage.stage_dataset("micro_puc_fixed", source, destination, laplacian_k=32)
    calls = 0
    real_copy = local_stage.copy_tree

    def counted_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(local_stage, "copy_tree", counted_copy)
    local_stage.stage_dataset("micro_puc_fixed", source, destination, laplacian_k=64)
    assert calls > 0
    assert json.loads((destination / ".pdebench-stage.json").read_text())["laplacian_k"] == 64

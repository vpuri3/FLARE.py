from __future__ import annotations

from pathlib import Path

from scripts.download_pdebench_dataset import (
    MESHGRAPHNETS_GCS_SPECS,
    download_meshgraphnets_gcs_dataset,
    verify_meshgraphnets_gcs_dataset,
)


def test_verify_meshgraphnets_gcs_dataset_flags_missing_and_wrong_size(tmp_path: Path) -> None:
    spec = MESHGRAPHNETS_GCS_SPECS["deforming_plate"]
    dst = tmp_path / spec.local_subdir
    dst.mkdir(parents=True)
    (dst / "meta.json").write_bytes(b"x" * spec.expected_files["meta.json"])
    (dst / "valid.tfrecord").write_bytes(b"short")

    missing = verify_meshgraphnets_gcs_dataset(dst, spec)
    assert "train.tfrecord" in missing
    assert any("valid.tfrecord" in item for item in missing)
    assert "meta.json" not in missing


def test_download_meshgraphnets_gcs_dataset_skips_when_present(tmp_path: Path, monkeypatch) -> None:
    spec = MESHGRAPHNETS_GCS_SPECS["deforming_plate"]
    dst = tmp_path / spec.local_subdir
    dst.mkdir(parents=True)
    for filename, size in spec.expected_files.items():
        (dst / filename).write_bytes(b"\0" * size)

    called = {"n": 0}

    def _fail_download(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("should not download when files already match expected sizes")

    monkeypatch.setattr(
        "scripts.download_pdebench_dataset._download_meshgraphnets_gcs_file",
        _fail_download,
    )
    download_meshgraphnets_gcs_dataset(tmp_path, spec)
    assert called["n"] == 0

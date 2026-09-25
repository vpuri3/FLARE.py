from __future__ import annotations

from scripts.download_drivaerml_surface import ALLOW_PATTERNS, DEFAULT_MAX_WORKERS, download_drivaerml_surface


def test_allow_patterns_boundary_only():
    assert ALLOW_PATTERNS == ["run_*/boundary_*.vtp"]


def test_default_max_workers_at_least_16():
    assert DEFAULT_MAX_WORKERS >= 16


def test_download_calls_snapshot_download(tmp_path, monkeypatch):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        "scripts.download_drivaerml_surface.snapshot_download",
        fake_snapshot_download,
    )
    download_drivaerml_surface(tmp_path, max_workers=32)
    assert calls["repo_id"] == "neashton/drivaerml"
    assert calls["repo_type"] == "dataset"
    assert calls["allow_patterns"] == ["run_*/boundary_*.vtp"]
    assert calls["max_workers"] == 32
    assert calls["local_dir"] == str(tmp_path.resolve())

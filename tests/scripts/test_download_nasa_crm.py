from __future__ import annotations

import subprocess
import sys
from email.message import Message
from pathlib import Path

import pytest

from scripts.download_nasa_crm import download_drive_file


def test_download_drive_file_uses_confirm_url(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "nested" / "sample.h5"
    partial = destination.with_suffix(destination.suffix + ".part")
    calls: list[tuple[str, Path]] = []

    def fake_urlretrieve(url: str, dest: Path):
        calls.append((url, dest))
        dest.write_bytes(b"downloaded")
        headers = Message()
        headers["Content-Length"] = str(len(b"downloaded"))
        return str(dest), headers

    monkeypatch.setattr("scripts.download_nasa_crm.request.urlretrieve", fake_urlretrieve)

    download_drive_file("drive-id", destination)

    assert calls == [
        (
            "https://drive.usercontent.google.com/download?id=drive-id&export=download&confirm=t",
            partial,
        )
    ]
    assert destination.read_bytes() == b"downloaded"
    assert not partial.exists()


def test_download_drive_file_skips_existing_files_larger_than_one_mib(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "sample.h5"
    destination.write_bytes(b"\0" * (1024 * 1024 + 1))

    def fail_urlretrieve(*_args, **_kwargs) -> None:
        raise AssertionError("existing file should be skipped")

    monkeypatch.setattr("scripts.download_nasa_crm.request.urlretrieve", fail_urlretrieve)

    download_drive_file("drive-id", destination)


def test_download_drive_file_removes_stale_part_before_download(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "sample.h5"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(b"stale")

    def fake_urlretrieve(_url: str, dest: Path):
        assert not dest.exists()
        dest.write_bytes(b"fresh")
        return str(dest), Message()

    monkeypatch.setattr("scripts.download_nasa_crm.request.urlretrieve", fake_urlretrieve)

    download_drive_file("drive-id", destination)

    assert destination.read_bytes() == b"fresh"


def test_download_drive_file_does_not_replace_final_on_failure(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "sample.h5"
    destination.write_bytes(b"existing")

    def fake_urlretrieve(_url: str, dest: Path) -> None:
        dest.write_bytes(b"partial")
        raise OSError("download interrupted")

    monkeypatch.setattr("scripts.download_nasa_crm.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(OSError, match="interrupted"):
        download_drive_file("drive-id", destination)

    assert destination.read_bytes() == b"existing"


def test_download_drive_file_rejects_content_length_mismatch(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "sample.h5"

    def fake_urlretrieve(_url: str, dest: Path):
        dest.write_bytes(b"short")
        headers = Message()
        headers["Content-Length"] = "100"
        return str(dest), headers

    monkeypatch.setattr("scripts.download_nasa_crm.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(ValueError, match="Content-Length"):
        download_drive_file("drive-id", destination)

    assert not destination.exists()


def test_download_script_imports_as_cli() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "download_nasa_crm.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NASA-CRM" in result.stdout

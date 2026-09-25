#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib import request

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pdebench.dataset.nasa_crm import assert_official_split  # noqa: E402

DRIVE_FILES = {
    "trainingData_NASA-CRM.h5": "1pWaHFoGuLuxo1i3TFXX1Iz7OdpJhtQIY",
    "testData_NASA-CRM.h5": "1uYlu-l5w-2q7pcoCm0hmuTUGO_bbZvWQ",
    "connectivity_NASA-CRM.h5": "13z1bQM08lhbqE8AHU1oAehyqGwWuGGY0",
    "h5Import_NASA-CRM.py": "1FX_Us474hpPv0aj8l1Q_KzDxA0FrlkQr",
    "README.md": "1ip879izK6EJcs6IPbJHKy0yur9WCcosD",
}
MIN_EXISTING_SIZE = 1024 * 1024


def download_drive_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > MIN_EXISTING_SIZE:
        print(f"Skipping {dest.name}: existing file is larger than 1 MiB")
        return

    partial = dest.with_suffix(dest.suffix + ".part")
    partial.unlink(missing_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    print(f"Downloading {dest.name}")
    _, headers = request.urlretrieve(url, partial)
    content_length = headers.get("Content-Length")
    if content_length is not None:
        expected_size = int(content_length)
        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Content-Length mismatch for {dest.name}: expected {expected_size} bytes, got {actual_size}"
            )
    partial.replace(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NASA-CRM AASM Case 4 HDF5 dataset from Google Drive.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/NASA_CRM"),
        help="Destination directory (default: data/NASA_CRM).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    for filename, file_id in DRIVE_FILES.items():
        download_drive_file(file_id, data_root / filename)

    subprocess.run(["ls", "-lh", str(data_root)], check=True)
    assert_official_split(data_root)


if __name__ == "__main__":
    main()

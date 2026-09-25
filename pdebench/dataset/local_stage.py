"""Safely stage Micro-PUC datasets from shared storage onto node-local storage."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

from scripts.fast_cp import copy_tree

_MARKER = ".pdebench-stage.json"


def _required_sources(dataset: str, source_root: Path) -> list[tuple[Path, Path]]:
    if dataset == "micro_puc":
        return [(source_root / "PeriodUnitCell", Path("PeriodUnitCell"))]
    if dataset == "micro_puc_fixed":
        return [
            (source_root / "PeriodUnitCell_fixed", Path("PeriodUnitCell_fixed")),
            (source_root / "PeriodUnitCell" / "sample_ids.npy", Path("PeriodUnitCell/sample_ids.npy")),
        ]
    raise ValueError(f"Local staging is unsupported for dataset={dataset!r}")


def _validate(dataset: str, root: Path, laplacian_k: int) -> None:
    if dataset == "micro_puc_fixed":
        required = [
            (root / "PeriodUnitCell_fixed" / "manifest.json", "fixed dataset manifest"),
            (root / "PeriodUnitCell" / "sample_ids.npy", "sample_ids.npy geometry mapping"),
        ]
        dataset_dir = root / "PeriodUnitCell_fixed"
    else:
        required = [(root / "PeriodUnitCell" / "sample_ids.npy", "sample_ids.npy")]
        dataset_dir = root / "PeriodUnitCell"
    for path, label in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required {label}: {path}")
    graph_files = list((dataset_dir / "static_cache" / "graph_lmdb").glob("**/data.mdb"))
    if not graph_files:
        raise FileNotFoundError(f"Missing graph LMDB cache under {dataset_dir}")
    if laplacian_k > 0:
        laplacian_root = dataset_dir / "static_cache" / "laplacian_lmdb"
        laplacian_files = list(laplacian_root.glob(f"**/K{laplacian_k}/**/data.mdb"))
        if not laplacian_files:
            raise FileNotFoundError(f"Missing requested K{laplacian_k} Laplacian LMDB cache under {laplacian_root}")


def _marker(dataset: str, source_root: Path, laplacian_k: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source_root": str(source_root.resolve()),
        "laplacian_k": int(laplacian_k),
    }


def _marker_matches(destination_root: Path, expected: dict[str, object]) -> bool:
    try:
        actual = json.loads((destination_root / _MARKER).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return actual == expected


def stage_dataset(dataset: str, source_root: Path, destination_root: Path, laplacian_k: int = 0) -> Path:
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    sources = _required_sources(dataset, source_root)
    _validate(dataset, source_root, laplacian_k)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    expected_marker = _marker(dataset, source_root, laplacian_k)
    lock_path = destination_root.parent / f".{destination_root.name}.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _marker_matches(destination_root, expected_marker):
            _validate(dataset, destination_root, laplacian_k)
            print(f"[local_stage] reuse dataset={dataset} root={destination_root}")
            return destination_root

        temporary = destination_root.parent / f".{destination_root.name}.tmp-{os.getpid()}"
        stale = destination_root.parent / f".{destination_root.name}.stale-{os.getpid()}"
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            for source, relative_destination in sources:
                destination = temporary / relative_destination
                if source.is_dir():
                    copied, skipped, failed = copy_tree(source, destination)
                    if failed:
                        raise OSError(f"Failed to copy {failed} files from {source}")
                    print(f"[local_stage] copied={copied} skipped={skipped} source={source}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            _validate(dataset, temporary, laplacian_k)
            (temporary / _MARKER).write_text(json.dumps(expected_marker, indent=2, sort_keys=True) + "\n")
            if destination_root.exists():
                os.replace(destination_root, stale)
            os.replace(temporary, destination_root)
            shutil.rmtree(stale, ignore_errors=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if stale.exists() and not destination_root.exists():
                os.replace(stale, destination_root)
            raise
    print(f"[local_stage] ready dataset={dataset} root={destination_root}")
    return destination_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("micro_puc", "micro_puc_fixed"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--laplacian-k", type=int, default=0)
    args = parser.parse_args()
    print(stage_dataset(args.dataset, args.source_root, args.destination_root, args.laplacian_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare DrivAerML surface VTPs as full-mesh NPY arrays."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

ARRAY_P = "pMeanTrim"
ARRAY_TAU = "wallShearStressMeanTrim"


def probe_field_names(vtp_path: Path) -> tuple[str, str]:
    mesh = pv.read(vtp_path)
    for name in (ARRAY_P, ARRAY_TAU):
        if name not in mesh.cell_data:
            raise KeyError(f"Missing {name} in {vtp_path}")
    return ARRAY_P, ARRAY_TAU


def discover_runs(data_root: Path, out_root: Path) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    found_run_dir = False
    for run_dir in sorted(data_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        found_run_dir = True
        if _train_run_prefix(out_root, run_dir.name) is not None:
            continue
        vtps = sorted(run_dir.glob("boundary_*.vtp"))
        if len(vtps) != 1:
            raise ValueError(f"expected exactly one boundary_*.vtp in {run_dir}, found {len(vtps)}")
        runs.append((run_dir.name, vtps[0]))
    if not found_run_dir:
        raise FileNotFoundError(f"No run_* directories under {data_root}")
    return runs


def finalize_run_outputs(*, vtp_path: Path, out_dir: Path, prefix: str, delete_vtp: bool) -> None:
    for field in ("points", "normals", "p", "tau"):
        if not (out_dir / f"{prefix}_{field}.npy").is_file():
            raise FileNotFoundError(f"missing {prefix}_{field}.npy under {out_dir}")
    if delete_vtp and vtp_path.is_file():
        vtp_path.unlink()


def _prep_vtp_worker(args: tuple[Path, Path, str, str, bool]) -> tuple[str, int, str]:
    vtp_path, out_dir, array_p, array_tau, delete_vtp = args
    mesh = pv.read(vtp_path)
    coords = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32)
    pressure = np.asarray(mesh.cell_data[array_p], dtype=np.float32).reshape(-1)
    tau = np.asarray(mesh.cell_data[array_tau], dtype=np.float32).reshape(-1, 3)
    n = coords.shape[0]

    prefix = vtp_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_points.npy", coords)
    np.save(out_dir / f"{prefix}_normals.npy", normals)
    np.save(out_dir / f"{prefix}_p.npy", pressure)
    np.save(out_dir / f"{prefix}_tau.npy", tau)
    finalize_run_outputs(
        vtp_path=vtp_path,
        out_dir=out_dir,
        prefix=prefix,
        delete_vtp=delete_vtp,
    )
    return out_dir.name, n, prefix


def _train_run_prefix(out_root: Path, run_name: str) -> str | None:
    run_dir = out_root / run_name
    if not run_dir.is_dir():
        return None
    points_files = sorted(run_dir.glob("*_points.npy"))
    if not points_files:
        return None
    prefix = points_files[0].name.removesuffix("_points.npy")
    for field in ("points", "normals", "p", "tau"):
        if not (run_dir / f"{prefix}_{field}.npy").is_file():
            return None
    return prefix


# Chunk size for float64 moment accumulation (mmap slices; avoids loading whole runs).
_STATS_CHUNK_CELLS = 1_000_000


def compute_train_stats(
    out_root: Path,
    train_runs: list[str],
    *,
    chunk_cells: int = _STATS_CHUNK_CELLS,
) -> dict[str, list[float]]:
    missing: list[str] = []
    run_prefixes: list[tuple[str, str]] = []

    for run_name in train_runs:
        prefix = _train_run_prefix(out_root, run_name)
        if prefix is None:
            missing.append(run_name)
        else:
            run_prefixes.append((run_name, prefix))

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Cannot compute train stats: {len(missing)}/{len(train_runs)} train runs "
            f"missing or incomplete under {out_root}: {preview}{suffix}"
        )

    if chunk_cells <= 0:
        raise ValueError(f"chunk_cells must be > 0, got {chunk_cells}")

    # Incremental float64 moments over mmap chunks. A full float32 concat+mean over
    # ~3e9 cells saturates and historically baked Y_MEAN≈-10 instead of ≈-230.
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    y_sum = np.zeros(4, dtype=np.float64)
    y_sum_sq = np.zeros(4, dtype=np.float64)
    y_count = 0
    for run_name, prefix in run_prefixes:
        run_dir = out_root / run_name
        xyz = np.load(run_dir / f"{prefix}_points.npy", mmap_mode="r")
        p = np.load(run_dir / f"{prefix}_p.npy", mmap_mode="r")
        tau = np.load(run_dir / f"{prefix}_tau.npy", mmap_mode="r")
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"expected points shape (N, 3) for {run_name}, got {xyz.shape}")
        if tau.ndim != 2 or tau.shape[1] != 3:
            raise ValueError(f"expected tau shape (N, 3) for {run_name}, got {tau.shape}")
        n = int(xyz.shape[0])
        if int(np.prod(p.shape)) != n or int(tau.shape[0]) != n:
            raise ValueError(f"mismatched field lengths for {run_name}")
        for start in range(0, n, chunk_cells):
            end = min(n, start + chunk_cells)
            xyz64 = np.asarray(xyz[start:end], dtype=np.float64)
            p64 = np.asarray(p[start:end], dtype=np.float64).reshape(-1)
            tau64 = np.asarray(tau[start:end], dtype=np.float64)
            y64 = np.empty((end - start, 4), dtype=np.float64)
            y64[:, 0] = p64
            y64[:, 1:] = tau64
            xyz_min = np.minimum(xyz_min, xyz64.min(axis=0))
            xyz_max = np.maximum(xyz_max, xyz64.max(axis=0))
            y_sum += y64.sum(axis=0)
            y_sum_sq += np.square(y64).sum(axis=0)
            y_count += end - start

    if y_count == 0:
        raise ValueError(f"no train cells under {out_root}")
    y_mean = y_sum / y_count
    y_var = np.maximum(y_sum_sq / y_count - np.square(y_mean), 0.0)
    y_std = np.sqrt(y_var)
    return {
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }


def prep_drivaerml_surface(
    data_root: Path,
    out_root: Path,
    *,
    split_json: Path | None = None,
    workers: int = 8,
    max_runs: int | None = None,
    delete_vtp: bool = True,
) -> dict:
    data_root = data_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(data_root, out_root)
    if max_runs is not None:
        runs = runs[:max_runs]

    if runs:
        array_p, array_tau = probe_field_names(runs[0][1])
    else:
        array_p, array_tau = ARRAY_P, ARRAY_TAU
    print(f"ARRAY_P = {array_p!r}")
    print(f"ARRAY_TAU = {array_tau!r}")

    per_run: dict[str, dict[str, int | str]] = {}
    for run_dir in sorted(data_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        prefix = _train_run_prefix(out_root, run_dir.name)
        if prefix is not None:
            points = np.load(out_root / run_dir.name / f"{prefix}_points.npy", mmap_mode="r")
            per_run[run_dir.name] = {"N": points.shape[0], "prefix": prefix}
    worker_args = [
        (vtp_path, out_root / run_name, array_p, array_tau, delete_vtp)
        for run_name, vtp_path in runs
    ]
    if workers <= 1:
        results = [_prep_vtp_worker(args) for args in worker_args]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_prep_vtp_worker, args) for args in worker_args]
            for future in as_completed(futures):
                results.append(future.result())

    for run_name, n, prefix in results:
        per_run[run_name] = {"N": n, "prefix": prefix}

    manifest: dict = {
        "layout": "full_mesh",
        "array_p": array_p,
        "array_tau": array_tau,
        "runs": per_run,
        "delete_vtp": delete_vtp,
    }
    if split_json is not None:
        split = json.loads(split_json.read_text())
        try:
            stats = compute_train_stats(out_root, split["train"])
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        manifest["stats"] = stats
        print(f"XYZ_MIN = {stats['xyz_min']}")
        print(f"XYZ_MAX = {stats['xyz_max']}")
        print(f"Y_MEAN = {stats['y_mean']}")
        print(f"Y_STD = {stats['y_std']}")

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path} ({len(per_run)} runs)")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DrivAerML surface VTPs as full-mesh NPY arrays.")
    parser.add_argument("--data-root", type=Path, default=Path("data/DrivAerML/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data/DrivAerML/surface_full"))
    parser.add_argument("--split-json", type=Path, default=Path("pdebench/dataset/splits/drivaerml.json"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-runs", type=int, default=None, help="Process at most N runs (smoke/debug).")
    parser.add_argument(
        "--delete-vtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete each source VTP after all four NPY outputs are verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prep_drivaerml_surface(
        args.data_root,
        args.out_root,
        split_json=args.split_json,
        workers=args.workers,
        max_runs=args.max_runs,
        delete_vtp=args.delete_vtp,
    )


if __name__ == "__main__":
    main()

"""Parallel laplacian_pt precompute and K64 co-sharded pack for elpl_v3."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from pdebench.dataset.plaid_core import load_plaid_readme_meta, split_labeled_train_test
from pdebench.dataset.plaid_datasets import PLAID_SPECS
from pdebench.dataset.plaid_elpl_v3.constants import CACHE_SCHEMA_VERSION
from pdebench.dataset.plaid_elpl_v3.laplacian import load_laplacian_bundle
from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload, save_shard_payload
from pdebench.dataset.plaid_elpl_v3.paths import shard_dir, sim_to_shard_path
from pdebench.dataset.plaid_elpl_v3.schema import LaplacianBundle, ShardPayload
from pdebench.dataset.plaid_laplacian import _ensure_plaid_laplacian, _load_plaid_laplacian
from pdebench.dataset.thread_limits import set_compute_thread_limits


def _laplacian_workers(num_tasks: int) -> int:
    cap = int(os.environ.get("PLAID_ELPL_LAPLACIAN_WORKERS", str(os.cpu_count() or 1)))
    return max(1, min(cap, os.cpu_count() or 1, int(num_tasks)))


def _num_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    visible = int(torch.cuda.device_count())
    if visible <= 0:
        return 0
    cap = int(os.environ.get("PLAID_ELPL_LAPLACIAN_NUM_GPUS", str(visible)))
    return max(1, min(cap, visible))


def _labeled_sim_ids(dataset_dir: str, split_seed: int) -> list[int]:
    spec = PLAID_SPECS["plaid_el_pl_dynamics"]
    meta = load_plaid_readme_meta(dataset_dir)
    labeled_ids = [int(i) for i in meta.split_map.get(spec.labeled_split, [])]
    train_ids, val_ids = split_labeled_train_test(labeled_ids, test_ratio=0.2, seed=split_seed)
    return sorted(set(train_ids) | set(val_ids))


def compute_laplacian_for_trajectory(
    traj: Any,
    *,
    dataset_dir: str | Path,
    bandwidth: float,
    laplacian_eig_dim: int,
    laplacian_spec: str,
    device: str | torch.device | None = None,
) -> LaplacianBundle:
    traj.ensure_edges(bandwidth=float(bandwidth))
    assert traj.edge_index is not None
    prev = os.environ.get("PLAID_LAPLACIAN_DEVICE")
    if device is not None:
        os.environ["PLAID_LAPLACIAN_DEVICE"] = str(device)
    try:
        _ensure_plaid_laplacian(
            dataset_dir=str(dataset_dir),
            dataset_name="plaid_el_pl_dynamics",
            sample_idx=int(traj.sim_id),
            edge_index=traj.edge_index,
            pos=traj.pos,
            cells=traj.cells,
            num_eigenvectors=int(laplacian_eig_dim),
            laplacian_spec=str(laplacian_spec),
        )
        eigenvalues, eigenvectors = _load_plaid_laplacian(
            dataset_dir=str(dataset_dir),
            dataset_name="plaid_el_pl_dynamics",
            sample_idx=int(traj.sim_id),
            num_eigenvectors=int(laplacian_eig_dim),
            laplacian_spec=str(laplacian_spec),
        )
    finally:
        if prev is None:
            os.environ.pop("PLAID_LAPLACIAN_DEVICE", None)
        else:
            os.environ["PLAID_LAPLACIAN_DEVICE"] = prev
    if eigenvectors is None:
        raise RuntimeError(f"Laplacian compute returned None for sim {traj.sim_id}.")
    return LaplacianBundle(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def _gpu_worker_entry(args: tuple[int, list[int], str, int, str, float, str, int]) -> tuple[int, int]:
    gpu_id, sim_ids, dataset_dir, laplacian_eig_dim, laplacian_spec, bandwidth, k0_shard_root, split_seed = args
    set_compute_thread_limits()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = f"cuda:{int(gpu_id)}"
    else:
        device = "cpu"

    sim_to_shard = pd.read_parquet(sim_to_shard_path(dataset_dir, split_seed=split_seed)).set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    built = 0
    for sim_id in sim_ids:
        row = sim_to_shard.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(Path(k0_shard_root) / f"shard_{shard_id:04d}.pt"))
        traj = shard_cache[shard_id].trajectories[local_idx]
        compute_laplacian_for_trajectory(
            traj,
            dataset_dir=dataset_dir,
            bandwidth=bandwidth,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            device=device,
        )
        built += 1
    return int(gpu_id), int(built)


def precompute_laplacian_pt_parallel(
    *,
    dataset_dir: str | Path,
    split_seed: int = 5,
    laplacian_eig_dim: int = 64,
    laplacian_spec: str = "graph",
    bandwidth: float | None = None,
    sim_ids: list[int] | None = None,
) -> int:
    dataset_dir = Path(dataset_dir)
    spec = PLAID_SPECS["plaid_el_pl_dynamics"]
    bandwidth = float(spec.bandwidth if bandwidth is None else bandwidth)
    sim_ids = sim_ids or _labeled_sim_ids(str(dataset_dir), split_seed)
    k0_root = shard_dir(dataset_dir, split_seed=split_seed, laplacian_eig_dim=0, laplacian_spec=laplacian_spec)
    if not k0_root.is_dir():
        raise FileNotFoundError(f"Missing K0 elpl_v3 shards at {k0_root}. Run trajectory precompute first.")

    num_gpus = _num_gpus()
    if num_gpus <= 0:
        chunks = [sim_ids]
        gpu_ids = [-1]
    else:
        gpu_ids = list(range(num_gpus))
        chunk = max(1, (len(sim_ids) + num_gpus - 1) // num_gpus)
        chunks = [sim_ids[i : i + chunk] for i in range(0, len(sim_ids), chunk)]

    tasks = [
        (
            gpu_ids[i % len(gpu_ids)],
            chunk,
            str(dataset_dir),
            int(laplacian_eig_dim),
            str(laplacian_spec),
            bandwidth,
            str(k0_root),
            int(split_seed),
        )
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    print(
        f"Precomputing laplacian_pt for {len(sim_ids)} sims "
        f"(K={laplacian_eig_dim} spec={laplacian_spec}) on {max(1, num_gpus)} GPU worker(s)."
    )
    ctx = mp.get_context("spawn")
    built = 0
    with ctx.Pool(processes=len(tasks)) as pool:
        for _gpu, count in tqdm(
            pool.imap_unordered(_gpu_worker_entry, tasks),
            total=len(tasks),
            desc="laplacian_pt GPUs",
            ncols=90,
        ):
            built += int(count)
    return int(built)


def _pack_shard_worker(args: tuple[str, str, str, int, str]) -> tuple[int, int]:
    k0_path, out_path, dataset_dir, laplacian_eig_dim, laplacian_spec = args
    if Path(out_path).is_file():
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        return int(payload["shard_id"]), len(payload.get("sim_ids", []))

    k0 = load_shard_payload(k0_path)
    laplacian = [
        load_laplacian_bundle(
            dataset_dir=dataset_dir,
            sim_id=int(traj.sim_id),
            laplacian_eig_dim=int(laplacian_eig_dim),
            laplacian_spec=str(laplacian_spec),
        )
        for traj in k0.trajectories
    ]
    if any(item is None for item in laplacian):
        missing = [int(k0.sim_ids[i]) for i, item in enumerate(laplacian) if item is None]
        raise FileNotFoundError(f"Missing laplacian_pt entries for sims: {missing[:8]} ...")

    shard = ShardPayload(
        schema_version=CACHE_SCHEMA_VERSION,
        shard_id=int(k0.shard_id),
        sim_ids=list(k0.sim_ids),
        trajectories=list(k0.trajectories),
        laplacian=laplacian,
    )
    save_shard_payload(out_path, shard)
    return int(k0.shard_id), len(k0.sim_ids)


def pack_co_sharded_laplacian_from_k0(
    *,
    dataset_dir: str | Path,
    split_seed: int = 5,
    laplacian_eig_dim: int = 64,
    laplacian_spec: str = "graph",
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    k0_root = shard_dir(dataset_dir, split_seed=split_seed, laplacian_eig_dim=0, laplacian_spec=laplacian_spec)
    out_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(k0_path),
            str(out_root / k0_path.name),
            str(dataset_dir),
            int(laplacian_eig_dim),
            str(laplacian_spec),
        )
        for k0_path in sorted(k0_root.glob("shard_*.pt"))
    ]

    num_workers = _laplacian_workers(len(tasks))
    print(f"Packing {len(tasks)} co-sharded laplacian shard(s) into {out_root} with {num_workers} worker(s).")
    if num_workers <= 1:
        for task in tqdm(tasks, desc="elpl_v3 K64 pack", ncols=90):
            _pack_shard_worker(task)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(num_workers, len(tasks))) as pool:
            for _sid, _n in tqdm(
                pool.imap_unordered(_pack_shard_worker, tasks),
                total=len(tasks),
                desc="elpl_v3 K64 pack",
                ncols=90,
            ):
                del _sid, _n
    return [out_root / Path(task[0]).name for task in tasks]


def build_elpl_v3_laplacian_cache(
    *,
    dataset_dir: str | Path,
    split_seed: int = 5,
    laplacian_eig_dim: int = 64,
    laplacian_spec: str = "graph",
) -> None:
    dataset_dir = Path(dataset_dir)
    precompute_laplacian_pt_parallel(
        dataset_dir=dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    paths = pack_co_sharded_laplacian_from_k0(
        dataset_dir=dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    total_mb = sum(p.stat().st_size for p in paths) / (1024 * 1024)
    print(f"Wrote {len(paths)} K{laplacian_eig_dim} shards ({total_mb:.1f} MB total).")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Precompute elpl_v3 K64 laplacian_pt + co-sharded pack.")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(os.environ.get("DATA_ROOT", "data")) / "plaid" / "2D_ElastoPlastoDynamics"),
    )
    parser.add_argument("--split-seed", type=int, default=int(os.environ.get("SPLIT_SEED", "5")))
    parser.add_argument("--laplacian-eig-dim", type=int, default=64)
    parser.add_argument("--laplacian-spec", default="graph")
    parser.add_argument("--skip-pt", action="store_true", help="Only pack co-sharded shards from existing laplacian_pt.")
    parser.add_argument("--skip-pack", action="store_true", help="Only precompute laplacian_pt files.")
    args = parser.parse_args()
    if not args.skip_pt:
        count = precompute_laplacian_pt_parallel(
            dataset_dir=args.dataset_dir,
            split_seed=args.split_seed,
            laplacian_eig_dim=args.laplacian_eig_dim,
            laplacian_spec=args.laplacian_spec,
        )
        print(f"laplacian_pt built/verified: {count}")
    if not args.skip_pack:
        paths = pack_co_sharded_laplacian_from_k0(
            dataset_dir=args.dataset_dir,
            split_seed=args.split_seed,
            laplacian_eig_dim=args.laplacian_eig_dim,
            laplacian_spec=args.laplacian_spec,
        )
        total_mb = sum(p.stat().st_size for p in paths) / (1024 * 1024)
        print(f"Wrote {len(paths)} K{args.laplacian_eig_dim} shards ({total_mb:.1f} MB total).")


if __name__ == "__main__":
    main()

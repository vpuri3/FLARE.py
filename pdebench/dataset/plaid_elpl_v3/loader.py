"""End-to-end loader for el-pl v3 cache."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any

import torch

from pdebench.dataset.distributed import distributed_barrier, distributed_rank
from pdebench.dataset.normalizer import NodeFeatureNormalizer
from pdebench.dataset.plaid_core import load_plaid_readme_meta, split_labeled_train_test
from pdebench.dataset.plaid_datasets import PLAID_SPECS, GraphListDataset, _require_mesh_dependencies
from pdebench.dataset.plaid_elpl_v3.build import build_shards
from pdebench.dataset.plaid_elpl_v3.dataset import ElPlTransitionDataset
from pdebench.dataset.plaid_elpl_v3.manifest import (
    build_sim_to_shard,
    build_transition_manifest,
    load_manifest_tables,
    transitions_per_sim,
    write_manifest_tables,
)
from pdebench.dataset.plaid_elpl_v3.norm import (
    ElPlNormStats,
    fit_norm_stats_from_shard_stream,
    load_norm_stats,
    save_norm_stats,
)
from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload
from pdebench.dataset.plaid_elpl_v3.paths import (
    is_cache_complete,
    manifest_dir,
    norm_stats_path,
    shard_dir,
    sim_to_shard_path,
    train_manifest_path,
    val_manifest_path,
    write_meta,
)


def _legacy_x_normalizer(stats: ElPlNormStats, *, use_sdf_features: bool) -> NodeFeatureNormalizer:
    """Monolithic x normalizer matching assembled node feature width."""
    pos_mean = stats.pos_normalizer.mean
    pos_std = stats.pos_normalizer.std
    if use_sdf_features:
        geom_norm = stats.geom_normalizer
        mean = torch.cat([pos_mean, geom_norm.mean, stats.u_normalizer.mean], dim=-1)
        std = torch.cat([pos_std, geom_norm.std, stats.u_normalizer.std], dim=-1)
    else:
        mean = torch.cat([pos_mean, stats.u_normalizer.mean], dim=-1)
        std = torch.cat([pos_std, stats.u_normalizer.std], dim=-1)
    return NodeFeatureNormalizer(mean=mean, std=std)


def _collect_train_trajectories(
    *,
    train_ids: list[int],
    sim_to_shard_df,
    shard_root: Path,
) -> tuple[list[Any], int]:
    from pdebench.dataset.plaid_elpl_v3.schema import TrajectoryBundle

    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    trajectories: list[TrajectoryBundle] = []
    transitions = None
    for sim_id in train_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        traj = shard_cache[shard_id].trajectories[local_idx]
        trajectories.append(traj)
        if transitions is None:
            transitions = transitions_per_sim(traj.timestep_list)
    if transitions is None:
        raise ValueError("No train trajectories found for norm fitting.")
    return trajectories, int(transitions)


def _build_elpl_v3_cache(
    *,
    raw: Any,
    dataset_dir: str,
    split_seed: int,
    train_ids: list[int],
    val_ids: list[int],
    public_test_ids: list[int],
    load_public_test: bool,
    bandwidth: float,
    laplacian_eig_dim: int,
    laplacian_spec: str,
    target_fields: tuple[str, ...],
) -> None:
    all_sim_ids = sorted(set(train_ids) | set(val_ids))
    optional_targets: set[int] = set()
    if load_public_test:
        all_sim_ids = sorted(set(all_sim_ids) | set(public_test_ids))
        optional_targets = set(int(v) for v in public_test_ids)

    build_shards(
        raw,
        all_sim_ids,
        dataset_dir=dataset_dir,
        split_seed=split_seed,
        bandwidth=bandwidth,
        optional_target_sim_ids=optional_targets,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )

    sim_to_shard_df = build_sim_to_shard(all_sim_ids)
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )

    times_by_sim: dict[int, list[float]] = {}
    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    for sim_id in all_sim_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        traj = shard_cache[shard_id].trajectories[local_idx]
        times_by_sim[int(sim_id)] = [float(v) for v in traj.times.tolist()]

    train_df = build_transition_manifest(train_ids, times_by_sim=times_by_sim)
    val_df = build_transition_manifest(val_ids, times_by_sim=times_by_sim)
    manifest_root = manifest_dir(dataset_dir, split_seed=split_seed)
    write_manifest_tables(
        train_df=train_df,
        val_df=val_df,
        sim_to_shard_df=sim_to_shard_df,
        train_path=train_manifest_path(dataset_dir, split_seed=split_seed),
        val_path=val_manifest_path(dataset_dir, split_seed=split_seed),
        sim_to_shard_path=sim_to_shard_path(dataset_dir, split_seed=split_seed),
    )

    train_trajectories, transitions = _collect_train_trajectories(
        train_ids=train_ids,
        sim_to_shard_df=sim_to_shard_df,
        shard_root=shard_root,
    )
    del train_trajectories
    stats = fit_norm_stats_from_shard_stream(
        train_ids=train_ids,
        sim_to_shard_df=sim_to_shard_df,
        shard_root=shard_root,
        transitions_per_sim=transitions,
    )
    save_norm_stats(norm_stats_path(dataset_dir, split_seed=split_seed), stats)
    write_meta(
        dataset_dir,
        split_seed=split_seed,
        bandwidth=bandwidth,
        target_fields=target_fields,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
        num_shards=len(list(shard_root.glob("shard_*.pt"))),
    )
    del manifest_root


def load_plaid_elpl_v3_dataset(
    *,
    data_root: str,
    split_seed: int,
    graph_backend: str,
    use_sdf_features: bool = True,
    max_samples: int = 0,
    load_public_test: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
):
    dataset_name = "plaid_el_pl_dynamics"
    _require_mesh_dependencies(graph_backend=graph_backend)
    spec = PLAID_SPECS[dataset_name]
    dataset_dir = os.path.join(data_root, spec.folder)
    readme_path = os.path.join(dataset_dir, "README.md")
    parquet_paths = sorted(glob(os.path.join(dataset_dir, "data", "all_samples-*.parquet")))
    if not os.path.exists(readme_path) or len(parquet_paths) == 0:
        raise FileNotFoundError(
            f"Could not find local PLAID dataset files under '{dataset_dir}'. "
            "Run scripts/download_dataset.py first."
        )

    meta = load_plaid_readme_meta(dataset_dir)
    from pdebench.dataset.plaid_core import resolve_target_fields

    target_fields = resolve_target_fields(spec.benchmark_fields, out_fields=list(meta.out_fields))
    if tuple(target_fields) != spec.benchmark_fields:
        raise ValueError(
            f"Unexpected target fields for {dataset_name}: {target_fields} (expected {spec.benchmark_fields})."
        )
    target_fields = tuple(target_fields)

    labeled_ids = [int(i) for i in meta.split_map.get(spec.labeled_split, [])]
    public_test_ids = [int(i) for i in meta.split_map.get(spec.test_split, [])]
    if len(labeled_ids) != spec.train_count or len(public_test_ids) != spec.test_count:
        raise ValueError(
            f"Unexpected PLAID split sizes for {dataset_name}: "
            f"train={len(labeled_ids)} (expected {spec.train_count}), "
            f"test={len(public_test_ids)} (expected {spec.test_count})."
        )

    full_split_sizes = dict(labeled=len(labeled_ids), public_test=len(public_test_ids))
    train_ids, val_ids = split_labeled_train_test(labeled_ids, test_ratio=0.2, seed=split_seed)
    if max_samples > 0:
        train_ids = train_ids[:max_samples]
        val_ids = val_ids[:max_samples]
        public_test_ids = public_test_ids[:max_samples]

    expected_shards = max(1, (len(set(train_ids) | set(val_ids)) + 127) // 128)
    del expected_shards
    cache_ready = is_cache_complete(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )

    if not cache_ready:
        if distributed_rank() != 0:
            distributed_barrier()
            cache_ready = is_cache_complete(
                dataset_dir,
                split_seed=split_seed,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
            )
            if not cache_ready:
                raise RuntimeError("PLAID el-pl v3 cache is missing after rank-0 build.")
        else:
            import datasets as hf_datasets

            raw = hf_datasets.load_dataset("parquet", data_files={"all_samples": parquet_paths}, split="all_samples")
            _build_elpl_v3_cache(
                raw=raw,
                dataset_dir=dataset_dir,
                split_seed=split_seed,
                train_ids=train_ids,
                val_ids=val_ids,
                public_test_ids=public_test_ids,
                load_public_test=load_public_test,
                bandwidth=spec.bandwidth,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
                target_fields=target_fields,
            )
            distributed_barrier()

    train_df, val_df, sim_to_shard_df = load_manifest_tables(
        train_path=train_manifest_path(dataset_dir, split_seed=split_seed),
        val_path=val_manifest_path(dataset_dir, split_seed=split_seed),
        sim_to_shard_path=sim_to_shard_path(dataset_dir, split_seed=split_seed),
    )
    stats = load_norm_stats(norm_stats_path(dataset_dir, split_seed=split_seed))
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    dataset_kwargs = dict(
        sim_to_shard=sim_to_shard_df,
        shard_root=shard_root,
        stats=stats,
        use_sdf_features=use_sdf_features,
        graph_backend=graph_backend,
        target_fields=target_fields,
        bandwidth=spec.bandwidth,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    train_data = ElPlTransitionDataset(manifest=train_df, **dataset_kwargs)
    val_data = ElPlTransitionDataset(manifest=val_df, **dataset_kwargs)

    test_data = GraphListDataset([])
    if load_public_test and public_test_ids:
        test_times = _times_by_sim_from_shards(public_test_ids, sim_to_shard_df, shard_root)
        test_df = build_transition_manifest(public_test_ids, times_by_sim=test_times)
        test_data = ElPlTransitionDataset(manifest=test_df, **dataset_kwargs)

    first_graph = train_data[0]
    pos_dim = int(first_graph.pos.shape[-1])
    x_dim = int(first_graph.x.shape[-1])
    node_feat_dim = x_dim - pos_dim
    num_input_scalars = int(first_graph.input_scalars.shape[-1])
    fun_dim = node_feat_dim + num_input_scalars
    c_in = pos_dim + fun_dim
    transitions_per_sim_count = len(train_df) // max(1, len(train_ids))

    metadata = dict(
        x_normalizer=_legacy_x_normalizer(stats, use_sdf_features=use_sdf_features),
        y_normalizer=stats.y_normalizer,
        input_scalar_normalizer=stats.input_scalar_normalizer,
        y_scalar_normalizer=None,
        elpl_norm_stats=stats,
        c_in=c_in,
        c_edge=int(first_graph.edge_attr.shape[-1]),
        c_out=int(first_graph.y.shape[-1]),
        space_dim=pos_dim,
        fun_dim=fun_dim,
        num_input_scalars=num_input_scalars,
        time_cond=True,
        max_length=max(int(train_data[i].x.shape[0]) for i in range(min(32, len(train_data)))),
        target_fields=list(target_fields),
        target_scalar_fields=[],
        plaid_use_sdf_features=bool(use_sdf_features),
        plaid_load_public_test=bool(load_public_test),
        plaid_max_samples=int(max_samples),
        plaid_laplacian_eig_dim=int(laplacian_eig_dim),
        plaid_laplacian_spec=str(laplacian_spec),
        plaid_loss_lbda=1.0,
        plaid_temporal_one_step=True,
        plaid_cache_format="elpl_v3",
        mesh_split_note=(
            f"Vi-Transf parity (elpl_v3): one graph per (t0->t1) transition from '{spec.labeled_split}' "
            f"with sklearn 80/20 seed={split_seed}. "
            f"Each simulation yields {transitions_per_sim_count} transition graphs."
        ),
        mesh_split_sizes=dict(
            train=len(train_data),
            val=len(val_data),
            test=len(test_data),
        ),
        mesh_full_split_sizes=full_split_sizes,
        mesh_train_ids=train_ids,
        mesh_val_ids=val_ids,
        mesh_public_test_ids=public_test_ids,
        mesh_graph_backend=graph_backend,
        mesh_test_data=test_data,
    )
    return train_data, val_data, metadata


def _times_by_sim_from_shards(sim_ids: list[int], sim_to_shard_df, shard_root: Path) -> dict[int, list[float]]:
    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    times_by_sim: dict[int, list[float]] = {}
    for sim_id in sim_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        traj = shard_cache[shard_id].trajectories[local_idx]
        times_by_sim[int(sim_id)] = [float(v) for v in traj.times.tolist()]
    return times_by_sim

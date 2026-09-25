"""End-to-end loader for plaid_elpl_terminal (reuses elpl_v3 trajectory shards)."""

from __future__ import annotations

import os
from glob import glob

import pandas as pd

from pdebench.dataset.distributed import distributed_barrier, distributed_rank
from pdebench.dataset.plaid_core import load_plaid_readme_meta, resolve_target_fields, split_labeled_train_test
from pdebench.dataset.plaid_datasets import PLAID_SPECS, GraphListDataset, _require_mesh_dependencies
from pdebench.dataset.plaid_elpl_terminal.constants import (
    DEFAULT_RUNTIME_Y_NORM,
    DEFAULT_TERMINAL_TARGET_FIELDS,
    parse_terminal_target_fields,
    terminal_target_field_indices,
)
from pdebench.dataset.plaid_elpl_terminal.dataset import ElPlTerminalDataset
from pdebench.dataset.plaid_elpl_terminal.manifest import (
    build_terminal_manifest,
    load_terminal_manifest_tables,
    write_terminal_manifest_tables,
)
from pdebench.dataset.plaid_elpl_terminal.norm import (
    fit_terminal_norm_stats_from_shard_stream,
    load_terminal_norm_stats,
    normalize_runtime_y_norm_mode,
    resolve_runtime_y_normalizer,
    save_terminal_norm_stats,
    slice_y_normalizer,
    x_normalizer_for_metadata,
)
from pdebench.dataset.plaid_elpl_terminal.paths import (
    is_terminal_cache_complete,
    norm_stats_path,
    resolve_sim_to_shard_path,
    train_manifest_path,
    val_manifest_path,
)
from pdebench.dataset.plaid_elpl_v3.loader import load_plaid_elpl_v3_dataset
from pdebench.dataset.plaid_elpl_v3.paths import is_cache_complete as is_v3_cache_complete
from pdebench.dataset.plaid_elpl_v3.paths import shard_dir, sim_to_shard_path


def _build_terminal_cache(
    *,
    dataset_dir: str,
    split_seed: int,
    train_ids: list[int],
    val_ids: list[int],
    use_sdf_features: bool,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> None:
    sim_to_shard_file = sim_to_shard_path(dataset_dir, split_seed=split_seed)
    if not sim_to_shard_file.is_file():
        raise RuntimeError(
            "elpl_v3 sim_to_shard manifest is missing; build elpl_v3 cache before terminal manifests."
        )
    sim_to_shard_df = pd.read_parquet(sim_to_shard_file)
    train_df = build_terminal_manifest(train_ids)
    val_df = build_terminal_manifest(val_ids)
    write_terminal_manifest_tables(
        train_df=train_df,
        val_df=val_df,
        train_path=train_manifest_path(dataset_dir, split_seed=split_seed),
        val_path=val_manifest_path(dataset_dir, split_seed=split_seed),
    )
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    stats = fit_terminal_norm_stats_from_shard_stream(
        train_ids=train_ids,
        sim_to_shard_df=sim_to_shard_df,
        shard_root=shard_root,
    )
    save_terminal_norm_stats(
        norm_stats_path(dataset_dir, split_seed=split_seed, use_sdf_features=use_sdf_features),
        stats,
    )


def _resolve_active_terminal_target_fields(
    *,
    available_fields: tuple[str, ...],
    terminal_target_fields: list[str] | None,
) -> tuple[str, ...]:
    env_raw = os.environ.get("PLAID_TERMINAL_TARGET_FIELDS", "").strip()
    if env_raw:
        raw = env_raw
    elif terminal_target_fields:
        raw = ",".join(terminal_target_fields)
    else:
        raw = None
    active = parse_terminal_target_fields(raw, default=DEFAULT_TERMINAL_TARGET_FIELDS)
    missing = [name for name in active if name not in available_fields]
    if missing:
        raise ValueError(
            f"Terminal target field(s) {missing} not available in dataset outputs {available_fields}."
        )
    return active


def load_plaid_elpl_terminal_dataset(
    *,
    data_root: str,
    split_seed: int,
    graph_backend: str,
    use_sdf_features: bool = False,
    max_samples: int = 0,
    load_public_test: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    y_norm_mode: str | None = None,
    terminal_target_fields: list[str] | None = None,
):
    dataset_name = "plaid_elpl_terminal"
    _require_mesh_dependencies(graph_backend=graph_backend)
    spec = PLAID_SPECS[dataset_name]
    dynamics_spec = PLAID_SPECS["plaid_el_pl_dynamics"]
    dataset_dir = os.path.join(data_root, spec.folder)
    readme_path = os.path.join(dataset_dir, "README.md")
    parquet_paths = sorted(glob(os.path.join(dataset_dir, "data", "all_samples-*.parquet")))
    if not os.path.exists(readme_path) or len(parquet_paths) == 0:
        raise FileNotFoundError(
            f"Could not find local PLAID dataset files under '{dataset_dir}'. "
            "Run scripts/download_dataset.py first."
        )

    meta = load_plaid_readme_meta(dataset_dir)
    available_fields = tuple(resolve_target_fields(spec.benchmark_fields, out_fields=list(meta.out_fields)))
    target_fields = _resolve_active_terminal_target_fields(
        available_fields=available_fields,
        terminal_target_fields=terminal_target_fields,
    )
    if not set(target_fields).issubset(set(available_fields)):
        raise ValueError(
            f"Unexpected target fields for {dataset_name}: {target_fields} (available {available_fields})."
        )
    channel_indices = terminal_target_field_indices(target_fields)

    labeled_ids = [int(i) for i in meta.split_map.get(dynamics_spec.labeled_split, [])]
    public_test_ids = [int(i) for i in meta.split_map.get(dynamics_spec.test_split, [])]
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

    v3_ready = is_v3_cache_complete(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    if not v3_ready:
        load_plaid_elpl_v3_dataset(
            data_root=data_root,
            split_seed=split_seed,
            graph_backend=graph_backend,
            use_sdf_features=False,
            max_samples=max_samples,
            load_public_test=False,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
        )

    terminal_ready = is_terminal_cache_complete(
        dataset_dir,
        split_seed=split_seed,
        use_sdf_features=use_sdf_features,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    if not terminal_ready:
        if distributed_rank() != 0:
            distributed_barrier()
            if not is_terminal_cache_complete(
                dataset_dir,
                split_seed=split_seed,
                use_sdf_features=use_sdf_features,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
            ):
                raise RuntimeError("PLAID el-pl terminal cache is missing after rank-0 build.")
        else:
            _build_terminal_cache(
                dataset_dir=dataset_dir,
                split_seed=split_seed,
                train_ids=train_ids,
                val_ids=val_ids,
                use_sdf_features=use_sdf_features,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
            )
            distributed_barrier()

    train_df, val_df = load_terminal_manifest_tables(
        train_path=train_manifest_path(dataset_dir, split_seed=split_seed),
        val_path=val_manifest_path(dataset_dir, split_seed=split_seed),
    )
    sim_to_shard_df = pd.read_parquet(resolve_sim_to_shard_path(dataset_dir, split_seed=split_seed))
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    stats = load_terminal_norm_stats(
        norm_stats_path(dataset_dir, split_seed=split_seed, use_sdf_features=use_sdf_features)
    )
    runtime_y_norm_mode = normalize_runtime_y_norm_mode(
        y_norm_mode or os.environ.get("PLAID_TERMINAL_Y_NORM", DEFAULT_RUNTIME_Y_NORM)
    )
    runtime_y_normalizer = slice_y_normalizer(
        resolve_runtime_y_normalizer(
            mode=runtime_y_norm_mode,
            dataset_dir=dataset_dir,
            split_seed=split_seed,
            use_sdf_features=use_sdf_features,
            cache_y_normalizer=stats.cache_y_normalizer,
            train_ids=train_ids,
            sim_to_shard_df=sim_to_shard_df,
            shard_root=shard_root,
        ),
        channel_indices,
    )
    cache_y_normalizer = slice_y_normalizer(stats.cache_y_normalizer, channel_indices)
    dataset_kwargs = dict(
        sim_to_shard=sim_to_shard_df,
        shard_root=shard_root,
        stats=stats,
        runtime_y_normalizer=runtime_y_normalizer,
        use_sdf_features=use_sdf_features,
        graph_backend=graph_backend,
        target_fields=target_fields,
        bandwidth=spec.bandwidth,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    train_data = ElPlTerminalDataset(manifest=train_df, **dataset_kwargs)
    val_data = ElPlTerminalDataset(manifest=val_df, **dataset_kwargs)

    test_data = GraphListDataset([])
    if load_public_test and public_test_ids:
        test_df = build_terminal_manifest(public_test_ids)
        test_data = ElPlTerminalDataset(manifest=test_df, **dataset_kwargs)

    first_graph = train_data[0]
    pos_dim = int(first_graph.pos.shape[-1])
    x_dim = int(first_graph.x.shape[-1])
    node_feat_dim = x_dim - pos_dim
    input_scalars = getattr(first_graph, "input_scalars", None)
    num_input_scalars = int(input_scalars.shape[-1]) if input_scalars is not None else 0
    fun_dim = node_feat_dim + num_input_scalars
    c_in = pos_dim + fun_dim

    metadata = dict(
        x_normalizer=x_normalizer_for_metadata(stats, use_sdf_features=use_sdf_features),
        y_normalizer=runtime_y_normalizer,
        cache_y_normalizer=cache_y_normalizer,
        input_scalar_normalizer=None,
        y_scalar_normalizer=None,
        elpl_terminal_norm_stats=stats,
        plaid_terminal_y_norm=runtime_y_norm_mode,
        c_in=c_in,
        c_edge=int(first_graph.edge_attr.shape[-1]),
        c_out=int(first_graph.y.shape[-1]),
        space_dim=pos_dim,
        fun_dim=fun_dim,
        num_input_scalars=num_input_scalars,
        time_cond=False,
        max_length=max(int(train_data[i].x.shape[0]) for i in range(min(32, len(train_data)))),
        target_fields=list(target_fields),
        target_scalar_fields=[],
        plaid_use_sdf_features=bool(use_sdf_features),
        plaid_load_public_test=bool(load_public_test),
        plaid_max_samples=int(max_samples),
        plaid_laplacian_eig_dim=int(laplacian_eig_dim),
        plaid_laplacian_spec=str(laplacian_spec),
        plaid_loss_lbda=1.0,
        plaid_temporal_one_step=False,
        plaid_terminal_prediction=True,
        plaid_terminal_target_fields=list(target_fields),
        plaid_cache_format="elpl_v3",
        mesh_split_note=(
            f"Terminal prediction (elpl_terminal): one graph per simulation from '{dynamics_spec.labeled_split}' "
            f"with sklearn 80/20 seed={split_seed}. Target fields={list(target_fields)} at t=0.04; "
            "mesh-only input (no u0, no time)."
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

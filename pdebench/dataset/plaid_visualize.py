"""PLAID mesh-static dataset visualization.

Owns the PLAID dataset names and the mesh-static PLAID sample loader for
``pdebench.dataset.visualize``. Rendering itself is not reimplemented here:
``main`` temporarily configures ``pdebench.dataset.ginot.visualize`` for a
PLAID-only invocation (dataset set + sample loader) and delegates to its
``main``, so plotting code stays owned by one module.

Usage:
  python -m pdebench.dataset.plaid_visualize --dataset plaid_tensile2d --mode raw
"""

from __future__ import annotations

from pathlib import Path

DATASETS = frozenset(
    {
        "plaid_tensile2d",
        "plaid_hyperelasticity",
        "plaid_el_pl_dynamics",
        "plaid_elpl_terminal",
    }
)


def _mesh_static_plaid_index_map(dataset, sim_ids: list[int]) -> dict[int, int]:
    manifest = getattr(dataset, "manifest", None)
    if manifest is not None:
        if len(dataset) != len(manifest):
            raise ValueError(
                f"PLAID graph/manifest count mismatch: dataset has {len(dataset)} graphs "
                f"but manifest has {len(manifest)} rows."
            )
        # Transition manifests contain multiple rows per simulation. Keep the
        # last transition so temporal visualizations show the final one-step target.
        return {int(row["sim_id"]): int(i) for i, row in manifest.iterrows()}
    if len(dataset) != len(sim_ids):
        raise ValueError(
            f"PLAID graph/ID count mismatch: dataset has {len(dataset)} graphs but metadata has {len(sim_ids)} IDs."
        )
    return {int(sim_id): i for i, sim_id in enumerate(sim_ids)}


def load_mesh_static_plaid_dataset_samples(
    dataset: str,
    data_root: Path,
    seed: int,
    max_samples: int,
    splits: list[str],
    *,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    use_sdf_features: bool = False,
) -> tuple[dict[str, list[dict]], list[str], dict, dict[str, object]]:
    from pdebench.dataset.ginot.visualize import _pyg_graph_to_viz_sample, random_sample_ids
    from pdebench.dataset.plaid_datasets import load_mesh_static_dataset

    print(
        f"  loading mesh-static PLAID dataset (laplacian K={int(laplacian_eig_dim)})...",
        flush=True,
    )
    train_ds, val_ds, metadata = load_mesh_static_dataset(
        dataset_name=dataset,
        data_root=str(data_root),
        split_seed=seed,
        graph_backend="pyg",
        use_sdf_features=bool(use_sdf_features),
        laplacian_eig_dim=int(laplacian_eig_dim),
        laplacian_spec=str(laplacian_spec),
    )
    split_datasets: dict[str, object] = {"train": train_ds, "test": val_ds}
    ids_source = {
        "train": [int(v) for v in metadata.get("mesh_train_ids", [])],
        "test": [int(v) for v in metadata.get("mesh_val_ids", [])],
    }
    split_samples: dict[str, list[dict]] = {}
    ids_by_split: dict[str, list[int]] = {}
    index_maps: dict[str, dict[int, int]] = {}
    for split in splits:
        if split not in split_datasets:
            raise ValueError(f"Split {split!r} is not available for dataset {dataset!r}.")
        ginot_ds = split_datasets[split]
        index_map = _mesh_static_plaid_index_map(ginot_ds, ids_source[split])
        index_maps[split] = index_map
        pool = [sim_id for sim_id in ids_source[split] if sim_id in index_map]
        chosen_ids = random_sample_ids(
            pool,
            max_samples=max_samples,
            seed=seed * 1009 + (17 if split == "train" else 53),
        )
        ids_by_split[split] = chosen_ids
        split_samples[split] = [
            _pyg_graph_to_viz_sample(ginot_ds[index_map[int(sim_id)]], metadata) for sim_id in chosen_ids
        ]
    target_fields = list(metadata.get("target_fields", []))
    return (
        split_samples,
        target_fields,
        {
            "num_samples": len(ids_source["train"]) + len(ids_source["test"]),
            "ids_by_split": ids_by_split,
            "index_maps": index_maps,
            "data_root": data_root,
            "seed": seed,
            "max_samples": max_samples,
            "dataset": dataset,
            "issue_stats": {},
            "original_issue_meshes": {},
            "y_normalizer": metadata["y_normalizer"],
            "graph_cache_dirs": {split_name: None for split_name in split_datasets},
        },
        split_datasets,
    )


def main() -> None:
    from pdebench.dataset.ginot import visualize as gv

    previous_datasets = gv.DATASETS
    previous_plaid_set = getattr(gv, "MESH_STATIC_PLAID_VIZ_DATASETS", frozenset())
    previous_loader = getattr(gv, "load_mesh_static_plaid_dataset_samples", None)
    gv.DATASETS = sorted(DATASETS)
    gv.MESH_STATIC_PLAID_VIZ_DATASETS = DATASETS
    gv.load_mesh_static_plaid_dataset_samples = load_mesh_static_plaid_dataset_samples
    try:
        gv.main()
    finally:
        gv.DATASETS = previous_datasets
        gv.MESH_STATIC_PLAID_VIZ_DATASETS = previous_plaid_set
        if previous_loader is not None:
            gv.load_mesh_static_plaid_dataset_samples = previous_loader


if __name__ == "__main__":
    main()

import os
import uuid
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# ============================================================================ #
# PLAID DATASET PREPROCESSING
#
# This module implements the static-mesh PLAID dataset loading and
# preprocessing pipeline used by PDEBench mesh experiments.
#
# Reference paper:
#   - PLAID: https://arxiv.org/abs/2505.02974
#     (PDF: https://arxiv.org/pdf/2505.02974)
#
# The feature construction and normalization in this file are intentionally kept
# aligned with PLAID benchmark conventions for plaid_tensile2d /
# plaid_hyperelasticity / plaid_el_pl_dynamics (aliases: tensile2d, hyperelasticity).
# ============================================================================ #
from pdebench.dataset.distributed import distributed_barrier, distributed_rank
from pdebench.dataset.laplacian.spec import laplacian_spec_slug
from pdebench.dataset.normalizer import NodeFeatureNormalizer
from pdebench.dataset.plaid_core import (
    build_edge_attr,
    build_edge_index_from_cells,
    build_plaid_benchmark_node_features,
    build_plaid_mgn_node_features,
    load_plaid_readme_meta,
    parse_plaid_sample_bytes,
    resolve_target_fields,
    split_labeled_train_test,
)
from pdebench.dataset.plaid_laplacian import (  # noqa: F401 — re-export
    _attach_laplacian_features,
    _ensure_plaid_laplacian,
    _load_plaid_laplacian,
    _plaid_laplacian_cache_file,
)
from pdebench.dataset.registry import resolve_dataset_name

_split_labeled_train_test = split_labeled_train_test

try:
    import datasets as hf_datasets
except ModuleNotFoundError as exc:
    hf_datasets = None
    _DATASETS_IMPORT_ERROR = exc

try:
    import torch_geometric as pyg
except ModuleNotFoundError as exc:
    pyg = None
    _PYG_IMPORT_ERROR = exc

__all__ = [
    "PLAID_DATASETS",
    "load_mesh_static_dataset",
    "NodeFeatureNormalizer",
]


# Canonical registry keys (aliases tensile2d / hyperelasticity resolve via registry).
PLAID_DATASETS = {"plaid_tensile2d", "plaid_hyperelasticity", "plaid_el_pl_dynamics", "plaid_elpl_terminal"}


@dataclass(frozen=True)
class PlaidSpec:
    folder: str
    labeled_split: str
    test_split: str
    train_count: int
    test_count: int
    benchmark_fields: tuple[str, ...]
    benchmark_scalar_outputs: tuple[str, ...]
    bandwidth: float
    use_plaid_benchmark_features: bool = False


PLAID_SPECS: dict[str, PlaidSpec] = {
    "plaid_tensile2d": PlaidSpec(
        folder=os.path.join("plaid", "Tensile2d"),
        labeled_split="train_500",
        test_split="test",
        train_count=500,
        test_count=200,
        benchmark_fields=("U1", "U2", "sig11", "sig22", "sig12"),
        benchmark_scalar_outputs=("max_von_mises", "max_U2_top", "max_sig22_top"),
        bandwidth=0.02976729880552192,
    ),
    "plaid_hyperelasticity": PlaidSpec(
        folder=os.path.join("plaid", "2D_Multiscale_Hyperelasticity"),
        labeled_split="DOE_train",
        test_split="DOE_test",
        train_count=764,
        test_count=376,
        benchmark_fields=("u1", "u2", "P11", "P12", "P22", "P21", "psi"),
        benchmark_scalar_outputs=("effective_energy",),
        bandwidth=0.03692578713441289,
        use_plaid_benchmark_features=True,
    ),
    "plaid_el_pl_dynamics": PlaidSpec(
        folder=os.path.join("plaid", "2D_ElastoPlastoDynamics"),
        labeled_split="train",
        test_split="test",
        train_count=1000,
        test_count=18,
        benchmark_fields=("U_x", "U_y"),
        benchmark_scalar_outputs=(),
        bandwidth=0.8813942074775696,
        use_plaid_benchmark_features=True,
    ),
    "plaid_elpl_terminal": PlaidSpec(
        folder=os.path.join("plaid", "2D_ElastoPlastoDynamics"),
        labeled_split="train",
        test_split="test",
        train_count=1000,
        test_count=18,
        benchmark_fields=("U_x", "U_y"),
        benchmark_scalar_outputs=(),
        bandwidth=0.8813942074775696,
        use_plaid_benchmark_features=True,
    ),
}


class GraphListDataset(Dataset):
    def __init__(self, graphs: list[Any]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _graph_get_x(graph):
    if hasattr(graph, "x"):
        return graph.x
    if hasattr(graph, "ndata") and ("x" in graph.ndata):
        return graph.ndata["x"]
    raise ValueError(f"Graph object does not contain node features 'x': {type(graph)}")


def _graph_set_x(graph, x: torch.Tensor):
    if hasattr(graph, "x"):
        graph.x = x
    elif hasattr(graph, "ndata"):
        graph.ndata["x"] = x
    else:
        raise ValueError(f"Graph object does not support setting node features 'x': {type(graph)}")


def _graph_get_y(graph):
    if hasattr(graph, "y"):
        return graph.y
    if hasattr(graph, "ndata") and ("y" in graph.ndata):
        return graph.ndata["y"]
    return None


def _graph_set_y(graph, y: torch.Tensor):
    if hasattr(graph, "y"):
        graph.y = y
    elif hasattr(graph, "ndata"):
        graph.ndata["y"] = y
    else:
        raise ValueError(f"Graph object does not support setting targets 'y': {type(graph)}")


def _graph_get_pos(graph):
    if hasattr(graph, "pos"):
        return graph.pos
    if hasattr(graph, "ndata") and ("pos" in graph.ndata):
        return graph.ndata["pos"]
    raise ValueError(f"Graph object does not contain node coordinates 'pos': {type(graph)}")


def _graph_get_edge_attr(graph):
    if hasattr(graph, "edge_attr"):
        return graph.edge_attr
    if hasattr(graph, "edata") and ("f" in graph.edata):
        return graph.edata["f"]
    if hasattr(graph, "edata") and ("edge_attr" in graph.edata):
        return graph.edata["edge_attr"]
    raise ValueError(f"Graph object does not contain edge features: {type(graph)}")


def _apply_graph_normalizers_once(
    graph_groups: list[list[Any]],
    x_normalizer: NodeFeatureNormalizer,
    y_normalizer: NodeFeatureNormalizer,
):
    seen = set()
    for graphs in graph_groups:
        for graph in graphs:
            graph_id = id(graph)
            if graph_id in seen:
                continue
            seen.add(graph_id)

            _graph_set_x(graph, x_normalizer.encode(_graph_get_x(graph)))
            y = _graph_get_y(graph)
            if y is not None:
                _graph_set_y(graph, y_normalizer.encode(y))


def _require_mesh_dependencies(graph_backend: str = "pyg"):
    if hf_datasets is None:
        raise ModuleNotFoundError(
            "datasets is required for mesh loaders. Install dependencies and retry."
        ) from _DATASETS_IMPORT_ERROR
    if graph_backend != "pyg":
        raise ValueError(f"Unsupported graph backend '{graph_backend}'. Choose: pyg.")
    if pyg is None:
        raise ModuleNotFoundError(
            "torch_geometric is required for mesh loaders/models with graph_backend='pyg'. "
            "Install the PyG stack and retry."
        ) from _PYG_IMPORT_ERROR


def _build_graph_sample(
    *,
    graph_backend: str,
    pos_t: torch.Tensor,
    x_t: torch.Tensor,
    y_t: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    cells_t: torch.Tensor,
    metadata: dict[str, Any],
):
    if graph_backend == "pyg":
        kwargs = dict(
            pos=pos_t.float(),
            x=x_t.float(),
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            cells=cells_t.long(),
            metadata=metadata,
            input_scalars=metadata["scalar_values_tensor"].float(),
            input_scalars_names=metadata["scalar_names"],
            output_fields_names=metadata["target_fields"],
            output_scalars_names=metadata["target_scalar_fields"],
            sample_id=metadata["sample_index"],
        )
        if y_t is not None:
            kwargs["y"] = y_t.float()
            kwargs["output_fields"] = y_t.float()
        if metadata["target_scalar_values_tensor"] is not None:
            kwargs["output_scalars"] = metadata["target_scalar_values_tensor"].float()
        del metadata["scalar_values_tensor"]
        del metadata["target_scalar_values_tensor"]
        return pyg.data.Data(**kwargs)

    raise ValueError(f"Unsupported graph backend '{graph_backend}'.")


def _plaid_static_graph_cache_file(
    *,
    dataset_dir: str,
    dataset_name: str,
    split_seed: int,
    graph_backend: str,
    use_sdf_features: bool,
    max_samples: int,
    load_public_test: bool,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> Path:
    sdf_tag = "sdf1" if bool(use_sdf_features) else "sdf0"
    public_tag = "public1" if bool(load_public_test) else "public0"
    n_scalars = len(PLAID_SPECS[dataset_name].benchmark_scalar_outputs)
    scalar_tag = f"sc{int(n_scalars)}"
    spec_slug = laplacian_spec_slug(str(laplacian_spec), int(laplacian_eig_dim)) if int(laplacian_eig_dim) > 0 else "K0"
    return (
        Path(dataset_dir)
        / "static_cache"
        / "pyg_graphs"
        / "v2"
        / dataset_name
        / graph_backend
        / f"split{int(split_seed)}"
        / sdf_tag
        / public_tag
        / scalar_tag
        / f"max{int(max_samples)}"
        / spec_slug
        / "graphs.pt"
    )


def _materialize_graph_tensors(graph: Any) -> None:
    """Copy mmap-backed cache tensors into RAM so PyG collate stays fast with num_workers=0."""
    for attr in ("x", "y", "pos", "edge_index", "edge_attr", "laplacian_eig", "laplacian_eigvals", "output_scalars"):
        if hasattr(graph, attr):
            value = getattr(graph, attr)
            if torch.is_tensor(value):
                setattr(graph, attr, value.clone())


def _materialize_plaid_graph_lists(*graph_lists: list[Any]) -> None:
    for graphs in graph_lists:
        for graph in graphs:
            _materialize_graph_tensors(graph)


def _load_plaid_static_graph_cache(cache_file: Path) -> tuple[GraphListDataset, GraphListDataset, dict[str, Any]] | None:
    if not cache_file.exists():
        return None
    payload = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != 1:
        return None
    train_graphs = payload["train_graphs"]
    val_graphs = payload["val_graphs"]
    test_graphs = payload.get("test_graphs", [])
    _materialize_plaid_graph_lists(train_graphs, val_graphs, test_graphs)
    metadata = payload["metadata"]
    metadata["mesh_test_data"] = GraphListDataset(test_graphs)
    return GraphListDataset(train_graphs), GraphListDataset(val_graphs), metadata


def _save_plaid_static_graph_cache(
    cache_file: Path,
    *,
    train_graphs: list[Any],
    val_graphs: list[Any],
    test_graphs: list[Any],
    metadata: dict[str, Any],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_for_cache = dict(metadata)
    metadata_for_cache.pop("mesh_test_data", None)
    tmp = cache_file.parent / f".{cache_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(
            {
                "version": 1,
                "train_graphs": train_graphs,
                "val_graphs": val_graphs,
                "test_graphs": test_graphs,
                "metadata": metadata_for_cache,
            },
            tmp,
        )
        os.replace(tmp, cache_file)
    finally:
        tmp.unlink(missing_ok=True)


def _parse_plaid_sample(
    sample_bytes: bytes,
    target_fields: list[str],
    scalar_names: list[str],
    dataset_name: str,
    sample_idx: int,
    bandwidth: float,
    require_targets: bool = True,
    graph_backend: str = "pyg",
    use_sdf_features: bool = True,
    dataset_dir: str | None = None,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
):
    parsed = parse_plaid_sample_bytes(
        sample_bytes,
        target_fields=target_fields,
        scalar_names=scalar_names,
        dataset_name=dataset_name,
        sample_idx=sample_idx,
        require_targets=require_targets,
    )
    if parsed is None:
        return None

    pos = parsed.pos
    cells = parsed.cells
    spec = PLAID_SPECS[dataset_name]
    x = (
        build_plaid_benchmark_node_features(parsed, use_sdf_features=use_sdf_features)
        if spec.use_plaid_benchmark_features
        else build_plaid_mgn_node_features(parsed)
    )
    edge_index = build_edge_index_from_cells(cells)
    pos_t = torch.from_numpy(pos)
    edge_attr = build_edge_attr(pos_t, edge_index, bandwidth=bandwidth).float()
    y_t = None if parsed.targets is None else torch.from_numpy(parsed.targets).float()
    cells_t = torch.from_numpy(cells).long()

    graph = _build_graph_sample(
        graph_backend=graph_backend,
        pos_t=pos_t,
        x_t=torch.from_numpy(x).float(),
        y_t=y_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cells_t=cells_t,
        metadata={
            "dataset": dataset_name,
            "sample_index": int(sample_idx),
            "target_fields": list(target_fields),
            "target_scalar_fields": list(spec.benchmark_scalar_outputs),
            "scalar_names": scalar_names,
            "scalar_values": parsed.scalar_values.tolist(),
            "scalar_values_tensor": torch.from_numpy(parsed.scalar_values).reshape(1, -1),
            "target_scalar_values_tensor": (
                None
                if parsed.target_scalars is None
                else torch.from_numpy(parsed.target_scalars).reshape(1, -1)
            ),
            "boundary_tags": list(parsed.boundary_tags),
            "boundary_ids": parsed.boundary_ids.tolist(),
            "raw_format": "pickled_cgns_tree",
        },
    )
    if int(laplacian_eig_dim) > 0:
        if dataset_dir is None:
            raise ValueError("dataset_dir is required when laplacian_eig_dim > 0.")
        _ensure_plaid_laplacian(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            sample_idx=sample_idx,
            edge_index=edge_index,
            pos=pos_t,
            cells=cells_t,
            num_eigenvectors=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
        )
        graph = _attach_laplacian_features(
            graph,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            sample_idx=sample_idx,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
        )
    return graph


def _load_plaid_dataset(
    dataset_name: str,
    data_root: str,
    split_seed: int,
    graph_backend: str,
    use_sdf_features: bool = True,
    max_samples: int = 0,
    load_public_test: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
):
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
    target_fields = _resolve_target_fields(spec, out_fields=list(meta.out_fields))

    labeled_ids = [int(i) for i in meta.split_map.get(spec.labeled_split, [])]
    public_test_ids = [int(i) for i in meta.split_map.get(spec.test_split, [])]
    if len(labeled_ids) == 0:
        raise ValueError(
            f"Could not find labeled split '{spec.labeled_split}' in {readme_path}."
        )
    if len(public_test_ids) == 0:
        raise ValueError(
            f"Could not find test split '{spec.test_split}' in {readme_path}."
        )

    if len(labeled_ids) != spec.train_count or len(public_test_ids) != spec.test_count:
        raise ValueError(
            f"Unexpected PLAID split sizes for {dataset_name}: "
            f"train '{spec.labeled_split}' has {len(labeled_ids)} samples "
            f"(expected {spec.train_count}), test '{spec.test_split}' has {len(public_test_ids)} samples "
            f"(expected {spec.test_count})."
        )

    full_split_sizes = dict(labeled=len(labeled_ids), public_test=len(public_test_ids))
    train_ids, val_ids = split_labeled_train_test(labeled_ids, test_ratio=0.2, seed=split_seed)
    if max_samples > 0:
        train_ids = train_ids[:max_samples]
        val_ids = val_ids[:max_samples]
        public_test_ids = public_test_ids[:max_samples]

    graph_cache_file = _plaid_static_graph_cache_file(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        split_seed=split_seed,
        graph_backend=graph_backend,
        use_sdf_features=use_sdf_features,
        max_samples=max_samples,
        load_public_test=load_public_test,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    cached = _load_plaid_static_graph_cache(graph_cache_file)
    if cached is not None:
        return cached

    if distributed_rank() != 0:
        distributed_barrier()
        cached = _load_plaid_static_graph_cache(graph_cache_file)
        if cached is None:
            raise RuntimeError(
                f"PLAID static graph cache is missing after rank-0 build: {graph_cache_file}"
            )
        return cached

    raw = hf_datasets.load_dataset("parquet", data_files={"all_samples": parquet_paths}, split="all_samples")

    def build_graphs(selected_ids: list[int], require_targets: bool):
        graphs = []
        for idx in selected_ids:
            graph = _parse_plaid_sample(
                sample_bytes=raw[idx]["sample"],
                target_fields=target_fields,
                scalar_names=list(meta.scalar_names),
                dataset_name=dataset_name,
                sample_idx=idx,
                bandwidth=spec.bandwidth,
                require_targets=require_targets,
                graph_backend=graph_backend,
                use_sdf_features=use_sdf_features,
                dataset_dir=dataset_dir,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
            )
            if graph is not None:
                graphs.append(graph)
        return graphs

    train_graphs = build_graphs(train_ids, require_targets=True)
    val_graphs = build_graphs(val_ids, require_targets=True)
    test_graphs = build_graphs(public_test_ids, require_targets=False) if load_public_test else []

    if len(train_graphs) != len(train_ids) or len(val_graphs) != len(val_ids):
        raise ValueError(
            f"Failed to load all requested PLAID graphs for {dataset_name}: "
            f"train loaded {len(train_graphs)}/{len(train_ids)}, "
            f"val loaded {len(val_graphs)}/{len(val_ids)}."
        )
    if load_public_test and len(test_graphs) != len(public_test_ids):
        raise ValueError(
            f"Failed to load all public PLAID test graphs for {dataset_name}: "
            f"test loaded {len(test_graphs)}/{len(public_test_ids)}."
        )

    if len(train_graphs) == 0 or len(val_graphs) == 0:
        raise ValueError(
            f"Invalid split for {dataset_name}. "
            f"Sizes: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}."
        )

    first_graph = train_graphs[0]
    first_x = _graph_get_x(first_graph)
    first_pos = _graph_get_pos(first_graph)
    c_in = first_x.shape[-1]
    x_mean = torch.zeros((1, c_in), dtype=first_x.dtype)
    x_std = torch.ones((1, c_in), dtype=first_x.dtype)
    pos_dim = int(first_pos.shape[-1])
    num_scalars = len(meta.scalar_names)
    scalar_offset = c_in - num_scalars if spec.use_plaid_benchmark_features else pos_dim + 9 + 1
    if spec.use_plaid_benchmark_features:
        x_field_train = torch.cat([_graph_get_x(g)[:, :scalar_offset] for g in train_graphs], dim=0)
        x_mean[:, :scalar_offset] = x_field_train.mean(dim=0, keepdim=True)
        x_std[:, :scalar_offset] = x_field_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    if (num_scalars > 0) and ((scalar_offset + num_scalars) <= c_in):
        scalar_train = torch.stack(
            [_graph_get_x(g)[0, scalar_offset:scalar_offset + num_scalars] for g in train_graphs],
            dim=0,
        )
        scalar_mean = scalar_train.mean(dim=0, keepdim=True)
        scalar_std = scalar_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        x_mean[:, scalar_offset:scalar_offset + num_scalars] = scalar_mean
        x_std[:, scalar_offset:scalar_offset + num_scalars] = scalar_std
    x_normalizer = NodeFeatureNormalizer(mean=x_mean, std=x_std)
    y_train = torch.cat([_graph_get_y(g) for g in train_graphs], dim=0)
    y_normalizer = NodeFeatureNormalizer(
        mean=y_train.mean(dim=0, keepdim=True),
        std=y_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8),
    )
    y_scalar_normalizer = None
    if spec.benchmark_scalar_outputs:
        y_scalars_train = torch.cat([g.output_scalars for g in train_graphs], dim=0)
        y_scalar_normalizer = NodeFeatureNormalizer(
            mean=y_scalars_train.mean(dim=0, keepdim=True),
            std=y_scalars_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8),
        )
    _apply_graph_normalizers_once(
        [train_graphs, val_graphs, test_graphs],
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )
    if y_scalar_normalizer is not None:
        for graphs in (train_graphs, val_graphs):
            for graph in graphs:
                graph.output_scalars = y_scalar_normalizer.encode(graph.output_scalars)

    first_y = _graph_get_y(first_graph)
    output_scalar_dim = len(spec.benchmark_scalar_outputs)

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        y_scalar_normalizer=y_scalar_normalizer,
        c_in=_graph_get_x(first_graph).shape[-1],
        c_edge=_graph_get_edge_attr(first_graph).shape[-1],
        c_out=first_y.shape[-1] + output_scalar_dim,
        space_dim=_graph_get_pos(first_graph).shape[-1],
        fun_dim=_graph_get_x(first_graph).shape[-1] - _graph_get_pos(first_graph).shape[-1],
        time_cond=False,
        max_length=max(_graph_get_x(g).shape[0] for g in train_graphs),
        target_fields=list(target_fields),
        target_scalar_fields=list(spec.benchmark_scalar_outputs),
        plaid_use_sdf_features=bool(use_sdf_features) if spec.use_plaid_benchmark_features else None,
        plaid_load_public_test=bool(load_public_test),
        plaid_max_samples=int(max_samples),
        plaid_laplacian_eig_dim=int(laplacian_eig_dim),
        plaid_laplacian_spec=str(laplacian_spec),
        plaid_loss_lbda=0.5,
        mesh_split_note=(
            f"Using PLAID benchmark sklearn 80/20 labeled split from '{spec.labeled_split}' with seed={split_seed}. "
            f"Public split '{spec.test_split}' is unlabeled locally and reserved for HF benchmark submission."
        ),
        mesh_split_sizes=dict(
            train=len(train_graphs),
            val=len(val_graphs),
            test=len(test_graphs),
        ),
        mesh_full_split_sizes=full_split_sizes,
        mesh_train_ids=train_ids,
        mesh_val_ids=val_ids,
        mesh_public_test_ids=public_test_ids,
        mesh_graph_backend=graph_backend,
        mesh_test_data=GraphListDataset(test_graphs),
    )
    _save_plaid_static_graph_cache(
        graph_cache_file,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        metadata=metadata,
    )
    distributed_barrier()
    cached = _load_plaid_static_graph_cache(graph_cache_file)
    if cached is None:
        raise RuntimeError(
            f"PLAID static graph cache is missing after rank-0 save: {graph_cache_file}"
        )
    return cached


def _resolve_target_fields(spec: PlaidSpec, out_fields: list[str]) -> list[str]:
    return resolve_target_fields(spec.benchmark_fields, out_fields=out_fields)


def load_mesh_static_dataset(
    dataset_name: str,
    data_root: str,
    split_seed: int = 0,
    graph_backend: str = "pyg",
    use_sdf_features: bool = True,
    max_samples: int = 0,
    load_public_test: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    y_norm_mode: str = "asinh_iqr",
    terminal_target_fields: list[str] | None = None,
):
    """Load a PLAID mesh-static / el-pl dataset by canonical name (aliases resolved)."""
    dataset_name = resolve_dataset_name(dataset_name.lower())
    if dataset_name == "plaid_elpl_terminal":
        from pdebench.dataset.plaid_elpl_terminal.loader import load_plaid_elpl_terminal_dataset

        return load_plaid_elpl_terminal_dataset(
            data_root=data_root,
            split_seed=split_seed,
            graph_backend=graph_backend,
            use_sdf_features=use_sdf_features,
            max_samples=max_samples,
            load_public_test=load_public_test,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            y_norm_mode=y_norm_mode,
            terminal_target_fields=terminal_target_fields,
        )
    if dataset_name == "plaid_el_pl_dynamics":
        from pdebench.dataset.plaid_elpl_v3.loader import load_plaid_elpl_v3_dataset

        return load_plaid_elpl_v3_dataset(
            data_root=data_root,
            split_seed=split_seed,
            graph_backend=graph_backend,
            use_sdf_features=use_sdf_features,
            max_samples=max_samples,
            load_public_test=load_public_test,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
        )
    if dataset_name in PLAID_SPECS:
        return _load_plaid_dataset(
            dataset_name=dataset_name,
            data_root=data_root,
            split_seed=split_seed,
            graph_backend=graph_backend,
            use_sdf_features=use_sdf_features,
            max_samples=max_samples,
            load_public_test=load_public_test,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
        )
    raise ValueError(f"Unsupported mesh dataset '{dataset_name}'.")

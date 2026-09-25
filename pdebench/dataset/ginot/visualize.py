#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pdebench.dataset.ginot import (  # noqa: E402
    GINOT_DATASETS,
    _as_float_array,
    _is_micro_puc_fixed_raw,
    _is_micro_puc_raw,
    _normalize_cells,
    _num_indexable_samples,
    _select_cells,
    load_ginot_dataset,
)
from pdebench.dataset.ginot.bumper_beam import TARGET_TIMES as BUMPER_BEAM_TARGET_TIMES
from pdebench.dataset.ginot.bumper_beam import decode_bumper_beam_target
from pdebench.dataset.ginot.deform_plate import (
    HANDLE,
    NORMAL,
    OBSTACLE,
    TARGET_STEP,
    load_deform_plate_equilibrium_arrays,
)
from pdebench.dataset.ginot.utils import is_bumper_beam_raw
from pdebench.dataset.laplacian import (  # noqa: E402
    DEFAULT_LAPLACIAN_EIGENVECTORS,
    DEFAULT_LAPLACIAN_SPECS,
    compute_laplacian_eigendecomp_part,
    compute_laplacian_feature_part,
    parse_laplacian_spec,
)
from pdebench.dataset.lpbf import LPBF_DATASETS

# Populated by pdebench.dataset.plaid_visualize.main() for a PLAID-only
# invocation; empty by default so plain ginot.visualize.main() never treats
# PLAID dataset names as valid GINOT datasets.
MESH_STATIC_PLAID_VIZ_DATASETS: frozenset[str] = frozenset()
DATASETS = sorted(GINOT_DATASETS | LPBF_DATASETS)
# Injected alongside MESH_STATIC_PLAID_VIZ_DATASETS by
# pdebench.dataset.plaid_visualize.main(); unused while that set is empty.
load_mesh_static_plaid_dataset_samples = None
MICRO_PUC_FIXED_RAW_LIMITS = (-0.005, 1.005)
DEFAULT_SPECTRAL_WORKERS = 16


def _resolve_spectral_workers(requested: int) -> int:
    if int(requested) == 1:
        return 1
    if int(requested) > 1:
        return int(requested)
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0")
    if slurm_cpus > 0:
        return max(1, min(slurm_cpus, 64))
    return max(1, min(os.cpu_count() or 1, DEFAULT_SPECTRAL_WORKERS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render random train/test samples for GINOT-style datasets, then compute graph Laplacian "
            "eigenmode visualizations one dataset at a time."
        )
    )
    parser.add_argument("--dataset", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--outdir", type=Path, default=Path("out/pdebench/ginot_dataset_viz"))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        "--samples-per-split",
        dest="max_samples",
        type=int,
        default=10,
        help="Randomly select this many samples from each requested split.",
    )
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-spectral", action="store_true")
    parser.add_argument(
        "--spectral-only",
        action="store_true",
        help="Skip raw field/geometry figures and render spectral overlays only.",
    )
    parser.add_argument("--spectral-modes", type=int, default=DEFAULT_LAPLACIAN_EIGENVECTORS)
    parser.add_argument(
        "--laplacian-specs",
        default=DEFAULT_LAPLACIAN_SPECS,
        help="Comma-separated Laplacian specs used for cached spectral features (e.g. graph:64,edge:64,fem-v:64).",
    )
    parser.add_argument(
        "--spectral-operators",
        nargs="+",
        default=None,
        choices=["graph", "edge", "fem_v", "fem-v", "fem_u", "fem-u"],
        help="Operators to plot; default is all operators listed in --laplacian-specs.",
    )
    parser.add_argument("--spectral-max-samples", type=int, default=None)
    parser.add_argument("--spectral-max-points", type=int, default=6000)
    parser.add_argument("--spectral-device", default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument(
        "--spectral-workers",
        type=int,
        default=0,
        help="Parallel workers for spectral cache IO and figure rendering (0 => auto from CPU count).",
    )
    parser.add_argument(
        "--micro-puc-fixed-resolved-samples",
        type=int,
        nargs="*",
        default=[],
        help="micro_puc_fixed sample ids to render under raw/resolved_unmatched_nodes after a mesh-fix issue is resolved.",
    )
    return parser.parse_args()


def random_sample_ids(ids: list[int], max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or max_samples >= len(ids):
        return list(ids)
    choice = np.random.default_rng(seed).choice(np.asarray(ids, dtype=np.int64), size=int(max_samples), replace=False)
    return sorted(int(x) for x in choice.tolist())


_VIZ_SAMPLE_PNG_RE = re.compile(r"^sample_(\d+)(?:_.+)?\.png$")


def _extract_sample_id_from_viz_png(path: Path) -> int | None:
    match = _VIZ_SAMPLE_PNG_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _viz_subdirs_for_args(args: argparse.Namespace) -> list[str]:
    subdirs: list[str] = []
    if not args.spectral_only:
        subdirs.extend(["raw", "raw/unmatched_nodes", "raw/resolved_unmatched_nodes"])
    if not args.no_spectral:
        operators = _spectral_operators_from_args(args)
        for operator in operators:
            subdirs.append(_spectral_output_subdir(operator))
    return subdirs


def _purge_dataset_viz_outputs(
    out_root: Path,
    dataset: str,
    splits: list[str],
    *,
    ids_by_split: dict[str, list[int]],
    args: argparse.Namespace,
) -> int:
    """Drop stale figure PNGs so reruns with a fixed seed do not accumulate old samples."""
    keep_ids: set[int] = set()
    for sample_ids in ids_by_split.values():
        keep_ids.update(int(sample_id) for sample_id in sample_ids)

    removed = 0
    for split in splits:
        for subdir in _viz_subdirs_for_args(args):
            viz_dir = out_root / dataset / split / subdir
            if not viz_dir.is_dir():
                continue
            for path in sorted(viz_dir.glob("*.png")):
                sample_id = _extract_sample_id_from_viz_png(path)
                if args.overwrite or sample_id is None or sample_id not in keep_ids:
                    path.unlink()
                    removed += 1
    if removed:
        print(f"  purged {removed} stale figure(s) under {out_root / dataset}", flush=True)
    return removed


def _ginot_sample_to_viz(
    tensor_sample: dict,
    raw,
    idx: int,
    metadata: dict,
    *,
    perimeter_edges: bool,
) -> dict:
    pos_np = metadata["pos_normalizer"].decode(tensor_sample["pos"]).detach().cpu().numpy()
    y = metadata["y_normalizer"].decode(tensor_sample["y"]) if raw.normalize_targets else tensor_sample["y"]
    if is_bumper_beam_raw(raw):
        # Position channels are min-max encoded via pos_normalizer; restore physical xyz for plots.
        y = decode_bumper_beam_target(y, metadata["pos_normalizer"])
    y_np = y.detach().cpu().numpy()
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    try:
        raw_pos = _as_float_array(raw.query_points[idx], dims=raw.space_dim)
        if raw_pos.shape == pos_np.shape:
            pos_np = raw_pos
    except (FileNotFoundError, IndexError, TypeError, RuntimeError):
        pass
    try:
        cells_np = _normalize_cells(_select_cells(raw, idx), num_nodes=int(pos_np.shape[0]))
    except (FileNotFoundError, IndexError, TypeError, RuntimeError, ValueError):
        cells_np = None
    if y_np.shape[0] != pos_np.shape[0]:
        try:
            y_np = _as_float_array(raw.targets[idx])
            if y_np.ndim == 1:
                y_np = y_np[:, None]
        except (FileNotFoundError, IndexError, TypeError, RuntimeError):
            pass
    out = {
        "pos": pos_np.astype(np.float32, copy=False),
        "y": y_np.astype(np.float32, copy=False),
        "cells": np.empty((0, 0), dtype=np.int64) if cells_np is None else cells_np,
        "sample_id": int(idx),
        "perimeter_edges": bool(perimeter_edges),
        "pos_encoded": tensor_sample["pos"].detach().cpu().numpy().astype(np.float32, copy=False),
    }
    if "edge_index" in tensor_sample:
        out["edge_index"] = tensor_sample["edge_index"]
    if "laplacian_eig" in tensor_sample:
        out["laplacian_eig"] = tensor_sample["laplacian_eig"]
    if "laplacian_eigvals" in tensor_sample:
        out["laplacian_eigvals"] = tensor_sample["laplacian_eigvals"]
    return out


def _micro_puc_fixed_stats_path(data_root: Path) -> Path:
    return data_root / "PeriodUnitCell_fixed" / "processing_stats.jsonl"


def _load_micro_puc_fixed_all_stats(data_root: Path) -> dict[int, dict]:
    stats_path = _micro_puc_fixed_stats_path(data_root)
    if not stats_path.exists():
        return {}
    stats: dict[int, dict] = {}
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            stats[int(item["sample_id"])] = item
    return stats


def _load_micro_puc_fixed_issue_stats(data_root: Path) -> dict[int, dict]:
    return {
        sample_id: item
        for sample_id, item in _load_micro_puc_fixed_all_stats(data_root).items()
        if (
            int(item.get("x1_unmatched_nodes", 0)) > 0
            or int(item.get("y1_unmatched_nodes", 0)) > 0
            or int(item.get("self_loops_after_remap", 0)) > 0
        )
    }


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_micro_puc_fixed_original_issue_meshes(data_root: Path, issue_stats: dict[int, dict]) -> dict[int, dict]:
    if not issue_stats:
        return {}
    src_root = data_root / "PeriodUnitCell"
    coords_path = src_root / "mesh_coords.pkl"
    cells_path = src_root / "mesh_cells10K.pkl"
    if not coords_path.exists() or not cells_path.exists():
        return {}
    coords = _load_pickle(coords_path)
    cells = _load_pickle(cells_path)
    out: dict[int, dict] = {}
    for sample_id, stats in issue_stats.items():
        canonical_row = int(stats["canonical_row"])
        if sample_id >= len(coords) or canonical_row >= len(cells):
            continue
        pos = _as_float_array(coords[int(sample_id)], dims=2).astype(np.float32, copy=False)
        raw_cells = _normalize_cells(cells[canonical_row], num_nodes=int(pos.shape[0]))
        out[int(sample_id)] = {
            "pos": pos,
            "cells": np.empty((0, 0), dtype=np.int64) if raw_cells is None else raw_cells,
        }
    return out


def _micro_puc_fixed_issue_title(stats: dict | None) -> str:
    if not stats:
        return ""
    return (
        f"\nfix stats: x1={int(stats.get('x1_unmatched_nodes', 0))} "
        f"y1={int(stats.get('y1_unmatched_nodes', 0))} loops={int(stats.get('self_loops_after_remap', 0))}"
        f"\nsrc={int(stats.get('source_geometry_id', -1))} canonical={int(stats.get('canonical_row', -1))}"
    )


def _pyg_graph_to_viz_sample(graph, metadata: dict) -> dict:
    y_norm = metadata["y_normalizer"]
    y_np = y_norm.decode(graph.y).detach().cpu().numpy()
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    cells = getattr(graph, "cells", None)
    if cells is not None and torch.is_tensor(cells):
        cells_np = cells.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        cells_np = np.empty((0, 0), dtype=np.int64)
    sample_id = int(getattr(graph, "sample_id", graph.metadata.get("sample_index", 0)))
    out = {
        "pos": graph.pos.detach().cpu().numpy().astype(np.float32, copy=False),
        "y": y_np.astype(np.float32, copy=False),
        "cells": cells_np,
        "sample_id": sample_id,
        "perimeter_edges": False,
        "pos_encoded": graph.x.detach().cpu().numpy().astype(np.float32, copy=False),
        "edge_index": graph.edge_index.detach().cpu(),
    }
    lap_eig = getattr(graph, "laplacian_eig", None)
    if lap_eig is not None:
        out["laplacian_eig"] = lap_eig.detach().cpu() if torch.is_tensor(lap_eig) else lap_eig
    lap_vals = getattr(graph, "laplacian_eigvals", None)
    if lap_vals is not None:
        if torch.is_tensor(lap_vals):
            out["laplacian_eigvals"] = lap_vals.detach().cpu().reshape(-1)
        else:
            out["laplacian_eigvals"] = np.asarray(lap_vals).reshape(-1)
    return out


def _load_ginot_split_datasets(
    dataset: str,
    data_root: Path,
    seed: int,
    max_samples: int,
    *,
    laplacian_eig_dim: int,
    laplacian_spec: str,
):
    del max_samples
    return load_ginot_dataset(
        dataset_name=dataset,
        data_root=str(data_root),
        split_seed=seed,
        include_edges=True,
        laplacian_eig_dim=int(laplacian_eig_dim),
        laplacian_spec=str(laplacian_spec),
        bucketed_batches=False,
        build_collate_metadata=False,
    )


def _load_selected_split_samples(
    ginot_ds,
    raw,
    metadata: dict,
    sample_ids: list[int],
    index_map: dict[int, int],
    *,
    perimeter_edges: bool,
    issue_stats: dict[int, dict] | None = None,
    original_issue_meshes: dict[int, dict] | None = None,
) -> list[dict]:
    samples = []
    for idx in sample_ids:
        sample = _ginot_sample_to_viz(
            ginot_ds[index_map[int(idx)]],
            raw,
            int(idx),
            metadata,
            perimeter_edges=perimeter_edges,
        )
        if issue_stats and int(idx) in issue_stats:
            sample["micro_puc_fixed_issue_stats"] = issue_stats[int(idx)]
        if original_issue_meshes and int(idx) in original_issue_meshes:
            sample["micro_puc_fixed_original_mesh"] = original_issue_meshes[int(idx)]
        samples.append(sample)
    return samples


def _merge_laplacian_from_dataloader(viz_sample: dict, tensor_sample: dict) -> dict:
    merged = dict(viz_sample)
    if "laplacian_eig" in tensor_sample:
        merged["laplacian_eig"] = tensor_sample["laplacian_eig"]
    if "laplacian_eigvals" in tensor_sample:
        merged["laplacian_eigvals"] = tensor_sample["laplacian_eigvals"]
    if "edge_index" in tensor_sample:
        merged["edge_index"] = tensor_sample["edge_index"]
    return merged


def load_dataset_samples(
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
    if dataset in MESH_STATIC_PLAID_VIZ_DATASETS:
        return load_mesh_static_plaid_dataset_samples(
            dataset,
            data_root,
            seed,
            max_samples,
            splits,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            use_sdf_features=use_sdf_features,
        )
    print(
        "  loading GINOT dataloader (geometry via full graph cache)...",
        flush=True,
    )
    train_ds, test_ds, metadata = _load_ginot_split_datasets(
        dataset,
        data_root,
        seed,
        max_samples,
        laplacian_eig_dim=0,
        laplacian_spec="graph",
    )
    print(f"  dataloader ready (train={len(train_ds)}, test={len(test_ds) if test_ds is not None else 0})", flush=True)
    raw = train_ds.raw
    perimeter_edges = _is_micro_puc_raw(raw) or _is_micro_puc_fixed_raw(raw)
    issue_stats = _load_micro_puc_fixed_issue_stats(data_root) if dataset == "micro_puc_fixed" else {}
    original_issue_meshes = (
        _load_micro_puc_fixed_original_issue_meshes(data_root, issue_stats) if dataset == "micro_puc_fixed" else {}
    )
    split_datasets: dict[str, object] = {"train": train_ds}
    if test_ds is not None:
        split_datasets["test"] = test_ds
    split_samples: dict[str, list[dict]] = {}
    ids_by_split: dict[str, list[int]] = {}
    index_maps: dict[str, dict[int, int]] = {}
    for split in splits:
        if split not in split_datasets:
            raise ValueError(f"Split {split!r} is not available for dataset {dataset!r}.")
        ginot_ds = split_datasets[split]
        index_map = {int(idx): item for item, idx in enumerate(ginot_ds.indices)}
        index_maps[split] = index_map
        chosen_ids = random_sample_ids(
            [int(idx) for idx in ginot_ds.indices],
            max_samples=max_samples,
            seed=seed * 1009 + (17 if split == "train" else 53),
        )
        ids_by_split[split] = chosen_ids
        split_samples[split] = _load_selected_split_samples(
            ginot_ds,
            raw,
            metadata,
            chosen_ids,
            index_map,
            perimeter_edges=perimeter_edges,
            issue_stats=issue_stats,
            original_issue_meshes=original_issue_meshes,
        )
    return (
        split_samples,
        list(raw.target_fields),
        {
            "num_samples": _num_indexable_samples(raw),
            "ids_by_split": ids_by_split,
            "index_maps": index_maps,
            "data_root": data_root,
            "seed": seed,
            "max_samples": max_samples,
            "dataset": dataset,
            "issue_stats": issue_stats,
            "original_issue_meshes": original_issue_meshes,
            "y_normalizer": metadata["y_normalizer"],
            "graph_cache_dirs": {
                split_name: getattr(split_datasets[split_name], "graph_cache_dir", None) for split_name in split_datasets
            },
        },
        split_datasets,
    )


def load_micro_puc_fixed_issue_samples(
    split_datasets: dict[str, object],
    meta: dict,
    metadata: dict,
) -> dict[str, list[dict]]:
    issue_stats = dict(meta.get("issue_stats") or {})
    original_issue_meshes = dict(meta.get("original_issue_meshes") or {})
    if not issue_stats:
        return {}
    train_ds = split_datasets["train"]
    raw = train_ds.raw
    perimeter_edges = _is_micro_puc_raw(raw) or _is_micro_puc_fixed_raw(raw)
    issue_samples: dict[str, list[dict]] = {}
    for split in meta["index_maps"]:
        ginot_ds = split_datasets[split]
        index_map = meta["index_maps"][split]
        selected = sorted(sample_id for sample_id in issue_stats if sample_id in index_map)
        if not selected:
            continue
        issue_samples[split] = _load_selected_split_samples(
            ginot_ds,
            raw,
            metadata,
            selected,
            index_map,
            perimeter_edges=perimeter_edges,
            issue_stats=issue_stats,
            original_issue_meshes=original_issue_meshes,
        )
    return issue_samples


def load_micro_puc_fixed_selected_samples(
    split_datasets: dict[str, object],
    meta: dict,
    metadata: dict,
    selected_stats: dict[int, dict],
) -> dict[str, list[dict]]:
    if not selected_stats:
        return {}
    train_ds = split_datasets["train"]
    raw = train_ds.raw
    perimeter_edges = _is_micro_puc_raw(raw) or _is_micro_puc_fixed_raw(raw)
    original_meshes = _load_micro_puc_fixed_original_issue_meshes(Path(meta["data_root"]), selected_stats)
    selected_samples: dict[str, list[dict]] = {}
    for split in meta["index_maps"]:
        ginot_ds = split_datasets[split]
        index_map = meta["index_maps"][split]
        selected = sorted(sample_id for sample_id in selected_stats if sample_id in index_map)
        if not selected:
            continue
        selected_samples[split] = _load_selected_split_samples(
            ginot_ds,
            raw,
            metadata,
            selected,
            index_map,
            perimeter_edges=perimeter_edges,
            issue_stats=selected_stats,
            original_issue_meshes=original_meshes,
        )
    return selected_samples


def _coerce_sample_dict_numpy(sample: dict) -> dict:
    out: dict = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            out[key] = value
        else:
            out[key] = value
    return out


def _fetch_spectral_sample_from_dataloader(
    base: dict,
    ginot_ds,
    item: int,
    *,
    dataset: str,
    split: str,
    sample_id: int,
    operator: str,
    laplacian_spec: str,
) -> dict:
    try:
        tensor_sample = ginot_ds[item]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing Laplacian cache for {dataset} split={split} sample_id={sample_id} "
            f"operator={operator!r} (spec={laplacian_spec!r}). "
            "Run python -m pdebench.dataset.laplacian.precompute with matching --laplacian-specs."
        ) from exc
    return _merge_laplacian_from_dataloader(base, tensor_sample)


_SPECTRAL_LOAD_WORKER_STATE: dict[str, object] = {}


def _init_spectral_load_worker(
    dataset: str,
    data_root: str,
    seed: int,
    max_samples: int,
    operator: str,
    spectral_modes: int,
    laplacian_spec: str,
) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)
    train_ds, test_ds, _ = _load_ginot_split_datasets(
        dataset,
        Path(data_root),
        seed,
        max_samples,
        laplacian_eig_dim=int(spectral_modes),
        laplacian_spec=str(laplacian_spec),
    )
    split_datasets: dict[str, object] = {"train": train_ds}
    if test_ds is not None:
        split_datasets["test"] = test_ds
    _SPECTRAL_LOAD_WORKER_STATE.clear()
    _SPECTRAL_LOAD_WORKER_STATE["dataset"] = dataset
    _SPECTRAL_LOAD_WORKER_STATE["operator"] = operator
    _SPECTRAL_LOAD_WORKER_STATE["laplacian_spec"] = laplacian_spec
    _SPECTRAL_LOAD_WORKER_STATE["split_datasets"] = split_datasets


def _load_spectral_sample_task(task: tuple[str, dict, int, int]) -> tuple[str, dict]:
    split, base, item, sample_id = task
    split_datasets = _SPECTRAL_LOAD_WORKER_STATE["split_datasets"]
    ginot_ds = split_datasets[split]
    return split, _fetch_spectral_sample_from_dataloader(
        base,
        ginot_ds,
        item,
        dataset=str(_SPECTRAL_LOAD_WORKER_STATE["dataset"]),
        split=split,
        sample_id=sample_id,
        operator=str(_SPECTRAL_LOAD_WORKER_STATE["operator"]),
        laplacian_spec=str(_SPECTRAL_LOAD_WORKER_STATE["laplacian_spec"]),
    )


def load_spectral_samples_for_operator(
    dataset: str,
    data_root: Path,
    seed: int,
    max_samples: int,
    splits: list[str],
    split_samples: dict[str, list[dict]],
    index_maps: dict[str, dict[int, int]],
    operator: str,
    spectral_modes: int,
    *,
    load_workers: int = 1,
) -> dict[str, list[dict]]:
    if dataset in MESH_STATIC_PLAID_VIZ_DATASETS:
        return {split: list(split_samples[split]) for split in splits}
    laplacian_spec = f"{operator}:{int(spectral_modes)}"
    enriched: dict[str, list[dict]] = {split: [] for split in splits}
    tasks: list[tuple[str, dict, int, int]] = []
    for split in splits:
        index_map = index_maps[split]
        for base in split_samples[split]:
            sample_id = int(base["sample_id"])
            item = index_map[sample_id]
            tasks.append((split, base, item, sample_id))
    if int(load_workers) <= 1 or len(tasks) <= 1:
        train_ds, test_ds, _ = _load_ginot_split_datasets(
            dataset,
            data_root,
            seed,
            max_samples,
            laplacian_eig_dim=int(spectral_modes),
            laplacian_spec=laplacian_spec,
        )
        split_datasets: dict[str, object] = {"train": train_ds}
        if test_ds is not None:
            split_datasets["test"] = test_ds
        for split, base, item, sample_id in tasks:
            ginot_ds = split_datasets[split]
            enriched[split].append(
                _fetch_spectral_sample_from_dataloader(
                    base,
                    ginot_ds,
                    item,
                    dataset=dataset,
                    split=split,
                    sample_id=sample_id,
                    operator=operator,
                    laplacian_spec=laplacian_spec,
                )
            )
        return enriched

    initargs = (
        dataset,
        str(data_root),
        int(seed),
        int(max_samples),
        operator,
        int(spectral_modes),
        laplacian_spec,
    )
    chunksize = max(1, len(tasks) // (int(load_workers) * 4))
    effective_load_workers = max(1, min(int(load_workers), len(tasks), 8))
    with ProcessPoolExecutor(
        max_workers=effective_load_workers,
        initializer=_init_spectral_load_worker,
        initargs=initargs,
    ) as pool:
        for split, sample in pool.map(_load_spectral_sample_task, tasks, chunksize=chunksize):
            enriched[split].append(sample)
    return enriched


def maybe_subsample(num_points: int, max_points: int, seed: int) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)
    return np.sort(np.random.default_rng(seed).choice(num_points, size=max_points, replace=False))


def triangulate_cells(cells: np.ndarray) -> np.ndarray:
    if cells.size == 0 or cells.ndim != 2 or cells.shape[1] < 3:
        return np.empty((0, 3), dtype=np.int64)
    cells = cells[np.all(cells >= 0, axis=1)]
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if cells.shape[1] >= 3 and np.all(cells[:, 0] == cells.shape[1] - 1):
        cells = cells[:, 1:]
    if cells.shape[1] == 3:
        return cells.astype(np.int64, copy=False)
    return np.concatenate([cells[:, [0, i, i + 1]] for i in range(1, cells.shape[1] - 1)], axis=0).astype(np.int64, copy=False)


def edge_segments_from_cells(pos: np.ndarray, cells: np.ndarray, max_edges: int = 25000) -> np.ndarray:
    if cells.size == 0 or cells.ndim != 2 or cells.shape[1] < 2:
        return np.empty((0, 2, pos.shape[1]), dtype=np.float32)
    cells = cells[np.all(cells >= 0, axis=1)]
    if cells.shape[1] >= 3 and np.all(cells[:, 0] == cells.shape[1] - 1):
        cells = cells[:, 1:]
    edges = []
    for i, j in combinations(range(cells.shape[1]), 2):
        pair = cells[:, [i, j]]
        pair = pair[(pair[:, 0] < pos.shape[0]) & (pair[:, 1] < pos.shape[0]) & (pair[:, 0] != pair[:, 1])]
        if len(pair):
            edges.append(np.sort(pair, axis=1))
    if not edges:
        return np.empty((0, 2, pos.shape[1]), dtype=np.float32)
    edge_idx = np.unique(np.concatenate(edges, axis=0), axis=0)
    if len(edge_idx) > max_edges:
        choice = np.sort(np.random.default_rng(0).choice(len(edge_idx), size=max_edges, replace=False))
        edge_idx = edge_idx[choice]
    return pos[edge_idx]


def edge_segments_from_cells_2d(pos: np.ndarray, cells: np.ndarray, max_edges: int = 60000) -> np.ndarray:
    if cells.size == 0 or cells.ndim != 2 or cells.shape[1] < 2:
        return np.empty((0, 2, 2), dtype=np.float32)
    cells = cells[np.all(cells >= 0, axis=1)]
    if cells.size == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    if cells.shape[1] >= 3 and np.all(cells[:, 0] == cells.shape[1] - 1):
        cells = cells[:, 1:]
    if cells.shape[1] == 2:
        edge_pairs = [cells[:, [0, 1]]]
    else:
        edge_pairs = [cells[:, [i, (i + 1) % cells.shape[1]]] for i in range(cells.shape[1])]
    edge_idx = np.concatenate(edge_pairs, axis=0)
    edge_idx = edge_idx[(edge_idx[:, 0] < pos.shape[0]) & (edge_idx[:, 1] < pos.shape[0]) & (edge_idx[:, 0] != edge_idx[:, 1])]
    if edge_idx.size == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    edge_idx = np.unique(np.sort(edge_idx, axis=1), axis=0)
    if len(edge_idx) > max_edges:
        choice = np.sort(np.random.default_rng(0).choice(len(edge_idx), size=max_edges, replace=False))
        edge_idx = edge_idx[choice]
    return pos[edge_idx, :2]


def set_3d_equalish(ax, pos: np.ndarray, *, margin_frac: float = 1.0):
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = float(margin_frac) * max(0.5 * float(np.max(maxs - mins)), 1e-12)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def panel_grid(num_panels: int) -> tuple[int, int]:
    ncols = int(math.ceil(math.sqrt(num_panels)))
    nrows = int(math.ceil(num_panels / ncols))
    return nrows, ncols


def axis_span_title(pos: np.ndarray) -> str:
    labels = ["x", "y", "z"]
    return "  ".join(f"{labels[i]}:[{pos[:, i].min():.4g}, {pos[:, i].max():.4g}]" for i in range(pos.shape[1]))


def add_2d_geometry(ax, pos: np.ndarray, cells: np.ndarray, title: str, point_size: float):
    try:
        segments = edge_segments_from_cells_2d(pos, cells)
        if len(segments):
            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.12, alpha=0.75))
            ax.autoscale_view()
        else:
            ax.scatter(pos[:, 0], pos[:, 1], s=point_size, c="black", alpha=0.65)
    except Exception as exc:
        ax.scatter(pos[:, 0], pos[:, 1], s=point_size, c="black", alpha=0.65)
        title += f"\nmesh failed: {type(exc).__name__}"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def add_2d_field(ax, pos: np.ndarray, cells: np.ndarray, values: np.ndarray, title: str, point_size: float):
    if np.all(np.isnan(values)):
        ax.scatter(pos[:, 0], pos[:, 1], s=point_size, c="0.75", alpha=0.35)
        ax.set_title(f"{title}\n(unlabeled)")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    try:
        triangles = triangulate_cells(cells)
        if len(triangles) and values.shape[0] == pos.shape[0]:
            artist = ax.tripcolor(
                mtri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles), values, shading="flat", cmap="viridis"
            )
        else:
            artist = ax.scatter(pos[:, 0], pos[:, 1], c=values, s=point_size, cmap="viridis")
    except Exception:
        artist = ax.scatter(pos[:, 0], pos[:, 1], c=values, s=point_size, cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(artist, ax=ax, fraction=0.04, pad=0.02)


def set_micro_puc_fixed_2d_limits(ax):
    lo, hi = MICRO_PUC_FIXED_RAW_LIMITS
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def apply_micro_puc_fixed_2d_limits(axes, dataset: str, is_3d: bool) -> None:
    if dataset != "micro_puc_fixed" or is_3d:
        return
    for ax in np.array(axes, dtype=object).reshape(-1):
        if ax.axison:
            set_micro_puc_fixed_2d_limits(ax)


def add_3d_geometry(ax, pos: np.ndarray, cells: np.ndarray, title: str, point_size: float):
    try:
        segments = edge_segments_from_cells(pos, cells)
        if len(segments):
            ax.add_collection3d(Line3DCollection(segments, colors="black", linewidths=0.08, alpha=0.35))
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=max(point_size * 0.4, 0.2), c="black", alpha=0.25, depthshade=False)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=point_size, c="black", alpha=0.55, depthshade=False)
    except Exception as exc:
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=point_size, c="black", alpha=0.55, depthshade=False)
        title += f"\nmesh failed: {type(exc).__name__}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_3d_equalish(ax, pos)


def add_3d_field(ax, pos: np.ndarray, values: np.ndarray, title: str, point_size: float):
    if np.all(np.isnan(values)):
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=point_size, c="0.75", alpha=0.35, depthshade=False)
        ax.set_title(f"{title}\n(unlabeled)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_3d_equalish(ax, pos)
        return
    artist = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=values, s=point_size, cmap="viridis", depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_3d_equalish(ax, pos)
    plt.colorbar(artist, ax=ax, fraction=0.04, pad=0.02)


def _deform_plate_node_colors(node_type: np.ndarray) -> np.ndarray:
    colors = np.zeros((int(node_type.shape[0]), 3), dtype=np.float32)
    colors[node_type == NORMAL] = np.array([0.20, 0.45, 0.95], dtype=np.float32)
    colors[node_type == HANDLE] = np.array([0.95, 0.55, 0.10], dtype=np.float32)
    colors[node_type == OBSTACLE] = np.array([0.90, 0.15, 0.15], dtype=np.float32)
    return colors


def add_3d_colored_geometry(
    ax,
    pos: np.ndarray,
    cells: np.ndarray,
    colors: np.ndarray,
    title: str,
    point_size: float,
):
    try:
        segments = edge_segments_from_cells(pos, cells)
        if len(segments):
            ax.add_collection3d(Line3DCollection(segments, colors="black", linewidths=0.08, alpha=0.25))
    except Exception:
        pass
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=point_size, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_3d_equalish(ax, pos)


def visualize_deform_plate_sample(
    sample: dict,
    raw,
    split: str,
    out_path: Path,
    args: argparse.Namespace,
):
    sample_id = int(sample["sample_id"])
    eq = load_deform_plate_equilibrium_arrays(raw, sample_id)
    x0 = np.asarray(eq["x0"], dtype=np.float32)
    u_prescribed = np.asarray(eq["u_prescribed"], dtype=np.float32)
    u_target = np.asarray(eq["u_target"], dtype=np.float32)
    cells = np.asarray(eq["cells"], dtype=np.int64)
    node_type = np.asarray(eq["node_type"], dtype=np.int64).reshape(-1)
    free_mask = np.asarray(eq["free_mask"], dtype=bool)
    colors = _deform_plate_node_colors(node_type)

    idx = maybe_subsample(x0.shape[0], args.max_points, seed=max(sample_id, 0))
    pos0 = x0[idx]
    pos_prescribed = (x0 + u_prescribed)[idx]
    pos_final = (x0 + u_target)[idx]
    cells_plot = cells if len(idx) == x0.shape[0] else np.empty((0, 0), dtype=np.int64)
    colors_plot = colors[idx]

    disp_mag = np.linalg.norm(u_target, axis=-1)
    disp_mag = np.where(free_mask, disp_mag, np.nan)
    disp_plot = disp_mag[idx]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.5), dpi=args.dpi, subplot_kw={"projection": "3d"})
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    add_3d_colored_geometry(
        axes_arr[0],
        pos0,
        cells_plot,
        colors_plot,
        "initial configuration x0\nblue=free, orange=handle, red=actuator",
        args.point_size,
    )
    add_3d_colored_geometry(
        axes_arr[1],
        pos_prescribed,
        cells_plot,
        colors_plot,
        "prescribed BC pose x0 + u_prescribed",
        args.point_size,
    )
    add_3d_colored_geometry(
        axes_arr[2],
        pos_final,
        cells_plot,
        colors_plot,
        f"loaded quasi-static equilibrium x0 + u_target (step {TARGET_STEP})",
        args.point_size,
    )
    add_3d_field(
        axes_arr[3],
        pos_final,
        disp_plot,
        "||u_target|| on free nodes",
        args.point_size,
    )
    if len(idx) < x0.shape[0]:
        fig.text(0.5, 0.01, f"plotting {len(idx)}/{x0.shape[0]} nodes", ha="center", fontsize=9)
    fig.suptitle(
        f"deform_plate | {split} | sample_id={sample_id} | nodes={x0.shape[0]} | tets={cells.shape[0]}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _bumper_beam_deformed_positions(y_full: np.ndarray, target_fields: list[str], time: int) -> np.ndarray:
    """Assemble deformed xyz at ``time`` from bumper_beam target channels."""
    ix = _field_index(target_fields, f"position_x_t{time}")
    iy = _field_index(target_fields, f"position_y_t{time}")
    iz = _field_index(target_fields, f"position_z_t{time}")
    if ix is None or iy is None or iz is None:
        raise KeyError(f"bumper_beam missing position channels for t={time}")
    return np.stack((y_full[:, ix], y_full[:, iy], y_full[:, iz]), axis=1)


def visualize_bumper_beam_sample(
    sample: dict,
    target_fields: list[str],
    split: str,
    out_path: Path,
    args: argparse.Namespace,
):
    """Raw bumper_beam panels: query mesh, then per-time deformed mesh + strain/stress."""
    pos_full = np.asarray(sample["pos"], dtype=np.float32)
    y_full = np.asarray(sample["y"], dtype=np.float32)
    cells = sample["cells"]
    sample_id = int(sample["sample_id"])
    idx = maybe_subsample(pos_full.shape[0], args.max_points, seed=max(sample_id, 0))
    cells_for_plot = cells if len(idx) == pos_full.shape[0] else np.empty((0, 0), dtype=np.int64)

    times = [int(t) for t in BUMPER_BEAM_TARGET_TIMES]
    deformed_full = {time: _bumper_beam_deformed_positions(y_full, target_fields, time) for time in times}
    limits_pos = np.concatenate([pos_full, *[deformed_full[time] for time in times]], axis=0)

    nrows = 1 + len(times)
    ncols = 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.0 * nrows),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"},
    )
    axes_arr = np.asarray(axes, dtype=object)

    query = pos_full[idx]
    geom_title = "query mesh (t=0)\n" + axis_span_title(pos_full)
    if len(idx) < pos_full.shape[0]:
        geom_title += f"\nplotting {len(idx)}/{pos_full.shape[0]} points"
    add_3d_geometry(axes_arr[0, 0], query, cells_for_plot, geom_title, args.point_size)
    set_3d_equalish(axes_arr[0, 0], limits_pos)
    axes_arr[0, 1].set_axis_off()
    axes_arr[0, 2].set_axis_off()

    for row, time in enumerate(times, start=1):
        pos_t_full = deformed_full[time]
        pos_t = pos_t_full[idx]
        mesh_title = f"mesh t={time}\n" + axis_span_title(pos_t_full)
        add_3d_geometry(axes_arr[row, 0], pos_t, cells_for_plot, mesh_title, args.point_size)
        set_3d_equalish(axes_arr[row, 0], limits_pos)

        for col, field in enumerate(
            (f"effective_plastic_strain_t{time}", f"stress_vm_t{time}"),
            start=1,
        ):
            field_idx = _field_index(target_fields, field)
            if field_idx is None:
                axes_arr[row, col].text(0.5, 0.5, f"{field} unavailable", ha="center", va="center")
                axes_arr[row, col].set_axis_off()
                continue
            values = y_full[idx, field_idx].reshape(-1)
            add_3d_field(axes_arr[row, col], pos_t, values, field, args.point_size)
            set_3d_equalish(axes_arr[row, col], limits_pos)

    fig.suptitle(
        f"bumper_beam | {split} | sample_id={sample_id} | points={pos_full.shape[0]} | times={times}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.01, 1.0, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def visualize_sample(
    sample: dict, target_fields: list[str], dataset: str, split: str, out_path: Path, args: argparse.Namespace
):
    pos_full = sample["pos"]
    y_full = sample["y"]
    cells = sample["cells"]
    sample_id = int(sample["sample_id"])
    idx = maybe_subsample(pos_full.shape[0], args.max_points, seed=max(sample_id, 0))
    pos = pos_full[idx]
    y = y_full[idx]
    cells_for_plot = cells if len(idx) == pos_full.shape[0] else np.empty((0, 0), dtype=np.int64)
    is_3d = pos.shape[1] >= 3
    num_panels = 1 + len(target_fields)
    nrows, ncols = panel_grid(num_panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.4 * nrows),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"} if is_3d else None,
    )
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    geom_title = "query mesh\n" + axis_span_title(pos_full)
    geom_title += _micro_puc_fixed_issue_title(sample.get("micro_puc_fixed_issue_stats"))
    if len(idx) < pos_full.shape[0]:
        geom_title += f"\nplotting {len(idx)}/{pos_full.shape[0]} points"
    if is_3d:
        add_3d_geometry(axes_arr[0], pos, cells_for_plot, geom_title, args.point_size)
    else:
        add_2d_geometry(axes_arr[0], pos, cells_for_plot, geom_title, args.point_size)
    for field_idx, field in enumerate(target_fields):
        values = y[:, field_idx].reshape(-1)
        if is_3d:
            add_3d_field(axes_arr[field_idx + 1], pos, values, field, args.point_size)
        else:
            add_2d_field(axes_arr[field_idx + 1], pos, cells_for_plot, values, field, args.point_size)
    for ax in axes_arr[num_panels:]:
        ax.set_axis_off()
    apply_micro_puc_fixed_2d_limits(axes_arr[:num_panels], dataset, is_3d)
    fig.suptitle(
        f"{dataset} | {split} | sample_id={sample_id} | points={pos_full.shape[0]} | dim={pos_full.shape[1]}", fontsize=12
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _field_index(target_fields: list[str], field: str) -> int | None:
    try:
        return list(target_fields).index(field)
    except ValueError:
        return None


def visualize_micro_puc_fixed_issue_sample(
    sample: dict,
    target_fields: list[str],
    split: str,
    out_path: Path,
    args: argparse.Namespace,
    *,
    status: str = "unmatched-node mesh fix",
):
    original = sample.get("micro_puc_fixed_original_mesh")
    if original is None:
        visualize_sample(sample, target_fields, "micro_puc_fixed", split, out_path, args)
        return

    sample_id = int(sample["sample_id"])
    fixed_pos = sample["pos"]
    fixed_y = sample["y"]
    fixed_cells = sample["cells"]
    original_pos = original["pos"]
    original_cells = original["cells"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.2), dpi=args.dpi)
    axes_arr = np.array(axes, dtype=object).reshape(-1)

    add_2d_geometry(
        axes_arr[0],
        original_pos,
        original_cells,
        "original mesh\n" + axis_span_title(original_pos),
        args.point_size,
    )
    add_2d_geometry(
        axes_arr[1],
        fixed_pos,
        fixed_cells,
        "fixed mesh\n" + axis_span_title(fixed_pos) + _micro_puc_fixed_issue_title(sample.get("micro_puc_fixed_issue_stats")),
        args.point_size,
    )

    for ax, field in zip(axes_arr[2:], ("disp_x", "disp_y")):
        idx = _field_index(target_fields, field)
        if idx is None:
            ax.text(0.5, 0.5, f"{field} unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        add_2d_field(ax, fixed_pos, fixed_cells, fixed_y[:, idx].reshape(-1), field, args.point_size)

    apply_micro_puc_fixed_2d_limits(axes_arr, "micro_puc_fixed", is_3d=False)
    fig.suptitle(
        f"micro_puc_fixed | {split} | sample_id={sample_id} | {status}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def spectral_subsample(features: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if features.shape[0] <= max_points:
        return features
    idx = np.sort(np.random.default_rng(seed).choice(features.shape[0], size=max_points, replace=False))
    return features[idx]


def _sample_edge_index(sample: dict, device: torch.device) -> torch.Tensor:
    edge_index = sample.get("edge_index")
    if edge_index is None:
        raise ValueError(
            f"sample_id={int(sample['sample_id'])} is missing graph cache edge_index; "
            "load samples with include_edges=True."
        )
    return edge_index.to(device=device)


def _sample_pos_raw(sample: dict) -> np.ndarray:
    """Physical query-point coordinates for plotting (never minmax-normalized graph-cache pos)."""
    return np.asarray(sample["pos"], dtype=np.float32)


def _sample_pos_encoded(sample: dict, device: torch.device) -> torch.Tensor:
    pos = sample.get("pos_encoded", sample["pos"])
    return torch.from_numpy(np.asarray(pos, dtype=np.float32).copy()).to(device=device, dtype=torch.float32)


def format_eigenvalue(value: float) -> str:
    return f"{float(value):.2e}"


def spectral_mode_title(mode_idx: int, eigenvalues: np.ndarray | None) -> str:
    title = f"mode {mode_idx + 1}"
    if eigenvalues is not None and mode_idx < len(eigenvalues):
        title += rf", $\lambda$={format_eigenvalue(eigenvalues[mode_idx])}"
    return title


def compute_spectral_feature(sample: dict, operator: str, modes: int, device: torch.device) -> np.ndarray:
    cached = sample.get("laplacian_eig")
    if cached is not None:
        if isinstance(cached, torch.Tensor):
            return cached[:, : int(modes)].detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(cached[:, : int(modes)], dtype=np.float32)

    edge_index = _sample_edge_index(sample, device)
    pos = _sample_pos_encoded(sample, device)
    if edge_index.numel() == 0:
        raise ValueError(f"sample_id={int(sample['sample_id'])} produced an empty mesh graph.")
    with torch.no_grad():
        feat = compute_laplacian_feature_part(edge_index, pos, None, operator, int(modes))
    return feat.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_spectral_eigenvalues(sample: dict, operator: str, modes: int, device: torch.device) -> np.ndarray:
    cached = sample.get("laplacian_eigvals")
    if cached is not None:
        if isinstance(cached, torch.Tensor):
            return cached[: int(modes)].detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(cached[: int(modes)], dtype=np.float64)

    edge_index = _sample_edge_index(sample, device)
    pos = _sample_pos_encoded(sample, device)
    if edge_index.numel() == 0:
        raise ValueError(f"sample_id={int(sample['sample_id'])} produced an empty mesh graph.")
    with torch.no_grad():
        eigenvalues, _ = compute_laplacian_eigendecomp_part(edge_index, pos, None, operator, int(modes))
    return eigenvalues.detach().cpu().numpy().astype(np.float64, copy=False)


_SPECTRAL_GRID = 8
_SPECTRAL_MODES_PER_FIGURE = _SPECTRAL_GRID * _SPECTRAL_GRID
_SPECTRAL_SUBPLOT_WSPACE = 0.05
_SPECTRAL_SUBPLOT_HSPACE = 0.05
_SPECTRAL_TITLE_PAD = -1.5
_SPECTRAL_FIG_EDGE = 0.02
_SPECTRAL_FIG_TOP = 0.92
_SPECTRAL_SUPTITLE_Y = 0.99
_SPECTRAL_TIGHT_PAD = 0.27


def _spectral_figure_specs(*, num_modes: int) -> list[tuple[str, int]]:
    chunk = _SPECTRAL_MODES_PER_FIGURE
    specs: list[tuple[str, int]] = []
    start = 0
    while start < int(num_modes):
        end = min(start + chunk, int(num_modes))
        specs.append((f"modes_{start + 1}_{end}", start))
        start += chunk
    return specs


def _style_spectral_axis(ax, *, is_3d: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if is_3d:
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
    else:
        ax.set_aspect("equal")


def _set_spectral_panel_title(ax, title: str) -> None:
    ax.set_title(title, fontsize=6, pad=_SPECTRAL_TITLE_PAD, loc="center")


def plot_spectral_overlay(
    sample: dict,
    features: np.ndarray,
    operator: str,
    dataset: str,
    split: str,
    out_path: Path,
    args: argparse.Namespace,
    *,
    mode_start: int = 0,
    eigenvalues: np.ndarray | None = None,
):
    pos_full = _sample_pos_raw(sample)
    cells = sample["cells"]
    sample_id = int(sample["sample_id"])
    nrows, ncols = _SPECTRAL_GRID, _SPECTRAL_GRID
    plots_per_figure = _SPECTRAL_MODES_PER_FIGURE
    available_modes = max(0, features.shape[1] - int(mode_start))
    num_modes = min(plots_per_figure, available_modes)
    is_3d = pos_full.shape[1] >= 3
    idx = maybe_subsample(pos_full.shape[0], args.spectral_max_points, seed=sample_id)
    pos = pos_full[idx]
    feat = features[idx]
    cells_for_plot = cells if len(idx) == pos_full.shape[0] else np.empty((0, 0), dtype=np.int64)
    axis_centers = None
    axis_radius = None
    if is_3d:
        mins = pos_full.min(axis=0)
        maxs = pos_full.max(axis=0)
        axis_centers = 0.5 * (mins + maxs)
        axis_radius = 0.92 * max(0.5 * float(np.max(maxs - mins)), 1e-12)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.75 * ncols, 1.65 * nrows),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"} if is_3d else None,
        gridspec_kw={"wspace": _SPECTRAL_SUBPLOT_WSPACE, "hspace": _SPECTRAL_SUBPLOT_HSPACE},
    )
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    for panel_idx, ax in enumerate(axes_arr):
        if panel_idx >= num_modes:
            ax.set_axis_off()
            continue
        mode_idx = int(mode_start) + panel_idx
        values = feat[:, mode_idx].reshape(-1)
        vmax = max(float(np.nanmax(np.abs(values))), 1e-6)
        title = spectral_mode_title(mode_idx, eigenvalues)
        if is_3d:
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                c=values,
                s=args.point_size,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                depthshade=False,
                rasterized=True,
            )
            ax.set_xlim(axis_centers[0] - axis_radius, axis_centers[0] + axis_radius)
            ax.set_ylim(axis_centers[1] - axis_radius, axis_centers[1] + axis_radius)
            ax.set_zlim(axis_centers[2] - axis_radius, axis_centers[2] + axis_radius)
            _style_spectral_axis(ax, is_3d=True)
            _set_spectral_panel_title(ax, title)
        else:
            try:
                triangles = triangulate_cells(cells_for_plot)
                if len(triangles):
                    ax.tripcolor(
                        mtri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles),
                        values,
                        shading="flat",
                        cmap="coolwarm",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                else:
                    ax.scatter(pos[:, 0], pos[:, 1], c=values, s=args.point_size, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            except Exception:
                ax.scatter(pos[:, 0], pos[:, 1], c=values, s=args.point_size, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            _style_spectral_axis(ax, is_3d=False)
            _set_spectral_panel_title(ax, title)
            if dataset == "micro_puc_fixed":
                set_micro_puc_fixed_2d_limits(ax)

    mode_end = int(mode_start) + num_modes
    fig.suptitle(
        f"{dataset} | {split} | sample_id={sample_id} | {operator} modes {int(mode_start) + 1}-{mode_end}",
        fontsize=7,
        y=_SPECTRAL_SUPTITLE_Y,
    )
    if is_3d:
        fig.subplots_adjust(
            left=_SPECTRAL_FIG_EDGE,
            right=1.0 - _SPECTRAL_FIG_EDGE,
            bottom=_SPECTRAL_FIG_EDGE,
            top=_SPECTRAL_FIG_TOP,
            wspace=_SPECTRAL_SUBPLOT_WSPACE,
            hspace=_SPECTRAL_SUBPLOT_HSPACE,
        )
    else:
        fig.tight_layout(
            rect=[0.0, 0.0, 1.0, 0.96],
            pad=_SPECTRAL_TIGHT_PAD,
            w_pad=_SPECTRAL_TIGHT_PAD,
            h_pad=_SPECTRAL_TIGHT_PAD,
        )
        fig.subplots_adjust(wspace=_SPECTRAL_SUBPLOT_WSPACE, hspace=_SPECTRAL_SUBPLOT_HSPACE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _spectral_operators_from_args(args: argparse.Namespace) -> list[str]:
    if args.spectral_operators:
        return [str(op) for op in args.spectral_operators]
    return [name for name, _ in parse_laplacian_spec(args.laplacian_specs, args.spectral_modes)]


def _spectral_output_subdir(operator: str) -> str:
    return f"spectral_{operator}"


@dataclass(frozen=True)
class SpectralPlotJob:
    log_line: str
    out_path: str
    sample: dict
    features: np.ndarray
    eigenvalues: np.ndarray | None
    operator: str
    dataset: str
    split: str
    mode_start: int
    dpi: int
    point_size: float
    spectral_max_points: int
    error_text: str | None = None


def _spectral_plot_args_from_job(job: SpectralPlotJob) -> SimpleNamespace:
    return SimpleNamespace(
        dpi=job.dpi,
        point_size=job.point_size,
        spectral_max_points=job.spectral_max_points,
    )


def _save_spectral_error_figure(out_path: Path | str, message: str, *, dpi: int) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.savefig(out_path)
    plt.close(fig)


def _execute_spectral_plot_job(job: SpectralPlotJob) -> str:
    import matplotlib

    matplotlib.use("Agg")
    if job.error_text is not None:
        _save_spectral_error_figure(job.out_path, job.error_text, dpi=job.dpi)
        return job.log_line
    plot_spectral_overlay(
        job.sample,
        job.features,
        job.operator,
        job.dataset,
        job.split,
        Path(job.out_path),
        _spectral_plot_args_from_job(job),
        mode_start=job.mode_start,
        eigenvalues=job.eigenvalues,
    )
    return job.log_line


def _run_spectral_plot_jobs(jobs: list[SpectralPlotJob], workers: int) -> None:
    if not jobs:
        return
    if int(workers) <= 1:
        for job in jobs:
            print(_execute_spectral_plot_job(job), flush=True)
        return
    chunksize = max(1, len(jobs) // (int(workers) * 4))
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for log_line in pool.map(_execute_spectral_plot_job, jobs, chunksize=chunksize):
            print(log_line, flush=True)


def _append_spectral_overlay_jobs(
    jobs: list[SpectralPlotJob],
    *,
    sample: dict,
    features: np.ndarray,
    eigenvalues: np.ndarray | None,
    operator: str,
    dataset: str,
    split: str,
    out_root: Path,
    args: argparse.Namespace,
    sample_id: int,
) -> None:
    figure_specs = _spectral_figure_specs(num_modes=int(features.shape[1]))
    worker_sample = _coerce_sample_dict_numpy(sample)
    for suffix, mode_start in figure_specs:
        if mode_start >= features.shape[1]:
            continue
        out_path = (
            out_root
            / dataset
            / split
            / _spectral_output_subdir(operator)
            / f"sample_{sample_id:05d}_{operator}_{suffix}.png"
        )
        if out_path.exists() and not args.overwrite:
            continue
        log_label = f"{dataset}/{split}/sample_{sample_id:05d}/{operator} ({suffix})"
        jobs.append(
            SpectralPlotJob(
                log_line=f"    spectral overlay: {log_label}",
                out_path=str(out_path),
                sample=worker_sample,
                features=np.asarray(features, dtype=np.float32),
                eigenvalues=None if eigenvalues is None else np.asarray(eigenvalues, dtype=np.float64),
                operator=operator,
                dataset=dataset,
                split=split,
                mode_start=int(mode_start),
                dpi=int(args.dpi),
                point_size=float(args.point_size),
                spectral_max_points=int(args.spectral_max_points),
            )
        )


def _append_spectral_error_job(
    jobs: list[SpectralPlotJob],
    *,
    dataset: str,
    split: str,
    out_root: Path,
    args: argparse.Namespace,
    operator: str,
    sample_id: int,
    exc: Exception,
) -> None:
    err_suffix = f"modes_1_{min(args.spectral_modes, _SPECTRAL_MODES_PER_FIGURE)}"
    filename = f"sample_{sample_id:05d}_{operator}_{err_suffix}.png"
    out_path = out_root / dataset / split / _spectral_output_subdir(operator) / filename
    if out_path.exists() and not args.overwrite:
        return
    jobs.append(
        SpectralPlotJob(
            log_line=f"    spectral overlay failed: {dataset}/{split}/{operator}",
            out_path=str(out_path),
            sample={},
            features=np.empty((0, 0), dtype=np.float32),
            eigenvalues=None,
            operator=operator,
            dataset=dataset,
            split=split,
            mode_start=0,
            dpi=int(args.dpi),
            point_size=float(args.point_size),
            spectral_max_points=int(args.spectral_max_points),
            error_text=f"{operator} failed\n{type(exc).__name__}\n{exc}",
        )
    )


def plot_spectral_features(
    dataset: str,
    split_samples: dict[str, list[dict]],
    out_root: Path,
    args: argparse.Namespace,
    meta: dict,
):
    spectral_max_samples = args.spectral_max_samples if args.spectral_max_samples is not None else args.max_samples
    workers = _resolve_spectral_workers(args.spectral_workers)
    device_name = args.spectral_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    jobs: list[SpectralPlotJob] = []
    for operator in _spectral_operators_from_args(args):
        operator_samples = load_spectral_samples_for_operator(
            str(meta["dataset"]),
            Path(meta["data_root"]),
            int(meta["seed"]),
            int(meta["max_samples"]),
            args.splits,
            split_samples,
            meta["index_maps"],
            operator,
            args.spectral_modes,
            load_workers=workers,
        )
        for split in args.splits:
            for sample in operator_samples[split][:spectral_max_samples]:
                sample_id = int(sample["sample_id"])
                try:
                    features = compute_spectral_feature(sample, operator, args.spectral_modes, device=device)
                    eigenvalues = compute_spectral_eigenvalues(sample, operator, args.spectral_modes, device=device)
                except Exception as exc:
                    _append_spectral_error_job(
                        jobs,
                        dataset=dataset,
                        split=split,
                        out_root=out_root,
                        args=args,
                        operator=operator,
                        sample_id=sample_id,
                        exc=exc,
                    )
                    continue

                _append_spectral_overlay_jobs(
                    jobs,
                    sample=sample,
                    features=features,
                    eigenvalues=eigenvalues,
                    operator=operator,
                    dataset=dataset,
                    split=split,
                    out_root=out_root,
                    args=args,
                    sample_id=sample_id,
                )

    print(
        f"  spectral: rendering {len(jobs)} figure(s) with {workers} worker(s)",
        flush=True,
    )
    _run_spectral_plot_jobs(jobs, workers)


def plot_spectral_matrix_summary(
    dataset: str,
    split_samples: dict[str, list[dict]],
    out_path: Path,
    args: argparse.Namespace,
    meta: dict,
):
    spectral_max_samples = min(args.spectral_max_samples if args.spectral_max_samples is not None else args.max_samples, 1)
    operators = _spectral_operators_from_args(args)
    rows = [(split, sample) for split in args.splits for sample in split_samples[split][:spectral_max_samples]]
    if not rows:
        return
    device_name = args.spectral_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    fig, axes = plt.subplots(
        len(rows), len(operators), figsize=(4.6 * len(operators), 2.0 * len(rows)), dpi=args.dpi, squeeze=False
    )
    operator_sample_maps: dict[str, dict[tuple[str, int], dict]] = {}
    for operator in operators:
        operator_samples = load_spectral_samples_for_operator(
            str(meta["dataset"]),
            Path(meta["data_root"]),
            int(meta["seed"]),
            int(meta["max_samples"]),
            args.splits,
            split_samples,
            meta["index_maps"],
            operator,
            args.spectral_modes,
        )
        operator_sample_maps[operator] = {
            (split, int(sample["sample_id"])): sample
            for split in args.splits
            for sample in operator_samples[split]
        }
    for row_idx, (split, base_sample) in enumerate(rows):
        sample_id = int(base_sample["sample_id"])
        for col_idx, operator in enumerate(operators):
            sample = operator_sample_maps[operator][(split, sample_id)]
            ax = axes[row_idx, col_idx]
            try:
                feat = compute_spectral_feature(sample, operator, args.spectral_modes, device=device)
                feat = spectral_subsample(feat, args.spectral_max_points, seed=sample_id)
                vmax = max(float(np.nanmax(np.abs(feat))), 1e-6)
                ax.imshow(feat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.set_title(f"{operator}:{args.spectral_modes}" if row_idx == 0 else "")
                ax.set_ylabel(f"{split} {sample_id}\nnode")
                ax.set_xlabel("mode")
            except Exception as exc:
                ax.text(0.5, 0.5, f"{operator} failed\n{type(exc).__name__}\n{exc}", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
    fig.suptitle(
        f"{dataset} spectral feature matrix sanity check | rows=samples | columns=operators | device={device}", fontsize=13
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _viz_laplacian_load_params(args: argparse.Namespace) -> tuple[int, str, bool]:
    laplacian_eig_dim = 0
    laplacian_spec = "graph"
    if not args.no_spectral:
        operators = _spectral_operators_from_args(args)
        if operators:
            parsed = parse_laplacian_spec(args.laplacian_specs, args.spectral_modes)
            for name, dim in parsed:
                if name == operators[0]:
                    laplacian_spec = name
                    laplacian_eig_dim = int(dim)
                    break
    use_sdf_features = os.environ.get("PLAID_USE_SDF_FEATURES", "false").strip().lower() in {"1", "true", "yes"}
    return laplacian_eig_dim, laplacian_spec, use_sdf_features


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.outdir.resolve()
    laplacian_eig_dim, laplacian_spec, use_sdf_features = _viz_laplacian_load_params(args)
    for dataset in args.dataset:
        print(f"\n[{dataset}] loading random {args.max_samples} sample(s) for splits={args.splits}", flush=True)
        split_samples, target_fields, meta, split_datasets = load_dataset_samples(
            dataset,
            data_root,
            args.split_seed,
            args.max_samples,
            args.splits,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            use_sdf_features=use_sdf_features,
        )
        print(f"  total_samples={meta['num_samples']} target_fields={target_fields}", flush=True)
        _purge_dataset_viz_outputs(
            out_root,
            dataset,
            args.splits,
            ids_by_split=meta["ids_by_split"],
            args=args,
        )
        raw = getattr(split_datasets["train"], "raw", None)
        for split in args.splits:
            print(f"  {split}: sample_ids={meta['ids_by_split'][split]}", flush=True)
            if args.spectral_only:
                continue
            for sample in split_samples[split]:
                out_path = out_root / dataset / split / "raw" / f"sample_{int(sample['sample_id']):05d}.png"
                if out_path.exists() and not args.overwrite:
                    continue
                if dataset == "deform_plate":
                    visualize_deform_plate_sample(sample, raw, split, out_path, args)
                elif dataset == "bumper_beam":
                    visualize_bumper_beam_sample(sample, target_fields, split, out_path, args)
                else:
                    visualize_sample(sample, target_fields, dataset, split, out_path, args)
        if dataset == "micro_puc_fixed" and not args.spectral_only:
            issue_samples = load_micro_puc_fixed_issue_samples(split_datasets, meta, {"y_normalizer": meta["y_normalizer"]})
            issue_count = sum(len(samples) for samples in issue_samples.values())
            resolved_ids = sorted(set(int(sample_id) for sample_id in args.micro_puc_fixed_resolved_samples))
            all_stats = _load_micro_puc_fixed_all_stats(data_root) if resolved_ids else {}
            resolved_stats = {sample_id: all_stats[sample_id] for sample_id in resolved_ids if sample_id in all_stats}
            resolved_samples = load_micro_puc_fixed_selected_samples(
                split_datasets,
                meta,
                {"y_normalizer": meta["y_normalizer"]},
                resolved_stats,
            )
            resolved_count = sum(len(samples) for samples in resolved_samples.values())
            if args.overwrite:
                for split in args.splits:
                    for dirname in ("unmatched_nodes", "resolved_unmatched_nodes"):
                        issue_dir = out_root / dataset / split / "raw" / dirname
                        if not issue_dir.is_dir():
                            continue
                        for stale_path in issue_dir.glob("sample_*.png"):
                            stale_path.unlink()
            if issue_count:
                print(f"  raw/unmatched_nodes: rendering {issue_count} mesh-fix issue sample(s)", flush=True)
            for split, samples in issue_samples.items():
                for sample in samples:
                    out_path = (
                        out_root / dataset / split / "raw" / "unmatched_nodes" / f"sample_{int(sample['sample_id']):05d}.png"
                    )
                    if out_path.exists() and not args.overwrite:
                        continue
                    visualize_micro_puc_fixed_issue_sample(sample, target_fields, split, out_path, args)
            if resolved_count:
                print(f"  raw/resolved_unmatched_nodes: rendering {resolved_count} resolved mesh-fix sample(s)", flush=True)
            missing_resolved = sorted(set(resolved_ids) - set(resolved_stats))
            if missing_resolved:
                print(f"  raw/resolved_unmatched_nodes: missing stats for sample_ids={missing_resolved}", flush=True)
            for split, samples in resolved_samples.items():
                for sample in samples:
                    out_path = (
                        out_root
                        / dataset
                        / split
                        / "raw"
                        / "resolved_unmatched_nodes"
                        / f"sample_{int(sample['sample_id']):05d}.png"
                    )
                    if out_path.exists() and not args.overwrite:
                        continue
                    visualize_micro_puc_fixed_issue_sample(
                        sample,
                        target_fields,
                        split,
                        out_path,
                        args,
                        status="resolved unmatched-node mesh fix",
                    )
        if not args.no_spectral:
            operators = _spectral_operators_from_args(args)
            print(
                f"  spectral: overlaying {args.spectral_modes} modes in {_SPECTRAL_GRID}x{_SPECTRAL_GRID} figures "
                f"for {operators} via GINOT dataloader (specs={args.laplacian_specs}); "
                f"workers={_resolve_spectral_workers(args.spectral_workers)}",
                flush=True,
            )
            plot_spectral_features(dataset, split_samples, out_root, args, meta)
    written_roots = sorted({str(out_root / dataset) for dataset in args.dataset})
    print("\nDone. Figures written under:")
    for root in written_roots:
        print(f"  {root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch_geometric as pyg

import pdebench

LPBF_DATASETS = frozenset({"lpbf"})
LPBF_DIRNAME = "lpbf"
LPBF_REPO_ID = "vedantpuri/LPBF_FLARE"
# Matches am.dataset.transform.DatasetTransform.pos_scale / disp_scale.
LPBF_POS_SCALE = torch.tensor([30.0, 30.0, 60.0])
LPBF_DISP_SCALE = 1.0


def _per_graph_disp_stats(train_dataset) -> tuple[torch.Tensor, torch.Tensor]:
    mean_disp = torch.zeros(1, dtype=torch.float32)
    std_disp = torch.zeros(1, dtype=torch.float32)
    for graph in train_dataset:
        disp = graph.y
        if disp.ndim == 1:
            disp = disp.unsqueeze(-1)
        mean_disp += disp.mean(dim=0).to(dtype=torch.float32)
        std_disp += disp.std(dim=0).to(dtype=torch.float32)
    n = max(len(train_dataset), 1)
    mean_disp /= n
    std_disp = (std_disp / n).clamp_min(1e-8)
    return mean_disp.reshape(1, 1), std_disp.reshape(1, 1)


def make_lpbf_y_normalizer(train_dataset) -> pdebench.UnitGaussianNormalizer:
    mean_disp, std_disp = _per_graph_disp_stats(train_dataset)
    y_normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(1, 1))
    y_normalizer.mean = mean_disp
    y_normalizer.std = std_disp
    return y_normalizer


# Pre-Jul4 LPBF metadata: rand y_normalizer; displacement stats assigned to x_normalizer.
_LPBF_LEGACY_BUGGY_NORMALIZER = True
# torch.rand(3, 1, generator=torch.Generator().manual_seed(0)) — frozen for reproducibility.
_LPBF_LEGACY_Y_NORM_RAND = torch.tensor(
    [
        [0.49625658988952637],
        [0.7682217955589294],
        [0.08847743272781372],
    ],
    dtype=torch.float32,
)
# Derived once from UnitGaussianNormalizer(_LPBF_LEGACY_Y_NORM_RAND); frozen for reproducibility.
_LPBF_LEGACY_Y_NORM = pdebench.UnitGaussianNormalizer(_LPBF_LEGACY_Y_NORM_RAND)
LPBF_LEGACY_Y_NORM_MEAN = _LPBF_LEGACY_Y_NORM.mean.clone()
LPBF_LEGACY_Y_NORM_STD = _LPBF_LEGACY_Y_NORM.std.clone()


def lpbf_legacy_y_normalizer() -> pdebench.UnitGaussianNormalizer:
    """Hardcoded pre-Jul4 LPBF y_normalizer (shared FLARE metadata)."""
    y_normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(1, 1))
    y_normalizer.mean = LPBF_LEGACY_Y_NORM_MEAN
    y_normalizer.std = LPBF_LEGACY_Y_NORM_STD
    return y_normalizer


def lpbf_ginot_y_normalizer():
    """GINOT/GLT y_normalizer: hardcoded legacy mean/std as StandardNormalizer."""
    from pdebench.dataset.ginot.types import StandardNormalizer

    return StandardNormalizer(mean=LPBF_LEGACY_Y_NORM_MEAN.clone(), std=LPBF_LEGACY_Y_NORM_STD.clone())


def lpbf_ginot_pos_normalizer():
    """GINOT pos/boundary normalizer matching FinaltimeDatasetTransform pos_scale."""
    from pdebench.dataset.ginot.types import StandardNormalizer

    return StandardNormalizer(
        mean=torch.zeros(1, 3, dtype=torch.float32),
        std=LPBF_POS_SCALE.reshape(1, 3).to(dtype=torch.float32),
    )


def lpbf_ginot_normalizers() -> tuple:
    """FLARE-compatible LPBF normalizers for the GINOT/GLT path (pos Standard, y Standard)."""
    pos_normalizer = lpbf_ginot_pos_normalizer()
    y_normalizer = lpbf_ginot_y_normalizer() if _LPBF_LEGACY_BUGGY_NORMALIZER else lpbf_y_normalizer()
    return pos_normalizer, pos_normalizer, y_normalizer


def _legacy_buggy_lpbf_normalizers(
    train_dataset,
) -> tuple[pdebench.IdentityNormalizer, pdebench.UnitGaussianNormalizer]:
    """Pre-6473f4d1 LPBF metadata: stats computed but assigned to x_normalizer, not y."""
    mean_disp = 0.0
    std_disp = 0.0
    for graph in train_dataset:
        disp = graph.y
        mean_disp += disp.mean(dim=0)
        std_disp += disp.std(dim=0)
    mean_disp /= len(train_dataset)
    std_disp /= len(train_dataset)

    x_normalizer = pdebench.IdentityNormalizer()
    y_normalizer = lpbf_legacy_y_normalizer()
    x_normalizer.mean = mean_disp
    x_normalizer.std = std_disp
    return x_normalizer, y_normalizer


def _lpbf_metadata_normalizers(train_dataset) -> tuple[pdebench.IdentityNormalizer, pdebench.UnitGaussianNormalizer]:
    if _LPBF_LEGACY_BUGGY_NORMALIZER:
        return _legacy_buggy_lpbf_normalizers(train_dataset)
    return pdebench.IdentityNormalizer(), make_lpbf_y_normalizer(train_dataset)


def lpbf_metadata_normalizers(
    train_dataset,
) -> tuple[pdebench.IdentityNormalizer, pdebench.UnitGaussianNormalizer]:
    """LPBF metadata normalizers (see _LPBF_LEGACY_BUGGY_NORMALIZER)."""
    return _lpbf_metadata_normalizers(train_dataset)


def lpbf_y_normalizer() -> pdebench.UnitGaussianNormalizer:
    """Canonical LPBF y_normalizer — same object wiring as load_lpbf_dataset metadata."""
    if _LPBF_LEGACY_BUGGY_NORMALIZER:
        return lpbf_legacy_y_normalizer()
    import am

    transform = am.FinaltimeDatasetTransform(disp=True, vmstr=False, mesh=False)
    train_dataset = create_lpbf_dataset(split="train", transform=transform)
    _x_normalizer, y_normalizer = lpbf_metadata_normalizers(train_dataset)
    del _x_normalizer
    return y_normalizer


def resolve_lpbf_batch_format(
    *,
    mixed_precision: bool,
    explicit: str | None = None,
    use_context_parallel: bool = False,
) -> str:
    """Choose LPBF FLARE batch layout: ``varlen`` (flat) or ``padded`` (+ mask).

    Context parallel is incompatible with packed flash-attn varlen, so CP forces
    the padded layout unless the caller explicitly requests ``varlen`` (error).
    """
    if explicit is not None:
        fmt = str(explicit).lower()
        if fmt not in {"padded", "varlen"}:
            raise ValueError(f"LPBF batch format must be 'padded' or 'varlen', got {explicit!r}")
        if use_context_parallel and fmt == "varlen":
            raise ValueError("LPBF varlen batch format is incompatible with context parallel")
        return fmt
    if use_context_parallel:
        return "padded"
    return "varlen" if bool(mixed_precision) else "padded"


def make_lpbf_collate_fn(batch_format: str):
    fmt = resolve_lpbf_batch_format(mixed_precision=False, explicit=batch_format)
    return lpbf_collate_varlen if fmt == "varlen" else lpbf_collate_padded


def _lpbf_as_node_matrix(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"LPBF node tensor must be [N] or [N, C]; got shape {tuple(t.shape)}")
    return t


def lpbf_collate_padded(batch: list) -> dict:
    """Pad variable-length LPBF graphs to ``[B, N_max, C]`` with a boolean mask."""
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty LPBF batch.")
    xs = [_lpbf_as_node_matrix(g.x) for g in batch]
    ys = [_lpbf_as_node_matrix(g.y) for g in batch]
    lengths = [int(x.shape[0]) for x in xs]
    max_n = max(lengths)
    c_in = int(xs[0].shape[-1])
    c_out = int(ys[0].shape[-1])
    bsz = len(batch)
    x = xs[0].new_zeros((bsz, max_n, c_in))
    y = ys[0].new_zeros((bsz, max_n, c_out))
    mask = torch.zeros((bsz, max_n), dtype=torch.bool)
    for i, (xi, yi, n) in enumerate(zip(xs, ys, lengths, strict=True)):
        x[i, :n] = xi
        y[i, :n] = yi
        mask[i, :n] = True
    return {
        "x": x,
        "y": y,
        "mask": mask,
        "num_graphs": bsz,
        "format": "padded",
    }


def lpbf_collate_varlen(batch: list) -> dict:
    """Pack variable-length LPBF graphs into flat ``[N_tot, C]`` + ``cu_seqlens``."""
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty LPBF batch.")
    xs = [_lpbf_as_node_matrix(g.x) for g in batch]
    ys = [_lpbf_as_node_matrix(g.y) for g in batch]
    lengths = [int(x.shape[0]) for x in xs]
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    cu = torch.zeros(len(batch) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)
    batch_index = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)],
        dim=0,
    )
    return {
        "x": x,
        "y": y,
        "cu_seqlens": cu,
        "max_seqlen": int(max(lengths)),
        "batch_index": batch_index,
        "num_graphs": len(batch),
        "format": "varlen",
    }


def lpbf_warped_rel_l2(
    yh: torch.Tensor,
    y: torch.Tensor,
    y_normalizer,
    *,
    batch_index: torch.Tensor | None = None,
    num_graphs: int | None = None,
    mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-graph channel-mean Rel-L2 after decoding both sides (unified field loss)."""
    from pdebench.dataset.loss import compute_field_loss
    from pdebench.dataset.sample import LossSpec

    loss_spec = LossSpec(mask="_mask") if mask is not None else LossSpec()
    masks = {"_mask": mask} if mask is not None else None
    return compute_field_loss(
        yh,
        y,
        y_normalizer,
        loss_spec,
        masks=masks,
        batch_index=batch_index,
        num_graphs=num_graphs,
        cu_seqlens=cu_seqlens,
    )


def lpbf_flare_graph_loss(model, graph, y_normalizer) -> torch.Tensor:
    device = next(model.parameters()).device
    x = _lpbf_as_node_matrix(graph.x).unsqueeze(0).to(device)
    y = _lpbf_as_node_matrix(graph.y).unsqueeze(0).to(device)
    yh = model(x)
    return lpbf_warped_rel_l2(yh, y, y_normalizer)


def lpbf_flare_batch_loss(
    model,
    batch,
    y_normalizer,
) -> torch.Tensor:
    """FLARE LPBF loss for single graphs, PyG batches, or padded/varlen dict batches."""
    if isinstance(batch, dict):
        device = next(model.parameters()).device
        fmt = batch["format"]
        if fmt not in {"padded", "varlen"}:
            raise ValueError(f"LPBF batch format must be 'padded' or 'varlen', got {fmt!r}")
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if fmt == "varlen":
            cu = batch["cu_seqlens"].to(device=device, dtype=torch.int32)
            max_seqlen = int(batch["max_seqlen"])
            yh = model(
                x,
                use_flash_varlen=True,
                cu_seqlens=cu,
                max_seqlen=max_seqlen,
            )
            return lpbf_warped_rel_l2(
                yh,
                y,
                y_normalizer,
                batch_index=batch.get("batch_index"),
                num_graphs=int(batch.get("num_graphs", cu.numel() - 1)),
                cu_seqlens=cu,
            )
        mask = batch["mask"].to(device=device)
        yh = model(x, mask=mask)
        return lpbf_warped_rel_l2(yh, y, y_normalizer, mask=mask)

    if hasattr(batch, "to_data_list") and int(getattr(batch, "num_graphs", 1)) > 1:
        # Legacy PyG Batch without custom collate: fall back to per-graph forwards.
        losses = [lpbf_flare_graph_loss(model, graph, y_normalizer) for graph in batch.to_data_list()]
        return torch.stack(losses).mean()
    return lpbf_flare_graph_loss(model, batch, y_normalizer)


# ======================================================================#
# Module-level so DataLoader workers can pickle the dataset (num_workers > 0).
class LPBFDataset(pyg.data.Dataset):
    """FLARE HF LPBF graphs (``vedantpuri/LPBF_FLARE``). Built via ``create_lpbf_dataset``."""

    def __init__(self, split="train", transform=None):
        import datasets

        assert split in ["train", "test"], f"Invalid split: {split}. Must be one of: 'train', 'test'."

        self.repo_id = LPBF_REPO_ID

        print(f"Initializing {split} dataset...")

        # Fast initialization: load dataset index first (lightweight)
        import time

        start_time = time.time()
        self.dataset = datasets.load_dataset(self.repo_id, split=split, keep_in_memory=True)
        dataset_time = time.time() - start_time
        print(f"Dataset index load: {dataset_time:.2f}s")

        # Lazy cache initialization - only download when needed
        self._cache_dir = None

        print(f"✅ Loaded {len(self.dataset)} samples for {split} split")

        super().__init__(None, transform=transform)

    @property
    def cache_dir(self):
        """Lazy load cache directory - only download when first sample is accessed."""
        if self._cache_dir is None:
            import huggingface_hub

            print("Downloading repository files on first access...")
            import time

            start_time = time.time()
            self._cache_dir = huggingface_hub.snapshot_download(self.repo_id, repo_type="dataset")
            download_time = time.time() - start_time
            print(f"Repository download/cache: {download_time:.2f}s")
            print(f"Cache directory: {self._cache_dir}")
        return self._cache_dir

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        # Get file path from index
        entry = self.dataset[idx]
        rel_path = entry["file"]
        npz_path = os.path.join(self.cache_dir, rel_path)

        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)
        graph = pyg.data.Data()

        # Convert to tensors efficiently
        for key, value in data.items():
            if key == "_metadata":
                graph["metadata"] = json.loads(value[0])["metadata"]
            else:
                if value.dtype.kind == "f":
                    tensor = torch.from_numpy(value.astype(np.float32))
                else:
                    if value.dtype != np.int64:
                        tensor = torch.from_numpy(value.astype(np.int64))
                    else:
                        tensor = torch.from_numpy(value)
                graph[key] = tensor

        # Set standard attributes
        graph.x = graph.pos
        graph.y = graph.disp[:, 2]

        return graph


def create_lpbf_dataset(*args, **kwargs):
    return LPBFDataset(*args, **kwargs)


# ======================================================================#
# GINOT graph-cache helpers (precompute / GLT edges path)
# ======================================================================#


def lpbf_dataset_dir(data_root: str) -> Path:
    return Path(data_root) / LPBF_DIRNAME


class _LpbfStore:
    def __init__(self) -> None:
        import datasets

        self.repo_id = LPBF_REPO_ID
        self._train = datasets.load_dataset(self.repo_id, split="train", keep_in_memory=True)
        self._test = datasets.load_dataset(self.repo_id, split="test", keep_in_memory=True)
        self._index: list[tuple[str, int]] = []
        for idx in range(len(self._train)):
            self._index.append(("train", idx))
        for idx in range(len(self._test)):
            self._index.append(("test", idx))
        self._cache_dir: str | None = None

    def __len__(self) -> int:
        return len(self._index)

    @property
    def cache_dir(self) -> str:
        if self._cache_dir is None:
            import huggingface_hub

            self._cache_dir = huggingface_hub.snapshot_download(self.repo_id, repo_type="dataset")
        return self._cache_dir

    @lru_cache(maxsize=512)
    def _load(self, global_idx: int) -> dict[str, np.ndarray]:
        split, local_idx = self._index[int(global_idx)]
        dataset = self._train if split == "train" else self._test
        rel_path = dataset[int(local_idx)]["file"]
        npz_path = os.path.join(self.cache_dir, rel_path)
        with np.load(npz_path, allow_pickle=True) as data:
            return {key: np.asarray(data[key]) for key in data.files if key != "_metadata"}


class _LpbfSequence:
    def __init__(self, store: _LpbfStore, field: str):
        self.store = store
        self.field = str(field)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int):
        sample = self.store._load(int(idx))
        if self.field == "query_points":
            return sample["pos"]
        if self.field == "point_clouds":
            return sample["pos"]
        if self.field == "targets":
            return sample["disp"][:, 2:3]
        raise KeyError(self.field)


class _LpbfEdgeIndexSequence:
    def __init__(self, store: _LpbfStore):
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int):
        return self.store._load(int(idx))["edge_index"]


def resolve_lpbf_splits(data_root: str, split_seed: int = 0) -> tuple[list[int], list[int]]:
    del split_seed, data_root
    store = _LpbfStore()
    train_ids: list[int] = []
    test_ids: list[int] = []
    for idx, (split, _) in enumerate(store._index):
        if split == "train":
            train_ids.append(idx)
        else:
            test_ids.append(idx)
    if not train_ids or not test_ids:
        raise ValueError("lpbf requires non-empty HuggingFace train and test splits.")
    return train_ids, test_ids


def load_lpbf_ginot(data_root: str):
    from pdebench.dataset.ginot.types import GinotRawDataset
    from pdebench.dataset.ginot.utils import target_fields

    root = lpbf_dataset_dir(data_root)
    root.mkdir(parents=True, exist_ok=True)
    store = _LpbfStore()
    return GinotRawDataset(
        dataset_dir=str(root),
        query_points=_LpbfSequence(store, "query_points"),
        point_clouds=_LpbfSequence(store, "point_clouds"),
        targets=_LpbfSequence(store, "targets"),
        cells=None,
        precomputed_edge_index=_LpbfEdgeIndexSequence(store),
        input_params=None,
        target_fields=target_fields("disp_z", 1),
        space_dim=3,
        target_normalizer=None,
        normalize_pos=True,
        normalize_boundary_pos=True,
        normalize_targets=False,
    )


# LPBF keeps raw z-displacement in GINOT batches (normalize_targets=False).
# Training loss applies y_normalizer.decode to both prediction and target, matching
# model_type=flare's warped rel-L2 — not physical displacement rel-L2.


def compute_lpbf_normalizers(raw, train_ids: list[int]):
    del raw, train_ids
    pos_normalizer = lpbf_ginot_pos_normalizer()
    return pos_normalizer, pos_normalizer, lpbf_ginot_y_normalizer()


def make_lpbf_statsfun(_cfg, _metadata):
    """LPBF GLT/graph-cache stats: reuse trainer.batch_lossfun (same as FLARE fallback_statsfun)."""
    del _cfg, _metadata

    @torch.no_grad()
    def statsfun(trainer, loader, split=None):
        del split
        model = trainer.model.module if getattr(trainer, "DDP", False) else trainer.model
        model.eval()
        rel_total = 0.0
        num_graphs_total = 0

        for batch in loader:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                loss = trainer.batch_lossfun(trainer, model, batch)
            num_graphs = int(batch["num_graphs"])
            rel_total += float(loss.item()) * num_graphs
            num_graphs_total += num_graphs

        if trainer.DDP:
            import torch.distributed as dist

            reduced = []
            for value in [rel_total, num_graphs_total]:
                tensor = torch.tensor(value, device=trainer.device)
                dist.all_reduce(tensor, dist.ReduceOp.SUM)
                reduced.append(tensor.item())
            rel_total, num_graphs_total = reduced

        if num_graphs_total == 0:
            return None, {"rel_l2": None}
        rel_l2 = rel_total / num_graphs_total
        return rel_l2, {"rel_l2": rel_l2}

    return statsfun


# LPBF: shared FLARE + GINOT normalizers via lpbf_metadata_normalizers / lpbf_y_normalizer.
# ======================================================================#
def load_lpbf_dataset(
    data_root: str | None = None,
    feature_request=None,
    *,
    include_edges: bool = False,
    use_flash_varlen: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    include_padded_boundary: bool = False,
    mesh_split_seed: int = 0,
    max_samples: int = 0,
):
    """Load LPBF for FLARE (HF+transform) or GLT (graph-cache via load_ginot_dataset)."""
    if feature_request is not None:
        include_edges = bool(feature_request.edges)
        include_padded_boundary = bool(feature_request.boundary)
        laplacian_eig_dim = int(feature_request.laplacian_k)
        laplacian_spec = str(feature_request.laplacian_spec)

    if include_edges:
        from pdebench.dataset.ginot.loader import load_ginot_dataset

        _ = use_flash_varlen  # edges/GLT path always uses flash-attn varlen packing
        root = data_root if data_root is not None else os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
        )
        return load_ginot_dataset(
            dataset_name="lpbf",
            data_root=root,
            split_seed=mesh_split_seed,
            max_samples=max_samples,
            include_edges=True,
            use_flash_varlen=True,  # packed GLT/graph-cache path
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            include_padded_boundary=include_padded_boundary,
        )

    import am

    transform = am.FinaltimeDatasetTransform(disp=True, vmstr=False, mesh=False)

    train_dataset = create_lpbf_dataset(split="train", transform=transform)
    test_dataset = create_lpbf_dataset(split="test", transform=transform)

    x_normalizer, y_normalizer = lpbf_metadata_normalizers(train_dataset)
    # y_normalizer must match lpbf_y_normalizer().

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        c_in=3,
        c_edge=3,
        c_out=1,
        time_cond=False,
        max_length=50_000,
    )

    return train_dataset, test_dataset, metadata

import torch
import torch.distributed as dist
from tqdm import tqdm

MESH_GRAPH_MODELS = {
    "glt",
    "meshgraphnet",
    "rigno",
    "gito",
    "geo_transolver",
}
MESH_SEQUENCE_MODELS = {
    "transolver",
    "flare",
    "flare_experimental",
    "flare_ablations",
    "mixer_backbone",
    "flarepp",
    "luna",
}
# Datasets that pass scalar time/context via input_scalars on the mesh GLT path
# (graph-level context; not plasticity rollout). See plaid_elpl_v3.assemble.
MESH_GRAPH_TIME_COND_DATASETS = frozenset({"plaid_el_pl_dynamics"})


def mesh_static_supports_time_cond(dataset_name: str) -> bool:
    return dataset_name in MESH_GRAPH_TIME_COND_DATASETS


# ======================================================================#
def mesh_sequence_collate_fn(batch):
    if len(batch) == 0:
        return []

    lengths = [int(sample.x.shape[0]) for sample in batch]
    if len(set(lengths)) != 1:
        raise NotImplementedError(
            "Batching variable-length mesh samples for sequence models (Transolver/FLARE) "
            "is not implemented. Add masking support or use batch_size=1."
        )

    x = torch.stack([sample.x for sample in batch], dim=0)
    y = torch.stack([sample.y for sample in batch], dim=0)
    return [x, y]


# ======================================================================#
def _mesh_graph_batch_targets(batch):
    if hasattr(batch, "y"):
        y = batch.y
        batch_index = batch.batch
        num_graphs = batch.num_graphs
        return y, batch_index, num_graphs

    raise ValueError(f"Unsupported graph batch type: {type(batch)}")


def _mesh_glt_forward(model, batch):
    if not hasattr(batch, "x") or not hasattr(batch, "edge_index") or not hasattr(batch, "ptr"):
        raise ValueError(f"Static mesh GLT expects a PyG Batch with x/edge_index/ptr, got {type(batch)}")

    pos = batch.x[:, :2]
    feats = batch.x[:, 2:] if batch.x.shape[-1] > 2 else None
    # Graph-level context (el-pl time) lives on ``batch.context`` and is concatenated
    # here. Static PLAID bakes scalars into ``x`` and also stores ``input_scalars``
    # for metadata — only concat when ``context`` is present.
    context = getattr(batch, "context", None)
    if context is not None:
        scalars = context
        if scalars.ndim == 2:
            scalars = scalars[batch.batch]
        scalar_feats = scalars.reshape(scalars.shape[0], -1)
        feats = scalar_feats if feats is None else torch.cat([feats, scalar_feats], dim=-1)
    ptr = batch.ptr.to(device=pos.device, dtype=torch.int32)
    lengths = ptr[1:] - ptr[:-1]
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    yh = model(
        pos=pos,
        feats=feats,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, "edge_attr", None),
        batch_index=batch.batch,
        use_flash_varlen=True,
        cu_seqlens=ptr,
        max_seqlen=max_seqlen,
        num_total_nodes=int(ptr[-1].item()) if ptr.numel() else int(pos.shape[0]),
        topology_features=getattr(batch, "laplacian_eig", None),
        topology_eigenvalues=getattr(batch, "laplacian_eigvals", None),
    )
    y, batch_index, num_graphs = _mesh_graph_batch_targets(batch)
    if torch.is_tensor(batch_index) and (batch_index.device != yh.device):
        batch_index = batch_index.to(yh.device)
    if torch.is_tensor(y) and (y.device != yh.device):
        y = y.to(yh.device)
    return yh, y, batch_index, num_graphs


# ======================================================================#
def mesh_model_forward(cfg, model, batch):
    model_type = cfg.model.model
    if model_type == "glt":
        return _mesh_glt_forward(model, batch)
    if model_type in MESH_GRAPH_MODELS:
        yh = model(batch)
        y, batch_index, num_graphs = _mesh_graph_batch_targets(batch)
        if torch.is_tensor(batch_index) and (batch_index.device != yh.device):
            batch_index = batch_index.to(yh.device)
        if torch.is_tensor(y) and (y.device != yh.device):
            y = y.to(yh.device)
        return yh, y, batch_index, num_graphs

    if model_type in MESH_SEQUENCE_MODELS:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            x, y = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch format for mesh sequence model '{model_type}': {type(batch)}")

        yh = model(x)
        if yh.ndim == 2:
            yh = yh.unsqueeze(0)
        if y.ndim == 2:
            y = y.unsqueeze(0)

        if yh.ndim != 3 or y.ndim != 3:
            raise ValueError(
                f"Expected [B, N, C] tensors for mesh sequence model '{model_type}', "
                f"got yh={tuple(yh.shape)}, y={tuple(y.shape)}."
            )

        bsz, npts = yh.shape[0], yh.shape[1]
        batch_index = torch.arange(bsz, device=yh.device).repeat_interleave(npts)

        yh = yh.reshape(-1, yh.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        return yh, y, batch_index, bsz

    raise NotImplementedError(
        f"Dataset '{cfg.dataset.dataset}' is a static mesh dataset, but model_type='{model_type}' "
        "does not implement mesh batch formatting."
    )


# ======================================================================#
def mesh_per_graph_plaid_rrmse(yh, y, batch_index, num_graphs):
    vals = []
    for gid in range(num_graphs):
        mask = batch_index == gid
        yh_g = yh[mask]
        y_g = y[mask]
        n_nodes = max(int(y_g.shape[0]), 1)

        field_vals = []
        for j in range(y_g.shape[-1]):
            ref = y_g[:, j]
            denom = (n_nodes * torch.max(torch.abs(ref)) ** 2).clamp_min(1e-12)
            field_vals.append(torch.sqrt(torch.sum((yh_g[:, j] - ref) ** 2) / denom))
        vals.append(torch.stack(field_vals).mean())
    return torch.stack(vals)


def mesh_batch_plaid_scalar_rrmse(scalar_pred, scalar_target):
    """Paper scalar RRMSE: mean over batch of |err|/|ref| (per scalar then mean)."""
    if scalar_pred.numel() == 0:
        return torch.zeros((), device=scalar_pred.device, dtype=scalar_pred.dtype)
    denom = scalar_target.abs().clamp_min(1e-12)
    per_sample = ((scalar_pred - scalar_target).abs() / denom).mean(dim=-1)
    return per_sample.mean()


def mesh_batch_plaid_scaled_mse(yh, y, batch, batch_index, num_graphs, field_dim, scalar_dim, lbda):
    """Vi-Transf train loss: λ·field_MSE + (1−λ)·scalar_MSE in normalized space."""
    y_fields = y[:, :field_dim] if y.shape[-1] >= field_dim else y
    field_loss = torch.nn.functional.mse_loss(yh[:, :field_dim], y_fields)
    if int(scalar_dim) > 0:
        scalar_pred = []
        for gid in range(int(num_graphs)):
            mask = batch_index == gid
            scalar_pred.append(yh[mask, field_dim : field_dim + int(scalar_dim)].mean(dim=0))
        scalar_pred = torch.stack(scalar_pred, dim=0)
        scalar_target = batch.output_scalars.to(device=scalar_pred.device, dtype=scalar_pred.dtype)
        if scalar_target.ndim == 1:
            scalar_target = scalar_target.unsqueeze(0)
        scalar_loss = torch.nn.functional.mse_loss(scalar_pred, scalar_target)
    else:
        scalar_loss = torch.zeros((), device=yh.device, dtype=yh.dtype)
    loss = float(lbda) * field_loss + (1.0 - float(lbda)) * scalar_loss
    return loss, field_loss, scalar_loss


# ======================================================================#
def make_mesh_static_statsfun(cfg, metadata):
    @torch.no_grad()
    def statsfun(trainer, loader, split: str):
        print_iterator = trainer.verbose and (trainer.GLOBAL_RANK == 0) and trainer.print_iterator
        batch_iterator = (
            tqdm(loader, desc=f"Evaluating ({split}) dataset", ncols=80, smoothing=0.0, miniters=1)
            if print_iterator
            else loader
        )

        loss_total = 0.0
        field_mse_total = 0.0
        scalar_mse_total = 0.0
        plaid_rrmse_total = 0.0
        scalar_rrmse_total = 0.0
        num_graphs_total = 0

        field_dim = len(metadata.get("target_fields", []))
        scalar_dim = len(metadata.get("target_scalar_fields", []))
        lbda = float(metadata.get("plaid_loss_lbda", 1.0 if scalar_dim == 0 else 0.5))
        y_normalizer = metadata["y_normalizer"].to(trainer.device)
        y_scalar_normalizer = metadata.get("y_scalar_normalizer")
        if y_scalar_normalizer is not None and hasattr(y_scalar_normalizer, "to"):
            y_scalar_normalizer = y_scalar_normalizer.to(trainer.device)

        for batch in batch_iterator:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                yh, y, batch_index, num_graphs = mesh_model_forward(cfg, trainer.model, batch)
            if y is None:
                # Public PLAID test split is unlabeled in local copies.
                continue

            loss, field_mse, scalar_mse = mesh_batch_plaid_scaled_mse(
                yh,
                y,
                batch,
                batch_index=batch_index,
                num_graphs=num_graphs,
                field_dim=field_dim,
                scalar_dim=scalar_dim,
                lbda=lbda,
            )
            yh_fields = y_normalizer.decode(yh[:, :field_dim])
            y_fields = y_normalizer.decode(y[:, :field_dim] if y.shape[-1] >= field_dim else y)
            plaid_rrmses = mesh_per_graph_plaid_rrmse(
                yh_fields, y_fields, batch_index=batch_index, num_graphs=num_graphs
            )

            n_g = int(num_graphs)
            loss_total += float(loss.item()) * n_g
            field_mse_total += float(field_mse.item()) * n_g
            scalar_mse_total += float(scalar_mse.item()) * n_g
            plaid_rrmse_total += plaid_rrmses.sum().item()

            if scalar_dim > 0 and hasattr(batch, "output_scalars"):
                scalar_pred = []
                for gid in range(n_g):
                    mask = batch_index == gid
                    scalar_pred.append(yh[mask, field_dim : field_dim + scalar_dim].mean(dim=0))
                scalar_pred = torch.stack(scalar_pred, dim=0)
                scalar_target = batch.output_scalars.to(device=scalar_pred.device, dtype=scalar_pred.dtype)
                if y_scalar_normalizer is not None:
                    scalar_pred = y_scalar_normalizer.decode(scalar_pred)
                    scalar_target = y_scalar_normalizer.decode(scalar_target)
                scalar_rrmse_total += float(mesh_batch_plaid_scalar_rrmse(scalar_pred, scalar_target).item()) * n_g

            num_graphs_total += n_g

        if trainer.DDP:
            stats = [
                loss_total,
                field_mse_total,
                scalar_mse_total,
                plaid_rrmse_total,
                scalar_rrmse_total,
                num_graphs_total,
            ]
            reduced = []
            for value in stats:
                tensor = torch.tensor(value, device=trainer.device)
                dist.all_reduce(tensor, dist.ReduceOp.SUM)
                reduced.append(tensor.item())
            (
                loss_total,
                field_mse_total,
                scalar_mse_total,
                plaid_rrmse_total,
                scalar_rrmse_total,
                num_graphs_total,
            ) = reduced

        if num_graphs_total == 0:
            return float("nan"), dict(
                plaid_loss=float("nan"),
                plaid_field_mse=float("nan"),
                plaid_scalar_mse=float("nan"),
                plaid_rrmse=float("nan"),
                plaid_scalar_rrmse=float("nan"),
                total_error=float("nan"),
            )

        plaid_loss = loss_total / num_graphs_total
        plaid_field_mse = field_mse_total / num_graphs_total
        plaid_scalar_mse = scalar_mse_total / num_graphs_total
        plaid_rrmse = plaid_rrmse_total / num_graphs_total
        plaid_scalar_rrmse = (
            scalar_rrmse_total / num_graphs_total if scalar_dim > 0 else float("nan")
        )
        if scalar_dim > 0:
            total_error = 0.5 * (plaid_rrmse + plaid_scalar_rrmse)
        else:
            total_error = plaid_rrmse
        return plaid_loss, dict(
            plaid_loss=plaid_loss,
            plaid_field_mse=plaid_field_mse,
            plaid_scalar_mse=plaid_scalar_mse,
            plaid_rrmse=plaid_rrmse,
            plaid_scalar_rrmse=plaid_scalar_rrmse,
            total_error=total_error,
        )

    return statsfun

#
import gc
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

import mlutils
import pdebench

__all__ = [
    'ahmedml_surface_metric_sums',
    'drivaerml_surface_metric_sums',
    '_reduce_drivaerml_surface_metric_sums',
    'surface_batch_loss',
    'make_ahmedml_surface_statsfun',
    'make_drivaerml_surface_statsfun',
    'make_navier_stokes_statsfun',
    'make_plasticity_statsfun',
    'RelL2Callback',
    'AhmedMLSurfaceRelL2Callback',
    'DrivAerMLSurfaceRelL2Callback',
    'NavierStokesCallback',
    'PlasticityCallback',
    'MeshStaticCallback',
    'ScoresCallback',
    'MixerDiagnosticsCallback',
]


_SURFACE_REL_L2_NAMES = ("full_rel_l2", "pressure_rel_l2", "wall_shear_rel_l2")


def ahmedml_surface_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Squared-error and target-energy sums for physical AhmedML fields."""
    return _surface_metric_sums(pred, target)


def _reduce_ahmedml_surface_metric_sums(
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cp_state,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return _reduce_surface_metric_sums(metric_sums, cp_state)


def drivaerml_surface_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Squared-error and target-energy sums for physical DrivAerML fields."""
    return _surface_metric_sums(pred, target)


def _reduce_drivaerml_surface_metric_sums(
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cp_state,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return _reduce_surface_metric_sums(metric_sums, cp_state)


def _surface_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Squared-error / target-energy sums for physical surface fields."""
    pred = pred.float()
    target = target.float()
    pred_tau = torch.linalg.vector_norm(pred[..., 1:], dim=-1)
    target_tau = torch.linalg.vector_norm(target[..., 1:], dim=-1)
    return {
        "full_rel_l2": ((pred - target).square().sum(), target.square().sum()),
        "pressure_rel_l2": ((pred[..., 0] - target[..., 0]).square().sum(), target[..., 0].square().sum()),
        "wall_shear_rel_l2": ((pred_tau - target_tau).square().sum(), target_tau.square().sum()),
    }


def _reduce_surface_metric_sums(
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cp_state,
    *,
    differentiable: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if cp_state is None or cp_state.cp_size <= 1:
        return metric_sums
    if cp_state.cp_group is None:
        raise RuntimeError("Context parallel group is not initialized.")
    import torch.distributed.nn.functional as dist_nn

    reduced = {}
    for name, (numerator, denominator) in metric_sums.items():
        if differentiable and (numerator.requires_grad or denominator.requires_grad):
            num = dist_nn.all_reduce(numerator, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
            den = dist_nn.all_reduce(denominator, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
            reduced[name] = num, den
        else:
            pair = torch.stack((numerator, denominator))
            dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
            reduced[name] = pair[0], pair[1]
    return reduced


def surface_batch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    rel_l2_loss: bool,
    y_normalizer,
    cp_state=None,
) -> torch.Tensor:
    """Per-batch / full-mesh train objective for AhmedML / DrivAerML surfaces.

    ``rel_l2_loss=False`` → normalized MSE (no decode).
    ``rel_l2_loss=True`` → ``0.5 * Rel-L2(pressure) + 0.5 * Rel-L2(wall-shear mag)``
    after decode to physical units.
    """
    from pdebench.distributed import cp_reduced_mse_loss

    if not rel_l2_loss:
        if cp_state is not None and getattr(cp_state, "cp_size", 1) > 1:
            return cp_reduced_mse_loss(pred, target, cp_state)
        return F.mse_loss(pred.float(), target.float())

    normalizer = y_normalizer.to(target.device)
    pred_phys = normalizer.decode(pred)
    target_phys = normalizer.decode(target)
    sums = _surface_metric_sums(pred_phys, target_phys)
    sums = _reduce_surface_metric_sums(sums, cp_state, differentiable=True)
    eps = pred.new_tensor(1e-12)
    p_num, p_den = sums["pressure_rel_l2"]
    t_num, t_den = sums["wall_shear_rel_l2"]
    pressure_rel = torch.sqrt(p_num / p_den.clamp_min(eps))
    wall_shear_rel = torch.sqrt(t_num / t_den.clamp_min(eps))
    return 0.5 * (pressure_rel + wall_shear_rel)


def _surface_run_dataloader(trainer, run_dataset):
    """Full-mesh Rel-L2 loader; reuse trainer dataloader worker settings."""
    num_workers = max(int(getattr(trainer, "num_workers", 0) or 0), 0)
    kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(trainer, "is_cuda", False)),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        prefetch = getattr(trainer, "prefetch_factor", None)
        if prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
        if kwargs["pin_memory"]:
            import multiprocessing as mp

            kwargs["multiprocessing_context"] = mp.get_context("spawn")
    return torch.utils.data.DataLoader(run_dataset, **kwargs)


def _surface_rel_l2_from_sums(
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    return {name: float(torch.sqrt(numerator / denominator)) for name, (numerator, denominator) in metric_sums.items()}


def _accumulate_surface_rel_l2(
    totals: dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    cp_state,
) -> None:
    sums = _reduce_surface_metric_sums(_surface_metric_sums(pred, target), cp_state)
    for name, value in _surface_rel_l2_from_sums(sums).items():
        totals[name] += value


def _assert_surface_stats_cp_counts_synced(run_count: int, cp_state, device: torch.device) -> None:
    """Detect CP desync (mismatched run loops → hung/raced all_reduce)."""
    if cp_state is None or cp_state.cp_size <= 1:
        return
    if cp_state.cp_group is None:
        raise RuntimeError("Context parallel group is not initialized.")
    counts = torch.tensor([run_count], device=device, dtype=torch.long)
    counts_min = counts.clone()
    counts_max = counts.clone()
    dist.all_reduce(counts_min, op=dist.ReduceOp.MIN, group=cp_state.cp_group)
    dist.all_reduce(counts_max, op=dist.ReduceOp.MAX, group=cp_state.cp_group)
    if not torch.equal(counts_min, counts_max):
        raise RuntimeError(
            "surface statsfun CP desync: run counts differ across ranks "
            f"(min={counts_min.tolist()}, max={counts_max.tolist()})."
        )


def _make_surface_statsfun(
    metadata,
    *,
    run_key_prefix: str,
    cp_state=None,
):
    """Full-mesh stats matching the surface train objective + Rel-L2 diagnostics.

    Primary returned loss follows ``metadata['rel_l2_loss']``:
    - False → mean per-run normalized MSE (no decode)
    - True → mean per-run ``0.5 * Rel-L2(p) + 0.5 * Rel-L2(wall-shear mag)`` (decoded)

    Stats always include ``mse`` and physical ``full_rel_l2`` / ``pressure_rel_l2`` /
    ``wall_shear_rel_l2``.
    """
    y_normalizer = metadata["y_normalizer"]
    rel_l2_loss = bool(metadata.get("rel_l2_loss", False))

    @torch.no_grad()
    def statsfun(trainer, loader, split: str):
        # Trainer fullbatch uses split='val' for the held-out loader; surface metadata
        # stores full-mesh runs under *_test_run_data (train/test splits).
        run_split = "test" if split == "val" else split
        run_dataset = metadata.get(f"{run_key_prefix}_{run_split}_run_data")
        if run_dataset is not None:
            loader = _surface_run_dataloader(trainer, run_dataset)
        model = trainer.model
        was_training = model.training
        model.eval()
        preprocess_fn = getattr(trainer, "preprocess_fn_", None)
        sync_device = torch.device("cpu")
        try:
            loss_total = 0.0
            mse_total = 0.0
            totals = {name: 0.0 for name in _SURFACE_REL_L2_NAMES}
            run_count = 0
            for batch in loader:
                batch = trainer.move_to_device(batch)
                x_full, target_full = batch
                sync_device = x_full.device

                mesh_batch = (x_full, target_full)
                if preprocess_fn is not None:
                    mesh_batch = preprocess_fn(mesh_batch)
                x, target = mesh_batch
                with trainer.auto_cast:
                    pred = model(x)

                mse = surface_batch_loss(
                    pred,
                    target,
                    rel_l2_loss=False,
                    y_normalizer=y_normalizer,
                    cp_state=cp_state,
                )
                mse_total += float(mse.item())

                normalizer = y_normalizer.to(target.device)
                pred_phys = normalizer.decode(pred)
                target_phys = normalizer.decode(target)
                sums = _reduce_surface_metric_sums(_surface_metric_sums(pred_phys, target_phys), cp_state)
                rels = _surface_rel_l2_from_sums(sums)
                for name, value in rels.items():
                    totals[name] += value

                if rel_l2_loss:
                    loss_total += 0.5 * (rels["pressure_rel_l2"] + rels["wall_shear_rel_l2"])
                else:
                    loss_total += float(mse.item())
                run_count += 1

            _assert_surface_stats_cp_counts_synced(run_count, cp_state, sync_device)

            if run_count == 0:
                stats = {"mse": float("nan"), **{name: float("nan") for name in totals}}
                return float("nan"), stats

            avg_loss = loss_total / run_count
            stats = {name: value / run_count for name, value in totals.items()}
            stats["mse"] = mse_total / run_count
            return avg_loss, stats
        finally:
            model.train(was_training)

    return statsfun


def make_ahmedml_surface_statsfun(metadata, cp_state=None):
    """AhmedML full-mesh loss (train-aligned) + physical Rel-L2 stats."""
    return _make_surface_statsfun(metadata, run_key_prefix="ahmedml", cp_state=cp_state)


def make_drivaerml_surface_statsfun(metadata, cp_state=None):
    """DrivAerML full-mesh loss (train-aligned) + physical Rel-L2 stats."""
    return _make_surface_statsfun(metadata, run_key_prefix="drivaerml", cp_state=cp_state)


def make_navier_stokes_statsfun():
    lossfun = pdebench.RelL2Loss()

    @torch.no_grad()
    def statsfun(trainer, loader, split: str):
        print_iterator = trainer.verbose and (trainer.GLOBAL_RANK == 0) and trainer.print_iterator
        batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80, smoothing=0.0, miniters=1) \
            if print_iterator else loader

        teacher_forcing = split == 'train'
        N = 0
        step_total = 0.0
        full_total = 0.0

        for batch in batch_iterator:
            batch = trainer.move_to_device(batch)
            pos, history, target = batch

            with trainer.auto_cast:
                _, step_loss, full_loss = pdebench.rollout_navier_stokes(
                    trainer.model,
                    pos,
                    history,
                    target,
                    lossfun=lossfun,
                    teacher_forcing=teacher_forcing,
                )

            n = trainer.get_batch_size(batch, loader)
            N += n
            step_total += (step_loss.item() / target.shape[-1]) * n
            full_total += full_loss.item() * n

        if trainer.DDP:
            stats = [step_total, full_total, N]
            reduced = []
            for value in stats:
                tensor = torch.tensor(value, device=trainer.device)
                dist.all_reduce(tensor, dist.ReduceOp.SUM)
                reduced.append(tensor.item())
            step_total, full_total, N = reduced

        if N == 0:
            return float('nan'), dict(step_rel_l2=float('nan'), full_rel_l2=float('nan'))

        step_avg = step_total / N
        full_avg = full_total / N

        return full_avg, dict(step_rel_l2=step_avg, full_rel_l2=full_avg)

    return statsfun


def make_plasticity_statsfun():
    lossfun = pdebench.RelL2Loss()

    @torch.no_grad()
    def statsfun(trainer, loader, split: str):
        print_iterator = trainer.verbose and (trainer.GLOBAL_RANK == 0) and trainer.print_iterator
        batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80, smoothing=0.0, miniters=1) \
            if print_iterator else loader

        N = 0
        step_total = 0.0
        full_total = 0.0

        for batch in batch_iterator:
            batch = trainer.move_to_device(batch)
            pos, time_grid, features, target = batch

            with trainer.auto_cast:
                _, step_loss, full_loss = pdebench.rollout_plasticity(
                    trainer.model,
                    pos,
                    time_grid,
                    features,
                    target,
                    lossfun=lossfun,
                )

            n = trainer.get_batch_size(batch, loader)
            N += n
            step_total += (step_loss.item() / target.shape[-1]) * n
            full_total += full_loss.item() * n

        if trainer.DDP:
            stats = [step_total, full_total, N]
            reduced = []
            for value in stats:
                tensor = torch.tensor(value, device=trainer.device)
                dist.all_reduce(tensor, dist.ReduceOp.SUM)
                reduced.append(tensor.item())
            step_total, full_total, N = reduced

        if N == 0:
            return float('nan'), dict(step_rel_l2=float('nan'), full_rel_l2=float('nan'))

        step_avg = step_total / N
        full_avg = full_total / N

        return full_avg, dict(step_rel_l2=step_avg, full_rel_l2=full_avg)

    return statsfun


def _rel_l2_by_field_metrics(
    yh: torch.Tensor,
    y: torch.Tensor,
    field_metrics: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Per-field Rel-L2. Specs: kind=slice | vector_norm (per-node L2 then Rel-L2)."""
    lossfun = pdebench.RelL2Loss()
    out: dict[str, torch.Tensor] = {}
    for name, spec in field_metrics.items():
        kind = spec["kind"]
        sl = spec["slice"]
        if kind == "slice":
            out[name] = lossfun(yh[..., sl], y[..., sl])
        elif kind == "vector_norm":
            pred = torch.linalg.vector_norm(yh[..., sl], ord=2, dim=-1, keepdim=True)
            tgt = torch.linalg.vector_norm(y[..., sl], ord=2, dim=-1, keepdim=True)
            out[name] = lossfun(pred, tgt)
        else:
            raise ValueError(f"Unknown y_field_metrics kind={kind!r} for field={name!r}")
    return out


def _rel_l2_by_slices(
    yh: torch.Tensor,
    y: torch.Tensor,
    field_slices: dict[str, slice],
) -> dict[str, torch.Tensor]:
    return _rel_l2_by_field_metrics(
        yh,
        y,
        {name: {"kind": "slice", "slice": sl} for name, sl in field_slices.items()},
    )


def _coerce_y_field_metrics(
    y_field_metrics: dict[str, dict] | None,
    y_field_slices: dict[str, slice] | None,
) -> dict[str, dict] | None:
    if y_field_metrics is not None:
        return y_field_metrics
    if y_field_slices is None:
        return None
    return {name: {"kind": "slice", "slice": sl} for name, sl in y_field_slices.items()}


#======================================================================#
class RelL2Callback(mlutils.Callback):
    def __init__(
        self,
        case_dir: str,
        dataset: str,
        x_normalizer,
        y_normalizer,
        y_field_slices: dict[str, slice] | None = None,
        y_field_metrics: dict[str, dict] | None = None,
    ):
        super().__init__(case_dir)
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.dataset = dataset
        self.y_field_slices = y_field_slices
        self.y_field_metrics = _coerce_y_field_metrics(y_field_metrics, y_field_slices)

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):

        trainer.model.eval()
        device = trainer.device

        lossfun = pdebench.RelL2Loss()
        y_normalizer = self.y_normalizer.to(device)

        _N, _rel_error, _r2 = 0, 0., []
        N_, rel_error_, r2_ = 0, 0., []
        train_field_errors = {name: 0.0 for name in self.y_field_metrics or {}}
        test_field_errors = {name: 0.0 for name in self.y_field_metrics or {}}

        for batch in trainer._loader_:
            x, y = batch[0].to(device), batch[1].to(device)
            with trainer.auto_cast:
                yh = trainer.model(x)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            loss = lossfun(yh,y)

            _n = trainer.get_batch_size(batch, trainer._loader_)
            _N += _n
            _rel_error += loss.item() * _n
            if self.y_field_metrics:
                for name, field_loss in _rel_l2_by_field_metrics(yh, y, self.y_field_metrics).items():
                    train_field_errors[name] += field_loss.item() * _n
            r2val = mlutils.r2(yh, y)
            _r2.append(r2val)
            del x, y, yh

        for batch in trainer.loader_:
            x, y = batch[0].to(device), batch[1].to(device)
            with trainer.auto_cast:
                yh = trainer.model(x)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            loss = lossfun(yh,y)

            n_ = trainer.get_batch_size(batch, trainer.loader_)
            N_ += n_
            rel_error_ += loss.item() * n_
            if self.y_field_metrics:
                for name, field_loss in _rel_l2_by_field_metrics(yh, y, self.y_field_metrics).items():
                    test_field_errors[name] += field_loss.item() * n_
            r2val = mlutils.r2(yh, y)
            r2_.append(r2val)
            del x, y, yh

        _r2 = torch.tensor(_r2)
        r2_ = torch.tensor(r2_)

        if trainer.DDP:
            # relative error
            pre_ddp, post_ddp = [_rel_error, rel_error_, _N, N_], []
            for p in pre_ddp:
                p = torch.tensor(p, device=trainer.device)
                dist.all_reduce(p, dist.ReduceOp.SUM)
                post_ddp.append(p.item())
            _rel_error, rel_error_, _N, N_ = post_ddp
            for field_errors in (train_field_errors, test_field_errors):
                for name, value in field_errors.items():
                    reduced = torch.tensor(value, device=trainer.device)
                    dist.all_reduce(reduced, dist.ReduceOp.SUM)
                    field_errors[name] = reduced.item()

            # R-Squared
            _r2 = _r2.to(device)
            r2_ = r2_.to(device)

            _r2_list = [torch.zeros_like(_r2) for _ in range(dist.get_world_size())]
            r2_list_ = [torch.zeros_like(r2_) for _ in range(dist.get_world_size())]

            dist.all_gather(_r2_list, _r2)
            dist.all_gather(r2_list_, r2_)

            _r2 = torch.cat(_r2_list, dim=0)
            r2_ = torch.cat(r2_list_, dim=0)

        _rel_error /= _N
        rel_error_ /= N_
        train_field_errors = {name: value / _N for name, value in train_field_errors.items()}
        test_field_errors = {name: value / N_ for name, value in test_field_errors.items()}

        # save rel_error.json
        if trainer.GLOBAL_RANK == 0:
            print(f'Relative Error (train / test): {_rel_error:.8e} / {rel_error_:.8e}')
            metrics = {'train_rel_error': _rel_error, 'test_rel_error': rel_error_}
            for name in train_field_errors:
                metrics[f'train_rel_error_{name}'] = train_field_errors[name]
                metrics[f'test_rel_error_{name}'] = test_field_errors[name]
                print(
                    f'Relative Error {name} (train / test): '
                    f'{train_field_errors[name]:.8e} / {test_field_errors[name]:.8e}'
                )
            with open(os.path.join(ckpt_dir, 'rel_error.json'), 'w') as f:
                json.dump(metrics, f)

            with open(os.path.join(ckpt_dir, '..', 'rel_error.json'), 'w') as f:
                json.dump(metrics, f)

        if trainer.GLOBAL_RANK == 0:
            print(f'Mean R2 (train / test): {_r2.mean():.4f} / {r2_.mean():.4f}')

        return


def _write_surface_rel_l2_metrics(
    *,
    label: str,
    trainer: mlutils.Trainer,
    ckpt_dir: str,
    stat_vals: dict,
) -> None:
    """Persist full-mesh Rel-L2 paper columns from fullbatch stats."""
    if trainer.GLOBAL_RANK != 0:
        return

    metrics = {}
    required = _SURFACE_REL_L2_NAMES
    for suffix in ("", "_ema"):
        train_stats = stat_vals.get(f"train_stats{suffix}")
        test_stats = stat_vals.get(f"test_stats{suffix}")
        # Disabled fullbatch returns {} (not None); skip until paper keys exist.
        if not isinstance(train_stats, dict) or not isinstance(test_stats, dict):
            continue
        if not all(name in train_stats and name in test_stats for name in required):
            continue
        for stat_name in required:
            metrics[f"train_{stat_name}{suffix}"] = train_stats[stat_name]
            metrics[f"test_{stat_name}{suffix}"] = test_stats[stat_name]

    if "train_pressure_rel_l2" not in metrics or "test_pressure_rel_l2" not in metrics:
        return

    print(
        f"{label} pressure Rel-L2 (train / test): "
        f"{metrics['train_pressure_rel_l2']:.8e} / {metrics['test_pressure_rel_l2']:.8e}"
    )
    print(
        f"{label} wall-shear Rel-L2 (train / test): "
        f"{metrics['train_wall_shear_rel_l2']:.8e} / {metrics['test_wall_shear_rel_l2']:.8e}"
    )
    print(
        f"{label} full-mesh Rel-L2 diagnostic (train / test): "
        f"{metrics['train_full_rel_l2']:.8e} / {metrics['test_full_rel_l2']:.8e}"
    )
    with open(os.path.join(ckpt_dir, "rel_error.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(ckpt_dir, "..", "rel_error.json"), "w") as f:
        json.dump(metrics, f)


class AhmedMLSurfaceRelL2Callback(RelL2Callback):
    """Write AhmedML paper columns from fullbatch full-mesh Rel-L2 stats.

    Paper Table 4 columns are ``pressure_rel_l2`` (surface pressure) and
    ``wall_shear_rel_l2`` (wall-shear magnitude). ``full_rel_l2`` is diagnostic.
    """

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        _write_surface_rel_l2_metrics(
            label="AhmedML", trainer=trainer, ckpt_dir=ckpt_dir, stat_vals=stat_vals
        )


class DrivAerMLSurfaceRelL2Callback(RelL2Callback):
    """Write DrivAerML paper columns from fullbatch full-mesh Rel-L2 stats."""

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        _write_surface_rel_l2_metrics(
            label="DrivAerML", trainer=trainer, ckpt_dir=ckpt_dir, stat_vals=stat_vals
        )


class NavierStokesCallback(mlutils.Callback):
    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        train_stats = stat_vals.get('train_stats') or {}
        test_stats = stat_vals.get('test_stats') or {}

        train_rel = train_stats.get('full_rel_l2', stat_vals.get('train_loss'))
        test_rel = test_stats.get('full_rel_l2', stat_vals.get('test_loss'))

        if trainer.GLOBAL_RANK == 0:
            if train_rel is not None and test_rel is not None:
                print(f'Relative Error (train / test): {train_rel:.8e} / {test_rel:.8e}')

            payload = {
                'train_rel_error': train_rel,
                'test_rel_error': test_rel,
                'train_step_rel_error': train_stats.get('step_rel_l2'),
                'test_step_rel_error': test_stats.get('step_rel_l2'),
            }

            with open(os.path.join(ckpt_dir, 'rel_error.json'), 'w') as f:
                json.dump(payload, f)

            with open(os.path.join(ckpt_dir, '..', 'rel_error.json'), 'w') as f:
                json.dump(payload, f)

        return


class PlasticityCallback(mlutils.Callback):
    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        train_stats = stat_vals.get('train_stats') or {}
        test_stats = stat_vals.get('test_stats') or {}

        train_rel = train_stats.get('full_rel_l2', stat_vals.get('train_loss'))
        test_rel = test_stats.get('full_rel_l2', stat_vals.get('test_loss'))

        if trainer.GLOBAL_RANK == 0:
            if train_rel is not None and test_rel is not None:
                print(f'Relative Error (train / test): {train_rel:.8e} / {test_rel:.8e}')

            payload = {
                'train_rel_error': train_rel,
                'test_rel_error': test_rel,
                'train_step_rel_error': train_stats.get('step_rel_l2'),
                'test_step_rel_error': test_stats.get('step_rel_l2'),
            }

            with open(os.path.join(ckpt_dir, 'rel_error.json'), 'w') as f:
                json.dump(payload, f)

            with open(os.path.join(ckpt_dir, '..', 'rel_error.json'), 'w') as f:
                json.dump(payload, f)

        return


class MeshStaticCallback(mlutils.Callback):
    _GRAPH_MODELS = {
        "glt",
        "meshgraphnet",
        "rigno",
        "gito",
        "geo_transolver",
    }
    _SEQUENCE_MODELS = {"transolver", "flare", "flare_experimental", "flare_ablations", "mixer_backbone", "flarepp", "luna"}

    def __init__(
        self,
        case_dir: str,
        model_type: str,
        y_normalizer,
        y_scalar_normalizer=None,
        target_fields=None,
        target_scalar_fields=None,
        test_data=None,
        dataset_name: str | None = None,
    ):
        super().__init__(case_dir)
        self.model_type = model_type
        self.y_normalizer = y_normalizer
        self.y_scalar_normalizer = y_scalar_normalizer
        self.target_fields = list(target_fields or [])
        self.target_scalar_fields = list(target_scalar_fields or [])
        self.test_data = test_data
        self.dataset_name = dataset_name

    @staticmethod
    def _sequence_collate_fn(batch):
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

    def _model_forward(self, trainer: mlutils.Trainer, batch):
        if self.model_type == "glt":
            if not hasattr(batch, "x") or not hasattr(batch, "edge_index") or not hasattr(batch, "ptr"):
                raise ValueError(f"Static mesh GLT expects a PyG Batch with x/edge_index/ptr, got {type(batch)}")
            pos = batch.x[:, :2]
            feats = batch.x[:, 2:] if batch.x.shape[-1] > 2 else None
            ptr = batch.ptr.to(device=pos.device, dtype=torch.int32)
            lengths = ptr[1:] - ptr[:-1]
            yh = trainer.model(
                pos=pos,
                feats=feats,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch_index=batch.batch,
                use_flash_varlen=True,
                cu_seqlens=ptr,
                max_seqlen=int(lengths.max().item()) if lengths.numel() else 0,
                num_total_nodes=int(ptr[-1].item()) if ptr.numel() else int(pos.shape[0]),
                topology_features=getattr(batch, "laplacian_eig", None),
                topology_eigenvalues=getattr(batch, "laplacian_eigvals", None),
            )
            y = batch.y if hasattr(batch, "y") else None
            batch_index = batch.batch.to(yh.device) if torch.is_tensor(batch.batch) else batch.batch
            if torch.is_tensor(y) and y.device != yh.device:
                y = y.to(yh.device)
            return yh, y, batch_index, batch.num_graphs

        if self.model_type in self._GRAPH_MODELS:
            yh = trainer.model(batch)
            if hasattr(batch, "batch"):
                y = batch.y if hasattr(batch, "y") else None
                batch_index = batch.batch
                num_graphs = batch.num_graphs
            elif hasattr(batch, "ndata"):
                y = batch.ndata["y"] if ("y" in batch.ndata) else None
                counts = batch.batch_num_nodes().to(yh.device)
                num_graphs = int(len(counts))
                batch_index = torch.repeat_interleave(
                    torch.arange(num_graphs, device=yh.device, dtype=torch.long),
                    counts.long(),
                )
            else:
                raise ValueError(f"Unsupported graph batch type: {type(batch)}")
            if torch.is_tensor(batch_index) and (batch_index.device != yh.device):
                batch_index = batch_index.to(yh.device)
            if torch.is_tensor(y) and (y.device != yh.device):
                y = y.to(yh.device)
            return yh, y, batch_index, num_graphs

        if self.model_type in self._SEQUENCE_MODELS:
            if (
                isinstance(batch, (tuple, list))
                and len(batch) >= 2
                and torch.is_tensor(batch[0])
                and torch.is_tensor(batch[1])
            ):
                x, y = batch[0], batch[1]
            else:
                raise ValueError(
                    f"Unexpected batch format for mesh sequence model '{self.model_type}': {type(batch)}"
                )

            yh = trainer.model(x)

            if yh.ndim == 2:
                yh = yh.unsqueeze(0)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            if yh.ndim != 3 or y.ndim != 3:
                raise ValueError(
                    f"Expected [B, N, C] tensors for mesh sequence model '{self.model_type}', "
                    f"got yh={tuple(yh.shape)}, y={tuple(y.shape)}."
                )

            bsz, npts = yh.shape[0], yh.shape[1]
            batch_index = torch.arange(bsz, device=yh.device).repeat_interleave(npts)
            yh = yh.reshape(-1, yh.shape[-1])
            y = y.reshape(-1, y.shape[-1])
            return yh, y, batch_index, bsz

        raise NotImplementedError(f"MeshStaticCallback does not support model_type={self.model_type}.")

    @staticmethod
    def _per_graph_rel_l2(yh: torch.Tensor, y: torch.Tensor, batch_index: torch.Tensor, num_graphs: int):
        rel_l2s = []
        for gid in range(num_graphs):
            mask = batch_index == gid
            yh_g = yh[mask]
            y_g = y[mask]
            err = torch.sqrt(torch.sum((yh_g - y_g) ** 2))
            ref = torch.sqrt(torch.sum(y_g ** 2)) + 1e-12
            rel_l2s.append(err / ref)
        return torch.stack(rel_l2s)

    @staticmethod
    def _per_graph_plaid_rrmse(yh: torch.Tensor, y: torch.Tensor, batch_index: torch.Tensor, num_graphs: int):
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

    def _make_loader(self, trainer: mlutils.Trainer, dataset):
        if trainer.gnn_loader:
            if trainer.graph_loader_backend == "pyg":
                import torch_geometric as pyg
                return pyg.loader.DataLoader(
                    dataset,
                    batch_size=trainer.batch_size_,
                    shuffle=False,
                    num_workers=0,
                )
            raise ValueError(f"Unsupported graph loader backend: {trainer.graph_loader_backend}")

        collate_fn = self._sequence_collate_fn if self.model_type in self._SEQUENCE_MODELS else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=trainer.batch_size_,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    @torch.no_grad()
    def _evaluate_dataset(self, trainer: mlutils.Trainer, dataset):
        if dataset is None:
            return None

        loader = self._make_loader(trainer, dataset)

        rel_total = 0.0
        plaid_total = 0.0
        num_graphs_total = 0

        y_normalizer = self.y_normalizer.to(trainer.device)
        trainer.model.eval()
        for batch in loader:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                yh, y, batch_index, num_graphs = self._model_forward(trainer, batch)
            if y is None:
                return None
            yh = y_normalizer.decode(yh)
            y = y_normalizer.decode(y)
            rel_l2 = self._per_graph_rel_l2(yh, y, batch_index=batch_index, num_graphs=num_graphs)
            plaid_rrmse = self._per_graph_plaid_rrmse(yh, y, batch_index=batch_index, num_graphs=num_graphs)
            rel_total += rel_l2.sum().item()
            plaid_total += plaid_rrmse.sum().item()
            num_graphs_total += int(num_graphs)

        if num_graphs_total == 0:
            return None
        return dict(
            rel_l2=rel_total / num_graphs_total,
            plaid_rrmse=plaid_total / num_graphs_total,
        )

    @torch.no_grad()
    def _write_plaid_submission(self, trainer: mlutils.Trainer, ckpt_dir: str):
        if self.test_data is None or not self.target_fields:
            return None
        if trainer.GLOBAL_RANK != 0:
            return None

        loader = self._make_loader(trainer, self.test_data)
        y_normalizer = self.y_normalizer.to(trainer.device)
        y_scalar_normalizer = (
            None if self.y_scalar_normalizer is None else self.y_scalar_normalizer.to(trainer.device)
        )
        field_dim = len(self.target_fields)
        scalar_dim = len(self.target_scalar_fields)

        records_by_id = {}
        trainer.model.eval()
        for batch in loader:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                yh, _y, batch_index, num_graphs = self._model_forward(trainer, batch)
            yh_fields = y_normalizer.decode(yh[:, :field_dim])
            if scalar_dim > 0:
                scalar_values = []
                for gid in range(num_graphs):
                    mask = batch_index == gid
                    scalar_values.append(yh[mask, field_dim:field_dim + scalar_dim].mean(dim=0))
                scalar_values = torch.stack(scalar_values, dim=0)
                if y_scalar_normalizer is not None:
                    scalar_values = y_scalar_normalizer.decode(scalar_values)
            else:
                scalar_values = None

            ptr = batch.ptr.detach().cpu().tolist() if hasattr(batch, "ptr") else None
            sample_ids = batch.sample_id.detach().cpu().tolist() if hasattr(batch, "sample_id") else list(range(num_graphs))
            for gid in range(num_graphs):
                if ptr is None:
                    mask = batch_index == gid
                    fields_g = yh_fields[mask].detach().cpu()
                else:
                    fields_g = yh_fields[ptr[gid]:ptr[gid + 1]].detach().cpu()
                record = {}
                for field_idx, field_name in enumerate(self.target_fields):
                    record[field_name] = fields_g[:, field_idx].numpy()
                if scalar_values is not None:
                    scalars_g = scalar_values[gid].detach().cpu().numpy()
                    for scalar_idx, scalar_name in enumerate(self.target_scalar_fields):
                        record[scalar_name] = float(scalars_g[scalar_idx])
                records_by_id[int(sample_ids[gid])] = record

        ordered_records = [records_by_id[idx] for idx in sorted(records_by_id)]
        paths = []
        for root in (ckpt_dir, self.case_dir):
            path = os.path.join(root, "reference.pkl")
            with open(path, "wb") as f:
                pickle.dump(ordered_records, f)
            paths.append(path)
        return paths[0]

    def _maybe_submit_plaid_hf(self, reference_path: str | None, ckpt_dir: str):
        if reference_path is None or os.environ.get("PLAID_HF_SUBMIT") != "1":
            return

        from pdebench.dataset.plaid_hf import resolve_plaid_hf_benchmark_url

        token = os.environ.get("HF_TOKEN")
        model_id = os.environ.get("PLAID_HF_MODEL_ID", os.environ.get("HF_HUB_MODEL_ID", "FLARE-dev-GLT"))
        comment = os.environ.get("PLAID_HF_COMMENT", "FLARE-dev PDEBench submission")
        url = resolve_plaid_hf_benchmark_url(self.dataset_name)
        if not token:
            payload = {"error": "PLAID_HF_SUBMIT=1 but HF_TOKEN is not set."}
        else:
            import requests

            headers = {"Authorization": f"Bearer {token}"}
            with open(reference_path, "rb") as f:
                response = requests.post(
                    f"{url}/new_submission",
                    headers=headers,
                    data={"hub_model": model_id, "submission_comment": comment},
                    files={"submission_file": ("reference.pkl", f, "application/octet-stream")},
                    timeout=120,
                )
            payload = {
                "status_code": response.status_code,
                "response": response.text,
            }
            try:
                leaderboard_response = requests.post(
                    f"{url}/leaderboard",
                    json={"lb": "public"},
                    timeout=60,
                )
                payload["leaderboard_status_code"] = leaderboard_response.status_code
                payload["leaderboard_response"] = leaderboard_response.text
            except requests.RequestException as exc:
                payload["leaderboard_error"] = repr(exc)

        with open(os.path.join(ckpt_dir, "plaid_hf_submission.json"), "w") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(self.case_dir, "plaid_hf_submission.json"), "w") as f:
            json.dump(payload, f, indent=2)

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        train_stats = stat_vals.get("train_stats") or {}
        val_stats = stat_vals.get("test_stats") or {}
        test_stats = self._evaluate_dataset(trainer, self.test_data)
        reference_path = self._write_plaid_submission(trainer, ckpt_dir)
        self._maybe_submit_plaid_hf(reference_path, ckpt_dir)

        payload = dict(
            train_loss=stat_vals.get("train_loss"),
            val_loss=stat_vals.get("test_loss"),
            train_plaid_loss=train_stats.get("plaid_loss"),
            val_plaid_loss=val_stats.get("plaid_loss"),
            train_plaid_field_mse=train_stats.get("plaid_field_mse"),
            val_plaid_field_mse=val_stats.get("plaid_field_mse"),
            train_plaid_scalar_mse=train_stats.get("plaid_scalar_mse"),
            val_plaid_scalar_mse=val_stats.get("plaid_scalar_mse"),
            train_plaid_rrmse=train_stats.get("plaid_rrmse"),
            val_plaid_rrmse=val_stats.get("plaid_rrmse"),
            test_plaid_rrmse=None if test_stats is None else test_stats.get("plaid_rrmse"),
            train_plaid_scalar_rrmse=train_stats.get("plaid_scalar_rrmse"),
            val_plaid_scalar_rrmse=val_stats.get("plaid_scalar_rrmse"),
            train_total_error=train_stats.get("total_error"),
            val_total_error=val_stats.get("total_error"),
            # Optional diagnostic from unlabeled-test decode pass only (not train objective).
            test_rel_l2=None if test_stats is None else test_stats.get("rel_l2"),
        )

        if trainer.GLOBAL_RANK == 0:
            with open(os.path.join(ckpt_dir, "mesh_metrics.json"), "w") as f:
                json.dump(payload, f, indent=2)
            with open(os.path.join(self.case_dir, "mesh_metrics.json"), "w") as f:
                json.dump(payload, f, indent=2)

            rrmse_payload = {
                "train_plaid_rrmse": payload["train_plaid_rrmse"],
                "val_plaid_rrmse": payload["val_plaid_rrmse"],
                "test_plaid_rrmse": payload["test_plaid_rrmse"],
                "train_total_error": payload["train_total_error"],
                "val_total_error": payload["val_total_error"],
            }
            with open(os.path.join(ckpt_dir, "rel_error.json"), "w") as f:
                json.dump(rrmse_payload, f, indent=2)
            with open(os.path.join(self.case_dir, "rel_error.json"), "w") as f:
                json.dump(rrmse_payload, f, indent=2)

            if payload["train_plaid_loss"] is not None and payload["val_plaid_loss"] is not None:
                print(
                    "PLAID Vi-Transf loss (train/val): "
                    f"{payload['train_plaid_loss']:.6e} / {payload['val_plaid_loss']:.6e}"
                )
            if payload["train_plaid_field_mse"] is not None and payload["val_plaid_field_mse"] is not None:
                print(
                    "PLAID field MSE (train/val): "
                    f"{payload['train_plaid_field_mse']:.6e} / {payload['val_plaid_field_mse']:.6e}"
                )
            if payload["train_plaid_scalar_mse"] is not None and payload["val_plaid_scalar_mse"] is not None:
                print(
                    "PLAID scalar MSE (train/val): "
                    f"{payload['train_plaid_scalar_mse']:.6e} / {payload['val_plaid_scalar_mse']:.6e}"
                )
            train_rrmse = float("nan") if payload["train_plaid_rrmse"] is None else payload["train_plaid_rrmse"]
            val_rrmse = float("nan") if payload["val_plaid_rrmse"] is None else payload["val_plaid_rrmse"]
            test_rrmse = float("nan") if payload["test_plaid_rrmse"] is None else payload["test_plaid_rrmse"]
            print(
                "PLAID field RRMSE (train/val/test): "
                f"{train_rrmse:.6e} / {val_rrmse:.6e} / {test_rrmse:.6e}"
            )
            if payload["train_total_error"] is not None and payload["val_total_error"] is not None:
                print(
                    "PLAID total_error (train/val): "
                    f"{payload['train_total_error']:.6e} / {payload['val_total_error']:.6e}"
                )

        return

#======================================================================#
class ScoresCallback(mlutils.Callback):

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):

        trainer.model.eval()
        case_dir = self.case_dir

        assert trainer.WORLD_SIZE == 1, "ScoresCallback only supports single-rank evaluation"

        #--------------------------------#
        # get scores
        #--------------------------------#
        num_blocks = len(trainer.model.blocks)
        score_paths = [os.path.join(case_dir, 'scores', f'score_{block_idx}.pt') for block_idx in range(num_blocks)]

        if not all(os.path.exists(score_path) for score_path in score_paths):
            num_batches, MSE = 0, 0.0

            scores = [[] for _ in range(num_blocks)]

            for batch in trainer.loader_:
                x = batch[0].to(trainer.device)
                y = batch[1].to(trainer.device)
                yh, score = trainer.model(x, return_scores=True)

                n = trainer.get_batch_size(batch, trainer._loader_)
                num_batches += n
                MSE += ((yh - y).pow(2).mean() * n).item()
                for i in range(num_blocks):
                    scores[i].append(score[i].detach().cpu())

                del x, y, yh, score

            MSE = MSE / num_batches
            print()
            print(f"Train MSE: {MSE:.8e}")

            # save scores in case_dir/scores/score_<block_idx>.pt
            gc.collect()
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(case_dir, 'scores'), exist_ok=True)

            for block_idx in range(num_blocks):
                scores_ = torch.cat(scores[block_idx], dim=0)

                print(f"Saving scores to {os.path.join(case_dir, 'scores', f'score_{block_idx}.pt')}")
                torch.save(scores_, os.path.join(case_dir, 'scores', f'score_{block_idx}.pt'))

                del scores_
                scores[block_idx] = None
                gc.collect()
                torch.cuda.empty_cache()

        scores = [
            torch.load(
                os.path.join(case_dir, 'scores', f'score_{block_idx}.pt'),
                mmap=True,
                weights_only=True,
            )
            for block_idx in range(num_blocks)
        ]
        gc.collect()
        torch.cuda.empty_cache()

        #--------------------------------#
        # attention weights
        #--------------------------------#
        num_batches, num_heads, num_latents, num_points = scores[0].shape
        eigen_paths = [os.path.join(case_dir, 'eigen', f'eigen_{block_idx}.pt') for block_idx in range(num_blocks)]

        if not all(os.path.exists(eigen_path) for eigen_path in eigen_paths):
            eigenvals = [[] for _ in range(num_blocks)]
            eigenvecs = [[] for _ in range(num_blocks)]
            chunk_size = 2
            num_chunks = (num_batches + chunk_size - 1) // chunk_size

            for block_idx in range(num_blocks):
                gc.collect()
                torch.cuda.empty_cache()

                for chunk_idx in tqdm(range(num_chunks), desc=f"Processing block {block_idx}", ncols=80):
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, len(scores[block_idx]))
                    S = scores[block_idx][start:end].to(trainer.device)  # [B H M N]

                    S = S.clamp(min=-30, max=30)
                    A  = torch.exp(S)
                    rsum = A.sum(dim=-1) # [B H M]
                    csum = A.sum(dim=-2) # [B H N]
                    LN = torch.diag_embed(1. / csum) # [B H N N]
                    LM = torch.diag_embed(1. / rsum) # [B H M M]
                    LM_sqrt = torch.sqrt(LM)
                    LN_sqrt = torch.sqrt(LN)
                    B  = LM_sqrt @ A @ LN_sqrt
                    BBT = B @ B.mT # [B H M M]

                    if torch.isnan(BBT).any() or torch.isinf(BBT).any():
                        print(f"Block {block_idx}: Matrix contains NaN or Inf!")
                        print(f"S: {S.isnan().sum().item()}, {S.isinf().sum().item()}")
                        print(f"A: {A.isnan().sum().item()}, {A.isinf().sum().item()}")
                        print(f"rsum: {rsum.isnan().sum().item()}, {rsum.isinf().sum().item()}")
                        print(f"csum: {csum.isnan().sum().item()}, {csum.isinf().sum().item()}")
                        print(f"LN: {LN.isnan().sum().item()}, {LN.isinf().sum().item()}")
                        print(f"LM: {LM.isnan().sum().item()}, {LM.isinf().sum().item()}")
                        print(f"LM_sqrt: {LM_sqrt.isnan().sum().item()}, {LM_sqrt.isinf().sum().item()}")
                        print(f"LN_sqrt: {LN_sqrt.isnan().sum().item()}, {LN_sqrt.isinf().sum().item()}")
                        exit()

                    # SVD and eig decomposition are the same for symmetric matrices
                    U, SigmaSq, _ = torch.linalg.svd(BBT)
                    SigmaMat = torch.diag_embed(torch.sqrt(SigmaSq))

                    eigvals = SigmaSq
                    eigvecs = LN_sqrt @ B.mT @ U @ SigmaMat

                    eigenvals[block_idx].append(eigvals.detach().cpu())
                    eigenvecs[block_idx].append(eigvecs.detach().cpu())

                    del S, A, rsum, csum, LN, LM, LM_sqrt, LN_sqrt, B, BBT, SigmaMat, U, SigmaSq
                    del eigvals, eigvecs

            # save eigenvalues and eigenvectors in case_dir/eigen/eigen_<block_idx>.pt
            gc.collect()
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(case_dir, 'eigen'), exist_ok=True)

            for block_idx in range(num_blocks):
                eigenvals_ = torch.cat(eigenvals[block_idx], dim=0)
                eigenvecs_ = torch.cat(eigenvecs[block_idx], dim=0)

                print(
                    "Saving eigen(values/vectors) to "
                    f"{os.path.join(case_dir, 'eigen', f'eigen_{block_idx}.pt')}"
                )
                torch.save(
                    [eigenvals_, eigenvecs_],
                    os.path.join(case_dir, 'eigen', f'eigen_{block_idx}.pt'),
                )

                del eigenvals_, eigenvecs_
                eigenvals[block_idx] = None
                eigenvecs[block_idx] = None
                gc.collect()
                torch.cuda.empty_cache()

        eigen = [torch.load(eigen_path, mmap=True, weights_only=True) for eigen_path in eigen_paths]

        gc.collect()
        torch.cuda.empty_cache()

        #--------------------------------#
        # plot spectra
        #--------------------------------#

        eigenvals_means = [eigenvals.mean(dim=0) for (eigenvals, _) in eigen]

        nrows = math.ceil(math.sqrt(num_blocks))
        ncols = math.ceil(num_blocks / nrows)
        cutoff = int(num_latents * 1.1)

        # Plot mean eigenvalues across training cases
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        fig.suptitle('Eigenvalues of different heads for each block')
        for block in range(num_blocks):
            ax = axes.flat[block] if num_blocks > 1 else axes
            for h in range(num_heads):
                ax.plot(eigenvals_means[block][h, :cutoff])
            ax.axvline(x=num_latents-1, color='red', linestyle='--', label=f'Number of  Clusters = {num_latents}')
            ax.axhline(y=torch.finfo(torch.float32).eps, color='black', linestyle='--', label='Float32 Precision')
            ax.set_title(f'Block {block+1}')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Magnitude')
            # ax.legend()
            ax.set_yscale('log')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'eigenvals.png'))
        plt.close()

        #--------------------------------#
        # production quality spectra plot
        #--------------------------------#

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
        fontsize = 28

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale('linear')
            ax.set_yscale('log', base=10)
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.set_ylim(1e-8, 2e-0)
            ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])

        ax1.set_yticklabels(['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1e-0'])
        ax2.set_yticklabels(['', '', '', '', '', '', '', '', ''])
        ax3.set_yticklabels(['', '', '', '', '', '', '', '', ''])

        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)

        ax1.set_ylabel(r'Eigenvalue magnitude', fontsize=fontsize)
        ax1.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)
        ax2.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)
        ax3.set_xlabel(r'Eigenvalue Index', fontsize=fontsize)

        ax1.set_title(r'Block 1', fontsize=fontsize)
        ax2.set_title(r'Block 5', fontsize=fontsize)
        ax3.set_title(r'Block 8', fontsize=fontsize)

        linewidth = 2.5

        for block, ax in zip([0, 4, 7], [ax1, ax2, ax3]):
            for h in range(num_heads):
                ax.plot(eigenvals_means[block][h, :cutoff], linewidth=linewidth)

            ax.axvline(
                x=num_latents-1,
                color='red',
                linestyle='--',
                linewidth=linewidth,
                label=r'Number of latents = %d' % num_latents,
            )
            ax.axhline(
                y=torch.finfo(torch.float32).eps,
                color='black',
                linestyle='--',
                linewidth=linewidth,
                label=r'Float32 Precision',
            )
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'spectra.pdf'))
        plt.close()

        # #--------------------------------#
        # # Plot eigenvalues for first 10 test cases
        # #--------------------------------#

        # for case_idx in range(min(10, num_batches)):
        #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        #     fig.suptitle(f'Eigenvalues of different heads for each block - Case {case_idx}')

        #     for block in range(num_blocks):
        #         ax = axes.flat[block] if num_blocks > 1 else axes
        #         for h in range(num_heads):
        #             ax.plot(eigen[block][0][case_idx, h, :cutoff])
        #         ax.axvline(x=num_latents-1, color='red', linestyle='--', label=f'Number of  Clusters = {num_latents}')
        #         ax.axhline(y=torch.finfo(torch.float32).eps, color='black', linestyle='--', label='Float32 Precision')
        #         ax.set_title(f'Block {block+1}')
        #         ax.set_xlabel('Eigenvalue Index')
        #         ax.set_ylabel('Magnitude')
        #         ax.set_yscale('log')
        #         # ax.legend()
        #         # ax.set_ylim(bottom=torch.finfo(torch.float32).eps)
        #         ax.grid(True)

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(ckpt_dir, f'eigenvals{case_idx}.png'))
        #     plt.close()

        # #--------------------------------#
        # # cosine similarity of eigenvalue spectra
        # #--------------------------------#

        # nrows = math.ceil(math.sqrt(num_blocks))
        # ncols = math.ceil(num_blocks / nrows)
        # cutoff = int(num_latents * 1.1)

        # # Plot mean eigenvalues across training cases
        # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        # fig.suptitle('Similarity of eigenvalue spectra among heads for each block')
        # for block in range(num_blocks):
        #     ax = axes.flat[block] if num_blocks > 1 else axes

        #     eigvals = eigenvals_means[block] # [H, N]
        #     eigvals = eigvals.abs() / eigvals.abs().sum(dim=-1, keepdim=True) # abs is unnecessary, but just in case

        #     # cosine similarity between heads
        #     similarity = []
        #     for h1 in range(num_heads):
        #         for h2 in range(h1+1, num_heads):
        #             cos_sim = F.cosine_similarity(eigvals[h1], eigvals[h2], dim=-1)
        #             similarity.append(cos_sim)

        #     # Create empty matrix and fill upper triangle (excluding diagonal)
        #     similarity_matrix = torch.zeros(num_heads, num_heads)
        #     triu_indices = torch.triu_indices(num_heads, num_heads, offset=1)
        #     similarity_matrix[triu_indices[0], triu_indices[1]] = torch.tensor(similarity)
        #     # Make matrix symmetric by copying upper triangle to lower
        #     similarity_matrix = similarity_matrix + similarity_matrix.T
        #     # set diagonal to 1
        #     similarity_matrix.fill_diagonal_(1.0)

        #     # set colorbar range to -1 to 1
        #     im = ax.imshow(similarity_matrix, cmap='gray', aspect='auto', vmin=-1, vmax=1)
        #     ax.set_title(f'Block {block+1}')
        #     ax.set_xlabel('Head Index')
        #     ax.set_ylabel('Head Index')
        #     ax.grid(True)

        #     cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_ticks([-1, 0, 1])
        #     cbar.set_ticklabels(['-1', '0', '1'])

        #     ax.set_xticks(range(num_heads))
        #     ax.set_yticks(range(num_heads))
        #     ax.set_xticklabels(range(num_heads))
        #     ax.set_yticklabels(range(num_heads))

        # plt.tight_layout()
        # plt.savefig(os.path.join(ckpt_dir, 'eigenvals_similarity.png'))
        # plt.close()


        #--------------------------------#
        # visualize eigenvectors for individual cases
        #--------------------------------#
        # for case_idx in range(min(10, num_batches)):
        #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3))
        #     fig.suptitle(f'Eigenvectors of different heads for each block - Case {case_idx}')

        #     for block in range(L):
        #         ax = axes.flat[block] if L > 1 else axes
        #         for h in range(num_heads):
        #             ax.imshow(eigenvecs[block][case_idx, h, :, :])
        #             ax.set_title(f'Block {block+1}, Head {h}')
        #             ax.set_xlabel('Cluster Index')
        #             ax.set_ylabel('Point Index')
        #             ax.grid(True)

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(ckpt_dir, f'eigenvecs{case_idx}.png'))
        #     plt.close()

        #--------------------------------#
        # cosine similarity between eigenvectors for individual cases
        #--------------------------------#

        # cluster utilizaton
        # connection sparsity
        # cross-head correlation <-- how much do attention weights correlate across heads?
        # plot eigenvalue decay of W

        # # Attention sparsity: what proportion of attention weights are non-zero?
        # Att_sparsity = [(Att > 1e-2).sum(dim=-2).float().mean().item() * 100 for Att in Atts]

        # print(f"Att sparsity: {[round(s, 2) for s in Att_sparsity]}")

        # print(f"Cluster utilization: How evenly are clusters used?")
        # print(f"Want every cluster to be used equally often, avoiding scenarios where")
        # print(f"some clusters are always ignored (underutilized) or overly relied upon (overutilized).")
        # print(f"Sparsity: What proportion of clusters are invoked per point?")

        # # Cluster utilization (how evenly are clusters used)
        # target_use = 1 / M
        # threshold = 0.5 * target_use
        # cluster_use = [w.mean(dim=-1) for w in W_encodes] # [B H M]
        # underused = [(cluster_use[i] < (target_use - threshold)).float().mean().item() * 100 for i in range(B)]
        # overused = [(cluster_use[i] > (target_use + threshold)).float().mean().item() * 100 for i in range(B)]

        # print()
        # print(f"Cluster utilization stats:")
        # print(f"  Mean: {[round(s.mean().item(), 4) for s in cluster_use]} (Target: {target_use:.5f})")
        # print(f"  Std : {[round(s.std(dim=-1).mean().item(), 4) for s in cluster_use]}")
        # print(f"  Min : {[round(s.min().item(), 4) for s in cluster_use]}")
        # print(f"  Max : {[round(s.max().item(), 4) for s in cluster_use]}")
        # print(
        #     f"  % Underused (< {target_use - threshold:.5f}): {[round(u, 2) for u in underused]}. "
        #     f"Mean: {sum(underused) / len(underused):.4f}"
        # )
        # print(
        #     f"  % Overused  (> {target_use + threshold:.5f}): {[round(o, 2) for o in overused]}. "
        #     f"Mean: {sum(overused) / len(overused):.4f}"
        # )
        # print(f"  % Sparsity : {[round(s, 2) for s in Att_sparsity]}. Mean: {sum(Att_sparsity) / len(Att_sparsity):.4f}")
        # print()

        # mean_cluster_use = [w.mean(dim=[0,-1]) / target_use for w in W_encodes]

        # fig, axes = plt.subplots(B, 2, figsize=(10, 3*B))
        # fig.suptitle('Cluster Utilization')

        # for i in range(B):
        #     Att = Atts[i]
        #     ax = axes[i, 0]
        #     im = ax.imshow(Att[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=100)
        #     ax.set_title(f'Layer {i}: sparsity: {Att_sparsity[i]:.1f}%')
        #     ax.set_xlabel('Cluster Index')
        #     ax.set_ylabel('Head Index')
        #     fig.colorbar(im, ax=ax)
        #     im.cmap.set_over('red')
        #     im.cmap.set_under('blue')

        # plt.tight_layout()
        # plt.savefig(os.path.join(ckpt_dir, 'utilization.png'))
        # plt.savefig(os.path.join(ckpt_dir, '..', 'utilization.png'))

        return

#======================================================================#
class MixerDiagnosticsCallback:
    """Per-step stability diagnostics for `mixer_backbone` (LP screening wave).

    Reads `trainer.model.last_diagnostics`, populated by `MixerBackboneModel.forward`
    when `MixerBackboneConfig.diagnostics=True`. Registered on the trainer's
    `batch_end` event (not `mlutils.Callback`, which is the heavier periodic
    checkpoint/eval hook triggered every `stats_every` steps/epochs).

    Writes one JSON record per (subsampled) step to `<case_dir>/diagnostics.jsonl`
    and tracks `first_nonfinite_step` across train loss / grad norm / diagnostic RMS.
    """

    # nonfinite watch-list: RMS-style stability signals only (gate/entropy/cos are
    # informational, not stability alarms -- e.g. gate_value is NaN by design when off).
    _RMS_KEYS = (
        "residual_stream_rms", "mixer_out_rms", "mixer_over_stream",
        "rms_q_dynamic", "rms_k0", "rms_v0", "rms_k", "rms_v",
    )

    def __init__(self, case_dir: str, *, stats_every: int = 1, print_every: int | None = None):
        self.case_dir = case_dir
        self.stats_every = max(1, int(stats_every))
        self.print_every = print_every
        self.first_nonfinite_step: int | None = None
        os.makedirs(case_dir, exist_ok=True)
        self._jsonl_path = os.path.join(case_dir, 'diagnostics.jsonl')
        self._summary_path = os.path.join(case_dir, 'mixer_diagnostics_summary.json')

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _is_nonfinite(self, train_loss, grad_norm, mean: dict) -> bool:
        for value in (train_loss, grad_norm):
            if value is not None and not math.isfinite(value):
                return True
        for key in self._RMS_KEYS:
            value = mean.get(key)
            if value is not None and not math.isfinite(value):
                return True
        return False

    def __call__(self, trainer: mlutils.Trainer) -> None:
        if trainer.GLOBAL_RANK != 0:
            return

        model = self._unwrap_model(trainer.model)
        diagnostics = getattr(model, 'last_diagnostics', None)
        if diagnostics is None:
            return

        step = trainer.step
        if (step % self.stats_every) != 0:
            return

        mean = diagnostics.get('mean', {}) or {}
        train_loss = trainer.train_loss_per_batch[-1] if trainer.train_loss_per_batch else None
        grad_norm = trainer.grad_norm_per_step[-1] if trainer.grad_norm_per_step else None
        step_time = trainer.time_per_step[-1] if trainer.time_per_step else None
        max_memory_allocated = getattr(trainer, 'max_memory_allocated', None)
        peak_memory = max_memory_allocated[-1] if max_memory_allocated else None

        record = {
            'step': step,
            'epoch': trainer.epoch,
            'train_loss': train_loss,
            'grad_norm': grad_norm,
            'step_time': step_time,
            'peak_memory_allocated': peak_memory,
            **{f'mean_{key}': value for key, value in mean.items()},
        }

        nonfinite = self._is_nonfinite(train_loss, grad_norm, mean)
        if nonfinite and self.first_nonfinite_step is None:
            self.first_nonfinite_step = step
            print(f"[MixerDiagnostics] first nonfinite value detected at step {step}: {record}")

        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

        with open(self._summary_path, 'w') as f:
            json.dump({
                'first_nonfinite_step': self.first_nonfinite_step,
                'last_step': step,
                'last_record': record,
            }, f, indent=2)

        if self.print_every is None:
            log_rank_every_steps = getattr(trainer, 'log_rank_every_steps', 0) or 0
            self.print_every = log_rank_every_steps if log_rank_every_steps > 0 else max(1, trainer.stats_every)

        if nonfinite or (step % self.print_every) == 0:
            loss_str = f"{train_loss:.4e}" if train_loss is not None else "None"
            grad_str = f"{grad_norm:.4e}" if grad_norm is not None else "None"
            mean_str = " ".join(f"{key}={value:.4e}" for key, value in sorted(mean.items()))
            print(f"[MixerDiagnostics] step={step} loss={loss_str} grad_norm={grad_str} {mean_str}")

        return

#======================================================================#
def eig1(S):
    _, _, M, N = S.shape

    We = F.softmax(S, dim=-1) # sum over N
    Wd = F.softmax(S, dim=-2) # sum over M
    W = Wd.mT @ We

    # U, Sv, V = torch.linalg.svd(W, full_matrices=False)
    # eigvals = Sv[:,:,:M]
    # eigvecs = V[:,:,:M,:]

    eigvals, eigvecs = torch.linalg.eig(W) # [B H N], [B H N N]
    eigvals = eigvals[:,:,:M]
    eigvecs = eigvecs[:,:,:,:M]

    print(f'Eig1: eigvals: {eigvals.shape}, eigvecs: {eigvecs.shape}')

    return We, Wd, W, eigvals, eigvecs

def eig2(S):
    _, _, M, N = S.shape

    A  = torch.exp(S)
    rsum = A.sum(dim=-1) # [B H M]
    csum = A.sum(dim=-2) # [B H N]

    LN = torch.diag_embed(1. / csum) # [B H N N]
    LM = torch.diag_embed(1. / rsum) # [B H M M]

    We = LM @ A
    Wd = A @ LN

    W1 = Wd.mT @ We          # verified
    W2 = LN @ A.mT @ LM @ A  # verified

    LM_sqrt = torch.sqrt(LM)
    LN_sqrt = torch.sqrt(LN)
    LN_sqrt_inv = torch.diag_embed(1.0 / torch.diagonal(LN_sqrt, dim1=-2, dim2=-1))
    B  = LM_sqrt @ A @ LN_sqrt

    BTB = B.mT @ B
    W = LN_sqrt @ BTB @ LN_sqrt_inv

    tol = 1e-6
    assert (W1 - W).abs().max() < tol
    assert (W2 - W).abs().max() < tol

    # ### METHOD 1
    # U, Sv, V = torch.linalg.svd(B, full_matrices=False)
    # eigvals = Sv**2
    # eigvecs = (LN_sqrt @ V.mT).mT

    ### METHOD 2
    BBT = B @ B.mT # [B H M M]
    U, SigmaSq, _ = torch.linalg.svd(BBT) # SVD and eig decomposition are the same for symmetric matrices
    SigmaMat = torch.diag_embed(torch.sqrt(SigmaSq))

    eigvals = SigmaSq
    eigvecs = LN_sqrt @ B.mT @ U @ torch.linalg.inv(SigmaMat)

    print(f'Eig2: eigvals: {eigvals.shape}, eigvecs: {eigvecs.shape}')

    return We, Wd, W, eigvals, eigvecs

def main():
    B, H, M, N = 2, 4, 16, 50

    S = torch.rand(B, H, M, N)

    We1, Wd1, W1, eigvals1, eigvecs1 = eig1(S)
    We2, Wd2, W2, eigvals2, eigvecs2 = eig2(S)

    ranks1 = torch.linalg.matrix_rank(W1)
    ranks2 = torch.linalg.matrix_rank(W2)

    assert (ranks1 == M).all()
    assert (ranks2 == M).all()

    e1 = (We1 - We2).abs().max()
    e2 = (Wd1 - Wd2).abs().max()
    e3 = (W1  - W2 ).abs().max()
    e4 = (eigvals1 - eigvals2).norm(2) / eigvals1.numel()
    e5 = (eigvecs1 - eigvecs2).norm(2) / eigvecs1.numel()
    print(f"We: {e1:.4f}, Wd: {e2:.4f}, W: {e3:.4f}, eigvals: {e4:.4f}, eigvecs: {e5:.4f}")

if __name__ == "__main__":
    main()
#======================================================================#
#

from __future__ import annotations

import torch
import torch.distributed as dist

from pdebench.dataset.ginot.forward import ginot_model_forward, ginot_postprocess_displacement
from pdebench.dataset.loss import compute_packed_loss
from pdebench.dataset.sample import LossSpec


def field_squared_error_sum(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"Expected matching [N,C] tensors, got pred={tuple(pred.shape)} target={tuple(target.shape)}.")
    return (pred - target).square().sum(dim=0), int(pred.shape[0])


def make_ginot_statsfun(cfg, metadata):
    @torch.no_grad()
    def statsfun(trainer, loader, split=None):
        del split
        model = trainer.model.module if getattr(trainer, "DDP", False) else trainer.model
        model.eval()
        y_normalizer = metadata["y_normalizer"]
        rel_total = 0.0
        num_graphs_total = 0
        track_fields = metadata.get("dataset") == "bumper_beam"
        target_fields = tuple(metadata.get("target_fields", ()))
        field_sse = torch.zeros(len(target_fields), device=trainer.device) if track_fields else None
        field_count = 0

        for batch in loader:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                yh, y, batch_index, num_graphs = ginot_model_forward(cfg, model, batch)
            normalizer = y_normalizer.to(y.device)
            yh = ginot_postprocess_displacement(yh, batch, y_normalizer=normalizer)
            if field_sse is not None:
                batch_sse, batch_count = field_squared_error_sum(yh, y)
                field_sse += batch_sse.to(device=field_sse.device)
                field_count += batch_count
            loss_spec = LossSpec(mask="free_mask" if batch.get("flat_free_mask") is not None else None)
            masks = {"free_mask": batch["flat_free_mask"]} if loss_spec.mask else None
            # Mean over graphs in this batch; accumulate graph-weighted total.
            batch_mean = compute_packed_loss(
                yh,
                y,
                normalizer,
                loss_spec,
                batch_index=batch_index,
                num_graphs=num_graphs,
                masks=masks,
            )
            rel_total += float(batch_mean.item()) * int(num_graphs)
            num_graphs_total += int(num_graphs)

        if trainer.DDP:
            reduced = []
            for value in [rel_total, num_graphs_total]:
                tensor = torch.tensor(value, device=trainer.device)
                dist.all_reduce(tensor, dist.ReduceOp.SUM)
                reduced.append(tensor.item())
            rel_total, num_graphs_total = reduced
            if field_sse is not None:
                dist.all_reduce(field_sse, dist.ReduceOp.SUM)
                field_count_tensor = torch.tensor(field_count, device=trainer.device)
                dist.all_reduce(field_count_tensor, dist.ReduceOp.SUM)
                field_count = int(field_count_tensor.item())

        if num_graphs_total == 0:
            return None, {"rel_l2": None}
        rel_l2 = rel_total / num_graphs_total
        metrics = {"rel_l2": rel_l2}
        if field_sse is not None and field_count > 0:
            metrics.update({
                f"field_mse/{name}": float(field_sse[index].item() / field_count)
                for index, name in enumerate(target_fields)
            })
        return rel_l2, metrics

    return statsfun

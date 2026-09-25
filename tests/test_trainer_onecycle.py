from __future__ import annotations

import torch

from mlutils.trainer import _safe_one_cycle_pct_start


def _one_cycle_lrs(total_steps: int, pct_start: float) -> list[float]:
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=1e4,
        final_div_factor=1e4,
    )
    lrs = [schedule.get_last_lr()[0]]
    for _ in range(total_steps):
        opt.step()
        schedule.step()
        lrs.append(schedule.get_last_lr()[0])
    return lrs


def test_safe_one_cycle_pct_start_preserves_normal_schedules() -> None:
    for total_steps, pct_start in [
        (100, 0.1),
        (100, 0.2),
        (100, 0.3),
        (500, 0.1),
        (1000, 0.1),
    ]:
        adjusted_pct_start = _safe_one_cycle_pct_start(pct_start, total_steps)

        assert adjusted_pct_start == pct_start
        assert _one_cycle_lrs(total_steps, adjusted_pct_start) == _one_cycle_lrs(total_steps, pct_start)


def test_safe_one_cycle_pct_start_repairs_tiny_invalid_schedule() -> None:
    adjusted_pct_start = _safe_one_cycle_pct_start(0.1, 10)

    assert adjusted_pct_start > 0.1
    _one_cycle_lrs(10, adjusted_pct_start)

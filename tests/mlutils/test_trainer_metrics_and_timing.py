from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import mlutils
from mlutils.callbacks import Callback
from mlutils.metrics import format_scalar, normalize_metric
from mlutils.run_timer import RunTimer


class _TinyXY(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.tensor([float(idx)], dtype=torch.float32)
        return x, x


def _make_trainer(**kwargs) -> mlutils.Trainer:
    defaults = dict(
        model=torch.nn.Linear(1, 1),
        _data=_TinyXY(8),
        data_=_TinyXY(4),
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        batch_size_=4,
        steps=2,
        epochs=0,
        stats_on_start=False,
        _fullbatch_stats=True,
        fullbatch_stats_=True,
        print_iterator=False,
        verbose=False,
        num_workers=0,
        overlap_train_dataloader=False,
    )
    defaults.update(kwargs)
    trainer = mlutils.Trainer(**defaults)

    def batch_lossfun(trainer, model, batch):
        x, y = batch
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        return torch.nn.functional.mse_loss(model(x), y)

    trainer.batch_lossfun = batch_lossfun
    return trainer


def test_format_scalar_respects_precision() -> None:
    assert format_scalar(0.5, precision=4) == "5.0000e-01"
    assert format_scalar(None, precision=4) == "null"
    assert format_scalar(float("nan"), precision=4) == "null"


def test_statistics_returns_none_when_fullbatch_disabled() -> None:
    trainer = _make_trainer(_fullbatch_stats=False, fullbatch_stats_=False)
    trainer.make_dataloader()
    trainer.statistics()
    assert trainer.train_loss_fullbatch[-1] is None
    assert trainer.test_loss_fullbatch[-1] is None


def test_statistics_normalizes_nan_from_statsfun() -> None:
    def statsfun(trainer, loader, split=None):
        del trainer, loader, split
        return float("nan"), {}

    trainer = _make_trainer(statsfun=statsfun)
    trainer.make_dataloader()
    trainer.statistics()
    assert trainer.train_loss_fullbatch[-1] is None
    assert trainer.test_loss_fullbatch[-1] is None


def test_fallback_statsfun_empty_loader_returns_none() -> None:
    trainer = _make_trainer(_fullbatch_stats=False, fullbatch_stats_=False)
    trainer.make_dataloader()
    empty_loader = DataLoader(_TinyXY(0), batch_size=1)
    loss, stats = trainer.fallback_statsfun(empty_loader, split="train")
    assert loss is None
    assert stats == {}


def test_run_timer_marks_statistics_only_once() -> None:
    timer = RunTimer(enabled=True, rank=0, log_rank=0, print_marks=False)
    trainer = _make_trainer(run_timer=timer, _fullbatch_stats=False, fullbatch_stats_=False)
    trainer.make_dataloader()
    trainer.statistics()
    trainer.statistics()
    stat_marks = [name for name, _, _ in timer.marks if name.startswith("trainer_statistics")]
    assert stat_marks == ["trainer_statistics_start", "trainer_statistics_done"]


def test_record_cuda_memory_keeps_utilization_in_sync_with_peak() -> None:
    if not torch.cuda.is_available():
        return
    trainer = _make_trainer(device="cuda")
    trainer.record_cuda_memory()
    assert trainer.memory_utilization[-1] == trainer.max_memory_allocated[-1]


def test_stats_on_start_triggers_batch_end_callback_at_step_zero() -> None:
    events: list[str] = []

    def on_batch_end(trainer):
        events.append(f"batch_end:step={trainer.step}")

    trainer = _make_trainer(stats_on_start=True, steps=10, stats_every=1)
    trainer.add_callback("batch_end", on_batch_end)
    trainer.make_dataloader()
    trainer._run_train_start()
    assert events == ["batch_end:step=0"]


def test_callback_model_stats_use_null_for_empty_timing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp)
        trainer = _make_trainer(steps=10, stats_every=1)
        trainer.make_dataloader()
        callback = Callback(str(case_dir), save_every=1)
        callback(trainer, final=False)
        model_stats = json.loads((case_dir / "model_stats.json").read_text())
        assert model_stats["avg_time_per_step"] is None
        assert "NaN" not in json.dumps(model_stats)


def test_callback_skips_startup_fullbatch_stats_by_default() -> None:
    calls = 0

    def statsfun(trainer, loader, split=None):
        del trainer, loader, split
        nonlocal calls
        calls += 1
        return 1.0, {}

    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(statsfun=statsfun, epochs=10, steps=0, fullbatch_stats_on_start=False)
        trainer.make_dataloader()
        callback = Callback(str(tmp), save_every=10)
        callback(trainer, final=False)
    assert calls == 0


def test_callback_runs_startup_fullbatch_stats_when_enabled() -> None:
    calls = 0

    def statsfun(trainer, loader, split=None):
        del trainer, loader, split
        nonlocal calls
        calls += 1
        return 1.0, {}

    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(statsfun=statsfun, epochs=10, steps=0, fullbatch_stats_on_start=True)
        trainer.make_dataloader()
        callback = Callback(str(tmp), save_every=10)
        callback(trainer, final=False)
    assert calls > 0


def test_timing_mark_helper_delegates_to_run_timer() -> None:
    timer = RunTimer(enabled=True, rank=0, log_rank=0, print_marks=False)
    trainer = _make_trainer(run_timer=timer)
    trainer._timing_mark("custom_mark")
    assert timer.marks[-1][0] == "custom_mark"


def test_normalize_metric_on_fallback_finite_loss() -> None:
    trainer = _make_trainer(_fullbatch_stats=False, fullbatch_stats_=False)
    trainer.make_dataloader()
    loader = DataLoader(_TinyXY(4), batch_size=4)
    loss, _ = trainer.fallback_statsfun(loader, split="train")
    assert loss == normalize_metric(loss)
    assert isinstance(loss, float)

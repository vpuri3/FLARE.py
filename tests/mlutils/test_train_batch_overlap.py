from __future__ import annotations

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

import mlutils
from mlutils.utils import StepBudgetBatchSampler


class _TinyDataset(Dataset):
    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.tensor([float(idx)], dtype=torch.float32)


class _CompiledWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        raise AssertionError("compiled model should not be used for stats")


def test_step_budget_batch_sampler_cycles_until_budget() -> None:
    data = _TinyDataset(4)
    inner = BatchSampler(RandomSampler(data), batch_size=4, drop_last=False)
    sampler = StepBudgetBatchSampler(inner, total_batches=7, get_epoch=lambda: 0)
    batches = list(sampler)
    assert len(batches) == 7
    assert len(sampler) == 7


def test_continuous_auto_enables_for_single_batch_epoch() -> None:
    data = _TinyDataset(4)

    def batch_lossfun(trainer, model, batch):
        x = batch.unsqueeze(-1)
        return torch.nn.functional.mse_loss(model(x), x)

    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=6,
        epochs=0,
        continuous_train_batches=None,
        stats_on_start=False,
        _fullbatch_stats=False,
        fullbatch_stats_=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.batch_lossfun = batch_lossfun
    trainer.make_dataloader()
    assert trainer._use_continuous_train_batches is True
    assert trainer._batches_per_train_epoch == 1


def test_continuous_disabled_for_multi_batch_epoch() -> None:
    data = _TinyDataset(8)

    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=8,
        epochs=0,
        continuous_train_batches=None,
        stats_on_start=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.make_dataloader()
    assert trainer._use_continuous_train_batches is False
    assert trainer._batches_per_train_epoch == 2


def test_continuous_training_runs_requested_steps() -> None:
    data = _TinyDataset(4)

    def batch_lossfun(trainer, model, batch):
        x = batch.unsqueeze(-1)
        return torch.nn.functional.mse_loss(model(x), x)

    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=6,
        epochs=0,
        continuous_train_batches=True,
        overlap_train_dataloader=False,
        stats_on_start=False,
        _fullbatch_stats=False,
        fullbatch_stats_=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.batch_lossfun = batch_lossfun
    trainer.train()
    assert trainer.step == 6
    assert len(trainer.time_dataload_per_step) == 6
    assert len(trainer.time_per_epoch) == 5


def test_continuous_false_single_batch_epoch_still_runs_requested_steps() -> None:
    data = _TinyDataset(4)

    def batch_lossfun(trainer, model, batch):
        x = batch.unsqueeze(-1)
        return torch.nn.functional.mse_loss(model(x), x)

    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=6,
        epochs=0,
        continuous_train_batches=False,
        overlap_train_dataloader=False,
        stats_on_start=False,
        _fullbatch_stats=False,
        fullbatch_stats_=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.batch_lossfun = batch_lossfun
    trainer.train()
    assert trainer._use_continuous_train_batches is False
    assert trainer.step == 6
    assert len(trainer.time_dataload_per_step) == 6
    assert len(trainer.time_per_epoch) == 5


def test_continuous_true_installs_step_budget_batch_sampler() -> None:
    data = _TinyDataset(8)

    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=7,
        epochs=0,
        continuous_train_batches=True,
        stats_on_start=False,
        _fullbatch_stats=False,
        fullbatch_stats_=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.make_dataloader()
    assert trainer._use_continuous_train_batches is True
    assert isinstance(trainer._loader.batch_sampler, StepBudgetBatchSampler)
    assert len(trainer._loader.batch_sampler) == 7


def test_compiled_trainer_uses_eager_model_for_stats_by_default(monkeypatch) -> None:
    data = _TinyDataset(4)

    def fake_compile(module, **kwargs):
        assert kwargs == {}
        return _CompiledWrapper(module)

    def statsfun(trainer, loader, split=None):
        assert not isinstance(trainer.model, _CompiledWrapper)
        return 0.0, {}

    monkeypatch.setattr(torch, "compile", fake_compile)
    trainer = mlutils.Trainer(
        model=torch.nn.Linear(1, 1),
        _data=data,
        device="cpu",
        compile_model=True,
        compile_stats_model=False,
        mixed_precision=False,
        _batch_size=4,
        steps=1,
        epochs=0,
        statsfun=statsfun,
        stats_on_start=False,
        _fullbatch_stats=True,
        fullbatch_stats_=False,
        print_iterator=False,
        verbose=False,
        num_workers=0,
    )
    trainer.make_dataloader()
    trainer.statistics()

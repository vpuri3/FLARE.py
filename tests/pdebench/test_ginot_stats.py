from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import pdebench.dataset.ginot.stats as ginot_stats
from pdebench.dataset.ginot.stats import make_ginot_statsfun


def test_ginot_statsfun_empty_loader_returns_null_metrics() -> None:
    cfg = SimpleNamespace()
    metadata = {"y_normalizer": MagicMock()}
    metadata["y_normalizer"].to.side_effect = lambda y: y
    metadata["y_normalizer"].decode.side_effect = lambda y: y

    trainer = SimpleNamespace(
        DDP=False,
        model=MagicMock(),
        device=torch.device("cpu"),
        auto_cast=torch.autocast("cpu", enabled=False),
        move_to_device=lambda batch: batch,
    )

    statsfun = make_ginot_statsfun(cfg, metadata)
    empty_loader = DataLoader(TensorDataset(torch.zeros(0, 1)), batch_size=1)
    loss, stats = statsfun(trainer, empty_loader, split="val")
    assert loss is None
    assert stats == {"rel_l2": None}


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_ginot_statsfun_propagates_non_finite_batch_metric(monkeypatch, non_finite: float) -> None:
    metadata = {"y_normalizer": SimpleNamespace(to=lambda y: y)}
    trainer = SimpleNamespace(
        DDP=False,
        model=MagicMock(),
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
    )
    batches = [{"flat_free_mask": None}, {"flat_free_mask": None}]
    monkeypatch.setattr(
        ginot_stats,
        "ginot_model_forward",
        lambda _cfg, _model, _batch: (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long),
            1,
        ),
    )
    monkeypatch.setattr(ginot_stats, "ginot_postprocess_displacement", lambda yh, _batch, **_kwargs: yh)
    batch_metrics = iter([torch.tensor(1.0), torch.tensor(non_finite)])
    monkeypatch.setattr(ginot_stats, "compute_packed_loss", lambda *_args, **_kwargs: next(batch_metrics))

    statsfun = make_ginot_statsfun(SimpleNamespace(), metadata)
    loss, stats = statsfun(trainer, batches)
    assert not math.isfinite(loss)
    assert not math.isfinite(stats["rel_l2"])


def test_ginot_statsfun_bumper_beam_reports_channelwise_field_mse(monkeypatch) -> None:
    metadata = {
        "dataset": "bumper_beam",
        "target_fields": ("field_a", "field_b"),
        "y_normalizer": SimpleNamespace(to=lambda _device: None),
    }
    trainer = SimpleNamespace(
        DDP=False,
        model=MagicMock(),
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
    )
    batches = [{"flat_free_mask": None}, {"flat_free_mask": None}]
    outputs = iter(
        [
            (
                torch.tensor([[1.0, 3.0], [4.0, 8.0]]),
                torch.tensor([[0.0, 1.0], [2.0, 5.0]]),
                torch.zeros(2, dtype=torch.long),
                1,
            ),
            (
                torch.tensor([[3.0, 0.0]]),
                torch.tensor([[1.0, 1.0]]),
                torch.zeros(1, dtype=torch.long),
                3,
            ),
        ]
    )
    monkeypatch.setattr(ginot_stats, "ginot_model_forward", lambda _cfg, _model, _batch: next(outputs))
    monkeypatch.setattr(ginot_stats, "ginot_postprocess_displacement", lambda yh, _batch, **_kwargs: yh)
    batch_metrics = iter([torch.tensor(2.0), torch.tensor(4.0)])
    monkeypatch.setattr(ginot_stats, "compute_packed_loss", lambda *_args, **_kwargs: next(batch_metrics))

    statsfun = make_ginot_statsfun(SimpleNamespace(), metadata)
    loss, stats = statsfun(trainer, batches)

    assert loss == pytest.approx(3.5)
    assert stats["rel_l2"] == pytest.approx(3.5)
    assert stats["field_mse/field_a"] == pytest.approx(3.0)
    assert stats["field_mse/field_b"] == pytest.approx(14.0 / 3.0)


def test_ginot_statsfun_non_bumper_keeps_legacy_metric_shape(monkeypatch) -> None:
    metadata = {
        "dataset": "poisson_unstructured",
        "target_fields": ("field_a", "field_b"),
        "y_normalizer": SimpleNamespace(to=lambda _device: None),
    }
    trainer = SimpleNamespace(
        DDP=False,
        model=MagicMock(),
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
    )
    monkeypatch.setattr(
        ginot_stats,
        "ginot_model_forward",
        lambda _cfg, _model, _batch: (
            torch.tensor([[1.0, 3.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, dtype=torch.long),
            2,
        ),
    )
    monkeypatch.setattr(ginot_stats, "ginot_postprocess_displacement", lambda yh, _batch, **_kwargs: yh)
    monkeypatch.setattr(ginot_stats, "compute_packed_loss", lambda *_args, **_kwargs: torch.tensor(1.25))

    statsfun = make_ginot_statsfun(SimpleNamespace(), metadata)
    loss, stats = statsfun(trainer, [{"flat_free_mask": None}])

    assert loss == pytest.approx(1.25)
    assert set(stats) == {"rel_l2"}
    assert stats["rel_l2"] == pytest.approx(1.25)


def test_ginot_statsfun_bumper_beam_allreduces_field_state_in_ddp(monkeypatch) -> None:
    metadata = {
        "dataset": "bumper_beam",
        "target_fields": ("field_a", "field_b"),
        "y_normalizer": SimpleNamespace(to=lambda _device: None),
    }
    trainer = SimpleNamespace(
        DDP=True,
        model=SimpleNamespace(module=MagicMock()),
        device=torch.device("cpu"),
        auto_cast=nullcontext(),
        move_to_device=lambda batch: batch,
    )
    monkeypatch.setattr(
        ginot_stats,
        "ginot_model_forward",
        lambda _cfg, _model, _batch: (
            torch.tensor([[1.0, 3.0], [4.0, 8.0]]),
            torch.tensor([[0.0, 1.0], [2.0, 5.0]]),
            torch.zeros(2, dtype=torch.long),
            2,
        ),
    )
    monkeypatch.setattr(ginot_stats, "ginot_postprocess_displacement", lambda yh, _batch, **_kwargs: yh)
    monkeypatch.setattr(ginot_stats, "compute_packed_loss", lambda *_args, **_kwargs: torch.tensor(7.0))
    updates = iter([10.0, 2.0, torch.tensor([1.0, 5.0]), 1.0])

    def _fake_all_reduce(tensor: torch.Tensor, _op) -> None:
        update = next(updates)
        tensor.add_(torch.as_tensor(update, dtype=tensor.dtype, device=tensor.device))

    monkeypatch.setattr(ginot_stats.dist, "all_reduce", _fake_all_reduce)

    statsfun = make_ginot_statsfun(SimpleNamespace(), metadata)
    loss, stats = statsfun(trainer, [{"flat_free_mask": None}])

    assert loss == pytest.approx(6.0)
    assert stats["rel_l2"] == pytest.approx(6.0)
    assert stats["field_mse/field_a"] == pytest.approx(2.0)
    assert stats["field_mse/field_b"] == pytest.approx(6.0)

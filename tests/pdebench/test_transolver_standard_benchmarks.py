from __future__ import annotations

import contextlib
import numpy as np
import pytest
import torch
import types

import pdebench
from pdebench.config import (
    Config,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
)
from pdebench.__main__ import make_model, make_navier_stokes_statsfun
from pdebench.dataset.utils import load_dataset, plasticity_random_collate_fn


class _LastHistoryFrame(torch.nn.Module):
    def forward(self, pos, fx=None):
        del pos
        return fx[..., -1:]


def make_cfg(
    *,
    use_puri2025flare_config: bool | None = None,
    run: dict | None = None,
    dataset: dict | None = None,
    training: dict | None = None,
    optimizer: dict | None = None,
    scheduler: dict | None = None,
    model: dict | None = None,
) -> Config:
    cfg = Config(
        run=RunConfig(**(run or {})),
        dataset=dataset or {},
        training=TrainingConfig(**(training or {})),
        optimizer=OptimizerConfig(**(optimizer or {})),
        scheduler=scheduler or {},
        model=model or {},
    )
    if use_puri2025flare_config is not None:
        cfg.use_puri2025flare_config = use_puri2025flare_config
    return cfg


def test_rollout_navier_stokes_teacher_forcing_matches_upstream_semantics() -> None:
    model = _LastHistoryFrame()
    pos = torch.zeros(1, 1, 2)
    history = torch.tensor([[[1.0, 2.0]]])
    target = torch.tensor([[[3.0, 4.0]]])
    lossfun = torch.nn.MSELoss()

    pred_tf, _, _ = pdebench.rollout_navier_stokes(
        model,
        pos,
        history,
        target,
        lossfun=lossfun,
        teacher_forcing=True,
    )
    pred_ar, _, _ = pdebench.rollout_navier_stokes(
        model,
        pos,
        history,
        target,
        lossfun=lossfun,
        teacher_forcing=False,
    )

    assert torch.equal(pred_tf, torch.tensor([[[2.0, 3.0]]]))
    assert torch.equal(pred_ar, torch.tensor([[[2.0, 2.0]]]))


def test_transolver_defaults_match_upstream_navier_stokes_script() -> None:
    cfg = make_cfg(
        dataset={"dataset": "navier_stokes"},
        model={"model": "transolver", "conv2d": True},
        use_puri2025flare_config=True,
    )
    metadata = dict(c_in=12, c_out=1, space_dim=2, fun_dim=10, H=64, W=64)

    cfg, model = make_model(cfg, metadata, GLOBAL_RANK=0)
    base_model = model.model

    assert isinstance(model, pdebench.NavierStokesModelAdapter)
    assert isinstance(base_model, pdebench.Transolver_Structured_Mesh_2D)
    assert cfg.training.batch_size == 2
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert cfg.model.conv2d is True
    assert cfg.model.unified_pos is True
    assert cfg.training.clip_grad_norm is None
    assert base_model.unified_pos is True
    assert base_model.preprocess.n_input == 10 + (base_model.ref * base_model.ref)


def test_transolver_defaults_match_upstream_elasticity_script() -> None:
    cfg = make_cfg(dataset={"dataset": "elasticity"}, model={"model": "transolver"}, use_puri2025flare_config=True)
    metadata = dict(c_in=2, c_out=1, space_dim=2, fun_dim=0)

    cfg, model = make_model(cfg, metadata, GLOBAL_RANK=0)

    assert isinstance(model, pdebench.Transolver)
    assert cfg.training.batch_size == 1
    assert cfg.scheduler.schedule == "CosineAnnealingLR"
    assert cfg.model.unified_pos is False
    assert cfg.training.clip_grad_norm == pytest.approx(0.1)
    assert model.preprocess.n_input == 2


def test_transolver_defaults_match_upstream_plasticity_script() -> None:
    cfg = make_cfg(
        dataset={"dataset": "plasticity"},
        model={"model": "transolver", "conv2d": True},
        use_puri2025flare_config=True,
    )
    metadata = dict(c_in=3, c_out=4, space_dim=2, fun_dim=1, H=101, W=31, time_cond=True)

    cfg, model = make_model(cfg, metadata, GLOBAL_RANK=0)
    base_model = model.model

    assert isinstance(model, pdebench.PlasticityModelAdapter)
    assert isinstance(base_model, pdebench.Transolver_Structured_Mesh_2D)
    assert cfg.training.batch_size == 2
    assert cfg.scheduler.schedule == "OneCycleLR"
    assert cfg.model.conv2d is True
    assert cfg.model.unified_pos is False
    assert cfg.training.clip_grad_norm == pytest.approx(0.1)
    assert base_model.Time_Input is True
    assert base_model.preprocess.n_input == 3


def test_transolver_standard_defaults_disable_non_upstream_runtime_features() -> None:
    cfg = make_cfg(dataset={"dataset": "plasticity"}, model={"model": "transolver"}, use_puri2025flare_config=True)
    metadata = dict(c_in=3, c_out=4, space_dim=2, fun_dim=1, H=101, W=31, time_cond=True)

    cfg, _ = make_model(cfg, metadata, GLOBAL_RANK=0)

    assert cfg.training.mixed_precision is False
    assert cfg.training.compile_model is False
    assert cfg.training.ema is False
    assert cfg.training.grad_accumulation_steps == 1


def test_load_dataset_navier_stokes_shapes(monkeypatch, tmp_path) -> None:
    import scipy.io

    base = np.zeros((1200, 1, 1, 20), dtype=np.float32)
    data = np.broadcast_to(base, (1200, 64, 64, 20))

    monkeypatch.setattr(scipy.io, "loadmat", lambda _: {"u": data})

    with pytest.warns(UserWarning, match="not writable"):
        train_data, test_data, metadata = load_dataset(
            "navier_stokes",
            DATADIR_BASE=str(tmp_path),
            PROJDIR=str(tmp_path),
        )

    pos, history, target = train_data[0]

    assert len(train_data) == 1000
    assert len(test_data) == 200
    assert tuple(pos.shape) == (64 * 64, 2)
    assert tuple(history.shape) == (64 * 64, 10)
    assert tuple(target.shape) == (64 * 64, 10)
    assert metadata["space_dim"] == 2
    assert metadata["fun_dim"] == 10
    assert metadata["rollout_steps"] == 10


def test_make_navier_stokes_statsfun_runs_on_dummy_loader() -> None:
    pos = torch.zeros(2, 4, 2)
    history = torch.ones(2, 4, 10)
    target = torch.ones(2, 4, 3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos, history, target),
        batch_size=1,
        shuffle=False,
    )

    class _CopyFirstChannel(torch.nn.Module):
        def forward(self, pos, fx=None):
            del pos
            return fx[..., :1]

    trainer = types.SimpleNamespace(
        model=_CopyFirstChannel(),
        _stats_tqdm=False,
        verbose=False,
        GLOBAL_RANK=0,
        print_iterator=False,
        move_to_device=lambda batch: batch,
        auto_cast=contextlib.nullcontext(),
        get_batch_size=lambda batch, loader: batch[0].shape[0],
        DDP=False,
    )
    trainer.batch_lossfun = lambda pred, tgt: pdebench.RelL2Loss()(pred, tgt)

    loss, stats = make_navier_stokes_statsfun()(trainer, loader, split="test")

    assert torch.isfinite(torch.as_tensor(loss))
    assert set(stats) == {"step_rel_l2", "full_rel_l2"}


def test_rollout_plasticity_matches_expected_time_conditioning() -> None:
    class _TimeShift(torch.nn.Module):
        def forward(self, pos, fx=None, T=None):
            del pos
            return fx + T[:, None]

    model = _TimeShift()
    pos = torch.zeros(1, 2, 2)
    time_grid = torch.tensor([[0.0, 1.0, 2.0]])
    features = torch.ones(1, 2, 1)
    target = torch.stack([
        torch.full((1, 2, 1), 1.0),
        torch.full((1, 2, 1), 2.0),
        torch.full((1, 2, 1), 3.0),
    ], dim=-1).reshape(1, 2, 1, 3)

    pred, step_loss, full_loss = pdebench.rollout_plasticity(
        model,
        pos,
        time_grid,
        features,
        target,
        lossfun=torch.nn.MSELoss(),
    )

    assert torch.equal(pred, target)
    assert step_loss.item() == pytest.approx(0.0)
    assert full_loss.item() == pytest.approx(0.0)


def test_plasticity_random_collate_fn_shuffles_time_consistently() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    time_grid = torch.tensor([0.0, 1.0, 2.0])
    features = torch.tensor([[10.0], [20.0]])
    target = torch.tensor([
        [[100.0, 101.0, 102.0]],
        [[200.0, 201.0, 202.0]],
    ])

    torch.manual_seed(0)
    batch = plasticity_random_collate_fn([(pos, time_grid, features, target)])
    batch_pos, batch_t, batch_features, batch_target = batch

    assert torch.equal(batch_pos[0], pos)
    assert torch.equal(batch_features[0], features)
    assert set(batch_t[0].tolist()) == {0.0, 1.0, 2.0}

    perm = batch_t[0].argsort()
    restored_target = batch_target[0][..., perm]
    assert torch.equal(restored_target, target)


def test_make_optimizer_plain_adamw_uses_single_upstream_param_group() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )

    optimizer = pdebench.make_optimizer_plain_adamw(
        model,
        lr=1e-3,
        weight_decay=1e-5,
        beta1=0.9,
        beta2=0.999,
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("model", "dataset", "metadata", "call_args"),
    [
        (
            "set_transformer",
            "navier_stokes",
            dict(c_in=12, c_out=1, space_dim=2, fun_dim=10, H=64, W=64, max_length=64 * 64, time_cond=False),
            (
                torch.zeros(2, 8, 2),
                torch.zeros(2, 8, 10),
            ),
        ),
        (
            "lno",
            "plasticity",
            dict(c_in=3, c_out=4, space_dim=2, fun_dim=1, H=101, W=31, max_length=101 * 31, time_cond=True),
            (
                torch.zeros(2, 8, 2),
                torch.zeros(2, 8, 1),
                torch.zeros(2, 1),
            ),
        ),
        (
            "gnot",
            "navier_stokes",
            dict(c_in=12, c_out=1, space_dim=2, fun_dim=10, H=4, W=4, max_length=16, time_cond=False),
            (
                torch.zeros(2, 16, 2),
                torch.zeros(2, 16, 10),
            ),
        ),
    ],
)
def test_non_transolver_models_build_for_rollout_benchmarks(model, dataset, metadata, call_args) -> None:
    cfg = make_cfg(
        dataset={"dataset": dataset},
        training={"compile_model": False, "ema": False},
        model={"model": model},
        use_puri2025flare_config=True,
    )

    cfg, model = make_model(cfg, metadata, GLOBAL_RANK=0)
    device = next(model.parameters()).device
    call_args = tuple(arg.to(device) for arg in call_args)
    out = model(*call_args)

    assert out.shape[0] == call_args[0].shape[0]
    assert out.shape[1] == call_args[0].shape[1]
    assert out.shape[-1] == metadata["c_out"]

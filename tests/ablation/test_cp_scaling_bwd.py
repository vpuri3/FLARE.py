from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_constants():
    from ablation import cp_scaling_bwd as cp

    assert cp.SEQ_LENGTHS == [500_000, 1_000_000]
    assert cp.CP_SIZES == [1, 2, 4]
    assert cp.LATENT_COUNTS == [64, 128]
    assert cp.MODELS == ("flare", "flarepp")
    assert cp.C_IN == 3 and cp.C_OUT == 1


def test_model_names():
    from ablation import cp_scaling_bwd as cp

    assert cp.model_name("flare", 64) == "FLARE (64 latents)"
    assert cp.model_name("flare", 128) == "FLARE (128 latents)"
    assert cp.model_name("flarepp", 64) == "FLARE++ (64 latents)"
    assert cp.model_name("flarepp", 128) == "FLARE++ (128 latents)"


def test_make_config_locked_knobs():
    from ablation import cp_scaling_bwd as cp

    for kind in cp.MODELS:
        for num_latents in cp.LATENT_COUNTS:
            cfg = cp.make_config(kind, num_latents)
            assert cfg.num_blocks == 8
            assert cfg.channel_dim == 128
            assert cfg.num_heads == 8
            assert cfg.rmsnorm is True
            assert cfg.out_proj_norm is True
            assert cfg.num_layers_in_out_proj == 2
            assert cfg.num_layers_ffn == 0
            assert cfg.ffn_mlp_ratio == 4.0
            assert cfg.num_latents == num_latents
            assert cfg.encoder_cp_backend == "flash"


def test_build_model_forward_cpu():
    from ablation import cp_scaling_bwd as cp

    for kind in cp.MODELS:
        model = cp.build_model(kind, 64)
        model.eval()
        y = model(torch.randn(1, 32, cp.C_IN))
        assert y.shape == (1, 32, cp.C_OUT)


def test_efficiency():
    from ablation import cp_scaling_bwd as cp

    assert cp.efficiency(100.0, 100.0, 1) == 1.0
    assert cp.efficiency(100.0, 50.0, 2) == pytest.approx(1.0)
    assert cp.efficiency(100.0, 60.0, 2) == pytest.approx(100.0 / (2 * 60.0))
    assert math.isnan(cp.efficiency(float("nan"), 50.0, 2))


def test_append_csv_row_uses_stable_schema(tmp_path):
    from ablation import cp_scaling_bwd as cp

    path = tmp_path / "results.csv"
    first = {
        "model_name": "FLARE (128 latents)",
        "N": 100_000,
        "P": 1,
        "time_ms": 100.0,
        "memory_gb": 1.0,
        "efficiency": 1.0,
        "num_valid_runs": 30,
    }
    second = dict(first, P=2, time_ms=60.0, efficiency=100.0 / 120.0)

    cp.append_csv_row(str(path), first)
    cp.append_csv_row(str(path), second)

    frame = pd.read_csv(path)
    assert frame.columns.tolist() == [
        "model_name",
        "N",
        "P",
        "time_ms",
        "memory_gb",
        "efficiency",
        "num_valid_runs",
    ]
    assert frame.to_dict("records") == [first, second]


def test_timing_constants():
    from ablation import cp_scaling_bwd as cp

    assert cp.WARMUP_STEPS == 50
    assert cp.TIMED_REPS == 30


def test_benchmark_cell_local_warms_up_before_timing(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    calls = []

    def record_warmup(model, x, target, steps, synchronize_fn):
        calls.extend(["warmup"] * steps)

    def record_timing(model, x, target, reps, synchronize_fn):
        calls.extend(["timed"] * reps)
        return 12.5

    monkeypatch.setattr(cp, "warmup_model", record_warmup)
    monkeypatch.setattr(cp, "timed_median_ms", record_timing)

    time_ms = cp.benchmark_cell_local(
        object(),
        object(),
        object(),
        warmup_steps=2,
        timed_reps=3,
        synchronize_fn=lambda: None,
    )

    assert calls == ["warmup", "warmup", "timed", "timed", "timed"]
    assert time_ms == 12.5


def test_warmup_model_runs_each_step_between_synchronizations(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    calls = []
    monkeypatch.setattr(cp, "run_step", lambda model, x, target: calls.append("step"))

    cp.warmup_model(object(), object(), object(), steps=2, synchronize_fn=lambda: calls.append("sync"))

    assert calls == ["sync", "step", "sync", "sync", "step", "sync"]


def test_run_step_replaces_existing_gradients():
    from ablation import cp_scaling_bwd as cp

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 99.0)

    cp.run_step(model, torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0]]))

    assert model.weight.grad is not None
    assert not torch.all(model.weight.grad == 99.0)


def test_timed_median_ms_uses_only_timed_step_samples(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    clock = iter([0.000, 0.003, 0.003, 0.004, 0.007, 0.012])
    calls = []
    monkeypatch.setattr(cp.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cp.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(cp, "run_step", lambda model, x, target: calls.append("step"))

    result = cp.timed_median_ms(
        object(), object(), object(), reps=3, synchronize_fn=lambda: calls.append("sync")
    )

    assert result == pytest.approx(3.0)
    assert calls == ["sync", "step", "sync"] * 3


def test_run_worker_initializes_shards_compiles_and_records_rank_zero(monkeypatch, tmp_path):
    from ablation import cp_scaling_bwd as cp
    from pdebench.distributed import context_parallel, utils

    calls = []
    cp_state = object()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)

        def to(self, device):
            calls.append(("to", device))
            return self

        def set_context_parallel(self, state):
            calls.append(("set_cp", state))

        def train(self, mode=True):
            calls.append(("train", mode))
            return self

    fake_model = FakeModel()
    initialized = iter([False])
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: next(initialized))
    monkeypatch.setattr(cp.dist, "init_process_group", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(cp.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(cp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(cp.torch.cuda, "set_device", lambda rank: calls.append(("set_device", rank)))
    monkeypatch.setattr(cp.torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda_seed", seed)))
    monkeypatch.setattr(cp.torch.cuda, "synchronize", lambda: calls.append("cuda_sync"))
    monkeypatch.setattr(
        cp.torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(("reset_peak", device))
    )
    monkeypatch.setattr(cp.torch.cuda, "max_memory_allocated", lambda device: 2 * 1024**3)
    monkeypatch.setattr(
        cp.mlutils,
        "configure_runtime",
        lambda seed, **kwargs: calls.append(("runtime", seed, kwargs)),
    )
    monkeypatch.setattr(
        context_parallel,
        "build_context_parallel_state",
        lambda size, sequence_length: calls.append(("build_cp", size, sequence_length)) or cp_state,
    )

    def shard(tensor, state, seq_dim):
        calls.append(("shard", tuple(tensor.shape), state, seq_dim))
        return tensor[:, :4]

    monkeypatch.setattr(utils, "shard_sequence_tensor", shard)
    monkeypatch.setattr(
        cp, "build_model", lambda kind, num_latents: calls.append(("build_model", kind, num_latents)) or fake_model
    )
    monkeypatch.setattr(
        cp.torch,
        "compile",
        lambda model: calls.append(("compile", model)) or model,
    )

    real_randn = torch.randn

    def cpu_randn(*shape, **kwargs):
        calls.append(("randn", shape, kwargs))
        kwargs.pop("device", None)
        return real_randn(*shape, **kwargs)

    monkeypatch.setattr(cp.torch, "randn", cpu_randn)

    def warmup(model, x, target, steps, sync):
        calls.append(("warmup", model, tuple(x.shape), tuple(target.shape), steps))
        assert x.requires_grad
        assert x.is_leaf

    monkeypatch.setattr(cp, "warmup_model", warmup)
    monkeypatch.setattr(
        cp,
        "timed_median_ms",
        lambda model, x, target, reps, sync: calls.append(("timed", reps)) or 12.5,
    )
    monkeypatch.setattr(cp, "OUT_CSV", str(tmp_path / "results.csv"))
    monkeypatch.setattr(cp, "append_csv_row", lambda path, row: calls.append(("append", path, row)))

    row = cp.run_worker("flare", N=8, cp_size=2, num_latents=64)

    assert calls.index(("compile", fake_model)) < next(
        index for index, call in enumerate(calls) if isinstance(call, tuple) and call[0] == "warmup"
    )
    assert next(index for index, call in enumerate(calls) if call[0] == "warmup") < calls.index(("timed", 30))
    assert [call for call in calls if isinstance(call, tuple) and call[0] == "shard"] == [
        ("shard", (1, 8, cp.C_IN), cp_state, 1),
        ("shard", (1, 8, cp.C_OUT), cp_state, 1),
    ]
    assert ("build_model", "flare", 64) in calls
    assert fake_model.weight.requires_grad
    assert {key: value for key, value in row.items() if key != "efficiency"} == {
        "model_name": "FLARE (64 latents)",
        "N": 8,
        "P": 2,
        "time_ms": 12.5,
        "memory_gb": 2.0,
        "num_valid_runs": 30,
    }
    assert math.isnan(row["efficiency"])
    append_calls = [call for call in calls if isinstance(call, tuple) and call[0] == "append"]
    assert len(append_calls) == 1
    assert append_calls[0][2] is row
    assert calls[-1] == "barrier"


def test_run_worker_nonzero_rank_does_not_append(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    monkeypatch.setattr(cp.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(cp.dist, "barrier", lambda: None)
    monkeypatch.setattr(cp, "_run_worker_cell", lambda kind, N, cp_size, num_latents: {"rank": 1})
    monkeypatch.setattr(cp, "append_csv_row", lambda path, row: pytest.fail("nonzero rank appended CSV"))

    assert cp.run_worker("flarepp", 8, 2, 128) == {"rank": 1}


def test_worker_cli_requires_cell_arguments_and_dispatches(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    with pytest.raises(SystemExit) as exc:
        cp.main(["--worker", "--model", "flare"])
    assert exc.value.code == 2

    calls = []
    monkeypatch.setattr(
        cp, "run_worker", lambda kind, N, cp_size, num_latents: calls.append((kind, N, cp_size, num_latents))
    )
    cp.main(["--worker", "--model", "flarepp", "--N", "1000", "--cp-size", "4", "--num-latents", "64"])
    assert calls == [("flarepp", 1000, 4, 64)]


def test_backfill_efficiency_uses_each_model_and_length_baseline():
    from ablation import cp_scaling_bwd as cp

    frame = pd.DataFrame(
        [
            {"model_name": "FLARE", "N": 100_000, "P": 1, "time_ms": 120.0, "efficiency": math.nan},
            {"model_name": "FLARE", "N": 100_000, "P": 2, "time_ms": 75.0, "efficiency": math.nan},
            {"model_name": "FLARE++", "N": 500_000, "P": 1, "time_ms": 300.0, "efficiency": math.nan},
            {"model_name": "FLARE++", "N": 500_000, "P": 4, "time_ms": 100.0, "efficiency": math.nan},
            {"model_name": "FLARE", "N": 500_000, "P": 2, "time_ms": 90.0, "efficiency": math.nan},
        ]
    )

    result = cp.backfill_efficiency(frame)

    assert result["efficiency"].iloc[:4].tolist() == pytest.approx([1.0, 0.8, 1.0, 0.75])
    assert math.isnan(result.at[4, "efficiency"])
    assert frame["efficiency"].isna().all()


def test_run_matrix_launches_torchrun_worker_and_backfills_csv(monkeypatch, tmp_path):
    from ablation import cp_scaling_bwd as cp

    output = tmp_path / "results.csv"
    calls = []
    monkeypatch.setattr(cp, "OUT_CSV", str(output))
    monkeypatch.setattr(cp, "MODELS", ("flare",))
    monkeypatch.setattr(cp, "LATENT_COUNTS", [64])
    monkeypatch.setattr(cp, "SEQ_LENGTHS", [500_000])
    monkeypatch.setattr(cp, "CP_SIZES", [1])

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        cp.append_csv_row(
            str(output),
            {
                "model_name": cp.model_name("flare", 64),
                "N": 500_000,
                "P": 1,
                "time_ms": 120.0,
                "memory_gb": 2.0,
                "efficiency": math.nan,
                "num_valid_runs": 30,
            },
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = cp.run_matrix()

    assert calls == [
        (
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=1",
                "ablation/cp_scaling_bwd.py",
                "--worker",
                "--model",
                "flare",
                "--N",
                "500000",
                "--cp-size",
                "1",
                "--num-latents",
                "64",
            ],
            cp.PROJDIR,
            True,
        )
    ]
    assert result["efficiency"].tolist() == [1.0]
    assert pd.read_csv(output)["efficiency"].tolist() == [1.0]


def test_run_matrix_filters_small_n_and_skips_existing_finite_rows(monkeypatch, tmp_path):
    from ablation import cp_scaling_bwd as cp

    output = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model_name": "FLARE (128 latents)",
                "N": 100_000,
                "P": 1,
                "time_ms": 10.0,
                "memory_gb": 1.0,
                "efficiency": 1.0,
                "num_valid_runs": 30,
            },
            {
                "model_name": "FLARE (128 latents)",
                "N": 500_000,
                "P": 1,
                "time_ms": 120.0,
                "memory_gb": 2.0,
                "efficiency": 1.0,
                "num_valid_runs": 30,
            },
        ]
    ).to_csv(output, index=False)

    calls = []
    monkeypatch.setattr(cp, "OUT_CSV", str(output))
    monkeypatch.setattr(cp, "MODELS", ("flare",))
    monkeypatch.setattr(cp, "LATENT_COUNTS", [128])
    monkeypatch.setattr(cp, "SEQ_LENGTHS", [500_000])
    monkeypatch.setattr(cp, "CP_SIZES", [1])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = cp.run_matrix()

    assert calls == []
    assert list(result["N"]) == [500_000]
    assert pd.read_csv(output)["N"].tolist() == [500_000]


def test_run_cli_dispatches_matrix(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    calls = []
    monkeypatch.setattr(cp, "run_matrix", lambda: calls.append("run"))

    cp.main(["--run"])

    assert calls == ["run"]


def test_plot_analysis_writes_three_panel_png_and_pdf(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)

    from ablation import cp_scaling_bwd as cp

    csv_path = tmp_path / "results.csv"
    png_path = tmp_path / "cp_scaling_bwd_fp16.png"
    pdf_path = tmp_path / "cp_scaling_bwd_fp16.pdf"
    rows = []
    for name, base_time in (
        ("FLARE (64 latents)", 100.0),
        ("FLARE (128 latents)", 120.0),
        ("FLARE++ (64 latents)", 110.0),
        ("FLARE++ (128 latents)", 150.0),
    ):
        for n, scale in ((500_000, 1.0), (1_000_000, 2.0)):
            for p in (1, 2, 4):
                time_ms = base_time * scale / p
                rows.append(
                    {
                        "model_name": name,
                        "N": n,
                        "P": p,
                        "time_ms": time_ms,
                        "memory_gb": 8.0 * scale / p,
                        "efficiency": 1.0,
                        "num_valid_runs": 30,
                    }
                )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    monkeypatch.setattr(cp, "OUT_CSV", str(csv_path))
    monkeypatch.setattr(cp, "OUT_PNG", str(png_path))
    monkeypatch.setattr(cp, "OUT_PDF", str(pdf_path))

    axes = cp.plot_analysis()

    assert len(axes) == 3
    assert [axis.get_ylabel() for axis in axes] == [
        "Step time (s)",
        "Parallel efficiency",
        "Peak memory (GB)",
    ]
    assert all(axis.get_xticks().tolist() == [1, 2, 4] for axis in axes)
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0


def test_plot_cli_dispatches_plot_analysis(monkeypatch):
    from ablation import cp_scaling_bwd as cp

    calls = []
    monkeypatch.setattr(cp, "plot_analysis", lambda: calls.append("plot"))

    cp.main(["--plot"])

    assert calls == ["plot"]


def test_cli_help_noop():
    script = REPO_ROOT / "ablation" / "cp_scaling_bwd.py"
    help_proc = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, text=True, capture_output=True
    )
    assert help_proc.returncode == 0
    noop = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, text=True, capture_output=True)
    assert noop.returncode == 0
    assert "No action specified" in noop.stdout

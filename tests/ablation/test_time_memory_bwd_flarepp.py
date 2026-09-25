from __future__ import annotations

import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_constants() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    assert tm.NUM_LATENTS == [64, 128, 256]
    assert tm.SEQ_LENGTHS == [
        1_000,
        50_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
        800_000,
        900_000,
        1_000_000,
    ]
    assert tm.SEQ_LENGTHS_MHA == tm.SEQ_LENGTHS
    assert tm.MEASURED_KINDS == ("flare", "simplifiedflarepp", "transolver3")
    assert tm.C_IN == 3 and tm.C_OUT == 1


def test_model_names() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    assert tm.model_name("mha", None) == "Full self-attention"
    assert tm.model_name("flare", 64) == "FLARE (64 latents)"
    assert tm.model_name("simplifiedflarepp", 128) == "Simplified FLARE++ (128 latents)"
    assert tm.model_name("flarepp", 64) == "FLARE++ (64 latents)"
    assert tm.model_name("simplifiedflarepp_qk0_off", 64) == "Simplified FLARE++ qk0_off (64 latents)"
    assert tm.model_name("transolver3", 256) == "Transolver 3 (256 slices)"


def test_build_backbone_shapes_and_mixer_kind() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    for kind in tm.MEASURED_KINDS:
        model = tm.build_backbone(kind, 64)
        model.eval()
        x = torch.randn(1, 16, tm.C_IN)
        y = model(x)
        assert y.shape == (1, 16, tm.C_OUT)
        assert model.blocks[0].mixer.__class__.__name__.lower().startswith(
            {"flare": "flare", "simplifiedflarepp": "simplifiedflarepp", "transolver3": "transolver3"}[kind]
        )

    mha = tm.build_backbone("mha")
    mha.eval()
    y = mha(torch.randn(1, 16, tm.C_IN))
    assert y.shape == (1, 16, tm.C_OUT)
    assert mha.blocks[0].mixer.__class__.__name__ == "MHAMixer"
    assert tm.make_backbone_config("mha").mixer.qk_norm is False

    anchored = tm.build_backbone("flarepp", 64)
    anchored.eval()
    y = anchored(torch.randn(1, 16, tm.C_IN))
    assert y.shape == (1, 16, tm.C_OUT)
    assert anchored.blocks[0].mixer.__class__.__name__ == "FLAREPPMixer"



def test_simplifiedflarepp_defaults() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    # Prefer checking the MixerBackboneConfig used to build:
    # build_backbone must set SimplifiedFLAREPPMixerConfig(qk_norm=False, qk0_norm=True, share_k0_v0=False)
    # If the mixer stores these as attributes, assert them; otherwise assert via a returned config helper.
    assert hasattr(tm, "make_backbone_config")
    bc = tm.make_backbone_config("simplifiedflarepp", 64)
    assert bc.mixer.kind == "simplifiedflarepp"
    assert bc.mixer.qk_norm is False
    assert bc.mixer.qk0_norm is True
    assert bc.mixer.share_k0_v0 is False
    assert bc.num_blocks == 8
    assert bc.channel_dim == 128
    assert bc.num_heads == 8
    assert bc.rmsnorm is True
    assert bc.out_proj_norm is True
    assert bc.num_layers_in_out_proj == 2
    assert bc.num_layers_ffn == 0
    assert bc.mlp_ratio_ffn == 4.0


def test_flarepp_share_k0_v0_true() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    assert hasattr(tm, "make_backbone_config")
    bc = tm.make_backbone_config("flarepp", 64)
    assert bc.mixer.kind == "flarepp"
    assert bc.mixer.share_k0_v0 is True
    assert bc.num_blocks == 8
    assert bc.channel_dim == 128
    assert bc.num_heads == 8
    mixer = tm.build_backbone("flarepp", 64).blocks[0].mixer
    assert mixer.share_k0_v0 is True
    assert mixer.v0_proj is None


def test_simplifiedflarepp_qk0_off_config() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    bc = tm.make_backbone_config("simplifiedflarepp_qk0_off", 64)
    assert bc.mixer.kind == "simplifiedflarepp"
    assert bc.mixer.qk_norm is False
    assert bc.mixer.qk0_norm is False
    assert bc.mixer.share_k0_v0 is False
    mixer = tm.build_backbone("simplifiedflarepp_qk0_off", 64).blocks[0].mixer
    assert mixer.__class__.__name__ == "SimplifiedFLAREPPMixer"
    assert isinstance(mixer.k0_norm, torch.nn.Identity)
    assert isinstance(mixer.q0_norm, torch.nn.Identity)


def test_mha_schema_rows() -> None:
    from ablation import time_memory_bwd_flarepp as tm

    rows = tm.mha_schema_rows(tm.SEQ_LENGTHS)
    assert len(rows) == len(tm.SEQ_LENGTHS)
    for row, n in zip(rows, tm.SEQ_LENGTHS):
        assert row["model_name"] == "Full self-attention"
        assert row["N"] == n
        assert math.isnan(row["time"]) and math.isnan(row["memory"])
        assert row["num_valid_runs"] == 0


def test_cli_help_and_noop() -> None:
    script = REPO_ROOT / "ablation" / "time_memory_bwd_flarepp.py"
    help_proc = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, text=True, capture_output=True
    )
    assert help_proc.returncode == 0, help_proc.stderr
    noop = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, text=True, capture_output=True)
    assert noop.returncode == 0, noop.stderr
    assert "No action specified" in noop.stdout
    assert "--mha" in help_proc.stdout
    assert "--anchored" in help_proc.stdout
    assert "--simplifiedflarepp-qk0-off" in help_proc.stdout


def test_benchmark_model_runs_backward_and_reports_peak_memory(monkeypatch) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    model = torch.nn.Linear(3, 1)
    x = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.randn(1, 4, 1)
    bench_args = {}

    def do_bench(fn, **kwargs):
        bench_args.update(kwargs)
        fn()
        return 12.5

    monkeypatch.setattr(tm.triton.testing, "do_bench", do_bench)
    monkeypatch.setattr(tm.torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(tm.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(tm.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(tm.torch.cuda, "max_memory_allocated", lambda: 2 * 1024**3)

    elapsed_ms, peak_gb = tm.benchmark_model(model, x, target)

    assert elapsed_ms == 12.5
    assert peak_gb == 2.0
    assert bench_args == {"warmup": 100, "rep": 1000, "return_mode": "median"}
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_prepare_measured_model_compiles_variant(monkeypatch) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    built = []
    compiled = []

    def build(kind, num_latents):
        model = torch.nn.Linear(1, 1)
        built.append((kind, num_latents, model))
        return model

    monkeypatch.setattr(tm, "build_backbone", build)
    monkeypatch.setattr(tm.torch, "compile", lambda model: compiled.append(model) or model)

    model, compiled_model = tm._prepare_measured_model("flare", 64, torch.device("cpu"))

    assert compiled_model is model
    assert compiled == [model]
    assert [entry[:2] for entry in built] == [("flare", 64)]
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_run_analysis_writes_mha_and_measured_rows(monkeypatch, tmp_path) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    class DummyModel:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    out_csv = tmp_path / "results" / "bench.csv"
    runtime_kwargs = {}
    original_randn = torch.randn
    benchmark_results = iter(
        [(3.5, 1.25), (4.5, 1.5), RuntimeError("out of memory"), (6.5, 2.0)]
    )
    events = []

    def configure_runtime(seed, **kwargs):
        runtime_kwargs.update(seed=seed, **kwargs)
        return {
            "profile": "speed",
            "seed": seed,
            "tf32": True,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
            "compile_model": True,
        }

    def fake_randn(*shape, **kwargs):
        return original_randn(*shape, requires_grad=kwargs.get("requires_grad", False))

    def prepare(kind, num_latents, device):
        name = tm.model_name(kind, num_latents)
        events.append(("prepare", name))
        model = DummyModel()
        return model, model

    def benchmark(model, x, target):
        result = next(benchmark_results)
        events.append(("benchmark", x.shape[1]))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(tm, "MEASURED_KINDS", ("flare", "simplifiedflarepp"))
    monkeypatch.setattr(tm, "NUM_LATENTS", [64])
    monkeypatch.setattr(tm, "SEQ_LENGTHS", [8, 16])
    monkeypatch.setattr(tm, "OUT_CSV", str(out_csv))
    monkeypatch.setattr(tm.mlutils, "configure_runtime", configure_runtime)
    monkeypatch.setattr(tm, "_prepare_measured_model", prepare)
    monkeypatch.setattr(tm, "benchmark_model", benchmark)
    monkeypatch.setattr(tm.torch, "randn", fake_randn)
    monkeypatch.setattr(tm.torch.cuda, "empty_cache", lambda: None)

    df = tm.run_analysis(torch.device("cpu"))

    assert runtime_kwargs == {
        "seed": 42,
        "mixed_precision": True,
        "deterministic": False,
        "compile_model": True,
    }
    assert events == [
        ("prepare", "FLARE (64 latents)"),
        ("benchmark", 8),
        ("benchmark", 16),
        ("prepare", "Simplified FLARE++ (64 latents)"),
        ("benchmark", 8),
        ("benchmark", 16),
    ]
    assert tm.torch._dynamo.config.recompile_limit == 1000
    first_measured = df.to_dict("records")[2]
    assert first_measured["model_name"] == "FLARE (64 latents)"
    assert first_measured["N"] == 8
    assert first_measured["time"] == 3.5
    assert first_measured["memory"] == 1.25
    assert math.isnan(first_measured["num_valid_runs"])
    second_measured = df.iloc[3]
    assert second_measured["model_name"] == "FLARE (64 latents)"
    assert second_measured["N"] == 16
    assert second_measured["time"] == 4.5
    assert second_measured["memory"] == 1.5
    assert math.isnan(second_measured["num_valid_runs"])
    assert math.isnan(df.iloc[0]["time"])
    assert math.isnan(df.iloc[1]["time"])
    assert math.isnan(df.iloc[4]["time"])
    assert df.iloc[4]["num_valid_runs"] == 0
    last_measured = df.iloc[5]
    assert last_measured["model_name"] == "Simplified FLARE++ (64 latents)"
    assert last_measured["N"] == 16
    assert last_measured["time"] == 6.5
    assert last_measured["memory"] == 2.0
    assert math.isnan(last_measured["num_valid_runs"])
    pd.testing.assert_frame_equal(pd.read_csv(out_csv), df)


def test_run_mha_analysis_merges_into_existing_csv(monkeypatch, tmp_path) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    class DummyModel:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    out_csv = tmp_path / "bench.csv"
    existing = pd.DataFrame(
        [
            {
                "model_name": "Full self-attention",
                "N": 8,
                "time": 9.0,
                "memory": 1.0,
                "num_valid_runs": None,
            },
            {
                "model_name": "Full self-attention",
                "N": 16,
                "time": float("nan"),
                "memory": float("nan"),
                "num_valid_runs": 0,
            },
            {
                "model_name": "Full self-attention",
                "N": 32,
                "time": float("nan"),
                "memory": float("nan"),
                "num_valid_runs": 0,
            },
            {
                "model_name": "FLARE (64 latents)",
                "N": 8,
                "time": 1.0,
                "memory": 2.0,
                "num_valid_runs": None,
            },
        ]
    )
    existing.to_csv(out_csv, index=False)
    results = iter([(20.0, 5.0), (30.0, 6.0)])
    prepared = []

    monkeypatch.setattr(tm, "OUT_CSV", str(out_csv))
    monkeypatch.setattr(tm, "SEQ_LENGTHS", [8, 16, 32])
    monkeypatch.setattr(tm, "SEQ_LENGTHS_MHA", [8, 16, 32])
    monkeypatch.setattr(
        tm.mlutils,
        "configure_runtime",
        lambda seed, **kwargs: {
            "profile": "speed",
            "seed": seed,
            "tf32": True,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
            "compile_model": True,
        },
    )

    def prepare(kind, num_latents, device):
        prepared.append((kind, num_latents))
        return DummyModel(), DummyModel()

    monkeypatch.setattr(tm, "_prepare_measured_model", prepare)
    monkeypatch.setattr(tm, "benchmark_model", lambda model, x, target: next(results))
    monkeypatch.setattr(tm.torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape))
    monkeypatch.setattr(tm.torch.cuda, "empty_cache", lambda: None)

    df = tm.run_mha_analysis(torch.device("cpu"))
    assert prepared == [("mha", None)]
    mha = df[df["model_name"] == "Full self-attention"].sort_values("N")
    assert list(mha["N"]) == [8, 16, 32]
    assert mha.iloc[0]["time"] == 9.0  # resumed prior measurement
    assert list(mha["time"].iloc[1:]) == [20.0, 30.0]
    flare = df[df["model_name"] == "FLARE (64 latents)"]
    assert len(flare) == 1
    assert flare.iloc[0]["time"] == 1.0


def test_run_anchored_analysis_merges_into_existing_csv(monkeypatch, tmp_path) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    class DummyModel:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    out_csv = tmp_path / "bench.csv"
    existing = pd.DataFrame(
        [
            {
                "model_name": "FLARE (64 latents)",
                "N": 8,
                "time": 1.0,
                "memory": 2.0,
                "num_valid_runs": None,
            },
            {
                "model_name": "Simplified FLARE++ (64 latents)",
                "N": 8,
                "time": 3.0,
                "memory": 4.0,
                "num_valid_runs": None,
            },
            {
                "model_name": "FLARE++ (64 latents)",
                "N": 8,
                "time": 5.0,
                "memory": 1.5,
                "num_valid_runs": None,
            },
        ]
    )
    existing.to_csv(out_csv, index=False)
    results = iter([(7.0, 2.5), (8.0, 3.5), (9.0, 4.5), (10.0, 5.5)])
    prepared = []

    monkeypatch.setattr(tm, "OUT_CSV", str(out_csv))
    monkeypatch.setattr(tm, "NUM_LATENTS", [64, 128])
    monkeypatch.setattr(tm, "SEQ_LENGTHS", [8, 16])
    monkeypatch.setattr(
        tm.mlutils,
        "configure_runtime",
        lambda seed, **kwargs: {
            "profile": "speed",
            "seed": seed,
            "tf32": True,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
            "compile_model": True,
        },
    )

    def prepare(kind, num_latents, device):
        prepared.append((kind, num_latents))
        return DummyModel(), DummyModel()

    monkeypatch.setattr(tm, "_prepare_measured_model", prepare)
    monkeypatch.setattr(tm, "benchmark_model", lambda model, x, target: next(results))
    monkeypatch.setattr(tm.torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape))
    monkeypatch.setattr(tm.torch.cuda, "empty_cache", lambda: None)

    df = tm.run_anchored_analysis(torch.device("cpu"))
    assert prepared == [("flarepp", 64), ("flarepp", 128)]
    flare = df[df["model_name"] == "FLARE (64 latents)"]
    assert len(flare) == 1 and flare.iloc[0]["time"] == 1.0
    simplifiedflarepp = df[df["model_name"] == "Simplified FLARE++ (64 latents)"]
    assert len(simplifiedflarepp) == 1 and simplifiedflarepp.iloc[0]["time"] == 3.0
    a64 = df[df["model_name"] == "FLARE++ (64 latents)"].sort_values("N")
    assert list(a64["N"]) == [8, 16]
    assert a64.iloc[0]["time"] == 5.0  # resumed prior measurement
    assert a64.iloc[1]["time"] == 7.0
    a128 = df[df["model_name"] == "FLARE++ (128 latents)"].sort_values("N")
    assert list(a128["N"]) == [8, 16]
    assert list(a128["time"]) == [8.0, 9.0]


def test_run_simplifiedflarepp_qk0_off_analysis_merges_into_existing_csv(monkeypatch, tmp_path) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    class DummyModel:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    out_csv = tmp_path / "bench.csv"
    existing = pd.DataFrame(
        [
            {
                "model_name": "Simplified FLARE++ (64 latents)",
                "N": 8,
                "time": 3.0,
                "memory": 4.0,
                "num_valid_runs": None,
            },
            {
                "model_name": "FLARE (64 latents)",
                "N": 8,
                "time": 1.0,
                "memory": 2.0,
                "num_valid_runs": None,
            },
        ]
    )
    existing.to_csv(out_csv, index=False)
    results = iter([(11.0, 3.0), (12.0, 3.5)])
    prepared = []

    monkeypatch.setattr(tm, "OUT_CSV", str(out_csv))
    monkeypatch.setattr(tm, "NUM_LATENTS", [64])
    monkeypatch.setattr(tm, "SEQ_LENGTHS", [8, 16])
    monkeypatch.setattr(
        tm.mlutils,
        "configure_runtime",
        lambda seed, **kwargs: {
            "profile": "speed",
            "seed": seed,
            "tf32": True,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
            "compile_model": True,
        },
    )

    def prepare(kind, num_latents, device):
        prepared.append((kind, num_latents))
        return DummyModel(), DummyModel()

    monkeypatch.setattr(tm, "_prepare_measured_model", prepare)
    monkeypatch.setattr(tm, "benchmark_model", lambda model, x, target: next(results))
    monkeypatch.setattr(tm.torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape))
    monkeypatch.setattr(tm.torch.cuda, "empty_cache", lambda: None)

    df = tm.run_simplifiedflarepp_qk0_off_analysis(torch.device("cpu"))
    assert prepared == [("simplifiedflarepp_qk0_off", 64)]
    simplifiedflarepp = df[df["model_name"] == "Simplified FLARE++ (64 latents)"]
    assert len(simplifiedflarepp) == 1 and simplifiedflarepp.iloc[0]["time"] == 3.0
    off = df[df["model_name"] == "Simplified FLARE++ qk0_off (64 latents)"].sort_values("N")
    assert list(off["N"]) == [8, 16]
    assert list(off["time"]) == [11.0, 12.0]


def test_run_cli_requires_cuda(monkeypatch) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    monkeypatch.setattr(tm.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA required for --run"):
        tm.main(["--run"])
    with pytest.raises(SystemExit, match="CUDA required for --mha"):
        tm.main(["--mha"])
    with pytest.raises(SystemExit, match="CUDA required for --anchored"):
        tm.main(["--anchored"])
    with pytest.raises(SystemExit, match="CUDA required for --simplifiedflarepp-qk0-off"):
        tm.main(["--simplifiedflarepp-qk0-off"])


def test_plot_from_synthetic_csv(tmp_path, monkeypatch) -> None:
    from ablation import time_memory_bwd_flarepp as tm

    rows = tm.mha_schema_rows([1000, 50000])
    for kind in (*tm.MEASURED_KINDS, "flarepp"):
        for m in tm.NUM_LATENTS:
            for n in (1000, 50000):
                rows.append(
                    {
                        "model_name": tm.model_name(kind, m),
                        "N": n,
                        "time": 1.0 + m / 1000.0,
                        "memory": 1.0 + n / 1e6,
                        "num_valid_runs": 10,
                    }
                )
    csv_path = tmp_path / "out.csv"
    png_path = tmp_path / "out.png"
    pdf_path = tmp_path / "out.pdf"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    monkeypatch.setattr(tm, "OUT_CSV", str(csv_path))
    monkeypatch.setattr(tm, "OUT_PNG", str(png_path))
    monkeypatch.setattr(tm, "OUT_PDF", str(pdf_path))

    tm.plot_analysis()

    assert png_path.exists() and pdf_path.exists()

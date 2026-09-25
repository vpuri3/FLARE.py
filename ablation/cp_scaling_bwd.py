"""Context-parallel strong-scaling fwd+bwd study for FLARE / FLARE++."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn

PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJDIR not in sys.path:
    sys.path.insert(0, PROJDIR)

import mlutils  # noqa: E402
from pdebench.models.flare import FlareConfig, FLAREModel  # noqa: E402
from pdebench.models.flarepp import FlarePPConfig, FLAREPPModel  # noqa: E402

OUT_CSV = os.path.join(PROJDIR, "out", "pdebench", "cp_scaling_bwd_fp16.csv")
OUT_PNG = os.path.join(PROJDIR, "figs", "cp_scaling_bwd_fp16.png")
OUT_PDF = os.path.join(PROJDIR, "figs", "cp_scaling_bwd_fp16.pdf")

SEQ_LENGTHS = [500_000, 1_000_000]
CP_SIZES = [1, 2, 4]
MODELS = ("flare", "flarepp")
LATENT_COUNTS = [64, 128]
C_IN, C_OUT = 3, 1
WARMUP_STEPS = 50
TIMED_REPS = 30
SEED = 42
CSV_COLUMNS = [
    "model_name",
    "N",
    "P",
    "time_ms",
    "memory_gb",
    "efficiency",
    "num_valid_runs",
]


def model_name(kind: str, num_latents: int) -> str:
    """Return the stable display name for a model kind and latent width."""
    if kind == "flare":
        return f"FLARE ({num_latents} latents)"
    if kind == "flarepp":
        return f"FLARE++ ({num_latents} latents)"
    raise ValueError(f"Unknown model kind: {kind!r}")


def make_config(kind: str, num_latents: int) -> FlareConfig | FlarePPConfig:
    """Build a model config with the study's locked knobs."""
    kwargs = {
        "num_blocks": 8,
        "channel_dim": 128,
        "num_heads": 8,
        "rmsnorm": True,
        "out_proj_norm": True,
        "num_layers_in_out_proj": 2,
        "num_layers_ffn": 0,
        "ffn_mlp_ratio": 4.0,
        "num_latents": num_latents,
        "encoder_cp_backend": "flash",
    }
    if kind == "flare":
        return FlareConfig(**kwargs)
    if kind == "flarepp":
        return FlarePPConfig(**kwargs)
    raise ValueError(f"Unknown model kind: {kind!r}")


def build_model(kind: str, num_latents: int) -> nn.Module:
    """Build a CPU model for the selected implementation."""
    metadata = {"c_in": C_IN, "c_out": C_OUT}
    if kind == "flare":
        return FLAREModel(make_config(kind, num_latents), metadata=metadata)
    if kind == "flarepp":
        return FLAREPPModel(make_config(kind, num_latents), metadata=metadata)
    raise ValueError(f"Unknown model kind: {kind!r}")


def filter_study_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose sequence length is in the current study grid."""
    return df[df["N"].isin(SEQ_LENGTHS)].reset_index(drop=True)


def existing_finite_keys(df: pd.DataFrame) -> set[tuple[str, int, int]]:
    """Return (model_name, N, P) keys that already have a finite timed measurement."""
    keys: set[tuple[str, int, int]] = set()
    if df.empty:
        return keys
    for row in df.itertuples(index=False):
        if math.isfinite(float(row.time_ms)):
            keys.add((str(row.model_name), int(row.N), int(row.P)))
    return keys


def efficiency(t1_ms: float, tp_ms: float, p: int) -> float:
    """Compute strong-scaling efficiency relative to the one-GPU time."""
    if p < 1 or not math.isfinite(t1_ms) or not math.isfinite(tp_ms) or tp_ms <= 0:
        return float("nan")
    if p == 1:
        return 1.0
    return float(t1_ms) / (float(p) * float(tp_ms))


def append_csv_row(path: str, row: dict[str, Any]) -> None:
    """Append one result using the stable study CSV schema."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    frame = pd.DataFrame([row], columns=CSV_COLUMNS)
    frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def backfill_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Fill each strong-scaling series using its one-GPU time baseline."""
    out = df.copy()
    for name in out["model_name"].unique():
        for n in out["N"].unique():
            mask = (out["model_name"] == name) & (out["N"] == n)
            sub = out.loc[mask]
            t1 = sub.loc[sub["P"] == 1, "time_ms"]
            t1v = float(t1.iloc[0]) if len(t1) else float("nan")
            for idx in sub.index:
                p = int(out.at[idx, "P"])
                out.at[idx, "efficiency"] = efficiency(t1v, float(out.at[idx, "time_ms"]), p)
    return out


def run_matrix() -> pd.DataFrame:
    """Launch missing study cells under torchrun, then backfill the result CSV."""
    if os.path.exists(OUT_CSV):
        existing = filter_study_rows(pd.read_csv(OUT_CSV))
        existing.to_csv(OUT_CSV, index=False)
    else:
        existing = pd.DataFrame(columns=CSV_COLUMNS)

    measured = existing_finite_keys(existing)
    for kind in MODELS:
        for num_latents in LATENT_COUNTS:
            for n in SEQ_LENGTHS:
                for p in CP_SIZES:
                    name = model_name(kind, num_latents)
                    if (name, int(n), int(p)) in measured:
                        continue
                    subprocess.run(
                        [
                            "torchrun",
                            "--standalone",
                            f"--nproc_per_node={p}",
                            "ablation/cp_scaling_bwd.py",
                            "--worker",
                            "--model",
                            kind,
                            "--N",
                            str(n),
                            "--cp-size",
                            str(p),
                            "--num-latents",
                            str(num_latents),
                        ],
                        cwd=PROJDIR,
                        check=True,
                    )

    out = backfill_efficiency(pd.read_csv(OUT_CSV))
    out.to_csv(OUT_CSV, index=False)
    return out


def plot_analysis() -> tuple[Any, Any, Any]:
    """Render step time, efficiency, and peak memory strong-scaling panels."""
    import matplotlib.pyplot as plt

    frame = pd.read_csv(OUT_CSV)
    sequence_lengths = sorted(frame["N"].unique())
    model_names = sorted(frame["model_name"].unique())
    colors = plt.get_cmap("tab10").colors
    color_by_n = {n: colors[index % len(colors)] for index, n in enumerate(sequence_lengths)}

    fig, axes_array = plt.subplots(1, 3, figsize=(20, 6))
    axes = tuple(axes_array)
    time_ax, efficiency_ax, memory_ax = axes

    for name in model_names:
        style = "--" if "FLARE++" in name else "-"
        for n in sequence_lengths:
            series = frame[(frame["model_name"] == name) & (frame["N"] == n)].sort_values("P")
            if series.empty:
                continue
            label = f"{name}, N={int(n):,}"
            color = color_by_n[n]
            plot_kwargs = {
                "color": color,
                "linestyle": style,
                "marker": "o",
                "linewidth": 2.0,
                "label": label,
            }
            time_ax.plot(series["P"], series["time_ms"] / 1e3, **plot_kwargs)
            efficiency_ax.plot(series["P"], series["efficiency"], **plot_kwargs)
            memory_ax.plot(series["P"], series["memory_gb"], **plot_kwargs)

            one_gpu = series.loc[series["P"] == 1, "time_ms"]
            if not one_gpu.empty and math.isfinite(float(one_gpu.iloc[0])):
                ideal_seconds = float(one_gpu.iloc[0]) / (series["P"] * 1e3)
                time_ax.plot(
                    series["P"],
                    ideal_seconds,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                    label="_nolegend_",
                )

    efficiency_ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
    ylabels = ("Step time (s)", "Parallel efficiency", "Peak memory (GB)")
    for axis, ylabel in zip(axes, ylabels, strict=True):
        axis.set_xlabel("#GPUs")
        axis.set_ylabel(ylabel)
        axis.set_xticks(CP_SIZES)
        axis.grid(True, alpha=0.3)

    handles, labels = time_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, min(4, len(labels))),
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    for path in (OUT_PNG, OUT_PDF):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return axes


def run_step(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> None:
    """Run one forward and backward step under CUDA fp16 autocast."""
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=torch.cuda.is_available(),
    ):
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    synchronize_fn: Callable[[], None],
) -> None:
    """Run synchronized warmup steps outside the timed sample window."""
    for _ in range(steps):
        synchronize_fn()
        run_step(model, x, target)
        synchronize_fn()


def timed_median_ms(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    reps: int,
    synchronize_fn: Callable[[], None],
) -> float:
    """Return the median synchronized fwd+bwd step time in milliseconds."""
    samples = []
    for _ in range(reps):
        synchronize_fn()
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_step(model, x, target)
            end.record()
            synchronize_fn()
            samples.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            run_step(model, x, target)
            synchronize_fn()
            samples.append((time.perf_counter() - start_time) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark_cell_local(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    warmup_steps: int = WARMUP_STEPS,
    timed_reps: int = TIMED_REPS,
    synchronize_fn: Callable[[], None],
) -> float:
    """Warm up a single-rank cell, then measure a separate timed window."""
    warmup_model(model, x, target, warmup_steps, synchronize_fn)
    return timed_median_ms(model, x, target, timed_reps, synchronize_fn)


def _run_worker_cell(kind: str, N: int, cp_size: int, num_latents: int) -> dict[str, Any]:
    """Execute one distributed CP benchmark cell and return its result."""
    from pdebench.distributed.context_parallel import build_context_parallel_state
    from pdebench.distributed.utils import shard_sequence_tensor

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    mlutils.configure_runtime(
        SEED,
        mixed_precision=True,
        deterministic=False,
        compile_model=True,
    )
    torch._dynamo.config.recompile_limit = 1000

    cp_state = build_context_parallel_state(cp_size, sequence_length=N)
    model = build_model(kind, num_latents).to(device)
    model.set_context_parallel(cp_state)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model = torch.compile(model)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    x_full = torch.randn(1, N, C_IN, device=device, requires_grad=True)
    target_full = torch.randn(1, N, C_OUT, device=device)
    x = shard_sequence_tensor(x_full, cp_state, seq_dim=1)
    x = x.detach().requires_grad_(True)
    target = shard_sequence_tensor(target_full, cp_state, seq_dim=1)

    def sync() -> None:
        dist.barrier()
        torch.cuda.synchronize()

    warmup_model(model, x, target, WARMUP_STEPS, sync)
    torch.cuda.reset_peak_memory_stats(device)
    time_ms = timed_median_ms(model, x, target, TIMED_REPS, sync)
    memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    row = {
        "model_name": model_name(kind, num_latents),
        "N": int(N),
        "P": int(cp_size),
        "time_ms": float(time_ms),
        "memory_gb": float(memory_gb),
        "efficiency": float("nan"),
        "num_valid_runs": TIMED_REPS,
    }
    return row


def run_worker(kind: str, N: int, cp_size: int, num_latents: int) -> dict[str, Any]:
    """Run one torchrun worker cell; rank zero appends the result CSV row."""
    row = _run_worker_cell(kind, N, cp_size, num_latents)
    if dist.get_rank() == 0:
        append_csv_row(OUT_CSV, row)
        print(row)
    dist.barrier()
    return row


def clean_artifacts() -> None:
    """Remove CSV and figure artifacts for this study."""
    for path in (OUT_CSV, OUT_PNG, OUT_PDF):
        if os.path.exists(path):
            os.remove(path)


def main(argv: list[str] | None = None) -> None:
    """Parse the study CLI and dispatch worker cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--worker", action="store_true", help="torchrun cell worker")
    parser.add_argument("--model", choices=list(MODELS), default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--cp-size", type=int, default=None)
    parser.add_argument("--num-latents", type=int, choices=LATENT_COUNTS, default=None)
    args = parser.parse_args(argv)

    if args.worker:
        missing = [
            flag
            for flag, value in (
                ("--model", args.model),
                ("--N", args.N),
                ("--cp-size", args.cp_size),
                ("--num-latents", args.num_latents),
            )
            if value is None
        ]
        if missing:
            parser.error(f"--worker requires {', '.join(missing)}")
        run_worker(args.model, args.N, args.cp_size, args.num_latents)
        return
    if args.run:
        run_matrix()
        return
    if args.plot:
        plot_analysis()
        return
    if args.clean:
        clean_artifacts()
        return
    print("No action specified; use --help for available options.")


if __name__ == "__main__":
    main()

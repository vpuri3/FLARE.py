"""Full MixerBackboneModel fwd+bwd time/memory sweep (FLARE / Simplified FLARE++ / FLARE++ / Transolver 3)."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Optional

PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJDIR not in sys.path:
    sys.path.insert(0, PROJDIR)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton.testing  # noqa: E402

import mlutils  # noqa: E402
from pdebench.models.mixer_backbone import (  # noqa: E402
    FLAREMixerConfig,
    FLAREPPMixerConfig,
    MHAMixerConfig,
    MixerBackboneConfig,
    MixerBackboneModel,
    SimplifiedFLAREPPMixerConfig,
    Transolver3MixerConfig,
)

OUT_CSV = os.path.join(PROJDIR, "out", "pdebench", "time_memory_bwd_flarepp_fp16.csv")
OUT_PNG = os.path.join(PROJDIR, "figs", "time_memory_bwd_flarepp_fp16.png")
OUT_PDF = os.path.join(PROJDIR, "figs", "time_memory_bwd_flarepp_fp16.pdf")

SEQ_LENGTHS = [
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
SEQ_LENGTHS_MHA = list(SEQ_LENGTHS)
NUM_LATENTS = [64, 128, 256]
MEASURED_KINDS = ("flare", "simplifiedflarepp", "transolver3")
C_IN = 3
C_OUT = 1
CHANNEL_DIM = 128
NUM_BLOCKS = 8
NUM_HEADS = 8
NUM_LAYERS_IN_OUT_PROJ = 2
NUM_LAYERS_FFN = 0
MLP_RATIO_FFN = 4.0
RMSNORM = True
OUT_PROJ_NORM = True
BENCHMARK_WARMUP_MS = 100
BENCHMARK_REP_MS = 1000
DYNAMO_RECOMPILE_LIMIT = 1000


def model_name(kind: str, num_latents: Optional[int]) -> str:
    if kind == "mha":
        return "Full self-attention"
    if kind == "flare":
        return f"FLARE ({num_latents} latents)"
    if kind == "simplifiedflarepp":
        return f"Simplified FLARE++ ({num_latents} latents)"
    if kind == "flarepp":
        return f"FLARE++ ({num_latents} latents)"
    if kind == "simplifiedflarepp_qk0_off":
        return f"Simplified FLARE++ qk0_off ({num_latents} latents)"
    if kind == "transolver3":
        return f"Transolver 3 ({num_latents} slices)"
    raise ValueError(f"unknown kind: {kind!r}")


def make_backbone_config(kind: str, num_latents: Optional[int] = None) -> MixerBackboneConfig:
    if kind == "mha":
        mixer = MHAMixerConfig(qk_norm=False)
    elif kind == "flare":
        if num_latents is None:
            raise ValueError("flare requires num_latents")
        mixer = FLAREMixerConfig(num_latents=num_latents, qk_norm=False)
    elif kind == "simplifiedflarepp":
        if num_latents is None:
            raise ValueError("simplifiedflarepp requires num_latents")
        mixer = SimplifiedFLAREPPMixerConfig(
            num_latents=num_latents, qk_norm=False, qk0_norm=True, share_k0_v0=False
        )
    elif kind == "flarepp":
        if num_latents is None:
            raise ValueError("flarepp requires num_latents")
        mixer = FLAREPPMixerConfig(num_latents=num_latents, share_k0_v0=True)
    elif kind == "simplifiedflarepp_qk0_off":
        if num_latents is None:
            raise ValueError("simplifiedflarepp_qk0_off requires num_latents")
        mixer = SimplifiedFLAREPPMixerConfig(
            num_latents=num_latents, qk_norm=False, qk0_norm=False, share_k0_v0=False
        )
    elif kind == "transolver3":
        if num_latents is None:
            raise ValueError("transolver3 requires num_latents")
        mixer = Transolver3MixerConfig(num_latents=num_latents)
    else:
        raise ValueError(f"unsupported kind: {kind!r}")
    return MixerBackboneConfig(
        num_blocks=NUM_BLOCKS,
        channel_dim=CHANNEL_DIM,
        num_heads=NUM_HEADS,
        rmsnorm=RMSNORM,
        out_proj_norm=OUT_PROJ_NORM,
        num_layers_in_out_proj=NUM_LAYERS_IN_OUT_PROJ,
        num_layers_ffn=NUM_LAYERS_FFN,
        mlp_ratio_ffn=MLP_RATIO_FFN,
        mixer=mixer,
    )


def build_backbone(kind: str, num_latents: Optional[int] = None) -> MixerBackboneModel:
    return MixerBackboneModel(
        make_backbone_config(kind, num_latents),
        metadata={"c_in": C_IN, "c_out": C_OUT},
    )


def mha_schema_rows(seq_lengths: list[int]) -> list[dict]:
    return [
        {
            "model_name": model_name("mha", None),
            "N": int(n),
            "time": float("nan"),
            "memory": float("nan"),
            "num_valid_runs": 0,
        }
        for n in seq_lengths
    ]


@contextmanager
def cuda_memory_manager(model):
    try:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        yield
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def benchmark_model(
    model,
    x,
    target,
    *,
    warmup_ms=BENCHMARK_WARMUP_MS,
    rep_ms=BENCHMARK_REP_MS,
) -> tuple[float, float]:
    def forward_backward():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(x)
            loss = F.mse_loss(output, target)
        loss.backward()
        return loss.item()

    # Triton interprets warmup and rep as time budgets in milliseconds, not iteration counts.
    time_median = triton.testing.do_bench(
        forward_backward, warmup=warmup_ms, rep=rep_ms, return_mode="median"
    )
    torch.cuda.reset_peak_memory_stats()
    forward_backward()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    return time_median, peak_memory


def _prepare_measured_model(kind: str, num_latents: Optional[int], device: torch.device):
    model = build_backbone(kind, num_latents)
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, torch.compile(model)


def _configure_speed_runtime() -> dict:
    runtime = mlutils.configure_runtime(
        42, mixed_precision=True, deterministic=False, compile_model=True
    )
    print(
        "runtime_profile={profile} seed={seed} tf32={tf32} "
        "cudnn.benchmark={cudnn_benchmark} cudnn.deterministic={cudnn_deterministic} "
        "deterministic_algorithms={deterministic_algorithms} compile_model={compile_model}".format(
            **runtime
        )
    )
    if runtime["profile"] != "speed":
        raise RuntimeError(f"Expected speed runtime profile, got {runtime['profile']!r}")
    # A single model sees several input lengths, so leave ample room for shape-specialized graphs.
    # Models are compiled and released one at a time to avoid shared-forward recompile-limit
    # fallback and to keep the memory sweep honest.
    torch._dynamo.config.recompile_limit = DYNAMO_RECOMPILE_LIMIT
    return runtime


def _benchmark_model_over_lengths(
    compiled_model,
    name: str,
    seq_lengths: list[int],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    for seq_length in seq_lengths:
        print("=" * 80)
        print(f"N={seq_length:<7}: {name:<28}:", end=" ", flush=True)
        torch.cuda.empty_cache()
        x = torch.randn(1, seq_length, C_IN, device=device, requires_grad=True)
        target = torch.randn(1, seq_length, C_OUT, device=device)
        time_median = float("nan")
        peak_memory = float("nan")
        try:
            with cuda_memory_manager(compiled_model):
                time_median, peak_memory = benchmark_model(compiled_model, x, target)
                print(f"Time: {time_median:.3g}ms, Memory: {peak_memory:.3g}GB")
        except RuntimeError as error:
            print(f"Runtime error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")
        finally:
            del x, target
            torch.cuda.empty_cache()
        rows.append(
            {
                "model_name": name,
                "N": int(seq_length),
                "time": time_median,
                "memory": peak_memory,
                # do_bench chooses iterations dynamically from its millisecond budget.
                "num_valid_runs": None if time_median == time_median else 0,
            }
        )
    return rows


def run_analysis(device: torch.device | None = None) -> pd.DataFrame:
    if device is None:
        device = torch.device("cuda", 0)
    _configure_speed_runtime()
    data: list[dict] = []
    data.extend(mha_schema_rows(SEQ_LENGTHS))

    for kind in MEASURED_KINDS:
        for num_latents in NUM_LATENTS:
            model, compiled_model = _prepare_measured_model(kind, num_latents, device)
            name = model_name(kind, num_latents)
            try:
                data.extend(
                    _benchmark_model_over_lengths(compiled_model, name, SEQ_LENGTHS, device)
                )
            finally:
                del compiled_model, model
                torch.cuda.empty_cache()

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    return df


def run_mha_analysis(device: torch.device | None = None) -> pd.DataFrame:
    """Measure Full self-attention on SEQ_LENGTHS_MHA; merge into OUT_CSV if present.

    Already-measured finite MHA rows in OUT_CSV are kept and skipped (resume-friendly).
    """
    if device is None:
        device = torch.device("cuda", 0)
    _configure_speed_runtime()
    name = model_name("mha", None)

    existing_mha = pd.DataFrame(columns=["model_name", "N", "time", "memory", "num_valid_runs"])
    kept = pd.DataFrame(columns=["model_name", "N", "time", "memory", "num_valid_runs"])
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        kept = existing[existing["model_name"] != name].copy()
        existing_mha = existing[existing["model_name"] == name].copy()

    done_ns = set(
        int(n)
        for n, t in zip(existing_mha.get("N", []), existing_mha.get("time", []))
        if pd.notna(t)
    )
    todo = [n for n in SEQ_LENGTHS_MHA if int(n) not in done_ns]
    print(f"MHA resume: {len(done_ns)} done, {len(todo)} remaining: {todo}")

    new_rows: list[dict] = []
    if todo:
        model, compiled_model = _prepare_measured_model("mha", None, device)
        try:
            new_rows = _benchmark_model_over_lengths(compiled_model, name, todo, device)
        finally:
            del compiled_model, model
            torch.cuda.empty_cache()

    kept_measured = existing_mha[existing_mha["time"].notna()].copy() if len(existing_mha) else existing_mha
    still_missing = [n for n in SEQ_LENGTHS if int(n) not in done_ns and int(n) not in {r["N"] for r in new_rows}]
    # Rows that were just attempted but OOM'd are already in new_rows as NaN; don't double-add schema.
    attempted_ns = {int(r["N"]) for r in new_rows}
    schema_ns = [n for n in still_missing if int(n) not in attempted_ns]
    mha_df = pd.concat(
        [
            kept_measured,
            pd.DataFrame(new_rows),
            pd.DataFrame(mha_schema_rows(schema_ns)),
        ],
        ignore_index=True,
    )
    # Prefer newly measured finite values over older ones for the same N.
    mha_df = mha_df.sort_values(["N", "time"], na_position="last").drop_duplicates(subset=["N"], keep="first")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.concat([kept, mha_df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} (MHA merge)")
    return df


def _run_kind_merge_analysis(kind: str, merge_label: str, device: torch.device) -> pd.DataFrame:
    """Measure one mixer kind over NUM_LATENTS × SEQ_LENGTHS; merge into OUT_CSV.

    Rows for other model names are kept. Already-measured finite rows for this kind
    are kept and skipped (resume-friendly).
    """
    names = {model_name(kind, m) for m in NUM_LATENTS}

    existing = pd.DataFrame(columns=["model_name", "N", "time", "memory", "num_valid_runs"])
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)

    kept = existing[~existing["model_name"].isin(names)].copy()
    existing_kind = existing[existing["model_name"].isin(names)].copy()

    new_rows: list[dict] = []
    for num_latents in NUM_LATENTS:
        name = model_name(kind, num_latents)
        prior = existing_kind[existing_kind["model_name"] == name]
        done_ns = set(
            int(n)
            for n, t in zip(prior.get("N", []), prior.get("time", []))
            if pd.notna(t)
        )
        todo = [n for n in SEQ_LENGTHS if int(n) not in done_ns]
        print(f"{name} resume: {len(done_ns)} done, {len(todo)} remaining: {todo}")
        if not todo:
            continue
        model, compiled_model = _prepare_measured_model(kind, num_latents, device)
        try:
            new_rows.extend(_benchmark_model_over_lengths(compiled_model, name, todo, device))
        finally:
            del compiled_model, model
            torch.cuda.empty_cache()

    kept_measured = (
        existing_kind[existing_kind["time"].notna()].copy() if len(existing_kind) else existing_kind
    )
    kind_df = pd.concat([kept_measured, pd.DataFrame(new_rows)], ignore_index=True)
    if len(kind_df):
        kind_df = kind_df.sort_values(
            ["model_name", "N", "time"], na_position="last"
        ).drop_duplicates(subset=["model_name", "N"], keep="first")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.concat([kept, kind_df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({merge_label} merge)")
    return df


def run_anchored_analysis(device: torch.device | None = None) -> pd.DataFrame:
    """Measure FLARE++ (share_k0_v0=True); merge into OUT_CSV if present."""
    if device is None:
        device = torch.device("cuda", 0)
    _configure_speed_runtime()
    return _run_kind_merge_analysis("flarepp", "FLARE++", device)


def run_simplifiedflarepp_qk0_off_analysis(device: torch.device | None = None) -> pd.DataFrame:
    """Measure Simplified FLARE++ with qk0_norm=False; merge into OUT_CSV if present."""
    if device is None:
        device = torch.device("cuda", 0)
    _configure_speed_runtime()
    return _run_kind_merge_analysis("simplifiedflarepp_qk0_off", "Simplified FLARE++ qk0_off", device)


def plot_analysis() -> None:
    import subprocess

    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_CSV)
    try:
        subprocess.run(["latex", "--version"], capture_output=True, check=True)
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        print("Using LaTeX for plot rendering")
    except (subprocess.CalledProcessError, FileNotFoundError):
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            }
        )
        print("LaTeX not available, using default matplotlib fonts")

    fontsize = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    for ax in (ax1, ax2):
        ax.set_xscale("linear")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(r"Sequence Length", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax1.set_yscale("log", base=10)
    ax2.set_yscale("linear")
    ax2.set_ylim(0, 85)
    ax1.set_ylabel(r"Time (s)", fontsize=fontsize)
    ax2.set_ylabel(r"Memory (GB)", fontsize=fontsize)

    def _tick_label(n: int) -> str:
        if n >= 1_000_000 and n % 1_000_000 == 0:
            return f"{n // 1_000_000}m"
        if n >= 1000 and n % 1000 == 0:
            return f"{n // 1000}k"
        return str(n)

    x_ticks = sorted(df["N"].unique().tolist())
    x_tick_labels = [_tick_label(int(n)) for n in x_ticks]
    for ax in (ax1, ax2):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    ax2.axhline(y=80, color="black", linestyle="--", linewidth=3.0)

    marker_by_m = {64: "s", 128: "v", 256: "D"}
    plot_specs: list[tuple[str, str, str, str, str]] = []
    mha = df[df["model_name"] == "Full self-attention"]
    if not mha.empty and mha["time"].notna().any():
        plot_specs.append(("Full self-attention", r"Full self-attention", "black", "o", "-"))
    for m, marker in marker_by_m.items():
        plot_specs.append((f"FLARE ({m} latents)", rf"FLARE ({m} latents)", "red", marker, "-"))
        plot_specs.append(
            (f"Simplified FLARE++ ({m} latents)", rf"Simplified FLARE++ ({m} latents)", "magenta", marker, "-")
        )
        qk0_off_name = f"Simplified FLARE++ qk0_off ({m} latents)"
        if not df[df["model_name"] == qk0_off_name].empty:
            plot_specs.append(
                (qk0_off_name, rf"Simplified FLARE++ qk0-off ({m} latents)", "darkorange", marker, ":")
            )
        anchored_name = f"FLARE++ ({m} latents)"
        if not df[df["model_name"] == anchored_name].empty:
            plot_specs.append(
                (anchored_name, rf"FLARE++ ({m} latents)", "green", marker, "-.")
            )
        plot_specs.append(
            (f"Transolver 3 ({m} slices)", rf"Transolver 3 ({m} slices)", "blue", marker, "--")
        )

    for name, label, color, marker, linestyle in plot_specs:
        series = df[df["model_name"] == name].sort_values(by="N")
        if series.empty:
            raise ValueError(f"Missing expected model series: {name}")
        ax1.plot(
            series["N"],
            series["time"] / 1e3,
            label=label,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            color=color,
            markersize=8,
        )
        ax2.plot(
            series["N"],
            series["memory"],
            label=label,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            color=color,
            markersize=8,
        )

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, 0.00),
        columnspacing=0.8,
        handletextpad=0.3,
        bbox_transform=fig.transFigure,
        handlelength=1.5,
        markerscale=1.5,
    )
    ax1.set_title(r"Execution Time (Forward + Backward)", fontsize=fontsize)
    ax2.set_title(r"Peak Memory Usage (Forward + Backward)", fontsize=fontsize)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.26)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Full MixerBackboneModel fwd+bwd time/memory (FLARE / Simplified FLARE++ / FLARE++ / Transolver 3 / MHA)"
    )
    parser.add_argument("--run", action="store_true", help="Measure FLARE/Simplified FLARE++/Transolver3 through N=1M")
    parser.add_argument(
        "--mha",
        action="store_true",
        help="Full self-attention run through N=1M (skips already-measured N); merges into the CSV",
    )
    parser.add_argument(
        "--anchored",
        action="store_true",
        help="FLARE++ (share_k0_v0=True) through N=1M; merges into the CSV without replacing other models",
    )
    parser.add_argument(
        "--simplifiedflarepp-qk0-off",
        action="store_true",
        help="Simplified FLARE++ with qk0_norm=False through N=1M; merges into the CSV without replacing other models",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)

    if args.clean:
        for path in (OUT_CSV, OUT_PNG, OUT_PDF):
            if os.path.exists(path):
                print(f"Removing {path}")
                os.remove(path)
    if args.run:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --run")
        run_analysis()
    if args.mha:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --mha")
        run_mha_analysis()
    if args.anchored:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --anchored")
        run_anchored_analysis()
    if args.simplifiedflarepp_qk0_off:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --simplifiedflarepp-qk0-off")
        run_simplifiedflarepp_qk0_off_analysis()
    if args.plot:
        plot_analysis()
    if (
        not args.run
        and not args.mha
        and not args.anchored
        and not args.simplifiedflarepp_qk0_off
        and not args.plot
        and not args.clean
    ):
        print(
            "No action specified. Please specify either --run, --mha, --anchored, "
            "--simplifiedflarepp-qk0-off, --plot, or --clean."
        )


if __name__ == "__main__":
    main()

"""Offline dataloader + train-step benchmark.

Runs 100 real training steps (compile on, fullbatch stats off), then reports:

- total wall time
- compile/warmup overhead (first-50 excess vs latter-50 mean)
- per-step metrics from **latter 50 steps only**: time/step, data wait/step, wait fraction
- data wait total (all 100) and latter-50 wait total

FAIL when latter-50 ``wait/step > 10%``.

Usage:
  python -m pdebench.dataset.bench_dataloader --dataset bracket_lug --model glt \\
      --batch-size 8 --laplacian-k 0 --pe-inject-mode concat_input
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

WAIT_FRACTION_FAIL = 0.10
DEFAULT_NUM_STEPS = 100
DEFAULT_METRIC_TAIL = 50
SCHEMA_VERSION = 2
TIMELINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+): elapsed=([0-9.]+)s delta=([0-9.]+)s",
    re.MULTILINE,
)
TIMELINE_FIELDS = (
    "dataset_init_s",
    "model_init_s",
    "compile_setup_s",
    "dataloader_setup_s",
    "startup_stats_s",
    "first_batch_ready_s",
    "first_train_step_ready_s",
)


def verdict(wait_s: float | None, step_s: float | None, threshold: float = WAIT_FRACTION_FAIL) -> str:
    if wait_s is None or step_s is None or not math.isfinite(wait_s) or not math.isfinite(step_s) or step_s <= 0:
        return "FAIL"
    return "FAIL" if (wait_s / step_s) > threshold else "PASS"


def _as_float_list(snapshot: dict[str, Any], key: str) -> list[float]:
    values = snapshot.get(key)
    if values is None:
        return []
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr]


def parse_timeline(text: str) -> dict[str, float]:
    """Return the last elapsed timestamp recorded for each timeline marker."""
    return {match.group(1): float(match.group(2)) for match in TIMELINE_RE.finditer(text)}


def _elapsed_between(markers: dict[str, float], start: str, end: str) -> float | None:
    if start not in markers or end not in markers:
        return None
    value = markers[end] - markers[start]
    return value if math.isfinite(value) else None


def summarize_timeline(markers: dict[str, float]) -> dict[str, float | None]:
    """Derive setup and startup durations from elapsed timeline markers."""
    return {
        "dataset_init_s": _elapsed_between(markers, "dataset_load_start", "dataset_loaded"),
        "model_init_s": _elapsed_between(markers, "model_build_start", "model_built"),
        "compile_setup_s": _elapsed_between(markers, "trainer_construct_start", "trainer_compile"),
        "dataloader_setup_s": _elapsed_between(markers, "trainer_train_enter", "trainer_dataloader"),
        "startup_stats_s": _elapsed_between(markers, "trainer_statistics_start", "trainer_statistics_done"),
        "first_batch_ready_s": _elapsed_between(markers, "trainer_train_enter", "trainer_first_batch_ready"),
        "first_train_step_ready_s": _elapsed_between(
            markers, "trainer_train_enter", "trainer_first_train_step"
        ),
    }


derive_timeline_timing = summarize_timeline


def _finite_or_none(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: list[float]) -> float | None:
    return _finite_or_none(np.mean(values)) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return value


def _markdown_value(value: Any) -> Any:
    return "—" if value is None else value


def summarize_timing(
    time_per_step: list[float],
    time_dataload: list[float],
    *,
    total_wall_s: float,
    metric_tail: int = DEFAULT_METRIC_TAIL,
) -> dict[str, Any]:
    n = min(len(time_per_step), len(time_dataload))
    steps = time_per_step[:n]
    waits = time_dataload[:n]
    dropped = max(len(time_per_step), len(time_dataload)) - n
    tail = max(0, min(metric_tail, n))
    head = n - tail
    steps_tail = steps[head:]
    waits_tail = waits[head:]
    steps_head = steps[:head] if head else []
    waits_head = waits[:head] if head else []

    mean_step_tail = _mean(steps_tail)
    mean_wait_tail = _mean(waits_tail)
    wait_fraction_tail = (
        _finite_or_none(mean_wait_tail / mean_step_tail)
        if mean_wait_tail is not None and mean_step_tail is not None and mean_step_tail > 0
        else None
    )
    total_steps = [step + wait for step, wait in zip(steps, waits)]
    total_steps_tail = total_steps[head:]
    total_steps_head = total_steps[:head]
    steady_state_mean_total = _mean(total_steps_tail)
    warmup_mean_total = _mean(total_steps_head)

    # Warmup overhead is the first-segment excess against steady-state total iteration time.
    if head > 0 and steady_state_mean_total is not None:
        head_wall = _finite_or_none(sum(total_steps_head))
        warmup_overhead_s = (
            _finite_or_none(max(0.0, head_wall - head * steady_state_mean_total))
            if head_wall is not None
            else None
        )
    else:
        warmup_overhead_s = None

    steady_state_std = _finite_or_none(np.std(total_steps_tail)) if total_steps_tail else None
    steady_state_p95 = _finite_or_none(np.percentile(total_steps_tail, 95)) if total_steps_tail else None
    steady_state_p99 = _finite_or_none(np.percentile(total_steps_tail, 99)) if total_steps_tail else None
    steady_state_cv = (
        _finite_or_none(steady_state_std / steady_state_mean_total)
        if steady_state_std is not None and steady_state_mean_total is not None and steady_state_mean_total > 0
        else None
    )
    warmup_to_steady_ratio = (
        _finite_or_none(warmup_mean_total / steady_state_mean_total)
        if warmup_mean_total is not None and steady_state_mean_total is not None and steady_state_mean_total > 0
        else None
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "num_steps_recorded": n,
        "timing_samples_dropped": dropped,
        "metric_tail": tail,
        "warmup_steps": head,
        "steady_state_steps": tail,
        "total_wall_s": _finite_or_none(total_wall_s),
        "warmup_overhead_s": warmup_overhead_s,
        "compile_overhead_s": warmup_overhead_s,
        "warmup_mean_step_s": _mean(steps_head),
        "warmup_mean_total_step_s": warmup_mean_total,
        "steady_state_mean_total_step_s": steady_state_mean_total,
        "warmup_to_steady_step_ratio": warmup_to_steady_ratio,
        "steady_state_step_std_s": steady_state_std,
        "steady_state_step_p95_s": steady_state_p95,
        "steady_state_step_p99_s": steady_state_p99,
        "steady_state_step_cv": steady_state_cv,
        "first_step_s": _finite_or_none(steps[0] + waits[0]) if n else None,
        "time_per_step_s": mean_step_tail,
        "data_wait_per_step_s": mean_wait_tail,
        "data_wait_total_s": _finite_or_none(sum(waits)) if n else None,
        "data_wait_tail_total_s": _finite_or_none(sum(waits_tail)) if waits_tail else None,
        "wait_to_step_ratio": wait_fraction_tail,
        "wait_fraction": wait_fraction_tail,
        "threshold": WAIT_FRACTION_FAIL,
        "verdict": verdict(mean_wait_tail, mean_step_tail),
        "mean_step_head_s": _mean(steps_head),
        "mean_wait_head_s": _mean(waits_head),
    }
    if n == 0:
        result["error"] = "no timing samples in checkpoint"
    return result


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt")),
        key=lambda p: int(p.name.removeprefix("ckpt")),
    )
    if not ckpts:
        raise FileNotFoundError(f"No ckpt* under {run_dir}")
    return ckpts[-1] / "model.pt"


def _build_train_cmd(
    *,
    dataset: str,
    model: str,
    batch_size: int,
    num_steps: int,
    num_workers: int,
    laplacian_k: int,
    pe_inject_mode: str,
    exp_name: str,
    data_root: str,
    compile_model: bool,
    mixed_precision: bool,
    amp_dtype: str,
    graph_feats: bool | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "pdebench",
        "--run.train=true",
        f"--run.exp_name={exp_name}",
        f"--dataset.dataset={dataset}",
        f"--model.model={model}",
        "--training.epochs=0",
        f"--training.steps={num_steps}",
        f"--training.batch_size={batch_size}",
        f"--training.num_workers={num_workers}",
        "--training.fullbatch_stats_train=false",
        "--training.fullbatch_stats_test=false",
        "--training.fullbatch_stats_on_start=false",
        f"--training.compile_model={'true' if compile_model else 'false'}",
        f"--training.mixed_precision={'true' if mixed_precision else 'false'}",
        f"--training.amp_dtype={amp_dtype}",
        "--training.ema=false",
        f"--dataset.data_root={data_root}",
    ]
    if model == "glt":
        cmd.extend(
            [
                f"--model.pe_inject_mode={pe_inject_mode}",
                "--model.pe.kind=raw_eigen",
                f"--model.pe.num_eigenmodes={laplacian_k}",
            ]
        )
    del graph_feats  # GALE GeoTransolver no longer has a graph_feats edge path.
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_bench(
    dataset: str,
    *,
    model: str = "glt",
    data_root: str = "data",
    num_steps: int = DEFAULT_NUM_STEPS,
    metric_tail: int = DEFAULT_METRIC_TAIL,
    batch_size: int = 8,
    num_workers: int = 0,
    laplacian_k: int = 0,
    pe_inject_mode: str = "concat_input",
    compile_model: bool = True,
    mixed_precision: bool = True,
    amp_dtype: str = "bf16",
    outdir: Path | None = None,
    use_srun: bool = True,
    gpus_per_task: int = 1,
    cpus_per_task: int | None = None,
    graph_feats: bool | None = None,
    label: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    host = socket.gethostname().split(".")[0]
    outdir = outdir or Path("out/pdebench/dataloader_bench")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = label or model
    if model == "geo_transolver" and graph_feats is not None and label is None:
        tag = f"geo_transolver_gf{str(graph_feats).lower()}"
    exp_name = f"debug/dataloader_bench/{dataset}_{tag}_bs{batch_size}"
    case_dir = Path("out/pdebench") / exp_name
    if case_dir.exists():
        # Fresh run directory to avoid stale ckpt timing.
        import shutil

        shutil.rmtree(case_dir)

    cmd = _build_train_cmd(
        dataset=dataset,
        model=model,
        batch_size=batch_size,
        num_steps=num_steps,
        num_workers=num_workers,
        laplacian_k=laplacian_k,
        pe_inject_mode=pe_inject_mode,
        exp_name=exp_name,
        data_root=data_root,
        compile_model=compile_model,
        mixed_precision=mixed_precision,
        amp_dtype=amp_dtype,
        graph_feats=graph_feats,
        extra_args=extra_args,
    )
    if use_srun and os.environ.get("SLURM_JOB_ID"):
        cpus = int(cpus_per_task or os.environ.get("SLURM_CPUS_PER_TASK", "26"))
        # If CUDA_VISIBLE_DEVICES is already pinned, do NOT nest --gpus-per-task:
        # under a multi-GPU allocation that remaps every step onto physical GPU0.
        srun_cmd = [
            "srun",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            "--overlap",
        ]
        if not os.environ.get("CUDA_VISIBLE_DEVICES") and int(gpus_per_task or 0) > 0:
            srun_cmd.append(f"--gpus-per-task={gpus_per_task}")
        cmd = [*srun_cmd, *cmd]

    log_path = outdir / f"{dataset}_{tag}_{host}_train.log"
    t0 = time.perf_counter()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    total_wall_s = time.perf_counter() - t0

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model": model,
        "label": tag,
        "graph_feats": graph_feats,
        "pe_inject_mode": pe_inject_mode if model == "glt" else None,
        "laplacian_k": laplacian_k if model == "glt" else 0,
        "host": host,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_steps": num_steps,
        "metric_tail": metric_tail,
        "compile_model": compile_model,
        "mixed_precision": mixed_precision,
        "amp_dtype": amp_dtype,
        "exp_name": exp_name,
        "case_dir": str(case_dir),
        "log_path": str(log_path),
        "cmd": cmd,
        "returncode": proc.returncode,
        "subprocess_wall_s": total_wall_s,
    }
    try:
        timeline_text = log_path.read_text()
    except OSError:
        timeline_text = ""
    report.update(summarize_timeline(parse_timeline(timeline_text)))

    if proc.returncode != 0:
        report["verdict"] = "FAIL"
        report["error"] = f"training exited {proc.returncode}; see {log_path}"
        _write_report(outdir, dataset, host, report, tag=tag)
        return report

    try:
        ckpt = _latest_checkpoint(case_dir)
        snapshot = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["error"] = f"failed to load checkpoint timing: {exc}"
        _write_report(outdir, dataset, host, report, tag=tag)
        return report

    steps = _as_float_list(snapshot, "time_per_step")
    waits = _as_float_list(snapshot, "time_dataload_per_step")
    # Prefer trainer-recorded wall if present; else subprocess wall.
    timing = summarize_timing(steps, waits, total_wall_s=total_wall_s, metric_tail=metric_tail)
    report.update(timing)
    if report.get("time_per_step_s"):
        report["it_s_tail"] = _finite_or_none(1.0 / report["time_per_step_s"])
    _write_report(outdir, dataset, host, report, tag=tag)
    return report


def _write_report(
    outdir: Path,
    dataset: str,
    host: str,
    report: dict[str, Any],
    *,
    tag: str | None = None,
) -> None:
    stem = outdir / f"{dataset}_{tag or report.get('label') or report.get('model')}_{host}"
    stem.with_suffix(".json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    lines = [
        f"# Dataloader/train bench: `{dataset}` / `{report.get('label') or report.get('model')}` on `{host}`",
        "",
        f"- verdict: **{report.get('verdict', 'FAIL')}**",
        (
            f"- model={report.get('model')} graph_feats={report.get('graph_feats')} "
            f"pe_inject_mode={report.get('pe_inject_mode')} K={report.get('laplacian_k')}"
        ),
        f"- batch_size={report.get('batch_size')} workers={report.get('num_workers')} steps={report.get('num_steps')}",
        f"- total_wall_s={_markdown_value(report.get('total_wall_s', report.get('subprocess_wall_s')))}",
        f"- dataset_init_s={_markdown_value(report.get('dataset_init_s'))}",
        f"- model_init_s={_markdown_value(report.get('model_init_s'))}",
        f"- compile_setup_s={_markdown_value(report.get('compile_setup_s'))}",
        f"- dataloader_setup_s={_markdown_value(report.get('dataloader_setup_s'))}",
        f"- startup_stats_s={_markdown_value(report.get('startup_stats_s'))}",
        f"- first_batch_ready_s={_markdown_value(report.get('first_batch_ready_s'))}",
        f"- first_train_step_ready_s={_markdown_value(report.get('first_train_step_ready_s'))}",
        f"- warmup_steps={_markdown_value(report.get('warmup_steps'))}",
        f"- steady_state_steps={_markdown_value(report.get('steady_state_steps'))}",
        f"- warmup_overhead_s={_markdown_value(report.get('warmup_overhead_s'))}",
        f"- warmup_mean_step_s={_markdown_value(report.get('warmup_mean_step_s'))}",
        f"- warmup_mean_total_step_s={_markdown_value(report.get('warmup_mean_total_step_s'))}",
        f"- steady_state_mean_total_step_s={_markdown_value(report.get('steady_state_mean_total_step_s'))}",
        f"- warmup_to_steady_step_ratio={_markdown_value(report.get('warmup_to_steady_step_ratio'))}",
        f"- steady_state_step_std_s={_markdown_value(report.get('steady_state_step_std_s'))}",
        f"- steady_state_step_p95_s={_markdown_value(report.get('steady_state_step_p95_s'))}",
        f"- steady_state_step_p99_s={_markdown_value(report.get('steady_state_step_p99_s'))}",
        f"- steady_state_step_cv={_markdown_value(report.get('steady_state_step_cv'))}",
        f"- compile_overhead_s (alias)={_markdown_value(report.get('compile_overhead_s'))}",
        f"- time_per_step_s (tail)={_markdown_value(report.get('time_per_step_s'))}",
        f"- it_s_tail={_markdown_value(report.get('it_s_tail'))}",
        f"- data_wait_per_step_s (tail)={_markdown_value(report.get('data_wait_per_step_s'))}",
        f"- wait_to_step_ratio (tail)={_markdown_value(report.get('wait_to_step_ratio'))}",
        f"- wait_fraction (alias)={_markdown_value(report.get('wait_fraction'))}",
        f"- data_wait_total_s={_markdown_value(report.get('data_wait_total_s'))}",
        f"- timing_samples_dropped={_markdown_value(report.get('timing_samples_dropped'))}",
    ]
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    lines.append("")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="glt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p.add_argument("--metric-tail", type=int, default=DEFAULT_METRIC_TAIL)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--laplacian-k", type=int, default=0)
    p.add_argument("--pe-inject-mode", default="concat_input")
    p.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", default="bf16")
    p.add_argument("--outdir", type=Path, default=Path("out/pdebench/dataloader_bench"))
    p.add_argument("--no-srun", action="store_true")
    p.add_argument("--graph-feats", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--extra-arg", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_bench(
        args.dataset,
        model=args.model,
        data_root=args.data_root,
        num_steps=args.num_steps,
        metric_tail=args.metric_tail,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        laplacian_k=args.laplacian_k,
        pe_inject_mode=args.pe_inject_mode,
        compile_model=args.compile_model,
        mixed_precision=args.mixed_precision,
        amp_dtype=args.amp_dtype,
        outdir=args.outdir,
        use_srun=not args.no_srun,
        graph_feats=args.graph_feats,
        label=args.label,
        extra_args=args.extra_arg,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()

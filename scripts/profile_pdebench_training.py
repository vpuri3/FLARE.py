#!/usr/bin/env python
"""Summarize per-step timing from a pdebench Trainer checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt")),
        key=lambda p: int(p.name.removeprefix("ckpt")),
    )
    if not ckpts:
        raise FileNotFoundError(f"No ckptXX directories found under {run_dir}")
    return ckpts[-1] / "model.pt"


def _as_array(snapshot: dict, key: str) -> np.ndarray:
    values = snapshot.get(key)
    if values is None:
        raise KeyError(f"Checkpoint does not contain {key!r}")
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _summary(name: str, values: np.ndarray) -> dict[str, float]:
    finite = _finite(values)
    if finite.size == 0:
        return {"name": name, "n": values.size, "finite": 0}
    return {
        "name": name,
        "n": values.size,
        "finite": finite.size,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _print_summary(summary: dict[str, float]) -> None:
    if summary["finite"] == 0:
        print(f"{summary['name']:>14}: n={summary['n']} finite=0")
        return
    print(
        f"{summary['name']:>14}: "
        f"mean={summary['mean']:.6f}s median={summary['median']:.6f}s "
        f"p90={summary['p90']:.6f}s p95={summary['p95']:.6f}s "
        f"p99={summary['p99']:.6f}s max={summary['max']:.6f}s "
        f"finite={summary['finite']}/{summary['n']}"
    )


def _top_steps(total_wall: np.ndarray, arrays: dict[str, np.ndarray], k: int) -> list[int]:
    finite = np.where(np.isfinite(total_wall))[0]
    if finite.size == 0:
        return []
    order = finite[np.argsort(total_wall[finite])[-k:]][::-1]
    return [int(i) for i in order]


def _percent(part: float, whole: float) -> float:
    if not math.isfinite(part) or not math.isfinite(whole) or whole <= 0:
        return float("nan")
    return 100.0 * part / whole


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="pdebench run directory, e.g. out/pdebench/micro_puc_FLA_B8_C64_M128")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Specific model.pt checkpoint to analyze")
    parser.add_argument("--top-k", type=int, default=12, help="Number of slowest steps to print")
    args = parser.parse_args()

    ckpt = args.checkpoint or _latest_checkpoint(args.run_dir)
    snapshot = torch.load(ckpt, map_location="cpu", weights_only=False)

    dataload = _as_array(snapshot, "time_dataload_per_step")
    step = _as_array(snapshot, "time_per_step")
    model_eval = _as_array(snapshot, "time_model_eval_per_step")
    n = min(dataload.size, step.size, model_eval.size)
    dataload = dataload[:n]
    step = step[:n]
    model_eval = model_eval[:n]

    train_loss = snapshot.get("train_loss_per_batch", [])
    train_loss = np.asarray(train_loss[:n], dtype=np.float64) if len(train_loss) else np.full(n, np.nan)

    # time_per_step starts after data has been fetched. It includes forward/loss,
    # backward, optimizer, scheduler, callbacks, and bookkeeping.
    non_dataload = step
    total_wall = dataload + step
    other = np.maximum(non_dataload - model_eval, 0.0)

    print(f"checkpoint: {ckpt}")
    print(f"epoch={snapshot.get('epoch')} step={snapshot.get('step')} timed_steps={n}")
    print()
    for name, values in [
        ("total_wall", total_wall),
        ("dataload", dataload),
        ("non_dataload", non_dataload),
        ("model_eval", model_eval),
        ("backward_opt", other),
    ]:
        _print_summary(_summary(name, values))

    totals = {name: _summary(name, values) for name, values in {
        "total_wall": total_wall,
        "dataload": dataload,
        "model_eval": model_eval,
        "backward_opt": other,
    }.items()}
    mean_total = totals["total_wall"].get("mean", float("nan"))
    print()
    print("mean wall-time attribution:")
    for name in ["dataload", "model_eval", "backward_opt"]:
        mean = totals[name].get("mean", float("nan"))
        print(f"  {name:>12}: {mean:.6f}s ({_percent(mean, mean_total):.1f}%)")

    slow_threshold = totals["total_wall"].get("p95", float("nan"))
    if math.isfinite(slow_threshold):
        slow = total_wall >= slow_threshold
        print()
        print(f"slow-step attribution at total_wall >= p95 ({slow_threshold:.6f}s):")
        slow_total = float(np.mean(total_wall[slow]))
        for name, values in [("dataload", dataload), ("model_eval", model_eval), ("backward_opt", other)]:
            mean = float(np.mean(values[slow]))
            print(f"  {name:>12}: {mean:.6f}s ({_percent(mean, slow_total):.1f}%)")

    print()
    print(f"slowest {args.top_k} steps:")
    for i in _top_steps(total_wall, {"dataload": dataload, "model_eval": model_eval, "backward_opt": other}, args.top_k):
        print(
            f"  step={i:6d} total={total_wall[i]:.6f}s dataload={dataload[i]:.6f}s "
            f"model_eval={model_eval[i]:.6f}s backward_opt={other[i]:.6f}s loss={train_loss[i]:.6g}"
        )


if __name__ == "__main__":
    main()

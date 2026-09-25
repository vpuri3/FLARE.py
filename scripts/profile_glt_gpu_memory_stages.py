#!/usr/bin/env python3
"""Stage-wise GPU memory breakdown for GLT GINOT pdebench training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pdebench  # noqa: E402
from mlutils.trainer import Trainer  # noqa: E402
from pdebench.config import (  # noqa: E402
    Config,
    GinotDatasetConfig,
    GLTConfig,
    OptimizerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
)
from pdebench.dataset.ginot.stats import make_ginot_statsfun  # noqa: E402
from pdebench.models.graph_models.glt_pe import SpectralFilterPEConfig  # noqa: E402
from pdebench.models.model_factory import make_model  # noqa: E402
from pdebench.utils import make_optimizer_adamw  # noqa: E402


def _gb(x: int) -> float:
    return x / (1024**3)


def mem_snapshot(tag: str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"tag": tag}
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "allocated_gb": _gb(torch.cuda.memory_allocated()),
        "reserved_gb": _gb(torch.cuda.memory_reserved()),
        "max_allocated_gb": _gb(torch.cuda.max_memory_allocated()),
        "max_reserved_gb": _gb(torch.cuda.max_memory_reserved()),
    }


def print_snapshot(s: dict[str, float]) -> None:
    print(
        f"[{s['tag']}] "
        f"alloc={s.get('allocated_gb', 0):.3f} GiB "
        f"reserved={s.get('reserved_gb', 0):.3f} GiB "
        f"max_alloc={s.get('max_allocated_gb', 0):.3f} GiB "
        f"max_reserved={s.get('max_reserved_gb', 0):.3f} GiB"
    )


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def build_cfg(pe_spe_filter_type: str, pe_spe_mode: str, compile_model: bool) -> Config:
    return Config(
        run=RunConfig(train=True, exp_name="MEM_PROFILE", seed=0),
        dataset=GinotDatasetConfig(dataset="micro_puc_fixed"),
        model=GLTConfig(
            pe_inject_mode="concat_qk",
            pe_update=True,
            pe=SpectralFilterPEConfig(
                num_eigenmodes=32,
                filter_type=pe_spe_filter_type,
                mode=pe_spe_mode,
            ),
        ),
        training=TrainingConfig(
            batch_size=64,
            num_workers=16,
            prefetch_factor=8,
            compile_model=compile_model,
            mixed_precision=True,
            amp_dtype="bf16",
            overlap_train_dataloader=True,
            stats_on_start=False,
            fullbatch_stats_train=True,
            fullbatch_stats_test=True,
            epochs=0,
            steps=0,
            ema=False,
        ),
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=1e-3),
        scheduler=SchedulerConfig(schedule="OneCycleLR"),
    )


def run_profile(args: argparse.Namespace) -> list[dict[str, float]]:
    device = torch.device("cuda")
    rows: list[dict[str, float]] = []

    def mark(tag: str) -> None:
        rows.append(mem_snapshot(tag))
        print_snapshot(rows[-1])

    reset_peak()
    mark("00_cuda_init")

    cfg = build_cfg(args.pe_spe_filter_type, args.pe_spe_mode, args.compile_model)
    data_root = os.environ.get("DATADIR_BASE", "data")
    train_data, test_data, metadata = pdebench.load_dataset(
        "micro_puc_fixed",
        data_root,
        REPO_ROOT,
        ginot_include_edges=True,
        mesh_split_seed=0,
        ginot_laplacian_eig_dim=cfg.model.pe.num_eigenmodes,
        ginot_laplacian_spec=cfg.model.pe.laplacian_spec,
    )
    mark("01_dataset_loaded")
    print(f"train={len(train_data)} test={0 if test_data is None else len(test_data)}")

    cfg, model = make_model(cfg, metadata, 0)
    model = model.to(device)
    mark("02_model_on_gpu")

    if args.compile_model:
        model = torch.compile(model)
        mark("03_after_torch_compile")

    trainer = Trainer(
        model,
        train_data,
        test_data,
        device=device,
        mixed_precision=True,
        amp_dtype="bf16",
        compile_model=False,
        static_graph=True,
        _batch_size=64,
        batch_size_=64,
        _batch_size_=64,
        num_workers=args.num_workers,
        prefetch_factor=8,
        overlap_train_dataloader=True,
        make_optimizer=make_optimizer_adamw,
        weight_decay=1e-3,
        lr=1e-3,
        opt_beta1=0.9,
        opt_beta2=0.999,
        opt_eps=1e-8,
        Schedule="OneCycleLR",
        one_cycle_pct_start=0.10,
        one_cycle_div_factor=10000.0,
        one_cycle_final_div_factor=10000.0,
        epochs=0,
        steps=args.train_steps,
        stats_on_start=False,
        _fullbatch_stats=args.fullbatch_stats,
        fullbatch_stats_=args.fullbatch_stats and test_data is not None,
        statsfun=make_ginot_statsfun(cfg, metadata),
        _collate_fn=metadata.get("train_collate_fn"),
        collate_fn_=metadata.get("eval_collate_fn"),
        verbose=True,
    )
    mark("04_trainer_constructed")

    trainer.make_dataloader()
    mark("05_dataloader_built")

    if args.stats_on_start and args.fullbatch_stats:
        if args.compile_model:
            print("WARNING: skipping stats_on_start with compile_model=True (GLT varlen + compile breaks eval path). "
                  "Use run_glt_gpu_memory_ablation.sh for full stats ablation via pdebench.")
        else:
            reset_peak()
            trainer.statistics()
            mark("06_after_stats_on_start")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mark("07_after_empty_cache_post_stats")

    if args.train_steps > 0:
        reset_peak()
        trainer.steps = args.train_steps
        trainer.train()
        mark("08_after_train_steps")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mark("09_after_empty_cache_post_train")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-spe-filter-type", default="band")
    parser.add_argument("--pe-spe-mode", default="query")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stats-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fullbatch-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default="", help="Optional path to write stage snapshots as JSON")
    args = parser.parse_args()

    rows = run_profile(args)
    print("\n=== Reserved memory deltas vs cuda_init ===")
    base = rows[0].get("reserved_gb", 0)
    for row in rows[1:]:
        delta = row.get("reserved_gb", 0) - base
        print(f"  {row['tag']}: {row.get('reserved_gb', 0):.3f} GiB (Δ{delta:+.3f})")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

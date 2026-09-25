#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LRA_OUT_DIR = REPO_ROOT / "out" / "lra"
DEFAULT_PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MODELS = [
    "transformer",
    "flare",
    "linear",
    "hedgehog",
    "linformer",
    "performer",
    "normattention",
    "cosformer",
    "funnel_hf",
    "reformer_hf",
]
EXTRA_MODELS = [
    "flare_kvmlp",
    "flare_kvmlp_focus",
]
ALL_MODELS = DEFAULT_MODELS + EXTRA_MODELS
EXTERNAL_MODELS = {"funnel_hf", "reformer_hf"}
TASK_CONFIGS = {
    "listops": {
        "steps": 10_000,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "num_blocks": 4,
        "channel_dim": 128,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "learning_rate": 5e-4,
        "pool": "mean",
        "pos_embed": "abs",
    },
    "image": {
        "steps": 20_000,
        "batch_size": 32,
        "weight_decay": 5e-2,
        "num_blocks": 3,
        "channel_dim": 64,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "learning_rate": 1e-3,
        "pool": "mean",
        "pos_embed": "abs",
    },
    "retrieval": {
        "steps": 10_000,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "num_blocks": 4,
        "channel_dim": 128,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "learning_rate": 5e-4,
        "pool": "mean",
        "pos_embed": "abs",
    },
    "text": {
        "steps": 20_000,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "num_blocks": 4,
        "channel_dim": 128,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "learning_rate": 1e-5,
        "pool": "cls",
        "pos_embed": "rope",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-wise random search for non-pathfinder LRA tasks across the main baseline families."
    )
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--search-name",
        default=time.strftime("autoresearch_%Y%m%d_%H%M%S"),
        help="Output subdirectory under out/lra/search/<task>/.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys. Use 'all' for the default curated set.",
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help="Comma-separated visible GPU indices to use, or 'all'.",
    )
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON_BIN),
        help="Python interpreter used to launch `python -m lra`.",
    )
    parser.add_argument("--stage1-trials", type=int, default=4)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-topk", type=int, default=1)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pick(rng: random.Random, values):
    return values[rng.randrange(len(values))]


def format_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def has_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def parse_models(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_MODELS)
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in models if item not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {ALL_MODELS}")
    return models


def parse_gpus(raw: str, *, dry_run: bool) -> list[str]:
    if not torch.cuda.is_available():
        if dry_run:
            return ["0"]
        raise RuntimeError("CUDA is unavailable. LRA training here assumes a GPU device.")
    n_gpus = torch.cuda.device_count()
    if raw == "all":
        return [str(i) for i in range(n_gpus)]
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [gpu for gpu in gpus if not gpu.isdigit() or int(gpu) < 0 or int(gpu) >= n_gpus]
    if invalid:
        raise ValueError(f"Invalid GPU selection {invalid}. Visible GPUs: 0..{n_gpus - 1}")
    return gpus


def default_stage1_steps(task: str) -> int:
    return {
        "listops": 1_500,
        "image": 4_000,
        "retrieval": 1_500,
        "text": 3_000,
    }[task]


def build_base_config(task: str, rng: random.Random, *, steps: int) -> dict:
    task_cfg = TASK_CONFIGS[task]
    cfg = {
        "task": task,
        "epochs": 0,
        "steps": steps,
        "batch_size": task_cfg["batch_size"],
        "optimizer": "adamw",
        "weight_decay": task_cfg["weight_decay"],
        "emb_drop": pick(rng, [0.0, 0.05]),
        "cls_drop": pick(rng, [0.0, 0.05]),
        "attn_drop": pick(rng, [0.0, 0.05]),
        "proj_drop": pick(rng, [0.0, 0.05]),
        "num_blocks": task_cfg["num_blocks"],
        "channel_dim": task_cfg["channel_dim"],
        "num_heads": task_cfg["num_heads"],
        "compile_model": False,
        "static_graph": False,
        "mixed_precision": True,
        "num_workers": 4,
        "prefetch_factor": 2,
        "seed": pick(rng, [0, 1, 2]),
    }
    if task in {"text", "retrieval", "listops", "image"}:
        cfg["pool"] = task_cfg["pool"]
        cfg["pos_embed"] = task_cfg["pos_embed"]
    return cfg


def sample_trial_config(
    task: str,
    model_key: str,
    *,
    seed: int,
    steps: int,
    num_workers: int,
    prefetch_factor: int,
) -> dict:
    rng = random.Random(seed)
    task_cfg = TASK_CONFIGS[task]
    cfg = build_base_config(task, rng, steps=steps)
    cfg["num_workers"] = num_workers
    cfg["prefetch_factor"] = prefetch_factor
    default_lr = task_cfg["learning_rate"]
    default_mlp = task_cfg["mlp_ratio"]

    if model_key == "transformer":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4],
            "image": [3e-4, 5e-4, 8e-4, 1e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5, 3e-5],
        }[task]
        cfg.update(model_type="transformer", learning_rate=pick(rng, lr_choices), mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]))
    elif model_key == "flare":
        lr_choices = {
            "listops": [5e-5, 1e-4, 2e-4, 3e-4],
            "image": [2e-4, 3e-4, 5e-4, 8e-4],
            "retrieval": [5e-5, 1e-4, 2e-4, 3e-4],
            "text": [3e-6, 5e-6, 1e-5, 2e-5],
        }[task]
        cfg.update(
            model_type="flare",
            learning_rate=pick(rng, lr_choices),
            attn_scale=pick(rng, ["one", "sqrt"]),
            num_latents=pick(rng, [64, 128, 256]),
            q_norm=True,
            k_norm=True,
            num_layers_kv_proj=-1,
            kv_proj_mlp_ratio=1.0,
            num_layers_ffn=0,
            ffn_mlp_ratio=default_mlp,
        )
    elif model_key == "flare_kvmlp":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [5e-4, 8e-4, 1e-3, 1.2e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5],
        }[task]
        latent_choices = {
            "listops": [64, 128],
            "image": [128, 256],
            "retrieval": [64, 128, 256],
            "text": [64, 128],
        }[task]
        kv_layer_choices = {
            "listops": [2, 3],
            "image": [2, 3],
            "retrieval": [2, 3],
            "text": [2, 3],
        }[task]
        ffn_layer_choices = {
            "listops": [1, 2, 3],
            "image": [1, 2],
            "retrieval": [1, 2, 3],
            "text": [1, 2, 3],
        }[task]
        kv_ratio_choices = [1.0, 2.0]
        ffn_ratio_choices = {
            "listops": [1.0, 2.0],
            "image": [1.0, 2.0],
            "retrieval": [1.0, 2.0],
            "text": [1.0, 2.0],
        }[task]
        head_choices = {
            "listops": [8],
            "image": [4, 8],
            "retrieval": [4, 8],
            "text": [8],
        }[task]
        cfg.update(
            model_type="flare",
            learning_rate=pick(rng, lr_choices),
            num_heads=pick(rng, head_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            attn_scale=pick(rng, ["one", "sqrt"]),
            num_latents=pick(rng, latent_choices),
            q_norm=True,
            k_norm=True,
            num_layers_kv_proj=pick(rng, kv_layer_choices),
            kv_proj_mlp_ratio=pick(rng, kv_ratio_choices),
            num_layers_ffn=pick(rng, ffn_layer_choices),
            ffn_mlp_ratio=pick(rng, ffn_ratio_choices),
            compile_model=True,
            static_graph=True,
            mixed_precision=pick(rng, [True, False]) if task != "text" else True,
            num_workers=max(num_workers, 8),
            prefetch_factor=max(prefetch_factor, 4),
        )
    elif model_key == "flare_kvmlp_focus":
        if task == "image":
            cfg.update(
                model_type="flare",
                learning_rate=pick(rng, [1.4e-3, 1.6e-3, 1.8e-3]),
                num_heads=8,
                mlp_ratio=2.0,
                attn_scale=pick(rng, ["one", "sqrt"]),
                num_latents=pick(rng, [192, 256]),
                q_norm=True,
                k_norm=True,
                num_layers_kv_proj=pick(rng, [2, 3]),
                kv_proj_mlp_ratio=pick(rng, [1.0, 2.0]),
                num_layers_ffn=pick(rng, [2, 3]),
                ffn_mlp_ratio=pick(rng, [2.0, 3.0]),
                compile_model=True,
                static_graph=True,
                mixed_precision=False,
                num_workers=max(num_workers, 8),
                prefetch_factor=max(prefetch_factor, 4),
            )
        elif task == "text":
            cfg.update(
                model_type="flare",
                learning_rate=pick(rng, [1.2e-5, 1.6e-5, 2.0e-5]),
                num_heads=8,
                mlp_ratio=4.0,
                attn_scale=pick(rng, ["sqrt", "one"]),
                num_latents=pick(rng, [128, 192, 256]),
                q_norm=True,
                k_norm=True,
                num_layers_kv_proj=pick(rng, [2, 3]),
                kv_proj_mlp_ratio=pick(rng, [1.0, 2.0]),
                num_layers_ffn=pick(rng, [2, 3]),
                ffn_mlp_ratio=pick(rng, [1.0, 2.0]),
                compile_model=True,
                static_graph=True,
                mixed_precision=False,
                num_workers=max(num_workers, 8),
                prefetch_factor=max(prefetch_factor, 4),
            )
        else:
            raise ValueError(f"flare_kvmlp_focus is only configured for image/text, got task={task}")
    elif model_key == "linear":
        qk_norm = pick(rng, [True, False])
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4],
            "image": [3e-4, 5e-4, 8e-4, 1e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5, 3e-5],
        }[task]
        cfg.update(
            model_type="linear",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            kernel=pick(rng, ["identity", "elu", "elu_norm", "silu", "gelu", "relu"]),
            q_norm=qk_norm,
            k_norm=qk_norm,
        )
    elif model_key == "hedgehog":
        lr_choices = {
            "listops": [5e-5, 1e-4, 2e-4],
            "image": [2e-4, 3e-4, 5e-4],
            "retrieval": [5e-5, 1e-4, 2e-4],
            "text": [3e-6, 5e-6, 1e-5],
        }[task]
        cfg.update(
            model_type="linear",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            kernel="hedgehog",
            q_norm=True,
            k_norm=True,
            mixed_precision=False,
        )
    elif model_key == "linformer":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [2e-4, 3e-4, 5e-4, 8e-4],
            "retrieval": [1e-5, 2.5e-5, 4.2e-5, 1e-4],
            "text": [5e-7, 1e-6, 2e-6, 5e-6],
        }[task]
        k_choices = {
            "listops": [64, 96, 128, 192],
            "image": [128, 192, 256],
            "retrieval": [64, 96, 128, 192],
            "text": [64, 96, 128],
        }[task]
        if task == "retrieval":
            cfg["weight_decay"] = pick(rng, [1e-6, 1e-5, 1e-4])
        if task == "text":
            cfg["weight_decay"] = pick(rng, [1e-4, 1e-3, 5e-3])
        cfg.update(
            model_type="linformer",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            linformer_k=pick(rng, k_choices),
            linformer_share_kv=pick(rng, [False, True]),
        )
    elif model_key == "performer":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [3e-4, 5e-4, 8e-4, 1e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5, 3e-5],
        }[task]
        cfg.update(
            model_type="performer",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            performer_nb_features=pick(rng, [64, 128, 192, 256]),
            performer_feature_map=pick(rng, ["favor_plus", "favor_pp"]),
            performer_redraw_interval=0,
            performer_normalize_inputs=True,
        )
    elif model_key == "normattention":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [3e-4, 5e-4, 8e-4, 1e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5, 3e-5],
        }[task]
        cfg.update(
            model_type="normattention",
            learning_rate=pick(rng, lr_choices),
            num_layers_kv_proj=pick(rng, [-1, 1, 2]),
            kv_proj_mlp_ratio=pick(rng, [1.0, 2.0]),
            num_layers_ffn=pick(rng, [0, 1]),
            ffn_mlp_ratio=pick(rng, [1.0, 2.0, default_mlp]),
            qk_dim_ratio=pick(rng, [0.5, 1.0, 2.0]),
        )
    elif model_key == "cosformer":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [3e-4, 5e-4, 8e-4, 1e-3],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [5e-6, 1e-5, 2e-5, 3e-5],
        }[task]
        cfg.update(
            model_type="cosformer",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            mixed_precision=False,
        )
    elif model_key == "funnel_hf":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [2e-4, 3e-4, 5e-4, 8e-4],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [3e-6, 5e-6, 1e-5],
        }[task]
        if task == "text":
            cfg["batch_size"] = pick(rng, [4, 8])
        cfg.update(model_type="funnel_hf", learning_rate=pick(rng, lr_choices), mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]))
    elif model_key == "reformer_hf":
        lr_choices = {
            "listops": [1e-4, 2e-4, 3e-4, 5e-4],
            "image": [2e-4, 3e-4, 5e-4, 8e-4],
            "retrieval": [1e-4, 2e-4, 3e-4, 5e-4],
            "text": [3e-6, 5e-6, 1e-5, 2e-5],
        }[task]
        cfg.update(
            model_type="reformer_hf",
            learning_rate=pick(rng, lr_choices),
            mlp_ratio=pick(rng, [default_mlp / 2, default_mlp]),
            reformer_num_hashes=pick(rng, [1, 2, 4]),
        )
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    return cfg


def make_trial(
    *,
    task: str,
    search_name: str,
    model_key: str,
    stage: int,
    trial_index: int,
    stage_seed: int,
    steps: int,
    num_workers: int,
    prefetch_factor: int,
    parent_trial_id: str | None = None,
    inherited_config: dict | None = None,
) -> dict:
    trial_id = f"{model_key}_s{stage}_t{trial_index:02d}"
    exp_name = f"search/{task}/{search_name}/{model_key}/s{stage}_t{trial_index:02d}"
    if inherited_config is None:
        config = sample_trial_config(task, model_key, seed=stage_seed, steps=steps, num_workers=num_workers, prefetch_factor=prefetch_factor)
    else:
        config = dict(inherited_config)
        config["steps"] = steps
        config["num_workers"] = num_workers
        config["prefetch_factor"] = prefetch_factor
    config["exp_name"] = exp_name
    return {
        "trial_id": trial_id,
        "model_key": model_key,
        "stage": stage,
        "parent_trial_id": parent_trial_id,
        "exp_name": exp_name,
        "config": config,
        "status": "pending",
        "returncode": None,
        "gpu": None,
        "mode": None,
        "log_file": None,
        "best_test_sequence_accuracy": None,
        "latest_test_sequence_accuracy": None,
        "best_checkpoint": None,
    }


def load_stats(exp_dir: Path) -> tuple[float | None, float | None, str | None]:
    best_acc = None
    latest_acc = None
    best_ckpt = None
    for stats_file in sorted(exp_dir.glob("ckpt*/stats.json")):
        with open(stats_file, "r") as f:
            data = json.load(f)
        test_stats = data.get("test_stats") or {}
        acc = test_stats.get("sequence_accuracy")
        if acc is None or (isinstance(acc, float) and math.isnan(acc)):
            continue
        latest_acc = acc
        ckpt_name = stats_file.parent.name
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt_name
    return best_acc, latest_acc, best_ckpt


def max_checkpoint_index(exp_dir: Path) -> int:
    max_idx = -1
    for ckpt_dir in exp_dir.glob("ckpt*"):
        suffix = ckpt_dir.name.replace("ckpt", "")
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx


def refresh_trial_from_disk(trial: dict) -> None:
    exp_dir = LRA_OUT_DIR / trial["exp_name"]
    if not exp_dir.exists():
        return
    best_acc, latest_acc, best_ckpt = load_stats(exp_dir)
    trial["best_test_sequence_accuracy"] = best_acc
    trial["latest_test_sequence_accuracy"] = latest_acc
    trial["best_checkpoint"] = best_ckpt
    if max_checkpoint_index(exp_dir) >= 10 and best_acc is not None:
        trial["status"] = "completed"


def build_stage1_trials(args: argparse.Namespace, models: list[str]) -> list[dict]:
    trials = []
    for model_index, model_key in enumerate(models):
        for trial_index in range(args.stage1_trials):
            stage_seed = args.seed + 1000 * model_index + trial_index
            trials.append(
                make_trial(
                    task=args.task,
                    search_name=args.search_name,
                    model_key=model_key,
                    stage=1,
                    trial_index=trial_index,
                    stage_seed=stage_seed,
                    steps=args.stage1_steps,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
            )
    return trials


def maybe_add_stage2_trials(plan: dict, args: argparse.Namespace) -> None:
    if args.stage2_topk <= 0:
        return
    existing = {(trial["stage"], trial["trial_id"]) for trial in plan["trials"]}
    by_model = defaultdict(list)
    for trial in plan["trials"]:
        if trial["stage"] != 1 or trial["status"] != "completed":
            continue
        score = trial["best_test_sequence_accuracy"]
        if score is None:
            continue
        by_model[trial["model_key"]].append(trial)
    for model_key, trials in by_model.items():
        ranked = sorted(
            trials,
            key=lambda trial: (trial["best_test_sequence_accuracy"], trial["latest_test_sequence_accuracy"] or float("-inf")),
            reverse=True,
        )
        for promote_index, parent in enumerate(ranked[: min(args.stage2_topk, len(ranked))]):
            promoted_id = f"{model_key}_s2_t{promote_index:02d}"
            if (2, promoted_id) in existing:
                continue
            promoted = make_trial(
                task=args.task,
                search_name=args.search_name,
                model_key=model_key,
                stage=2,
                trial_index=promote_index,
                stage_seed=args.seed,
                steps=args.stage2_steps,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                parent_trial_id=parent["trial_id"],
                inherited_config=parent["config"],
            )
            promoted["trial_id"] = promoted_id
            promoted["exp_name"] = f"search/{args.task}/{args.search_name}/{model_key}/s2_t{promote_index:02d}"
            promoted["config"]["exp_name"] = promoted["exp_name"]
            plan["trials"].append(promoted)
            existing.add((2, promoted_id))


def load_or_create_plan(args: argparse.Namespace, models: list[str], search_dir: Path) -> dict:
    plan_file = search_dir / "plan.json"
    if plan_file.exists():
        with open(plan_file, "r") as f:
            plan = json.load(f)
    else:
        plan = {
            "task": args.task,
            "search_name": args.search_name,
            "models": models,
            "stage1_trials": args.stage1_trials,
            "stage1_steps": args.stage1_steps,
            "stage2_topk": args.stage2_topk,
            "stage2_steps": args.stage2_steps,
            "trials": build_stage1_trials(args, models),
        }
    for trial in plan["trials"]:
        if trial["status"] == "running":
            trial["status"] = "pending"
        refresh_trial_from_disk(trial)
    maybe_add_stage2_trials(plan, args)
    return plan


def save_plan(plan: dict, search_dir: Path) -> None:
    with open(search_dir / "plan.json", "w") as f:
        json.dump(plan, f, indent=2, sort_keys=False)


def trial_command(trial: dict, python_bin: str) -> list[str]:
    exp_dir = LRA_OUT_DIR / trial["exp_name"]
    config_file = exp_dir / "config.yaml"
    should_restart = exp_dir.exists() and config_file.exists() and trial["status"] != "completed"
    cmd = [python_bin, "-m", "lra"]
    if should_restart:
        cmd.extend(["--restart", "true", "--exp_name", trial["exp_name"]])
        trial["mode"] = "restart"
    else:
        cmd.extend(["--train", "true"])
        trial["mode"] = "train"
    if trial["mode"] == "train":
        for key, value in trial["config"].items():
            if value is None:
                continue
            cmd.extend([f"--{key}", format_cli_value(value)])
    return cmd


def launch_trial(trial: dict, gpu: str, python_bin: str, logs_dir: Path) -> tuple[subprocess.Popen, object]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{trial['trial_id']}.log"
    log_handle = open(log_file, "a", buffering=1)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    cmd = trial_command(trial, python_bin)
    print(f"[launch][task={trial['config']['task']}][gpu={gpu}] {trial['trial_id']} ({trial['mode']})")
    print("  " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    trial["status"] = "running"
    trial["gpu"] = gpu
    trial["log_file"] = str(log_file.relative_to(REPO_ROOT))
    return proc, log_handle


def finalize_trial(trial: dict, returncode: int) -> None:
    trial["returncode"] = returncode
    trial["gpu"] = None
    refresh_trial_from_disk(trial)
    if returncode == 0 and trial["best_test_sequence_accuracy"] is not None:
        trial["status"] = "completed"
        print(
            f"[done] {trial['trial_id']} best={100.0 * trial['best_test_sequence_accuracy']:.2f}% "
            f"latest={100.0 * (trial['latest_test_sequence_accuracy'] or 0.0):.2f}%"
        )
    else:
        trial["status"] = "failed"
        print(f"[fail] {trial['trial_id']} returncode={returncode} log={trial['log_file']}")


def run_stage(plan: dict, stage: int, gpus: list[str], python_bin: str, poll_seconds: float, search_dir: Path, dry_run: bool) -> None:
    logs_dir = search_dir / "logs"
    trials = [trial for trial in plan["trials"] if trial["stage"] == stage]
    for trial in trials:
        refresh_trial_from_disk(trial)
    save_plan(plan, search_dir)
    pending = [trial for trial in trials if trial["status"] == "pending"]
    if not pending:
        print(f"[stage {stage}] nothing pending")
        return
    print(f"[stage {stage}] pending={len(pending)} gpus={gpus}")
    if dry_run:
        return
    available_gpus = list(gpus)
    active = {}
    try:
        while pending or active:
            while pending and available_gpus:
                gpu = available_gpus.pop(0)
                trial = pending.pop(0)
                proc, log_handle = launch_trial(trial, gpu, python_bin, logs_dir)
                active[gpu] = {"trial": trial, "proc": proc, "log_handle": log_handle}
                save_plan(plan, search_dir)
            if not active:
                continue
            time.sleep(poll_seconds)
            for gpu, payload in list(active.items()):
                returncode = payload["proc"].poll()
                if returncode is None:
                    continue
                payload["log_handle"].close()
                finalize_trial(payload["trial"], returncode)
                available_gpus.append(gpu)
                del active[gpu]
                save_plan(plan, search_dir)
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active trials and saving plan.")
        for payload in active.values():
            payload["proc"].terminate()
            payload["log_handle"].close()
            payload["trial"]["status"] = "pending"
            payload["trial"]["gpu"] = None
        save_plan(plan, search_dir)
        raise


def write_summary(plan: dict, search_dir: Path) -> None:
    best_by_model = {}
    for trial in plan["trials"]:
        score = trial.get("best_test_sequence_accuracy")
        if score is None:
            continue
        current = best_by_model.get(trial["model_key"])
        if current is None or score > current["best_test_sequence_accuracy"]:
            best_by_model[trial["model_key"]] = trial
    leaderboard = []
    for trial in sorted(best_by_model.values(), key=lambda trial: (trial["best_test_sequence_accuracy"], trial["latest_test_sequence_accuracy"] or float("-inf")), reverse=True):
        leaderboard.append(
            {
                "model_key": trial["model_key"],
                "stage": trial["stage"],
                "trial_id": trial["trial_id"],
                "exp_name": trial["exp_name"],
                "best_test_sequence_accuracy": trial["best_test_sequence_accuracy"],
                "latest_test_sequence_accuracy": trial["latest_test_sequence_accuracy"],
                "best_checkpoint": trial["best_checkpoint"],
                "log_file": trial["log_file"],
                "config": trial["config"],
            }
        )
    summary = {
        "task": plan["task"],
        "search_name": plan["search_name"],
        "leaderboard": leaderboard,
        "completed_trials": sum(1 for trial in plan["trials"] if trial["status"] == "completed"),
        "failed_trials": sum(1 for trial in plan["trials"] if trial["status"] == "failed"),
        "pending_trials": sum(1 for trial in plan["trials"] if trial["status"] == "pending"),
    }
    with open(search_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print("\nLeaderboard")
    for item in leaderboard:
        print(f"  {item['model_key']:<14} stage={item['stage']} best={100.0 * item['best_test_sequence_accuracy']:.2f}% exp={item['exp_name']}")


def main() -> int:
    args = parse_args()
    if args.stage1_steps is None:
        args.stage1_steps = default_stage1_steps(args.task)
    if args.stage2_steps is None:
        args.stage2_steps = TASK_CONFIGS[args.task]["steps"]
    python_bin = Path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"Missing python interpreter: {python_bin}")
    models = parse_models(args.models)
    if not has_transformers():
        unavailable = sorted(set(models) & EXTERNAL_MODELS)
        if unavailable:
            print(f"Skipping external models without transformers: {unavailable}")
            models = [model for model in models if model not in EXTERNAL_MODELS]
    if not models:
        raise RuntimeError("No runnable models remain after dependency checks.")
    gpus = parse_gpus(args.gpus, dry_run=args.dry_run)
    search_dir = LRA_OUT_DIR / "search" / args.task / args.search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    plan = load_or_create_plan(args, models, search_dir)
    save_plan(plan, search_dir)
    run_stage(plan, 1, gpus, str(python_bin), args.poll_seconds, search_dir, args.dry_run)
    maybe_add_stage2_trials(plan, args)
    save_plan(plan, search_dir)
    run_stage(plan, 2, gpus, str(python_bin), args.poll_seconds, search_dir, args.dry_run)
    write_summary(plan, search_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

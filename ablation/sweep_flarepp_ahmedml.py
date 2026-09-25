"""Collect/plot AhmedML-surface FLARE vs FLARE++ B×M normalized MSE from ``out/pdebench/ahmedml_surface/``.

Grid: model∈{flare,flarepp}, B∈{2,4,8,12}, M∈{32,64,128}, C=128, H=8, IO=2,
FFN=0, OPN=true, MR=2, RMSNorm, WD=1e-4, CP=2, fp16. Case stems match
``run_flarepp.sh`` EXP_NAME used by the AhmedML CP=2 B×M sweep.

Metrics read ``stats.json`` ``train_loss``/``test_loss`` (normalized MSE).
Incomplete runs (no ``ckpt10``) contribute NaNs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
if _ABLATION_DIR not in sys.path:
    sys.path.insert(0, _ABLATION_DIR)

from sweep_flarepp import (  # noqa: E402, I001
    FIGDIR,
    _apply_shared_log_ylim,
    _best_normalized_mse,
    _resolve_case_dir,
)

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASEDIR = os.path.join(PROJDIR, "out", "pdebench", "ahmedml_surface")

CHANNEL_DIM = 128
NUM_HEADS = 8
NUM_LAYERS_IN_OUT_PROJ = 2
OUT_PROJ_NORM = True
NUM_LAYERS_FFN = 0
MLP_RATIO_FFN = 2.0
WEIGHT_DECAY = 1e-4
RMSNORM = True
CONTEXT_PARALLEL_SIZE = 2
PRECISION = "fp16"

NUM_BLOCKS = (2, 4, 8, 12)
NUM_LATENTS = (32, 64, 128)
MODELS = ("flare", "flarepp")

OUTER_LABEL = (
    f"$C$={CHANNEL_DIM}, $H$={NUM_HEADS}, "
    f"IO={NUM_LAYERS_IN_OUT_PROJ}, FFN={NUM_LAYERS_FFN}, "
    f"OPN={'true' if OUT_PROJ_NORM else 'false'}, MR={MLP_RATIO_FFN:g}, "
    f"RMSNorm, WD=$1\\times10^{{-4}}$, CP={CONTEXT_PARALLEL_SIZE}"
)


def _case_stem(model: str, num_blocks: int, num_latents: int) -> str:
    if model not in MODELS:
        raise ValueError(f"unknown model: {model!r}")
    opn = 1 if OUT_PROJ_NORM else 0
    return (
        f"ahmedml_surface_{model}_cp{CONTEXT_PARALLEL_SIZE}_C{CHANNEL_DIM}"
        f"_B{int(num_blocks)}_H{NUM_HEADS}_M{int(num_latents)}"
        f"_IO_{NUM_LAYERS_IN_OUT_PROJ}_FFN_{NUM_LAYERS_FFN}"
        f"_OPN_{opn}_MR_2p0_amp_fp16_rmsnorm_wd1e-4"
    )


def _series_name(model: str, num_latents: int) -> str:
    return f"{model} M={int(num_latents)}"


def collect_data(*, casedir: Optional[str] = None) -> pd.DataFrame:
    case_root = os.path.abspath(casedir) if casedir else CASEDIR
    rows = []
    for model in MODELS:
        for num_latents in NUM_LATENTS:
            for num_blocks in NUM_BLOCKS:
                stem = _case_stem(model, num_blocks, num_latents)
                case_path = _resolve_case_dir(stem, case_root=case_root)
                row = {
                    "outer_label": OUTER_LABEL,
                    "channel_dim": CHANNEL_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers_in_out_proj": NUM_LAYERS_IN_OUT_PROJ,
                    "out_proj_norm": OUT_PROJ_NORM,
                    "num_layers_ffn": NUM_LAYERS_FFN,
                    "mlp_ratio_ffn": MLP_RATIO_FFN,
                    "weight_decay": WEIGHT_DECAY,
                    "precision": PRECISION,
                    "rmsnorm": RMSNORM,
                    "context_parallel_size": CONTEXT_PARALLEL_SIZE,
                    "mixer": model,
                    "num_latents": num_latents,
                    "num_blocks": num_blocks,
                    "series": _series_name(model, num_latents),
                    "case_path": case_path,
                    "train_mse": float("nan"),
                    "test_mse": float("nan"),
                    "complete": False,
                }
                if case_path is not None:
                    row["train_mse"] = _best_normalized_mse(case_path, "train")
                    row["test_mse"] = _best_normalized_mse(case_path, "test")
                    row["complete"] = os.path.isdir(os.path.join(case_path, "ckpt10"))
                rows.append(row)

    df = pd.DataFrame(rows)
    n_complete = int(df["complete"].sum()) if len(df) else 0
    print(f"Collected {len(df)} grid cells ({n_complete} with ckpt10) from {case_root}")
    return df


def _style_kwargs(model: str, num_latents: int) -> dict:
    color = {"flare": "red", "flarepp": "green"}[model]
    linestyle = {32: "-", 64: "--", 128: ":"}[int(num_latents)]
    marker = {"flare": "s", "flarepp": "^"}[model]
    return {
        "marker": marker,
        "linestyle": linestyle,
        "color": color,
        "linewidth": 2.5,
        "markersize": 9,
    }


def plot_results(df: pd.DataFrame) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    out_name = "sweep_flarepp_ahmedml_surface.pdf"
    out_path = os.path.join(FIGDIR, out_name)
    csv_path = os.path.join(FIGDIR, out_name.replace(".pdf", ".csv"))

    if shutil.which("latex") is not None:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
        })

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    fontsize = 16

    for ax in (ax_train, ax_test):
        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xticks(list(NUM_BLOCKS))
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)

    ax_train.set_ylabel(r"Best normalized MSE (four outputs)", fontsize=fontsize)
    ax_train.set_title("Train", fontsize=fontsize)
    ax_test.set_title("Test", fontsize=fontsize)
    fig.suptitle(OUTER_LABEL, fontsize=fontsize - 4)

    handles = []
    labels = []
    for model in MODELS:
        for num_latents in NUM_LATENTS:
            series = _series_name(model, num_latents)
            sub = df[df["series"] == series].sort_values("num_blocks")
            if len(sub) == 0:
                continue
            kwargs = _style_kwargs(model, num_latents)
            train_ok = sub[np.isfinite(sub["train_mse"].to_numpy(dtype=float))]
            test_ok = sub[np.isfinite(sub["test_mse"].to_numpy(dtype=float))]
            if len(train_ok):
                (h_train,) = ax_train.plot(
                    train_ok["num_blocks"], train_ok["train_mse"], label=series, **kwargs,
                )
                handles.append(h_train)
                labels.append(series)
            if len(test_ok):
                ax_test.plot(test_ok["num_blocks"], test_ok["test_mse"], label=None, **kwargs)

    vals = np.concatenate([
        df["train_mse"].to_numpy(dtype=float),
        df["test_mse"].to_numpy(dtype=float),
    ])
    vals = vals[np.isfinite(vals)]
    if len(vals):
        ymin, ymax = float(vals.min()) * 0.8, float(vals.max()) * 1.4
        _apply_shared_log_ylim((ax_train, ax_test), ymin, ymax)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(handles)),
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=fontsize - 3,
            bbox_to_anchor=(0.5, 0.00),
            columnspacing=1.5,
            handletextpad=0.6,
            bbox_transform=fig.transFigure,
            handlelength=3.0,
            markerscale=1.0,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.82)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {out_path}")

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AhmedML-surface FLARE vs FLARE++ B×M normalized MSE plots (fp16, RMSNorm, WD=1e-4, CP=2)",
    )

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument("--eval", type=str_to_bool, default=False)
    parser.add_argument(
        "--casedir",
        type=str,
        default=None,
        help="Case root (default: out/pdebench/ahmedml_surface)",
    )
    args = parser.parse_args()

    if args.eval:
        dataframe = collect_data(casedir=args.casedir)
        cols = [
            "mixer", "num_latents", "num_blocks", "complete",
            "train_mse", "test_mse",
        ]
        print(dataframe[cols].to_string(index=False))
        plot_results(dataframe)
    else:
        print("No action specified. Please specify --eval true.")

    raise SystemExit(0)

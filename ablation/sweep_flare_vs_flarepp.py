"""FLARE vs FLARE++ B×M Rel-L2 plots (fp32) from ``out/pdebench/flarepp_standard/``.

For each dataset, collects ``flare`` / ``simplifiedflarepp`` / ``flarepp`` over
B∈{2,4,8} and the dataset's latent grid (elasticity: M∈{32,64,128}; darcy: +256),
merges those rows into ``figs/sweep_flarepp_{dataset}.csv``, and writes a 2×3
figure:

  left=simplifiedflarepp, mid=flare, right=flarepp; top=train Rel-L2, bottom=test Rel-L2

Shared x (blocks) and y (Rel-L2) across panels. Case stems match
``run_flarepp_standard.sh`` defaults (C=128 H=8 IO=2 FFN=0 OPN=true MR=2).
Anchored cases prefer local ``CASEDIR``, then ``--casedir``.
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
    CASEDIR,
    CHANNEL_DIM,
    FIGDIR,
    MLP_RATIO_FFN,
    NUM_BLOCKS,
    NUM_HEADS,
    NUM_LAYERS_FFN,
    NUM_LAYERS_IN_OUT_PROJ,
    OUTER_LABEL,
    OUT_PROJ_NORM,
    _apply_shared_log_ylim,
    _best_rel_error_across_ckpts,
    _case_stem,
    _legend_label,
    _resolve_case_dir,
    _series_name,
)

#======================================================================#
DATASETS = ("elasticity", "darcy", "airfoil_steady", "pipe")
DEFAULT_DATASET = "elasticity"
PRECISION = "fp32"
MIXERS = ("simplifiedflarepp", "flare", "flarepp")  # left → right panel order
LATENTS_BY_DATASET: dict[str, tuple[int, ...]] = {
    "elasticity": (32, 64, 128),
    "darcy": (32, 64, 128, 256),
    "airfoil_steady": (32, 64, 128),
    "pipe": (32, 64, 128),
}

# Merge key columns shared with sweep_flarepp.py CSV schema.
_KEY_COLS = ("precision", "mixer", "num_latents", "num_blocks")


def _roots_for_mixer(mixer: str, case_root: str) -> list[str]:
    """Case roots to search; anchored prefers local CASEDIR then ``case_root``."""
    primary = os.path.abspath(case_root)
    if mixer != "flarepp":
        return [primary]
    roots: list[str] = []
    for root in (os.path.abspath(CASEDIR), primary):
        if root not in roots:
            roots.append(root)
    return roots


def _resolve_mixer_case_dir(stem: str, mixer: str, case_root: str) -> Optional[str]:
    for root in _roots_for_mixer(mixer, case_root):
        path = _resolve_case_dir(stem, case_root=root)
        if path is not None:
            return path
    return None


#======================================================================#
def _latent_style(num_latents: int) -> dict:
    colors = {
        32: "#1f77b4",
        64: "#ff7f0e",
        128: "#2ca02c",
        256: "#d62728",
    }
    markers = {
        32: "o",
        64: "s",
        128: "^",
        256: "D",
    }
    linestyles = {
        32: "-",
        64: "--",
        128: "-.",
        256: ":",
    }
    m = int(num_latents)
    return {
        "color": colors.get(m, "black"),
        "marker": markers.get(m, "o"),
        "linestyle": linestyles.get(m, "-"),
        "linewidth": 2.5,
        "markersize": 9,
        "zorder": 3,
    }


def collect_flare_vs_flarepp(
    dataset: str,
    *,
    casedir: Optional[str] = None,
    latents: Optional[tuple[int, ...]] = None,
) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}; expected one of {DATASETS}")
    case_root = os.path.abspath(casedir) if casedir else CASEDIR
    ms = tuple(latents) if latents is not None else LATENTS_BY_DATASET[dataset]
    rows = []
    for mixer in MIXERS:
        for num_latents in ms:
            for num_blocks in NUM_BLOCKS:
                stem = _case_stem(dataset, mixer, num_latents, num_blocks, PRECISION)
                case_path = _resolve_mixer_case_dir(stem, mixer, case_root)
                row = {
                    "outer_label": OUTER_LABEL,
                    "channel_dim": CHANNEL_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers_in_out_proj": NUM_LAYERS_IN_OUT_PROJ,
                    "out_proj_norm": OUT_PROJ_NORM,
                    "num_layers_ffn": NUM_LAYERS_FFN,
                    "mlp_ratio_ffn": MLP_RATIO_FFN,
                    "precision": PRECISION,
                    "mixer": mixer,
                    "num_latents": float(num_latents),
                    "num_blocks": num_blocks,
                    "series": _series_name(mixer, num_latents),
                    "case_path": case_path,
                    "train_rel_error": float("nan"),
                    "test_rel_error": float("nan"),
                    "complete": False,
                }
                if case_path is not None:
                    row["train_rel_error"] = _best_rel_error_across_ckpts(case_path, "train")
                    row["test_rel_error"] = _best_rel_error_across_ckpts(case_path, "test")
                    row["complete"] = os.path.isdir(os.path.join(case_path, "ckpt10"))
                rows.append(row)

    df = pd.DataFrame(rows)
    n_complete = int(df["complete"].sum()) if len(df) else 0
    print(
        f"Collected {len(df)} FLARE/FLARE++/anchored cells "
        f"({n_complete} with ckpt10); primary casedir={case_root}"
    )
    return df


def merge_into_sweep_csv(new_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Upsert FLARE/FLARE++/anchored fp32 rows into ``figs/sweep_flarepp_{dataset}.csv``."""
    csv_path = os.path.join(FIGDIR, f"sweep_flarepp_{dataset}.csv")
    key_cols = list(_KEY_COLS)
    if os.path.isfile(csv_path):
        old = pd.read_csv(csv_path)
    else:
        old = pd.DataFrame(columns=new_df.columns)

    def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["precision"] = out["precision"].astype(str)
        out["mixer"] = out["mixer"].astype(str)
        # Preserve NaN latents (mha) as a distinct key via nullable float.
        out["num_latents"] = pd.to_numeric(out["num_latents"], errors="coerce")
        out["num_blocks"] = pd.to_numeric(out["num_blocks"], errors="coerce").astype("Int64")
        return out

    old_n = _norm_keys(old)
    new_n = _norm_keys(new_df)

    if len(old_n):
        old_idx = pd.MultiIndex.from_frame(old_n[key_cols])
        new_idx = pd.MultiIndex.from_frame(new_n[key_cols])
        retained = old_n.loc[~old_idx.isin(new_idx)].copy()
    else:
        retained = old_n

    merged = pd.concat([retained, new_n], ignore_index=True, sort=False)
    sort_cols = ["precision", "mixer", "num_latents", "num_blocks"]
    merged = merged.sort_values(sort_cols, kind="mergesort", na_position="first").reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    print(
        f"Wrote {csv_path} ({len(merged)} rows; "
        f"upserted {len(new_n)} FLARE/FLARE++/anchored cells)"
    )
    return merged


def plot_flare_vs_flarepp(df: pd.DataFrame, dataset: str) -> str:
    if len(df) == 0:
        print("ERROR: empty dataframe; nothing to plot.")
        return ""

    out_path = os.path.join(FIGDIR, f"sweep_flare_vs_flarepp_{dataset}.pdf")
    latents = LATENTS_BY_DATASET.get(dataset, (32, 64, 128))

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

    n_cols = len(MIXERS)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.0 * n_cols, 9.0), sharex=True, sharey=True)
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    fontsize = 16
    # axes[row, col]: row 0 train / 1 test; cols follow MIXERS
    col_mixers = list(MIXERS)
    legend_handles, legend_labels = [], []

    for col, mixer in enumerate(col_mixers):
        ax_train, ax_test = axes[0, col], axes[1, col]
        for ax in (ax_train, ax_test):
            ax.set_yscale("log")
            ax.set_xscale("linear")
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.set_xticks(list(NUM_BLOCKS))
            ax.tick_params(axis="both", which="major", labelsize=fontsize)

        ax_train.set_title(rf"{_legend_label(mixer, None)} — train", fontsize=fontsize - 2)
        ax_test.set_title("test", fontsize=fontsize - 2)
        ax_test.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)
        if col == 0:
            ax_train.set_ylabel(r"Best relative error", fontsize=fontsize)
            ax_test.set_ylabel(r"Best relative error", fontsize=fontsize)

        sub_m = df[df["mixer"] == mixer]
        for num_latents in latents:
            sub = sub_m[sub_m["num_latents"] == float(num_latents)].sort_values("num_blocks")
            if len(sub) == 0:
                continue
            kwargs = _latent_style(num_latents)
            train_ok = sub[np.isfinite(sub["train_rel_error"].to_numpy(dtype=float))]
            test_ok = sub[np.isfinite(sub["test_rel_error"].to_numpy(dtype=float))]
            label = rf"$M$={num_latents}"
            if len(train_ok):
                (h,) = ax_train.plot(
                    train_ok["num_blocks"], train_ok["train_rel_error"], label=label, **kwargs,
                )
                if col == 0 and label not in legend_labels:
                    legend_handles.append(h)
                    legend_labels.append(label)
            if len(test_ok):
                ax_test.plot(test_ok["num_blocks"], test_ok["test_rel_error"], label=None, **kwargs)

    vals = np.concatenate([
        df["train_rel_error"].to_numpy(dtype=float),
        df["test_rel_error"].to_numpy(dtype=float),
    ])
    vals = vals[np.isfinite(vals)]
    if len(vals):
        ymin, ymax = float(vals.min()) * 0.8, float(vals.max()) * 1.4
        _apply_shared_log_ylim(axes, ymin, ymax)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=fontsize - 2,
            bbox_to_anchor=(0.5, 0.00),
            columnspacing=1.5,
            handletextpad=0.6,
            bbox_transform=fig.transFigure,
            handlelength=3.0,
            markerscale=1.0,
        )

    fig.suptitle(OUTER_LABEL, fontsize=fontsize - 2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.88)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "FLARE vs FLARE++ vs anchored fp32 B×M Rel-L2 plots; "
            "upserts sweep_flarepp_{dataset}.csv"
        ),
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
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--casedir", type=str, default=None)
    parser.add_argument(
        "--latents",
        type=str,
        default=None,
        help="Comma-separated M list override (e.g. 32,64,128)",
    )
    args = parser.parse_args()

    if not args.eval:
        print("No action specified. Please specify --eval true.")
        raise SystemExit(0)

    latents = None
    if args.latents:
        latents = tuple(int(x) for x in args.latents.split(",") if x.strip())

    dataframe = collect_flare_vs_flarepp(args.dataset, casedir=args.casedir, latents=latents)
    cols = ["mixer", "num_latents", "num_blocks", "complete", "train_rel_error", "test_rel_error"]
    print(dataframe[cols].to_string(index=False))
    merge_into_sweep_csv(dataframe, args.dataset)
    plot_flare_vs_flarepp(dataframe, args.dataset)
    raise SystemExit(0)

"""Collect/plot B×M mixer sweeps from ``out/pdebench/flarepp_standard/``.

Sweeps ``mha`` / ``flare`` / ``luna`` / ``simplifiedflarepp`` / ``flarepp`` /
``transolver`` / ``transolver3`` on
elasticity, darcy, airfoil_steady, pipe for B∈{2,4,8} and M∈{64,128} (MHA: B
only), in fp32 and fp16. Case stems match today's ``run_flarepp_standard.sh``
defaults (C=128 H=8 IO=2 FFN=0 OPN=true MR=2 + mixer extras).

Metrics prefer each ckpt's ``rel_error.json`` over ``stats.json`` train/test
loss. Incomplete runs (no ``ckpt10``) contribute NaNs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASEDIR = os.path.join(PROJDIR, "out", "pdebench", "flarepp_standard")
FIGDIR = os.path.join(PROJDIR, "figs")
os.makedirs(FIGDIR, exist_ok=True)

DEFAULT_DATASET = "elasticity"
DATASETS = ("elasticity", "darcy", "airfoil_steady", "pipe")
NUM_BLOCKS = (2, 4, 8)
NUM_LATENTS = (64, 128)
PRECISIONS = ("fp32", "fp16")  # CSV / plot labels; stem uses amp_fp16 for fp16

# Locked outer config matching run_flarepp_standard.sh suite defaults.
CHANNEL_DIM = 128
NUM_HEADS = 8
NUM_LAYERS_IN_OUT_PROJ = 2
OUT_PROJ_NORM = True
NUM_LAYERS_FFN = 0
MLP_RATIO_FFN = 2.0

# (mixer, num_latents). MHA has no latents.
MIXER_SPECS: tuple[tuple[str, Optional[int]], ...] = (
    ("mha", None),
    ("flare", 64),
    ("flare", 128),
    ("luna", 64),
    ("luna", 128),
    ("simplifiedflarepp", 64),
    ("simplifiedflarepp", 128),
    ("flarepp", 64),
    ("flarepp", 128),
    ("transolver", 64),
    ("transolver", 128),
    ("transolver3", 64),
    ("transolver3", 128),
)

OUTER_LABEL = (
    f"$C$={CHANNEL_DIM}, $H$={NUM_HEADS}, "
    f"IO={NUM_LAYERS_IN_OUT_PROJ}, FFN={NUM_LAYERS_FFN}, "
    f"OPN={'true' if OUT_PROJ_NORM else 'false'}, MR={MLP_RATIO_FFN:g}"
)


#======================================================================#
def _mr_tag(mlp_ratio_ffn: float = MLP_RATIO_FFN) -> str:
    return f"{float(mlp_ratio_ffn)}".replace(".", "p")


def _prec_stem_tag(precision: str) -> str:
    """Map CSV precision label to run_flarepp_standard stem tag."""
    name = str(precision).lower()
    if name == "fp32":
        return "fp32"
    if name in ("fp16", "amp_fp16"):
        return "amp_fp16"
    raise ValueError(f"unknown precision: {precision!r}")


_LEGEND_NAME = {
    "mha": "MHA",
    "flare": "FLARE",
    "luna": "LUNA",
    "simplifiedflarepp": "Simplified FLARE++",
    "flarepp": "FLARE++",
    "transolver": "Transolver",
    "transolver3": "Transolver 3",
}


def _series_name(mixer: str, num_latents: Optional[int]) -> str:
    if mixer == "mha":
        return "mha"
    return f"{mixer} M={num_latents}"


def _legend_label(mixer: str, num_latents: Optional[int]) -> str:
    name = _LEGEND_NAME.get(mixer, mixer)
    if mixer == "mha" or num_latents is None:
        return name
    return rf"{name} $M$={int(num_latents)}"


def _mixer_extras(mixer: str) -> str:
    """Extras suffix matching today's run_flarepp_standard.sh defaults."""
    if mixer == "mha":
        return "_qknorm_false"
    if mixer == "flare":
        return "_qknorm_false"
    if mixer == "luna":
        return "_qknorm_false"
    if mixer == "simplifiedflarepp":
        return "_qknorm_false_qk0_true_v0_false_sharek0v0_false_gate_false"
    if mixer == "flarepp":
        return "_k_true_sharek0v0_true_gate_0.25_convex_false"
    if mixer in ("transolver", "transolver3"):
        return ""
    raise ValueError(f"unknown mixer: {mixer!r}")


def _case_stem(
    dataset: str,
    mixer: str,
    num_latents: Optional[int],
    num_blocks: int,
    precision: str,
) -> str:
    if mixer == "mha":
        mixer_tag = "mha"
    else:
        if num_latents is None:
            raise ValueError(f"{mixer} requires num_latents")
        mixer_tag = f"{mixer}_{int(num_latents)}"
    opn = 1 if OUT_PROJ_NORM else 0
    prec = _prec_stem_tag(precision)
    extras = _mixer_extras(mixer)
    return (
        f"{dataset}_{mixer_tag}_C{CHANNEL_DIM}_B{num_blocks}_H{NUM_HEADS}"
        f"_IO_{NUM_LAYERS_IN_OUT_PROJ}_FFN_{NUM_LAYERS_FFN}"
        f"_OPN_{opn}_MR_{_mr_tag()}_{prec}{extras}"
    )


def _resolve_case_dir(stem: str, *, case_root: Optional[str] = None) -> Optional[str]:
    """Prefer the directory that contains ``ckpt10`` (often ``stem_01``)."""
    root = case_root if case_root is not None else CASEDIR
    candidates = [
        os.path.join(root, f"{stem}_01"),
        os.path.join(root, stem),
    ]
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if re.fullmatch(re.escape(stem) + r"_\d+", name):
                candidates.append(os.path.join(root, name))
    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        if os.path.isdir(os.path.join(path, "ckpt10")):
            return path
    for path in candidates:
        if os.path.isdir(path) and any(name.startswith("ckpt") for name in os.listdir(path)):
            return path
    return None


def _best_rel_error_across_ckpts(case_path: str, split: str) -> float:
    """Min over ckpts of Rel-L2 for ``train`` or ``test``.

    Incomplete runs (no ``ckpt10``) return NaN. Prefers
    ``{split}_rel_error`` from ``rel_error.json``. If no ckpt has that file,
    falls back to ``stats.json`` ``{split}_loss`` / ``{split}_loss_ema``.
    """
    if not os.path.isdir(os.path.join(case_path, "ckpt10")):
        return float("nan")

    rel_key = f"{split}_rel_error"
    best = float("inf")
    found = False
    for name in sorted(os.listdir(case_path)):
        if not name.startswith("ckpt"):
            continue
        rel_path = os.path.join(case_path, name, "rel_error.json")
        if not os.path.isfile(rel_path):
            continue
        with open(rel_path) as f:
            rel = json.load(f)
        if rel.get(rel_key) is not None:
            found = True
            best = min(best, float(rel[rel_key]))
    if found:
        return best

    loss_key = f"{split}_loss"
    loss_ema_key = f"{split}_loss_ema"
    best = float("inf")
    found = False
    for name in sorted(os.listdir(case_path)):
        if not name.startswith("ckpt"):
            continue
        stats_path = os.path.join(case_path, name, "stats.json")
        if not os.path.isfile(stats_path):
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        vals = []
        if stats.get(loss_key) is not None:
            vals.append(float(stats[loss_key]))
        if stats.get(loss_ema_key) is not None:
            vals.append(float(stats[loss_ema_key]))
        if vals:
            found = True
            best = min(best, min(vals))
    return best if found else float("nan")


def _best_normalized_mse(case_path: str, split: str) -> float:
    """Min over ckpts of normalized MSE for ``train`` or ``test``.

    Incomplete runs (no ``ckpt10``) return NaN. Reads ``stats.json``
    ``{split}_loss``, then ``{split}_stats.mse``.
    """
    if not os.path.isdir(os.path.join(case_path, "ckpt10")):
        return float("nan")
    values = []
    for name in os.listdir(case_path):
        if not name.startswith("ckpt"):
            continue
        stats_path = os.path.join(case_path, name, "stats.json")
        if not os.path.isfile(stats_path):
            continue
        with open(stats_path) as handle:
            metrics = json.load(handle)
        value = metrics.get(f"{split}_loss")
        if value is None:
            value = metrics.get(f"{split}_stats", {}).get("mse")
        if value is not None:
            values.append(float(value))
    return min(values) if values else float("nan")


def collect_data(
    dataset: str = DEFAULT_DATASET,
    *,
    casedir: Optional[str] = None,
) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}; expected one of {DATASETS}")
    case_root = os.path.abspath(casedir) if casedir else CASEDIR
    rows = []
    for precision in PRECISIONS:
        for mixer, num_latents in MIXER_SPECS:
            for num_blocks in NUM_BLOCKS:
                stem = _case_stem(dataset, mixer, num_latents, num_blocks, precision)
                case_path = _resolve_case_dir(stem, case_root=case_root)
                series = _series_name(mixer, num_latents)
                row = {
                    "outer_label": OUTER_LABEL,
                    "channel_dim": CHANNEL_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers_in_out_proj": NUM_LAYERS_IN_OUT_PROJ,
                    "out_proj_norm": OUT_PROJ_NORM,
                    "num_layers_ffn": NUM_LAYERS_FFN,
                    "mlp_ratio_ffn": MLP_RATIO_FFN,
                    "precision": "fp16" if precision == "fp16" else "fp32",
                    "mixer": mixer,
                    "num_latents": num_latents if num_latents is not None else np.nan,
                    "num_blocks": num_blocks,
                    "series": series,
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
    print(f"Collected {len(df)} grid cells ({n_complete} with ckpt10) from {case_root}")
    return df


#======================================================================#
def _style_kwargs(mixer: str, num_latents: Optional[int]) -> dict:
    mixer_color = {
        "mha": "black",
        "flare": "red",
        "luna": "teal",
        "simplifiedflarepp": "green",
        "flarepp": "darkorange",
        "transolver": "blue",
        "transolver3": "purple",
    }
    latent_linestyle = {
        32: ":",
        64: "-",
        128: "--",
        256: "-.",
    }
    mixer_marker = {
        "mha": "o",
        "flare": "s",
        "luna": "x",
        "simplifiedflarepp": "^",
        "flarepp": "P",
        "transolver": "D",
        "transolver3": "v",
    }
    if mixer == "mha" or num_latents is None:
        linestyle = "-"
    else:
        linestyle = latent_linestyle.get(int(num_latents), "-")
    return {
        "marker": mixer_marker[mixer],
        "linestyle": linestyle,
        "color": mixer_color[mixer],
        "linewidth": 2.5,
        "markersize": 9,
        "zorder": 2 if mixer == "mha" else 3,
    }


def _log_tick_label(val: float) -> str:
    exp = int(np.floor(np.log10(val) + 1e-12))
    mant = int(round(val / (10.0 ** exp)))
    if mant == 10:
        mant = 1
        exp += 1
    if mant == 1:
        return rf"$10^{{{exp}}}$"
    return rf"${mant}\times10^{{{exp}}}$"


def _apply_shared_log_ylim(axes, ymin: float, ymax: float) -> None:
    """Shared log y-limits. Label 1, 2, and 5 in each decade so the scale stays readable."""
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin <= 0 or ymax <= 0:
        return
    lo = int(np.floor(np.log10(ymin)))
    hi = int(np.ceil(np.log10(ymax)))
    ticks = []
    for exp in range(lo, hi + 1):
        for mant in (1, 2, 5):
            tick = mant * (10.0 ** exp)
            if ymin <= tick <= ymax:
                ticks.append(tick)
    labels = [_log_tick_label(tick) for tick in ticks]
    for ax in np.ravel(axes):
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.minorticks_off()
        ax.tick_params(axis="y", which="major", labelsize=12)


def _plot_one_panel(ax_train, ax_test, df_prec: pd.DataFrame, fontsize: int):
    ax_train.set_ylabel(r"Best relative error", fontsize=fontsize)
    for ax in (ax_train, ax_test):
        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xticks(list(NUM_BLOCKS))
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

    handles = []
    labels = []
    order = {name: i for i, name in enumerate(
        ("mha", "flare", "luna", "simplifiedflarepp", "flarepp", "transolver", "transolver3")
    )}
    present = []
    for mixer, group in df_prec.groupby("mixer", sort=False):
        latents = sorted(int(v) for v in group["num_latents"].dropna().unique())
        if mixer == "mha" or not latents:
            present.append((str(mixer), None))
        else:
            present.extend((str(mixer), latent) for latent in latents)
    present.sort(key=lambda item: (order.get(item[0], 99), -1 if item[1] is None else item[1]))
    for mixer, num_latents in present:
        series = _series_name(mixer, num_latents)
        sub = df_prec[df_prec["series"] == series].sort_values("num_blocks")
        if len(sub) == 0:
            continue
        if not np.isfinite(sub["train_rel_error"].to_numpy(dtype=float)).any():
            continue
        kwargs = _style_kwargs(mixer, num_latents)
        train_ok = sub[np.isfinite(sub["train_rel_error"].to_numpy(dtype=float))]
        test_ok = sub[np.isfinite(sub["test_rel_error"].to_numpy(dtype=float))]
        legend = _legend_label(mixer, num_latents)
        (h_train,) = ax_train.plot(
            train_ok["num_blocks"], train_ok["train_rel_error"], label=legend, **kwargs,
        )
        ax_test.plot(test_ok["num_blocks"], test_ok["test_rel_error"], label=None, **kwargs)
        handles.append(h_train)
        labels.append(legend)
    return handles, labels


def plot_results(df: pd.DataFrame, dataset: str = DEFAULT_DATASET):
    if len(df) == 0:
        print("ERROR: empty dataframe; nothing to plot.")
        return

    out_name = f"sweep_flarepp_{dataset}.pdf"
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=True, sharey=True)
    fontsize = 16

    legend_handles, legend_labels = [], []
    row_specs = (
        (0, "fp32", "fp32"),
        (1, "fp16", "fp16"),
    )
    for row_idx, prec_key, prec_title in row_specs:
        ax_train, ax_test = axes[row_idx, 0], axes[row_idx, 1]
        df_prec = df[df["precision"] == prec_key]
        handles, labels = _plot_one_panel(ax_train, ax_test, df_prec, fontsize=fontsize)
        if not legend_handles and handles:
            legend_handles, legend_labels = handles, labels
        ax_train.set_title(rf"Train — {prec_title} — {OUTER_LABEL}", fontsize=fontsize - 2)
        ax_test.set_title(rf"Test — {prec_title} — {OUTER_LABEL}", fontsize=fontsize - 2)
        if row_idx == 1:
            ax_train.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)
            ax_test.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)

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

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {out_path}")

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLARE++ standard B×M mixer sweep Rel-L2 plots (fp32+fp16)",
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
    parser.add_argument(
        "--casedir",
        type=str,
        default=None,
        help="Case root (default: out/pdebench/flarepp_standard)",
    )
    args = parser.parse_args()

    if args.eval:
        dataframe = collect_data(args.dataset, casedir=args.casedir)
        cols = [
            "precision", "series", "num_blocks", "complete",
            "train_rel_error", "test_rel_error",
        ]
        print(dataframe[cols].to_string(index=False))
        plot_results(dataframe, args.dataset)
    else:
        print("No action specified. Please specify --eval true.")

    raise SystemExit(0)

#======================================================================#

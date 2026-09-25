#
"""Plot mixer-backbone train/test losses vs num_blocks, one row per outer config.

Outer configuration key:
  (channel_dim, num_heads, num_layers_in_out_proj, out_proj_norm, num_layers_ffn, mlp_ratio_ffn)

Runs use ``--model.model mixer_backbone`` with ``--model.mixer`` among
mha / flare / simplifiedflarepp / transolver. Case dirs live under
``out/pdebench/mixer_backbone/`` (legacy ``mixer_ablations/`` is searched
when the new root is absent). Training often teed ``train.log`` into the
planned case dir first, so mlutils wrote the actual run into a ``*_01``
sibling — both are searched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CASEDIR_NEW = os.path.join(PROJDIR, "out", "pdebench", "mixer_backbone")
_CASEDIR_LEGACY = os.path.join(PROJDIR, "out", "pdebench", "mixer_ablations")
CASEDIR = _CASEDIR_NEW if os.path.isdir(_CASEDIR_NEW) else _CASEDIR_LEGACY
FIGDIR = os.path.join(PROJDIR, "figs")
os.makedirs(FIGDIR, exist_ok=True)

NUM_BLOCKS = (2, 4, 8)
MIXER_SPECS = (
    ("mha", None),
    ("flare", 32),
    ("flare", 64),
    ("flare", 128),
    ("simplifiedflarepp", 32),
    ("simplifiedflarepp", 64),
    ("simplifiedflarepp", 128),
    ("transolver", 32),
    ("transolver", 64),
    ("transolver", 128),
)


@dataclass(frozen=True)
class OuterConfig:
    channel_dim: int
    num_heads: int
    num_layers_in_out_proj: int
    out_proj_norm: bool
    num_layers_ffn: int
    mlp_ratio_ffn: float = 1.0

    def label(self) -> str:
        opn = "true" if self.out_proj_norm else "false"
        return (
            f"$C$={self.channel_dim}, $H$={self.num_heads}, "
            f"IO={self.num_layers_in_out_proj}, FFN={self.num_layers_ffn}, "
            f"OPN={opn}, MR={self.mlp_ratio_ffn:g}"
        )

    def key(self) -> tuple:
        return (
            self.channel_dim,
            self.num_heads,
            self.num_layers_in_out_proj,
            self.out_proj_norm,
            self.num_layers_ffn,
            float(self.mlp_ratio_ffn),
        )


# Historical sweeps + FFN0 / MR∈{2,4} outers.
OUTER_CONFIGS = (
    OuterConfig(64, 8, 2, True, 3, 1.0),
    OuterConfig(64, 8, -1, False, -1, 1.0),
    OuterConfig(128, 8, -1, False, -1, 1.0),
    OuterConfig(128, 8, 2, True, 3, 1.0),
    OuterConfig(64, 8, -1, False, 0, 4.0),
    OuterConfig(128, 8, -1, False, 0, 4.0),
    OuterConfig(64, 8, -1, False, 0, 2.0),
    OuterConfig(128, 8, -1, False, 0, 2.0),
)

#======================================================================#
def _mr_tag(mlp_ratio_ffn: float) -> str:
    # Keep a stable token (1.0 -> 1p0) matching the sweep launcher.
    return f"{float(mlp_ratio_ffn)}".replace(".", "p")


def _case_stem(
    mixer: str,
    num_latents: Optional[int],
    num_blocks: int,
    outer: OuterConfig,
    *,
    include_opn_mr: bool = True,
) -> str:
    mixer_tag = "mha" if mixer == "mha" else f"{mixer}_{num_latents}"
    stem = (
        f"{mixer_tag}_C{outer.channel_dim}_B{num_blocks}_H{outer.num_heads}"
        f"_IO_{outer.num_layers_in_out_proj}_FFN_{outer.num_layers_ffn}"
    )
    if include_opn_mr:
        opn = 1 if outer.out_proj_norm else 0
        stem = f"{stem}_OPN_{opn}_MR_{_mr_tag(outer.mlp_ratio_ffn)}"
    return stem


def _resolve_case_dir(stem: str) -> Optional[str]:
    """Prefer the directory that contains ``ckpt10`` (often ``stem_01``)."""
    candidates = [
        os.path.join(CASEDIR, f"{stem}_01"),
        os.path.join(CASEDIR, stem),
    ]
    if os.path.isdir(CASEDIR):
        for name in sorted(os.listdir(CASEDIR)):
            if re.fullmatch(re.escape(stem) + r"_\d+", name):
                candidates.append(os.path.join(CASEDIR, name))
    seen = set()
    for path in candidates:
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        if os.path.isdir(os.path.join(path, "ckpt10")):
            return path
    # Incomplete-but-present run: still return a case dir if any candidate exists
    # with ckpts (so we can report incomplete), else None.
    for path in candidates:
        if os.path.isdir(path) and any(n.startswith("ckpt") for n in os.listdir(path)):
            return path
    return None


def _resolve_case_dir_for_outer(mixer: str, num_latents: Optional[int], num_blocks: int, outer: OuterConfig) -> Optional[str]:
    # Prefer new naming with OPN/MR; fall back to legacy stem (original sweep).
    for include in (True, False):
        stem = _case_stem(mixer, num_latents, num_blocks, outer, include_opn_mr=include)
        path = _resolve_case_dir(stem)
        if path is None:
            continue
        # If using legacy stem, verify config matches this outer (when present).
        if not include:
            cfg_path = os.path.join(path, "config.yaml")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                model = cfg.get("model", cfg)
                if (
                    int(model.get("channel_dim", -1)) != outer.channel_dim
                    or int(model.get("num_heads", -1)) != outer.num_heads
                    or int(model.get("num_layers_in_out_proj", -999)) != outer.num_layers_in_out_proj
                    or bool(model.get("out_proj_norm", True)) != outer.out_proj_norm
                    or int(model.get("num_layers_ffn", -999)) != outer.num_layers_ffn
                    or float(model.get("mlp_ratio_ffn", 1.0)) != float(outer.mlp_ratio_ffn)
                ):
                    continue
        return path
    return None


def _best_loss_across_ckpts(case_path: str, split: str) -> float:
    """Min over ckpts of min(regular, ema) for ``train`` or ``test``.

    Incomplete runs (no ``ckpt10``) return NaN. EMA keys are used when present.
    """
    if not os.path.isdir(os.path.join(case_path, "ckpt10")):
        return float("nan")

    key = f"{split}_loss"
    key_ema = f"{split}_loss_ema"
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
        if stats.get(key) is not None:
            vals.append(float(stats[key]))
        if stats.get(key_ema) is not None:
            vals.append(float(stats[key_ema]))
        if vals:
            found = True
            best = min(best, min(vals))
    return best if found else float("nan")


def collect_data(
    outer_configs: tuple[OuterConfig, ...] = OUTER_CONFIGS,
    *,
    case_dir: Optional[str] = None,
    mixer_specs: tuple[tuple[str, Optional[int]], ...] = MIXER_SPECS,
) -> pd.DataFrame:
    global CASEDIR
    if case_dir is not None:
        CASEDIR = case_dir
    rows = []
    for outer in outer_configs:
        for mixer, num_latents in mixer_specs:
            for num_blocks in NUM_BLOCKS:
                case_path = _resolve_case_dir_for_outer(mixer, num_latents, num_blocks, outer)
                series = "mha" if mixer == "mha" else f"{mixer} M={num_latents}"
                row = {
                    "outer_label": outer.label(),
                    "channel_dim": outer.channel_dim,
                    "num_heads": outer.num_heads,
                    "num_layers_in_out_proj": outer.num_layers_in_out_proj,
                    "out_proj_norm": outer.out_proj_norm,
                    "num_layers_ffn": outer.num_layers_ffn,
                    "mlp_ratio_ffn": outer.mlp_ratio_ffn,
                    "mixer": mixer,
                    "num_latents": num_latents if num_latents is not None else np.nan,
                    "num_blocks": num_blocks,
                    "series": series,
                    "case_path": case_path,
                    "train_loss": float("nan"),
                    "test_loss": float("nan"),
                    "complete": False,
                }
                if case_path is not None:
                    row["train_loss"] = _best_loss_across_ckpts(case_path, "train")
                    row["test_loss"] = _best_loss_across_ckpts(case_path, "test")
                    row["complete"] = os.path.isdir(os.path.join(case_path, "ckpt10"))
                rows.append(row)

    df = pd.DataFrame(rows)
    n_complete = int(df["complete"].sum()) if len(df) else 0
    print(f"Collected {len(df)} grid cells ({n_complete} with ckpt10) from {CASEDIR}")
    return df


#======================================================================#
def _style_kwargs(mixer: str, num_latents: Optional[int]) -> dict:
    mixer_color = {"mha": "black", "flare": "red", "simplifiedflarepp": "green", "transolver": "blue"}
    latent_linestyle = {
        32: "--",
        64: "-.",
        128: (0, (4.0, 1.2, 4.0, 1.2)),
    }
    mixer_marker = {"mha": "o", "flare": "s", "simplifiedflarepp": "^", "transolver": "D"}
    linestyle = "-" if mixer == "mha" else latent_linestyle[int(num_latents)]
    return {
        "marker": mixer_marker[mixer],
        "linestyle": linestyle,
        "color": mixer_color[mixer],
        "linewidth": 2.5,
        "markersize": 9,
    }


def _plot_one_outer(ax_train, ax_test, df_outer: pd.DataFrame, fontsize: int, show_ylabel: bool):
    if show_ylabel:
        ax_train.set_ylabel(r"Best relative error", fontsize=fontsize)

    for ax in (ax_train, ax_test):
        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xticks(list(NUM_BLOCKS))
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

    series_order = [
        ("mha", None),
        ("flare", 32),
        ("flare", 64),
        ("flare", 128),
        ("simplifiedflarepp", 32),
        ("simplifiedflarepp", 64),
        ("simplifiedflarepp", 128),
        ("transolver", 32),
        ("transolver", 64),
        ("transolver", 128),
    ]
    handles = []
    labels = []
    for mixer, num_latents in series_order:
        series = "mha" if mixer == "mha" else f"{mixer} M={num_latents}"
        sub = df_outer[df_outer["series"] == series].sort_values("num_blocks")
        sub = sub[np.isfinite(sub["train_loss"].to_numpy(dtype=float))]
        if len(sub) == 0:
            continue
        kwargs = _style_kwargs(mixer, num_latents)
        (h_train,) = ax_train.plot(sub["num_blocks"], sub["train_loss"], label=series, **kwargs)
        ax_test.plot(sub["num_blocks"], sub["test_loss"], label=None, **kwargs)
        handles.append(h_train)
        labels.append(series)

    return handles, labels


def plot_results(df: pd.DataFrame, out_name: str = "abl_mixer_elasticity.pdf"):
    if len(df) == 0:
        print("ERROR: empty dataframe; nothing to plot.")
        return

    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
        _ = plt.figure()
        plt.close()
    except Exception:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
        })

    # Only draw outer rows that have at least one finished run.
    outer_labels = []
    for outer in OUTER_CONFIGS:
        mask = df["outer_label"] == outer.label()
        if mask.any() and bool(df.loc[mask, "complete"].any()):
            outer_labels.append(outer)

    n_rows = max(len(outer_labels), 1)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.6 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.array([axes])
    fontsize = 16

    legend_handles, legend_labels = [], []
    for row_idx, outer in enumerate(outer_labels):
        ax_train, ax_test = axes[row_idx, 0], axes[row_idx, 1]
        df_outer = df[df["outer_label"] == outer.label()]
        handles, labels = _plot_one_outer(
            ax_train, ax_test, df_outer, fontsize=fontsize, show_ylabel=True,
        )
        if not legend_handles:
            legend_handles, legend_labels = handles, labels
        ax_train.set_title(rf"Train — {outer.label()}", fontsize=fontsize - 1)
        ax_test.set_title(rf"Test — {outer.label()}", fontsize=fontsize - 1)
        if row_idx == n_rows - 1:
            ax_train.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)
            ax_test.set_xlabel(r"Number of blocks ($B$)", fontsize=fontsize)

    # Shared y-limits across every panel from all finite train/test values.
    vals = np.concatenate([
        df["train_loss"].to_numpy(dtype=float),
        df["test_loss"].to_numpy(dtype=float),
    ])
    vals = vals[np.isfinite(vals)]
    if len(vals):
        ymin, ymax = float(vals.min()) * 0.8, float(vals.max()) * 1.4
        for ax in axes.ravel():
            ax.set_ylim(ymin, ymax)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=5,
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
    plt.subplots_adjust(bottom=max(0.08, 0.22 / n_rows))

    out_path = os.path.join(FIGDIR, out_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

    csv_path = os.path.join(FIGDIR, out_name.replace(".pdf", ".csv"))
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixer-backbone train/test loss plots")

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument("--eval", type=str_to_bool, default=False, help="Collect and plot results")
    parser.add_argument(
        "--case-dir",
        type=str,
        default=None,
        help="Case directory (default: out/pdebench/mixer_backbone, else legacy mixer_ablations)",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="abl_mixer_elasticity.pdf",
        help="Output PDF/CSV basename under figs/",
    )
    parser.add_argument(
        "--mixers",
        type=str,
        default="all",
        help="Comma-separated mixers to include, or 'all' (e.g. mha,simplifiedflarepp)",
    )
    args = parser.parse_args()

    if args.eval:
        if args.mixers.strip().lower() == "all":
            mixer_specs = MIXER_SPECS
        else:
            wanted = {m.strip().lower() for m in args.mixers.split(",") if m.strip()}
            mixer_specs = tuple(spec for spec in MIXER_SPECS if spec[0] in wanted)
            if not mixer_specs:
                raise SystemExit(f"No MIXER_SPECS match --mixers={args.mixers!r}")
        dataframe = collect_data(case_dir=args.case_dir, mixer_specs=mixer_specs)
        cols = [
            "outer_label", "series", "num_blocks", "complete", "train_loss", "test_loss",
        ]
        print(dataframe[cols].to_string(index=False))
        plot_results(dataframe, out_name=args.out_name)
    else:
        print("No action specified. Please specify --eval true.")

    raise SystemExit(0)

#======================================================================#

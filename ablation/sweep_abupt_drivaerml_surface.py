"""Collect and plot the DrivAerML surface AB-UPT anchor-by-depth MSE sweep."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ablation.sweep_flarepp import FIGDIR

PROJDIR = Path(__file__).resolve().parents[1]
CASEDIR = PROJDIR / "out" / "pdebench"
NUM_BLOCKS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
NUM_SURFACE_ANCHORS = (32, 64, 128, 256, 512)
_LEGACY_B2_CASES = {
    256: "drivaerml_abupt_surface_mixer_surfaceonly_B2_C128_H8_A256_oc2_seed0_A256",
    512: "drivaerml_abupt_surface_mixer_surfaceonly_B2_C128_H8_A512_oc2_seed0_A512",
}


def _grid_case_stem(num_surface_anchors: int, num_blocks: int) -> str:
    return (
        f"drivaerml_abupt_surface_grid_M{int(num_surface_anchors)}_B{int(num_blocks)}"
        f"_C128_H8_seed0_A{int(num_surface_anchors)}"
    )


def _legacy_b2_case_stem(num_surface_anchors: int) -> str:
    return _LEGACY_B2_CASES[int(num_surface_anchors)]


def _case_candidates(case_root: Path, num_surface_anchors: int, num_blocks: int) -> list[Path]:
    if num_blocks == 2 and num_surface_anchors in _LEGACY_B2_CASES:
        return [case_root / _legacy_b2_case_stem(num_surface_anchors)]
    stem = _grid_case_stem(num_surface_anchors, num_blocks)
    return [Path(path) for path in sorted(glob.glob(str(case_root / f"{stem}*"))) if Path(path).is_dir()]


def _completed_case(candidates: list[Path]) -> Path | None:
    completed = [case for case in candidates if (case / "ckpt10" / "stats.json").is_file()]
    return completed[-1] if completed else (candidates[-1] if candidates else None)


def _best_stats(case_path: Path | None) -> dict[str, float]:
    if case_path is None or not (case_path / "ckpt10" / "stats.json").is_file():
        return {}
    checkpoints = sorted(case_path.glob("ckpt*/stats.json"))
    records = []
    for stats_path in checkpoints:
        with stats_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("test_loss") is not None:
            records.append(record)
    return min(records, key=lambda record: float(record["test_loss"])) if records else {}


def _metric(record: dict, section: str, key: str) -> float:
    value = record.get(section, {}).get(key)
    return float(value) if value is not None else float("nan")


def collect_data(*, casedir: Path | str | None = None) -> pd.DataFrame:
    """Return one row per requested anchor count and total mixer depth."""
    case_root = Path(casedir) if casedir is not None else CASEDIR
    rows = []
    for num_surface_anchors in NUM_SURFACE_ANCHORS:
        for num_blocks in NUM_BLOCKS:
            case_path = _completed_case(_case_candidates(case_root, num_surface_anchors, num_blocks))
            record = _best_stats(case_path)
            rows.append(
                {
                    "mixer": "abupt_surface_mixer",
                    "channel_dim": 128,
                    "num_heads": 8,
                    "num_blocks": num_blocks,
                    "num_surface_anchors": num_surface_anchors,
                    "precision": "fp16",
                    "weight_decay": 1e-5,
                    "seed": 0,
                    "case_path": str(case_path) if case_path is not None else None,
                    "train_mse": float(record.get("train_loss", float("nan"))),
                    "test_mse": float(record.get("test_loss", float("nan"))),
                    "train_full_rel_l2": _metric(record, "train_stats", "full_rel_l2"),
                    "test_full_rel_l2": _metric(record, "test_stats", "full_rel_l2"),
                    "test_pressure_rel_l2": _metric(record, "test_stats", "pressure_rel_l2"),
                    "test_wall_shear_rel_l2": _metric(record, "test_stats", "wall_shear_rel_l2"),
                    "complete": bool(case_path and (case_path / "ckpt10" / "stats.json").is_file()),
                }
            )
    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame) -> tuple[Path, Path]:
    """Write the sweep CSV and a test-normalized-MSE curve."""
    figdir = Path(FIGDIR)
    figdir.mkdir(parents=True, exist_ok=True)
    csv_path = figdir / "sweep_abupt_drivaerml_surface.csv"
    pdf_path = figdir / "sweep_abupt_drivaerml_surface.pdf"
    df.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for num_surface_anchors in NUM_SURFACE_ANCHORS:
        subset = df[(df["num_surface_anchors"] == num_surface_anchors) & df["complete"]]
        axis.plot(
            subset["num_blocks"],
            subset["test_mse"],
            marker="o",
            label=f"M={num_surface_anchors}",
        )
    axis.set_xlabel("Number of mixer blocks (B)")
    axis.set_ylabel("Test normalized MSE")
    axis.set_xticks(NUM_BLOCKS)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Surface anchors")
    figure.tight_layout()
    figure.savefig(pdf_path)
    plt.close(figure)
    return csv_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write CSV and PDF artifacts")
    args = parser.parse_args()
    if not args.write:
        print("No action specified; pass --write to generate the AB-UPT sweep artifacts.")
        return
    df = collect_data()
    csv_path, pdf_path = write_results(df)
    print(f"Collected {len(df)} cells ({int(df['complete'].sum())} complete)")
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

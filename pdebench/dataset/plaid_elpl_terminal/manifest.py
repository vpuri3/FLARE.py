"""One-row-per-simulation manifest for terminal prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def terminal_length(*, num_sims: int) -> int:
    """Indexer length for ``plaid_elpl_terminal``: one sample per simulation."""
    return int(num_sims)


def build_terminal_manifest(sim_ids: list[int]) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for global_idx, sim_id in enumerate(sorted(int(v) for v in sim_ids)):
        rows.append({"global_idx": int(global_idx), "sim_id": int(sim_id)})
    return pd.DataFrame(rows)


def write_terminal_manifest_tables(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_path: str | Path,
    val_path: str | Path,
) -> None:
    for path, frame in ((train_path, train_df), (val_path, val_df)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)


def load_terminal_manifest_tables(
    *,
    train_path: str | Path,
    val_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(train_path), pd.read_parquet(val_path)

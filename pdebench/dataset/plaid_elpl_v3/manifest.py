"""Transition manifest tables for el-pl v3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdebench.dataset.plaid_elpl_v3.constants import TRAJECTORIES_PER_SHARD
from pdebench.dataset.plaid_elpl_v3.schema import ManifestRow


def partition_sim_ids(sim_ids: list[int], *, trajectories_per_shard: int = TRAJECTORIES_PER_SHARD) -> list[list[int]]:
    ordered = sorted(int(v) for v in sim_ids)
    if not ordered:
        return []
    chunk = max(1, int(trajectories_per_shard))
    return [ordered[i : i + chunk] for i in range(0, len(ordered), chunk)]


def build_sim_to_shard(sim_ids: list[int], *, trajectories_per_shard: int = TRAJECTORIES_PER_SHARD) -> pd.DataFrame:
    rows = []
    for shard_id, shard_sims in enumerate(partition_sim_ids(sim_ids, trajectories_per_shard=trajectories_per_shard)):
        for local_idx, sim_id in enumerate(shard_sims):
            rows.append({"sim_id": int(sim_id), "shard_id": int(shard_id), "local_idx": int(local_idx)})
    return pd.DataFrame(rows)


def transitions_per_sim(times: list[float] | tuple[float, ...]) -> int:
    return max(0, len(times) - 1)


def transition_length(*, num_sims: int, num_steps: int) -> int:
    """Indexer length for ``plaid_el_pl_dynamics``: ``#sims × (T − 1)``."""
    return int(num_sims) * max(0, int(num_steps) - 1)


def build_transition_manifest(
    sim_ids: list[int],
    *,
    times_by_sim: dict[int, list[float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    global_idx = 0
    for sim_id in sim_ids:
        times = [float(v) for v in times_by_sim[int(sim_id)]]
        for step_idx in range(transitions_per_sim(times)):
            rows.append(
                {
                    "global_idx": int(global_idx),
                    "sim_id": int(sim_id),
                    "step_idx": int(step_idx),
                    "t0": float(times[step_idx]),
                    "t1": float(times[step_idx + 1]),
                }
            )
            global_idx += 1
    return pd.DataFrame(rows)


def manifest_rows_from_frame(frame: pd.DataFrame) -> list[ManifestRow]:
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            ManifestRow(
                global_idx=int(record["global_idx"]),
                sim_id=int(record["sim_id"]),
                step_idx=int(record["step_idx"]),
                t0=float(record["t0"]),
                t1=float(record["t1"]),
            )
        )
    return rows


def write_manifest_tables(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sim_to_shard_df: pd.DataFrame,
    train_path: str | Path,
    val_path: str | Path,
    sim_to_shard_path: str | Path,
) -> None:
    for path, frame in (
        (train_path, train_df),
        (val_path, val_df),
        (sim_to_shard_path, sim_to_shard_df),
    ):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)


def load_manifest_tables(
    *,
    train_path: str | Path,
    val_path: str | Path,
    sim_to_shard_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(train_path),
        pd.read_parquet(val_path),
        pd.read_parquet(sim_to_shard_path),
    )

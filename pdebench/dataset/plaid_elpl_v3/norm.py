"""Channel-group normalizers for el-pl v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pdebench.dataset.normalizer import NodeFeatureNormalizer
from pdebench.dataset.plaid_elpl_v3.schema import TrajectoryBundle


@dataclass
class ElPlNormStats:
    pos_normalizer: NodeFeatureNormalizer
    geom_normalizer: NodeFeatureNormalizer
    u_normalizer: NodeFeatureNormalizer
    input_scalar_normalizer: NodeFeatureNormalizer
    y_normalizer: NodeFeatureNormalizer
    identity_geom_normalizer: NodeFeatureNormalizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos_normalizer": (self.pos_normalizer.mean, self.pos_normalizer.std),
            "geom_normalizer": (self.geom_normalizer.mean, self.geom_normalizer.std),
            "u_normalizer": (self.u_normalizer.mean, self.u_normalizer.std),
            "input_scalar_normalizer": (self.input_scalar_normalizer.mean, self.input_scalar_normalizer.std),
            "y_normalizer": (self.y_normalizer.mean, self.y_normalizer.std),
            "identity_geom_normalizer": (self.identity_geom_normalizer.mean, self.identity_geom_normalizer.std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ElPlNormStats":
        def _load(key: str) -> NodeFeatureNormalizer:
            mean, std = payload[key]
            return NodeFeatureNormalizer(mean=mean, std=std)

        return cls(
            pos_normalizer=_load("pos_normalizer"),
            geom_normalizer=_load("geom_normalizer"),
            u_normalizer=_load("u_normalizer"),
            input_scalar_normalizer=_load("input_scalar_normalizer"),
            y_normalizer=_load("y_normalizer"),
            identity_geom_normalizer=_load("identity_geom_normalizer"),
        )


def _identity_normalizer(dim: int) -> NodeFeatureNormalizer:
    return NodeFeatureNormalizer(
        mean=torch.zeros(1, int(dim)),
        std=torch.ones(1, int(dim)),
    )


def _streaming_normalizer(chunks: list[torch.Tensor], *, dim: int) -> NodeFeatureNormalizer:
    count = 0
    mean = torch.zeros(dim, dtype=torch.float64)
    m2 = torch.zeros(dim, dtype=torch.float64)
    for chunk in chunks:
        values = chunk.reshape(-1, dim).to(dtype=torch.float64, device="cpu")
        if values.numel() == 0:
            continue
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if count == 0:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue
        total = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / total)
        m2 = m2 + batch_m2 + (delta**2) * (count * batch_count / total)
        count = total
    if count == 0:
        raise ValueError("Cannot build normalizer from empty chunks.")
    std = torch.sqrt(m2 / count).clamp_min(1e-8)
    return NodeFeatureNormalizer(mean=mean.to(torch.float32).reshape(1, -1), std=std.to(torch.float32).reshape(1, -1))


def fit_norm_stats_from_trajectories(
    trajectories: list[TrajectoryBundle],
    *,
    transitions_per_sim: int,
) -> ElPlNormStats:
    pos_norm = _streaming_normalizer([traj.pos for traj in trajectories], dim=2)
    geom_norm = _streaming_normalizer(
        [torch.cat([traj.sdf, traj.proj], dim=-1) for traj in trajectories],
        dim=3,
    )

    u_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    scalar_chunks: list[torch.Tensor] = []
    for traj in trajectories:
        for step in range(int(transitions_per_sim)):
            u_chunks.append(traj.u_traj[step])
            y_chunks.append(traj.u_traj[step + 1])
            scalar_chunks.append(traj.times[step].reshape(1, 1))

    return ElPlNormStats(
        pos_normalizer=pos_norm,
        geom_normalizer=geom_norm,
        u_normalizer=_streaming_normalizer(u_chunks, dim=2),
        input_scalar_normalizer=_streaming_normalizer(scalar_chunks, dim=1),
        y_normalizer=_streaming_normalizer(y_chunks, dim=2),
        identity_geom_normalizer=_identity_normalizer(3),
    )


def fit_norm_stats_from_shard_stream(
    *,
    train_ids: list[int],
    sim_to_shard_df,
    shard_root: Path,
    transitions_per_sim: int,
) -> ElPlNormStats:
    from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload

    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}

    pos_chunks: list[torch.Tensor] = []
    geom_chunks: list[torch.Tensor] = []
    u_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    scalar_chunks: list[torch.Tensor] = []

    for sim_id in train_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        traj = shard_cache[shard_id].trajectories[local_idx]
        pos_chunks.append(traj.pos)
        geom_chunks.append(torch.cat([traj.sdf, traj.proj], dim=-1))
        for step in range(int(transitions_per_sim)):
            u_chunks.append(traj.u_traj[step])
            y_chunks.append(traj.u_traj[step + 1])
            scalar_chunks.append(traj.times[step].reshape(1, 1))
        del traj

    return ElPlNormStats(
        pos_normalizer=_streaming_normalizer(pos_chunks, dim=2),
        geom_normalizer=_streaming_normalizer(geom_chunks, dim=3),
        u_normalizer=_streaming_normalizer(u_chunks, dim=2),
        input_scalar_normalizer=_streaming_normalizer(scalar_chunks, dim=1),
        y_normalizer=_streaming_normalizer(y_chunks, dim=2),
        identity_geom_normalizer=_identity_normalizer(3),
    )


def save_norm_stats(path: str | Path, stats: ElPlNormStats) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(stats.to_dict(), tmp)
    tmp.replace(out)


def load_norm_stats(path: str | Path) -> ElPlNormStats:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ElPlNormStats.from_dict(payload)


def encode_on_cpu(normalizer: NodeFeatureNormalizer, x: torch.Tensor) -> torch.Tensor:
    return (x.cpu() - normalizer.mean.cpu()) / normalizer.std.cpu()


def encode_node_features(
    *,
    pos: torch.Tensor,
    geom: torch.Tensor | None,
    u_field: torch.Tensor,
    stats: ElPlNormStats,
    use_sdf_features: bool,
) -> torch.Tensor:
    pos_enc = encode_on_cpu(stats.pos_normalizer, pos)
    u_enc = encode_on_cpu(stats.u_normalizer, u_field)
    if not use_sdf_features:
        return torch.cat([pos_enc, u_enc], dim=-1)
    if geom is None:
        raise ValueError("geom is required when use_sdf_features=True.")
    geom_enc = encode_on_cpu(stats.geom_normalizer, geom)
    return torch.cat([pos_enc, geom_enc, u_enc], dim=-1)


def metadata_normalizers(stats: ElPlNormStats) -> dict[str, NodeFeatureNormalizer]:
    """Expose legacy metadata keys expected by the training stack."""
    return {
        "x_normalizer": stats.pos_normalizer,
        "y_normalizer": stats.y_normalizer,
        "input_scalar_normalizer": stats.input_scalar_normalizer,
        "y_scalar_normalizer": None,
    }

"""Normalizers for terminal el-pl graphs (mesh -> u(T))."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from pdebench.dataset.normalizer import NodeFeatureNormalizer
from pdebench.dataset.plaid_elpl_terminal.constants import (
    CACHE_Y_NORM_VERSION,
    DEFAULT_RUNTIME_Y_NORM,
    FINAL_STEP_IDX,
    ROBUST_SCALE_FACTOR,
    RUNTIME_ASINH_IQR_VERSION,
    RUNTIME_Y_NORM_MODES,
    UX_CLIP_HI,
    UX_CLIP_LO,
    Y_NORM_IQR_FLOOR,
    Y_NORM_SCALE_FLOOR,
)
from pdebench.dataset.plaid_elpl_v3.norm import _identity_normalizer, _streaming_normalizer, encode_on_cpu
from pdebench.dataset.plaid_elpl_v3.schema import TrajectoryBundle


@dataclass
class AsinhIQRChannelSpec:
    """Per-channel: optional clip, asinh scale s, then affine on asinh output."""

    s: float
    median: float
    iqr: float
    clip_lo: float | None = None
    clip_hi: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "s": float(self.s),
            "median": float(self.median),
            "iqr": float(self.iqr),
            "clip_lo": None if self.clip_lo is None else float(self.clip_lo),
            "clip_hi": None if self.clip_hi is None else float(self.clip_hi),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AsinhIQRChannelSpec":
        clip_lo = payload.get("clip_lo")
        clip_hi = payload.get("clip_hi")
        return cls(
            s=float(payload["s"]),
            median=float(payload["median"]),
            iqr=float(payload["iqr"]),
            clip_lo=None if clip_lo is None else float(clip_lo),
            clip_hi=None if clip_hi is None else float(clip_hi),
        )


class TerminalFieldNormalizer:
    """U_x/U_y: clip (U_x only) -> asinh(u/s) -> robust affine (median/IQR)."""

    def __init__(self, channels: tuple[AsinhIQRChannelSpec, ...]):
        if len(channels) == 0:
            raise ValueError("TerminalFieldNormalizer requires at least one channel.")
        self.channels = channels

    def to(self, device):
        return self

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != len(self.channels):
            raise ValueError(
                f"Expected y with {len(self.channels)} channels, got shape {tuple(y.shape)}."
            )
        out = []
        for ch, spec in enumerate(self.channels):
            u = y[..., ch]
            if spec.clip_lo is not None and spec.clip_hi is not None:
                u = torch.clamp(u, min=float(spec.clip_lo), max=float(spec.clip_hi))
            t = torch.asinh(u / float(spec.s))
            out.append((t - float(spec.median)) / max(float(spec.iqr), Y_NORM_IQR_FLOOR))
        return torch.stack(out, dim=-1)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != len(self.channels):
            raise ValueError(
                f"Expected y with {len(self.channels)} channels, got shape {tuple(y.shape)}."
            )
        out = []
        for ch, spec in enumerate(self.channels):
            t = y[..., ch] * max(float(spec.iqr), Y_NORM_IQR_FLOOR) + float(spec.median)
            u = float(spec.s) * torch.sinh(t)
            out.append(u)
        return torch.stack(out, dim=-1)


YFieldNormalizer = Union[NodeFeatureNormalizer, TerminalFieldNormalizer]


def slice_y_normalizer(normalizer: YFieldNormalizer, channel_indices: tuple[int, ...]) -> YFieldNormalizer:
    """Restrict a fitted y normalizer to a channel subset (order preserved)."""
    if len(channel_indices) == 0:
        raise ValueError("channel_indices must be non-empty.")
    if isinstance(normalizer, NodeFeatureNormalizer):
        idx = list(channel_indices)
        return NodeFeatureNormalizer(mean=normalizer.mean[:, idx], std=normalizer.std[:, idx])
    if isinstance(normalizer, TerminalFieldNormalizer):
        return TerminalFieldNormalizer(channels=tuple(normalizer.channels[i] for i in channel_indices))
    raise TypeError(f"Unsupported y normalizer type: {type(normalizer)!r}")


def _mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _robust_scale(values: np.ndarray) -> float:
    return max(float(ROBUST_SCALE_FACTOR * _mad(values)), Y_NORM_SCALE_FLOOR)


def _iqr(values: np.ndarray) -> float:
    p25, p75 = np.percentile(values, [25.0, 75.0])
    return max(float(p75 - p25), Y_NORM_IQR_FLOOR)


def _fit_ux_channel(ux: np.ndarray) -> AsinhIQRChannelSpec:
    clip_lo, clip_hi = float(UX_CLIP_LO), float(UX_CLIP_HI)
    uxc = np.clip(ux, clip_lo, clip_hi)
    s = _robust_scale(uxc)
    t = np.arcsinh(uxc / s)
    return AsinhIQRChannelSpec(
        s=s,
        median=float(np.median(t)),
        iqr=_iqr(t),
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )


def _fit_uy_channel(uy: np.ndarray) -> AsinhIQRChannelSpec:
    s = _robust_scale(uy)
    t = np.arcsinh(uy / s)
    return AsinhIQRChannelSpec(
        s=s,
        median=float(np.median(t)),
        iqr=_iqr(t),
    )


def fit_terminal_field_normalizer(y_chunks: list[torch.Tensor]) -> TerminalFieldNormalizer:
    if len(y_chunks) == 0:
        raise ValueError("Cannot fit terminal field normalizer from empty chunks.")
    y = torch.cat(y_chunks, dim=0).detach().cpu().numpy()
    if y.ndim != 2 or y.shape[-1] < 2:
        raise ValueError(f"Expected terminal targets [N, 2], got shape {y.shape}.")
    return TerminalFieldNormalizer(
        channels=(
            _fit_ux_channel(y[:, 0]),
            _fit_uy_channel(y[:, 1]),
        )
    )


def decode_y_physical(
    *,
    raw_y: torch.Tensor | None = None,
    cached_y: torch.Tensor | None = None,
    cache_y_normalizer: NodeFeatureNormalizer,
) -> torch.Tensor:
    if raw_y is not None:
        return raw_y
    if cached_y is not None:
        return cache_y_normalizer.decode(cached_y)
    raise ValueError("decode_y_physical requires raw_y or cached_y.")


def encode_y_for_runtime(
    *,
    raw_y: torch.Tensor | None = None,
    cached_y: torch.Tensor | None = None,
    cache_y_normalizer: NodeFeatureNormalizer,
    runtime_y_normalizer: YFieldNormalizer,
) -> torch.Tensor:
    """Decode frozen-cache y encoding (or use raw u) then apply runtime y normalization."""
    physical = decode_y_physical(
        raw_y=raw_y,
        cached_y=cached_y,
        cache_y_normalizer=cache_y_normalizer,
    )
    return runtime_y_normalizer.encode(physical)


def encode_y_on_cpu(
    *,
    raw_y: torch.Tensor | None = None,
    cached_y: torch.Tensor | None = None,
    cache_y_normalizer: NodeFeatureNormalizer,
    runtime_y_normalizer: YFieldNormalizer,
) -> torch.Tensor:
    return encode_y_for_runtime(
        raw_y=raw_y,
        cached_y=cached_y,
        cache_y_normalizer=cache_y_normalizer,
        runtime_y_normalizer=runtime_y_normalizer,
    ).cpu().float()


@dataclass
class ElPlTerminalNormStats:
    pos_normalizer: NodeFeatureNormalizer
    geom_normalizer: NodeFeatureNormalizer
    cache_y_normalizer: NodeFeatureNormalizer
    identity_geom_normalizer: NodeFeatureNormalizer

    @property
    def y_normalizer(self) -> NodeFeatureNormalizer:
        """Backward-compatible alias for the frozen cache y normalizer."""
        return self.cache_y_normalizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_norm_version": int(CACHE_Y_NORM_VERSION),
            "pos_normalizer": (self.pos_normalizer.mean, self.pos_normalizer.std),
            "geom_normalizer": (self.geom_normalizer.mean, self.geom_normalizer.std),
            "y_normalizer": (self.cache_y_normalizer.mean, self.cache_y_normalizer.std),
            "identity_geom_normalizer": (
                self.identity_geom_normalizer.mean,
                self.identity_geom_normalizer.std,
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ElPlTerminalNormStats":
        version = int(payload.get("y_norm_version", 1))
        if version != CACHE_Y_NORM_VERSION:
            raise ValueError(
                f"Unsupported terminal cache y_norm_version={version} "
                f"(expected {CACHE_Y_NORM_VERSION})."
            )

        def _load_node(key: str) -> NodeFeatureNormalizer:
            mean, std = payload[key]
            return NodeFeatureNormalizer(mean=mean, std=std)

        return cls(
            pos_normalizer=_load_node("pos_normalizer"),
            geom_normalizer=_load_node("geom_normalizer"),
            cache_y_normalizer=_load_node("y_normalizer"),
            identity_geom_normalizer=_load_node("identity_geom_normalizer"),
        )


def fit_terminal_norm_stats_from_trajectories(trajectories: list[TrajectoryBundle]) -> ElPlTerminalNormStats:
    pos_chunks = [traj.pos for traj in trajectories]
    geom_chunks = [torch.cat([traj.sdf, traj.proj], dim=-1) for traj in trajectories]
    y_chunks = [traj.u_traj[int(FINAL_STEP_IDX)] for traj in trajectories]
    return ElPlTerminalNormStats(
        pos_normalizer=_streaming_normalizer(pos_chunks, dim=2),
        geom_normalizer=_streaming_normalizer(geom_chunks, dim=3),
        cache_y_normalizer=_streaming_normalizer(y_chunks, dim=2),
        identity_geom_normalizer=_identity_normalizer(3),
    )


def fit_terminal_norm_stats_from_shard_stream(
    *,
    train_ids: list[int],
    sim_to_shard_df,
    shard_root: Path,
) -> ElPlTerminalNormStats:
    from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload

    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    trajectories: list[TrajectoryBundle] = []
    for sim_id in train_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        trajectories.append(shard_cache[shard_id].trajectories[local_idx])
    return fit_terminal_norm_stats_from_trajectories(trajectories)


def save_terminal_norm_stats(path: str | Path, stats: ElPlTerminalNormStats) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(stats.to_dict(), tmp)
    tmp.replace(out)


def load_terminal_norm_stats(path: str | Path) -> ElPlTerminalNormStats:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ElPlTerminalNormStats.from_dict(payload)


def save_runtime_y_normalizer(path: str | Path, normalizer: TerminalFieldNormalizer, *, mode: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": str(mode),
        "y_norm_version": int(RUNTIME_ASINH_IQR_VERSION),
        "channels": [ch.to_dict() for ch in normalizer.channels],
    }
    tmp = out.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def load_runtime_y_normalizer(path: str | Path) -> TerminalFieldNormalizer:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = int(payload.get("y_norm_version", RUNTIME_ASINH_IQR_VERSION))
    if version != RUNTIME_ASINH_IQR_VERSION:
        raise ValueError(
            f"Unsupported runtime y_norm_version={version} (expected {RUNTIME_ASINH_IQR_VERSION})."
        )
    return TerminalFieldNormalizer(
        channels=tuple(AsinhIQRChannelSpec.from_dict(ch) for ch in payload["channels"])
    )


def normalize_runtime_y_norm_mode(mode: str | None) -> str:
    resolved = str(mode or DEFAULT_RUNTIME_Y_NORM).strip().lower()
    if resolved not in RUNTIME_Y_NORM_MODES:
        raise ValueError(
            f"Unsupported terminal y norm mode '{resolved}'. Expected one of {RUNTIME_Y_NORM_MODES}."
        )
    return resolved


def resolve_runtime_y_normalizer(
    *,
    mode: str,
    dataset_dir: str | Path,
    split_seed: int,
    use_sdf_features: bool,
    cache_y_normalizer: NodeFeatureNormalizer,
    train_ids: list[int],
    sim_to_shard_df,
    shard_root: Path,
    fit_if_missing: bool = True,
) -> YFieldNormalizer:
    mode = normalize_runtime_y_norm_mode(mode)
    if mode == "cache":
        return cache_y_normalizer

    from pdebench.dataset.plaid_elpl_terminal.paths import runtime_y_norm_stats_path

    runtime_path = runtime_y_norm_stats_path(
        dataset_dir,
        split_seed=split_seed,
        use_sdf_features=use_sdf_features,
        y_norm_mode=mode,
    )
    if runtime_path.is_file():
        try:
            return load_runtime_y_normalizer(runtime_path)
        except ValueError as exc:
            if "y_norm_version" not in str(exc):
                raise
    elif not fit_if_missing:
        raise FileNotFoundError(f"Runtime y normalizer is missing: {runtime_path}")
    if mode == "asinh_iqr":
        y_chunks = _collect_final_y_chunks(
            train_ids=train_ids,
            sim_to_shard_df=sim_to_shard_df,
            shard_root=shard_root,
        )
        normalizer = fit_terminal_field_normalizer(y_chunks)
        save_runtime_y_normalizer(runtime_path, normalizer, mode=mode)
        return normalizer
    raise ValueError(f"Unhandled runtime y norm mode '{mode}'.")


def _collect_final_y_chunks(
    *,
    train_ids: list[int],
    sim_to_shard_df,
    shard_root: Path,
) -> list[torch.Tensor]:
    from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload

    lookup = sim_to_shard_df.set_index("sim_id")
    shard_cache: dict[int, Any] = {}
    y_chunks: list[torch.Tensor] = []
    for sim_id in train_ids:
        row = lookup.loc[int(sim_id)]
        shard_id = int(row["shard_id"])
        local_idx = int(row["local_idx"])
        if shard_id not in shard_cache:
            shard_cache[shard_id] = load_shard_payload(str(shard_root / f"shard_{shard_id:04d}.pt"))
        y_chunks.append(shard_cache[shard_id].trajectories[local_idx].u_traj[int(FINAL_STEP_IDX)])
    return y_chunks


def encode_static_node_features(
    *,
    pos: torch.Tensor,
    geom: torch.Tensor | None,
    stats: ElPlTerminalNormStats,
    use_sdf_features: bool,
) -> torch.Tensor:
    pos_enc = encode_on_cpu(stats.pos_normalizer, pos)
    if not use_sdf_features:
        return pos_enc
    if geom is None:
        raise ValueError("geom is required when use_sdf_features=True.")
    geom_enc = encode_on_cpu(stats.geom_normalizer, geom)
    return torch.cat([pos_enc, geom_enc], dim=-1)


def x_normalizer_for_metadata(stats: ElPlTerminalNormStats, *, use_sdf_features: bool) -> NodeFeatureNormalizer:
    if use_sdf_features:
        mean = torch.cat([stats.pos_normalizer.mean, stats.geom_normalizer.mean], dim=-1)
        std = torch.cat([stats.pos_normalizer.std, stats.geom_normalizer.std], dim=-1)
    else:
        mean = stats.pos_normalizer.mean
        std = stats.pos_normalizer.std
    return NodeFeatureNormalizer(mean=mean, std=std)

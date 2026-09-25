from __future__ import annotations

import numpy as np
import torch

from pdebench.dataset.plaid_datasets import NodeFeatureNormalizer
from pdebench.dataset.plaid_elpl_terminal.constants import DEFAULT_RUNTIME_Y_NORM, FINAL_STEP_IDX
from pdebench.dataset.plaid_elpl_terminal.norm import (
    AsinhIQRChannelSpec,
    TerminalFieldNormalizer,
    encode_y_for_runtime,
    fit_terminal_field_normalizer,
    fit_terminal_norm_stats_from_trajectories,
    normalize_runtime_y_norm_mode,
)
from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj


def test_terminal_field_normalizer_round_trip() -> None:
    normalizer = TerminalFieldNormalizer(
        channels=(
            AsinhIQRChannelSpec(s=8.0, median=0.5, iqr=1.2, clip_lo=-1.0, clip_hi=25.0),
            AsinhIQRChannelSpec(s=0.2, median=0.0, iqr=0.8),
        )
    )
    y = torch.tensor([[0.0, 0.0], [20.0, -3.0], [100.0, 5.0]], dtype=torch.float32)
    encoded = normalizer.encode(y)
    decoded = normalizer.decode(encoded)
    assert torch.allclose(decoded[0], y[0], atol=1e-5)
    assert torch.allclose(decoded[1, 0], torch.tensor(20.0), atol=1e-4)
    assert torch.allclose(decoded[:, 1], y[:, 1], atol=1e-4)


def test_fit_terminal_field_normalizer_from_train_like_chunks() -> None:
    from pdebench.dataset.plaid_elpl_terminal.constants import UX_CLIP_HI, UX_CLIP_LO

    ux = np.concatenate([np.zeros(1000), np.full(1000, 20.0), np.array([200.0])])
    uy = np.random.default_rng(0).normal(0.0, 1.0, size=2001).astype(np.float32)
    y = torch.from_numpy(np.stack([ux, uy], axis=-1))
    normalizer = fit_terminal_field_normalizer([y])
    encoded = normalizer.encode(y).numpy()
    assert encoded.shape == y.shape
    assert np.isfinite(encoded).all()
    assert normalizer.channels[0].clip_lo == UX_CLIP_LO
    assert normalizer.channels[0].clip_hi == UX_CLIP_HI


def test_fit_terminal_cache_stats_use_zscore_y() -> None:
    traj = _dummy_traj(sim_id=9)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    y = traj.u_traj[int(FINAL_STEP_IDX)]
    encoded = stats.cache_y_normalizer.encode(y)
    assert encoded.shape == y.shape


def test_runtime_reencode_from_cache_matches_raw_path() -> None:
    raw = torch.tensor([[0.5, -1.0], [20.0, 3.0]], dtype=torch.float32)
    cache_norm = NodeFeatureNormalizer(
        mean=torch.tensor([[1.0, 0.0]]),
        std=torch.tensor([[2.0, 1.0]]),
    )
    runtime_norm = TerminalFieldNormalizer(
        channels=(
            AsinhIQRChannelSpec(s=5.0, median=0.0, iqr=1.0),
            AsinhIQRChannelSpec(s=1.0, median=0.0, iqr=1.0),
        )
    )
    cached = cache_norm.encode(raw)
    from_raw = encode_y_for_runtime(
        raw_y=raw,
        cache_y_normalizer=cache_norm,
        runtime_y_normalizer=runtime_norm,
    )
    from_cached = encode_y_for_runtime(
        cached_y=cached,
        cache_y_normalizer=cache_norm,
        runtime_y_normalizer=runtime_norm,
    )
    assert torch.allclose(from_raw, from_cached, atol=1e-6)


def test_default_runtime_y_norm_mode_is_asinh_iqr() -> None:
    assert DEFAULT_RUNTIME_Y_NORM == "asinh_iqr"
    assert normalize_runtime_y_norm_mode(None) == "asinh_iqr"


def test_parse_terminal_target_fields_defaults_and_ux_only() -> None:
    from pdebench.dataset.plaid_elpl_terminal.constants import (
        DEFAULT_TERMINAL_TARGET_FIELDS,
        parse_terminal_target_fields,
    )

    assert DEFAULT_TERMINAL_TARGET_FIELDS == ("U_x",)
    assert parse_terminal_target_fields(None) == ("U_x",)
    assert parse_terminal_target_fields("U_x") == ("U_x",)
    assert parse_terminal_target_fields("U_x, U_y") == ("U_x", "U_y")


def test_slice_y_normalizer_keeps_selected_channels() -> None:
    from pdebench.dataset.plaid_elpl_terminal.norm import slice_y_normalizer

    cache_norm = NodeFeatureNormalizer(
        mean=torch.tensor([[9.0, 0.0]]),
        std=torch.tensor([[9.0, 1.7]]),
    )
    sliced = slice_y_normalizer(cache_norm, (0,))
    y = torch.tensor([[18.0, 5.0], [0.0, -2.0]])
    assert sliced.encode(y[:, :1]).shape == (2, 1)
    assert torch.allclose(sliced.decode(sliced.encode(y[:, :1])), y[:, :1])

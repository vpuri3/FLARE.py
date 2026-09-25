from __future__ import annotations

import torch

import pdebench
from pdebench.dataset.ginot.forward import ginot_per_graph_channel_rel_l2
from pdebench.dataset.lpbf import load_lpbf_dataset, lpbf_legacy_y_normalizer, lpbf_y_normalizer
from pdebench.dataset.utils import compile_stats_model_for_dataset, uses_ginot_pipeline


def test_lpbf_unified_path_routing() -> None:
    # FLARE and GLT both use the standalone lpbf.py entry (not GINOT registry).
    assert uses_ginot_pipeline("lpbf", model_type=None) is False
    assert uses_ginot_pipeline("lpbf", model_type="flare") is False
    assert uses_ginot_pipeline("lpbf", model_type="glt") is False


def test_lpbf_y_normalizer_matches_legacy_hardcoded_stats() -> None:
    _, _, meta = load_lpbf_dataset()
    flare_y = meta["y_normalizer"]
    legacy_y = lpbf_legacy_y_normalizer()
    assert isinstance(flare_y, pdebench.UnitGaussianNormalizer)
    assert torch.allclose(flare_y.mean, legacy_y.mean, atol=1e-6, rtol=0)
    assert torch.allclose(flare_y.std, legacy_y.std, atol=1e-6, rtol=0)
    assert torch.allclose(flare_y.mean, lpbf_y_normalizer().mean, atol=1e-6, rtol=0)


def test_lpbf_zero_oracle_train_mean_matches_channel_rel_l2_formula() -> None:
    """Same raw targets + decode-both rel-L2 must agree dataset-wide (oracle yh=0)."""
    flare_train, _, meta = load_lpbf_dataset()
    yn = meta["y_normalizer"]
    lf = pdebench.RelL2Loss()

    flare_vals: list[float] = []
    channel_vals: list[float] = []
    for graph in flare_train:
        y = graph.y.unsqueeze(0)
        zero = torch.zeros_like(y)
        flare_vals.append(float(lf(yn.decode(zero), yn.decode(y))))
        y_nodes = y.reshape(-1, 1)
        bi = torch.zeros(y_nodes.shape[0], dtype=torch.long)
        channel_vals.append(
            float(
                ginot_per_graph_channel_rel_l2(
                    yn.decode(torch.zeros_like(y_nodes)),
                    yn.decode(y_nodes),
                    batch_index=bi,
                    num_graphs=1,
                )[0]
            )
        )

    assert len(flare_vals) == len(channel_vals) == len(flare_train)
    assert max(abs(a - b) for a, b in zip(flare_vals, channel_vals, strict=True)) < 1e-5
    flare_mean = sum(flare_vals) / len(flare_vals)
    channel_mean = sum(channel_vals) / len(channel_vals)
    assert abs(flare_mean - channel_mean) < 1e-5


def test_lpbf_uses_compiled_stats_model_when_training_is_compiled() -> None:
    assert compile_stats_model_for_dataset("lpbf", "flare", True) is True
    # GLT uses varlen graph-cache batches → eager full-batch stats
    assert compile_stats_model_for_dataset("lpbf", "glt", True) is False
    assert compile_stats_model_for_dataset("deform_plate", "glt", True) is False

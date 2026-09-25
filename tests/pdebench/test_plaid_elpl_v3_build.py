from __future__ import annotations

from unittest.mock import MagicMock

from pdebench.dataset.plaid_elpl_v3.build import build_shards
from pdebench.dataset.plaid_elpl_v3.manifest import partition_sim_ids


def test_partition_three_sims_into_two_shards() -> None:
    parts = partition_sim_ids([10, 20, 30], trajectories_per_shard=2)
    assert parts == [[10, 20], [30]]


def _fake_parse(sample_bytes, *, sample_idx, bandwidth, require_targets=True):
    del sample_bytes, bandwidth, require_targets
    from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj

    return _dummy_traj(sim_id=int(sample_idx))


def test_build_shards_writes_expected_sim_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PLAID_ELPL_PRECOMPUTE_WORKERS", "1")
    sim_ids = [10, 20, 30]
    raw = MagicMock()
    raw.__getitem__.side_effect = lambda idx: {"sample": pickle_dumps(idx)}

    monkeypatch.setattr(
        "pdebench.dataset.plaid_elpl_v3.build.parse_trajectory_bundle",
        _fake_parse,
    )

    paths = build_shards(
        raw,
        sim_ids,
        dataset_dir=tmp_path,
        split_seed=5,
        bandwidth=1.0,
        laplacian_eig_dim=0,
        trajectories_per_shard=2,
    )
    assert len(paths) == 2
    payload = __import__("torch").load(paths[0], map_location="cpu", weights_only=False)
    assert payload["sim_ids"] == [10, 20]


def pickle_dumps(idx: int) -> bytes:
    import pickle

    return pickle.dumps({"idx": idx})

from __future__ import annotations

from pdebench.dataset.plaid_elpl_v3.manifest import build_sim_to_shard, build_transition_manifest, transitions_per_sim


def test_manifest_counts_for_seed5_splits() -> None:
    train_ids = list(range(800))
    val_ids = list(range(800, 1000))
    times_by_sim = {sim_id: [0.001 * i for i in range(41)] for sim_id in train_ids + val_ids}
    train_df = build_transition_manifest(train_ids, times_by_sim=times_by_sim)
    val_df = build_transition_manifest(val_ids, times_by_sim=times_by_sim)
    assert len(train_df) == 800 * 40
    assert len(val_df) == 200 * 40
    assert transitions_per_sim(times_by_sim[0]) == 40


def test_sim_to_shard_bijection() -> None:
    sim_ids = list(range(1000))
    df = build_sim_to_shard(sim_ids, trajectories_per_shard=128)
    assert len(df) == 1000
    assert df["sim_id"].is_unique
    assert set(df["shard_id"]) <= set(range(8))

from __future__ import annotations

from pdebench.dataset.plaid_elpl_terminal.manifest import build_terminal_manifest


def test_terminal_manifest_one_row_per_sim() -> None:
    train_ids = [3, 1, 2]
    df = build_terminal_manifest(train_ids)
    assert len(df) == 3
    assert list(df["sim_id"]) == [1, 2, 3]
    assert list(df["global_idx"]) == [0, 1, 2]

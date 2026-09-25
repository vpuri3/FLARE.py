"""Unit tests for el-pl transition/terminal indexer length math (no data required)."""

from __future__ import annotations

from pdebench.dataset.plaid_elpl_terminal.manifest import terminal_length
from pdebench.dataset.plaid_elpl_v3.manifest import transition_length


def test_transition_length() -> None:
    assert transition_length(num_sims=3, num_steps=41) == 120
    assert transition_length(num_sims=3, num_steps=41) == 3 * 40


def test_terminal_length() -> None:
    assert terminal_length(num_sims=3) == 3
    assert terminal_length(num_sims=1000) == 1000

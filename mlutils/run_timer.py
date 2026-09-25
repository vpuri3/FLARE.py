"""Wall-clock timing marks for pdebench / Trainer startup diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunTimer:
    """Lightweight perf_counter timeline; rank 0 prints ``[TIMING]`` lines by default."""

    enabled: bool = True
    rank: int = 0
    log_rank: int = 0
    print_marks: bool = True
    _t0: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _last: float = field(init=False, repr=False)
    marks: list[tuple[str, float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._last = self._t0

    def mark(self, name: str, *, message: str | None = None) -> float:
        now = time.perf_counter()
        elapsed = now - self._t0
        delta = now - self._last
        self._last = now
        if self.enabled:
            self.marks.append((name, elapsed, delta))
        if self.print_marks and self.enabled and self.rank == self.log_rank:
            suffix = f" — {message}" if message else ""
            print(
                f"[TIMING] {name}: elapsed={elapsed:.3f}s delta={delta:.3f}s{suffix}",
                flush=True,
            )
        return elapsed

    def elapsed(self, name: str | None = None) -> float:
        if name is None:
            return time.perf_counter() - self._t0
        for mark_name, elapsed, _ in self.marks:
            if mark_name == name:
                return elapsed
        return time.perf_counter() - self._t0

    def summary_lines(self) -> list[str]:
        lines = ["[TIMING] summary:"]
        for name, elapsed, delta in self.marks:
            lines.append(f"  {name}: elapsed={elapsed:.3f}s delta={delta:.3f}s")
        return lines

    def print_summary(self) -> None:
        if not self.enabled or self.rank != self.log_rank:
            return
        for line in self.summary_lines():
            print(line, flush=True)


def disabled_timer() -> RunTimer:
    return RunTimer(enabled=False, print_marks=False)

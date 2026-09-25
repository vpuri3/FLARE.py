from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Optional

_EPOCH_END = object()


class OverlappedTrainBatchStream:
    """Prefetch the next training batch on a worker thread while the GPU trains."""

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._loader_iter = iter(trainer._loader)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="train_batch_prefetch",
        )
        self._future: Optional[concurrent.futures.Future] = None

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _worker_fetch(self) -> tuple[Any, float]:
        while True:
            try:
                fetch_start = time.perf_counter()
                batch = next(self._loader_iter)
                return batch, time.perf_counter() - fetch_start
            except StopIteration:
                return _EPOCH_END, 0.0

    def prefetch(self) -> None:
        if self._future is not None:
            raise RuntimeError("OverlappedTrainBatchStream.prefetch called while a fetch is already in flight.")
        self._future = self._executor.submit(self._worker_fetch)

    def initial_batch(self) -> tuple[Any, float]:
        batch, wait = self._worker_fetch()
        if batch is _EPOCH_END:
            raise RuntimeError("Training dataset produced no batches.")
        return batch, wait

    def get_prefetched(self) -> tuple[Any | None, float]:
        if self._future is None:
            raise RuntimeError("OverlappedTrainBatchStream.get_prefetched called before prefetch.")

        wait_start = time.perf_counter()
        while True:
            batch, _worker_fetch_time = self._future.result()
            self._future = None
            if batch is not _EPOCH_END:
                return batch, time.perf_counter() - wait_start

            should_continue = self.trainer._advance_train_epoch()
            if not should_continue:
                return None, time.perf_counter() - wait_start

            self._loader_iter = iter(self.trainer._loader)
            self.trainer._set_sampler_epoch(self.trainer.epoch)
            wait_start = time.perf_counter()
            self.prefetch()

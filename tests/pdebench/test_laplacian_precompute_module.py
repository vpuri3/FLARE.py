from __future__ import annotations


def test_laplacian_precompute_module_importable() -> None:
    import pdebench.dataset.laplacian.precompute as m

    assert hasattr(m, "main") or hasattr(m, "_parse_args")


def test_ginot_precompute_shim_reexports() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import pdebench.dataset.ginot.precompute as ginot_pc
        import pdebench.dataset.laplacian.precompute as lap_pc

        assert ginot_pc._parse_args is lap_pc._parse_args


def test_gpu_worker_flushes_lane_queue_feeders_before_done(monkeypatch) -> None:
    import pdebench.dataset.laplacian.precompute as precompute

    events: list[str] = []

    class LaneQueue:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(f"{self.name}:close")

        def join_thread(self) -> None:
            events.append(f"{self.name}:join")

    class ProgressQueue:
        def put(self, item: tuple) -> None:
            events.append(str(item[0]))

    monkeypatch.setattr(precompute.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(precompute.torch, "set_num_threads", lambda _threads: None)

    precompute._gpu_compute_worker(
        dataset_name="dummy",
        data_root="data",
        graph_cache_dir="cache",
        laplacian_specs=["graph:1"],
        laplacian_dim=1,
        device_idx=0,
        sample_ids=[],
        write_batch_size=8,
        lane_queues=[LaneQueue("lane0"), LaneQueue("lane1")],
        lane_queue_index={},
        progress_queue=ProgressQueue(),
        direct_writes=False,
        no_write=False,
    )

    assert events == ["lane0:close", "lane0:join", "lane1:close", "lane1:join", "done"]

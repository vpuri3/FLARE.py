from pdebench.config import RunConfig


def test_run_config_deterministic_default_false() -> None:
    assert RunConfig().deterministic is False

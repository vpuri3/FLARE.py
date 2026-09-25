from pdebench.dataset.adapters import _DrivAerAdapter, _DrivAerMLSurfaceAdapter, get_adapter


def test_drivaerml_surface_uses_dedicated_adapter_not_legacy() -> None:
    adapter = get_adapter("drivaerml_surface")

    assert isinstance(adapter, _DrivAerMLSurfaceAdapter)
    assert not isinstance(adapter, _DrivAerAdapter)


def test_legacy_drivaerml_40k_still_uses_prefix_adapter() -> None:
    adapter = get_adapter("drivaerml_40k")

    assert isinstance(adapter, _DrivAerAdapter)

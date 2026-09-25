"""Unit tests for the visualization-bundle extraction and case selection.

These cover the parts that are pure numpy, which is where the silent errors
live: coordinate de-normalization, grid-shape agreement, channel-count
agreement, and the case-selection rule.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pdebench.vis import extract, select


class _Gaussian:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)


class _Identity:
    pass


def _elasticity_sample(n=972):
    rng = np.random.default_rng(0)
    return rng.random((n, 2), dtype=np.float32)


def test_elasticity_coords_are_passed_through():
    x = _elasticity_sample()
    geom = extract.extract_geometry("elasticity", x, {"x_normalizer": _Identity()})
    assert geom["coords"].shape == (972, 2)
    assert geom["grid_shape"] is None
    np.testing.assert_allclose(geom["coords"], x)


def test_pipe_coords_are_denormalized():
    """Pipe stores x_normalizer-encoded coordinates; physical ones are wanted."""
    h, w = 129, 129
    rng = np.random.default_rng(1)
    physical = rng.random((h * w, 2), dtype=np.float32) * np.float32([8.0, 2.0])
    norm = _Gaussian(mean=[4.0, 1.0], std=[2.0, 0.5])
    encoded = (physical - norm.mean) / norm.std

    geom = extract.extract_geometry(
        "pipe", encoded, {"x_normalizer": norm, "H": h, "W": w}
    )
    np.testing.assert_allclose(geom["coords"], physical, rtol=1e-5)
    assert geom["grid_shape"] == (h, w)


def test_airfoil_grid_shape_mismatch_is_loud():
    x = np.zeros((100, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="grid shape"):
        extract.extract_geometry("airfoil_steady", x, {"x_normalizer": _Identity(), "H": 221, "W": 51})


def test_darcy_carries_the_decoded_coefficient():
    h = w = 85
    pos = np.stack(np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)), axis=-1).reshape(-1, 2)
    coeff = np.full((h * w, 1), 2.0, dtype=np.float32)
    norm = _Gaussian(mean=[3.0], std=[4.0])
    x = np.concatenate([pos.astype(np.float32), (coeff - 3.0) / 4.0], axis=-1)

    geom = extract.extract_geometry("darcy", x, {"x_normalizer": norm, "H": h, "W": w})
    np.testing.assert_allclose(geom["inputs"]["a"], 2.0, rtol=1e-5)


@pytest.mark.parametrize("dataset", ["ahmedml_surface", "drivaerml_surface"])
def test_surface_coords_are_restored_to_physical_units(dataset):
    """Per-axis min-max does not preserve aspect ratio; it must be undone."""
    rng = np.random.default_rng(2)
    lo = np.array([-0.94, -1.13, -0.32], dtype=np.float32)
    hi = np.array([4.13, 1.13, 1.24], dtype=np.float32)
    physical = rng.uniform(lo, hi, size=(500, 3)).astype(np.float32)
    unit = (physical - lo) / (hi - lo)
    normals = rng.standard_normal((500, 3)).astype(np.float32)
    x = np.concatenate([unit, normals], axis=-1)

    geom = extract.extract_geometry(
        dataset, x, {"vis_xyz_min": lo, "vis_xyz_max": hi}
    )
    np.testing.assert_allclose(geom["coords"], physical, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(geom["normals"], normals)


def test_surface_extraction_without_minmax_is_loud():
    x = np.zeros((10, 6), dtype=np.float32)
    with pytest.raises(KeyError, match="vis_xyz_min"):
        extract.extract_geometry("ahmedml_surface", x, {})


def test_bundle_rejects_a_channel_count_mismatch():
    x = _elasticity_sample(64)
    y = np.zeros((64, 2), dtype=np.float32)  # elasticity has one output channel
    with pytest.raises(ValueError, match="output channels"):
        extract.extract_bundle(
            "elasticity", x=x, y_ref=y, predictions={}, metadata={"x_normalizer": _Identity()}
        )


def test_bundle_rejects_a_prediction_shape_mismatch():
    x = _elasticity_sample(64)
    y = np.zeros((64, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="prediction shape"):
        extract.extract_bundle(
            "elasticity",
            x=x,
            y_ref=y,
            predictions={"flare": np.zeros((32, 1), dtype=np.float32)},
            metadata={"x_normalizer": _Identity()},
        )


def test_bundle_keys_and_dtypes():
    x = _elasticity_sample(64)
    y = np.ones((64, 1), dtype=np.float32)
    bundle = extract.extract_bundle(
        "elasticity",
        x=x,
        y_ref=y,
        predictions={"flare": y * 0.9, "flarepp": y * 0.95},
        metadata={"x_normalizer": _Identity()},
    )
    assert set(bundle) >= {"coords", "y_ref", "y_flare", "y_flarepp", "field_names", "dataset"}
    assert bundle["coords"].dtype == np.float32
    assert list(bundle["field_names"]) == ["sigma"]


def test_surface_metrics_match_the_callback_definition():
    """wall shear is scored on the norm of channels 1:, not componentwise."""
    rng = np.random.default_rng(3)
    target = rng.standard_normal((200, 4))
    pred = target + 0.01 * rng.standard_normal((200, 4))

    metrics = select.sample_metrics("nasa_crm", pred, target)
    expected_tau = np.linalg.norm(
        np.linalg.norm(pred[:, 1:], axis=-1) - np.linalg.norm(target[:, 1:], axis=-1)
    ) / np.linalg.norm(np.linalg.norm(target[:, 1:], axis=-1))
    assert metrics["wall_shear_rel_l2"] == pytest.approx(expected_tau)
    assert set(metrics) == {"rel_l2", "full_rel_l2", "pressure_rel_l2", "wall_shear_rel_l2"}


def test_two_dimensional_metrics_are_the_plain_relative_l2():
    rng = np.random.default_rng(4)
    target = rng.standard_normal((100, 1))
    pred = target * 1.1
    metrics = select.sample_metrics("darcy", pred, target)
    assert set(metrics) == {"rel_l2"}
    assert metrics["rel_l2"] == pytest.approx(0.1, rel=1e-6)


def test_selected_cases_are_real_samples():
    errors_ref = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    errors_other = np.array([0.6, 0.4, 0.8, 0.35, 0.2])
    cases = select.select_cases(errors_ref, errors_other)

    assert cases["max"] == 2
    assert cases["median"] == 0  # sorted: .1 .3 .5 .7 .9 -> middle is 0.5 at index 0
    assert cases["best_gain"] == 1  # 0.4 - 0.1
    assert cases["worst_gain"] == 4  # 0.2 - 0.7
    assert all(0 <= i < errors_ref.size for i in cases.values())


def test_selection_needs_matching_array_shapes():
    with pytest.raises(ValueError, match="disagree"):
        select.select_cases(np.zeros(5), np.zeros(4))


def test_reproduction_check_flags_a_mismatch():
    ok = select.check_against_reported("darcy", "flarepp", "rel_l2", 0.0059, 0.0059)
    bad = select.check_against_reported("darcy", "flarepp", "rel_l2", 0.0074, 0.0059)
    assert "OK" in ok and "MISMATCH" not in ok
    assert "MISMATCH" in bad


def test_every_dataset_declares_its_fields_and_dimension():
    for name in extract.DATASETS:
        assert extract.field_names(name)
        assert extract.ndim(name) in (2, 3)
    assert extract.is_surface("nasa_crm")
    assert not extract.is_surface("darcy")


class TestCaseRootResolution:
    """Experiment dirs may live in a collaborator's tree, not this checkout."""

    @staticmethod
    def _make(tmp_path, *rel):
        for r in rel:
            (tmp_path / r).mkdir(parents=True, exist_ok=True)
            (tmp_path / r / "config.yaml").write_text("x: 1")

    def test_roots_are_ordered_deduped_and_read_from_env(self, tmp_path, monkeypatch):
        from pdebench.vis import runner

        a, b = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("PDEBENCH_CASE_ROOT", f"{b}{os.pathsep}{a}")
        roots = runner.case_roots([str(a)])
        assert roots[:2] == [a, b]  # --case-root first, then env, no duplicate a
        assert roots[-1] == runner.CASEDIR  # this repo is always the fallback

    def test_resolves_a_unique_name(self, tmp_path):
        from pdebench.vis import runner

        self._make(tmp_path, "a/exp01", "b/exp02")
        found = runner.resolve_case_dir("exp02", [tmp_path / "a", tmp_path / "b"])
        assert found == (tmp_path / "b" / "exp02").resolve()

    def test_an_ambiguous_name_is_an_error_not_a_first_match(self, tmp_path):
        """Two roots holding one name is how the wrong checkpoint gets plotted."""
        from pdebench.vis import runner

        self._make(tmp_path, "a/exp01", "b/exp01")
        with pytest.raises(FileNotFoundError, match="ambiguous"):
            runner.resolve_case_dir("exp01", [tmp_path / "a", tmp_path / "b"])

    def test_a_full_path_bypasses_the_search(self, tmp_path):
        from pdebench.vis import runner

        self._make(tmp_path, "b/exp02")
        target = tmp_path / "b" / "exp02"
        assert runner.resolve_case_dir(str(target), [tmp_path / "a"]) == target.resolve()

    def test_a_missing_name_lists_the_roots_it_searched(self, tmp_path):
        from pdebench.vis import runner

        roots = [tmp_path / "a", tmp_path / "b"]
        with pytest.raises(FileNotFoundError) as excinfo:
            runner.resolve_case_dir("nope", roots)
        assert all(str(root) in str(excinfo.value) for root in roots)

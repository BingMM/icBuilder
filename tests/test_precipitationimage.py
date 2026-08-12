"""Focused tests for the method-specific Product-2 boundary."""

#%% Imports and small test inputs

from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import icbuilder.precipitationimage as precipitation_module
from netCDF4 import Dataset

from icbuilder.precipitationimage import (
    PrecipitationImage,
    make_regrid_mapping,
    match_sensor_times,
    regrid_to_target,
)
from icbuilder.grids import make_image_grids


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_precipitation_orbit_files.py"
SPEC = importlib.util.spec_from_file_location("make_precipitation_orbit_files", SCRIPT)
ORBIT_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORBIT_SCRIPT)


class _Grid:
    def __init__(self, xi, eta):
        self.xi = np.asarray(xi, dtype=float)
        self.eta = np.asarray(eta, dtype=float)
        self.shape = self.xi.shape
        self.lat = 70 + self.eta
        self.lon = self.xi * 15
        self.projection = SimpleNamespace(
            position=np.array([0.0, 90.0]), orientation=np.array([0.0, 1.0])
        )
        self.L = 20_000_000.0
        self.W = 20_000_000.0
        self.Lres = 225_000.0
        self.Wres = 225_000.0
        self.R = 6_481_200.0


class _Binned:
    def __init__(self, sensor, time, grid, value):
        self.sensor = sensor
        self.time = np.asarray(time, dtype=object)
        shape = (len(time), *grid.shape)
        self.mu = np.full(shape, value, dtype=float)
        self.sigma = np.full(shape, 2.0, dtype=float)
        self.w = np.full(shape, 0.5, dtype=float)
        self.ssalon = np.arange(len(time), dtype=float) + 10
        self.grid = grid
        self.shape = shape
        self.correction = "SH" if sensor == "WIC" else "DG"
        self.los_correction = False
        self.counts = np.full(shape, 7, dtype=np.int32)


TARGET_GRID = _Grid([[0.25]], [[0.25]])
SI_GRID = _Grid([[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]])
SI_TO_TARGET = make_regrid_mapping(SI_GRID, TARGET_GRID)
BASE_TIME = datetime(2000, 1, 1, 1)


def _kp_series():
    return {
        "time": np.array(["2000-01-01T00:00:00"], dtype="datetime64[s]"),
        "kp": np.array([2.0]),
        "status": np.array(["def"]),
        "provenance": {"source": "test", "status": "def"},
    }


def _physics_result(inputs, ratio=False):
    shape = inputs["wic_corrected"].shape
    result = {
        "E0": np.full(shape, 2.5),
        "dE0": np.full(shape, 0.4),
        "Fe": np.full(shape, 3.0),
        "dFe": np.full(shape, 0.5),
        "varE0Fe": np.full(shape, -0.1),
    }
    if ratio:
        result.update({
            "R": np.full(shape, 4.0),
            "dR": np.full(shape, 0.6),
        })
    return result


def _zhang_paxton_stub(**inputs):
    assert set(("wic_corrected", "dwic_corrected")) <= set(inputs)
    assert "si13_corrected" not in inputs
    assert set(("kp", "mlt")) <= set(inputs)
    return _physics_result(inputs)


def _image_ratio_stub(**inputs):
    assert set((
        "wic_corrected", "dwic_corrected",
        "si13_corrected", "dsi13_corrected",
    )) <= set(inputs)
    assert "kp" not in inputs
    return _physics_result(inputs, ratio=True)


def _binned_inputs():
    wic = _Binned("WIC", [BASE_TIME], TARGET_GRID, 10.0)
    si12 = _Binned("SI12", [BASE_TIME + timedelta(seconds=1)], SI_GRID, 4.0)
    si13 = _Binned("SI13", [BASE_TIME + timedelta(seconds=2)], SI_GRID, 6.0)
    return wic, si12, si13


#%% Time support and interpolation

def test_time_matching_preserves_the_existing_two_second_rule():
    first = [BASE_TIME, BASE_TIME + timedelta(seconds=10)]
    second = [BASE_TIME + timedelta(seconds=2), BASE_TIME + timedelta(seconds=13)]

    time, indices = match_sensor_times(first, second)

    np.testing.assert_array_equal(time, [BASE_TIME + timedelta(seconds=2)])
    np.testing.assert_array_equal(indices, [[0, 0]])


def test_regrid_propagates_variance_with_squared_weights():
    values = np.array([[[0.0, 4.0], [8.0, 12.0]]])
    sigma = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    target = regrid_to_target(values, SI_TO_TARGET)
    target_sigma = regrid_to_target(
        sigma, SI_TO_TARGET, propagate_uncertainty=True
    )

    # Bilinear weights at (0.25, 0.25) are 0.5625, 0.1875, 0.1875, 0.0625.
    np.testing.assert_allclose(target, [[[3.0]]])
    expected_sigma = np.sqrt(
        0.5625**2 * 1**2 + 0.1875**2 * 2**2 +
        0.1875**2 * 3**2 + 0.0625**2 * 4**2
    )
    np.testing.assert_allclose(target_sigma, [[[expected_sigma]]])


def test_regrid_uses_nearest_cell_at_physical_boundary_only():
    boundary_grid = _Grid(
        [[-0.25, 0.25, 1.25, 1.75]],
        [[0.25, 0.25, 0.25, 0.25]],
    )
    mapping = make_regrid_mapping(SI_GRID, boundary_grid)
    values = np.array([[[0.0, 4.0], [8.0, 12.0]]])
    sigma = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    target = regrid_to_target(values, mapping)
    target_sigma = regrid_to_target(
        sigma, mapping, propagate_uncertainty=True
    )

    # -0.25 and 1.25 lie in the outer half of the SI boundary cells.
    np.testing.assert_allclose(target[0, 0, :3], [0.0, 3.0, 4.0])
    assert target_sigma[0, 0, 0] == 1.0
    assert target_sigma[0, 0, 2] == 2.0

    # 1.75 lies beyond the physical SI grid edge at 1.5.
    assert np.isnan(target[0, 0, 3])
    assert np.isnan(target_sigma[0, 0, 3])


def test_regrid_does_not_extrapolate_beyond_physical_source_grid():
    outside_grid = _Grid([[2.0]], [[2.0]])
    outside_mapping = make_regrid_mapping(SI_GRID, outside_grid)
    target = regrid_to_target(np.ones((1, 2, 2)), outside_mapping)
    target_sigma = regrid_to_target(
        np.ones((1, 2, 2)), outside_mapping, propagate_uncertainty=True
    )

    assert np.isnan(target).all()
    assert np.isnan(target_sigma).all()


def test_canonical_si_grid_covers_the_complete_wic_grid():
    wic_grid, si_grid = make_image_grids()
    mapping = make_regrid_mapping(si_grid, wic_grid)
    target = regrid_to_target(np.ones((1, *si_grid.shape)), mapping)

    assert len(mapping["target_indices"]) == 1156
    assert len(mapping["boundary_indices"]) == 140
    assert np.isfinite(target).all()


def test_regrid_treats_each_source_array_independently():
    values = np.array([[[0.0, 4.0], [8.0, np.nan]]])
    sigma = np.ones_like(values)

    target = regrid_to_target(values, SI_TO_TARGET)
    target_sigma = regrid_to_target(
        sigma, SI_TO_TARGET, propagate_uncertainty=True
    )

    assert np.isnan(target).all()
    assert np.isfinite(target_sigma).all()


#%% Method boundary

def test_default_physics_function_comes_from_icphysics(monkeypatch):
    wic, si12, _ = _binned_inputs()
    monkeypatch.setattr(
        precipitation_module,
        "precipitation_from_zhang_paxton",
        _zhang_paxton_stub,
    )

    image = PrecipitationImage(
        wic, si12, "zhang_paxton", kp_series=_kp_series()
    )
    assert image.physics_provenance["function"] == "_zhang_paxton_stub"


def test_filenames_and_default_kp_are_loaded(monkeypatch):
    wic, si12, si13 = _binned_inputs()
    files = {
        "wic.nc": wic,
        "si12.nc": si12,
        "si13.nc": si13,
    }
    monkeypatch.setattr(
        precipitation_module, "icload", lambda filename: files[str(filename)]
    )
    monkeypatch.setattr(precipitation_module, "load_gfz_kp", _kp_series)

    image = PrecipitationImage(
        "wic.nc",
        "si12.nc",
        "image_ratio",
        _image_ratio_stub,
        si13="si13.nc",
    )

    assert image.source_products == {
        "wic": "wic.nc",
        "si12": "si12.nc",
        "si13": "si13.nc",
    }
    np.testing.assert_allclose(image.kp, [2.0])


def test_method_specific_sensor_requirements():
    wic, si12, _ = _binned_inputs()

    with pytest.raises(ValueError, match="requires a SI13"):
        PrecipitationImage(
            wic, si12, "image_ratio", _image_ratio_stub,
            kp_series=_kp_series(),
        )


def test_zhang_paxton_uses_two_sensor_time_support_without_mutating_inputs():
    wic, si12, _ = _binned_inputs()
    original_si12 = si12.mu.copy()

    precipitation = PrecipitationImage(
        wic, si12, "zhang_paxton", _zhang_paxton_stub,
        kp_series=_kp_series(),
    )

    assert precipitation.time.tolist() == [BASE_TIME + timedelta(seconds=1)]
    assert set(precipitation.source_indices) == {"wic", "si12", "si13"}
    np.testing.assert_array_equal(precipitation.source_indices["si13"], [-1])
    assert np.isnan(precipitation.si13).all()
    assert np.isnan(precipitation.si13_corrected).all()
    np.testing.assert_allclose(
        precipitation.w,
        precipitation.wic_weight * precipitation.si12_weight,
    )
    np.testing.assert_array_equal(si12.mu, original_si12)
    np.testing.assert_array_equal(si12.counts, 7)
    assert not hasattr(precipitation, "counts")


def test_image_ratio_routes_all_three_prepared_sensors():
    wic, si12, si13 = _binned_inputs()

    precipitation = PrecipitationImage(
        wic, si12, "image_ratio", _image_ratio_stub,
        si13=si13, kp_series=_kp_series(),
    )

    assert precipitation.time.tolist() == [BASE_TIME + timedelta(seconds=2)]
    assert set(precipitation.source_indices) == {"wic", "si12", "si13"}
    np.testing.assert_allclose(precipitation.R, 4.0)
    assert precipitation.wic_weight.shape == precipitation.shape
    assert precipitation.si12_weight.shape == precipitation.shape
    assert precipitation.si13_weight.shape == precipitation.shape
    np.testing.assert_allclose(
        precipitation.w,
        precipitation.wic_weight * precipitation.si12_weight
        * precipitation.si13_weight,
    )


def test_zhang_paxton_attaches_si13_without_reducing_time_support():
    wic = _Binned(
        "WIC", [BASE_TIME, BASE_TIME + timedelta(seconds=10)], TARGET_GRID, 10.0
    )
    si12 = _Binned(
        "SI12",
        [BASE_TIME + timedelta(seconds=1), BASE_TIME + timedelta(seconds=11)],
        SI_GRID,
        4.0,
    )
    si13 = _Binned(
        "SI13", [BASE_TIME + timedelta(seconds=2)], SI_GRID, 6.0
    )

    precipitation = PrecipitationImage(
        wic, si12, "zhang_paxton", _zhang_paxton_stub,
        si13=si13, kp_series=_kp_series(),
    )

    assert precipitation.time.size == 2
    np.testing.assert_array_equal(precipitation.source_indices["si13"], [0, -1])
    assert np.isfinite(precipitation.si13[0]).all()
    assert np.isnan(precipitation.si13[1]).all()


def test_physics_results_may_not_contain_conductance_or_counts():
    wic, si12, _ = _binned_inputs()

    def invalid_physics(**inputs):
        result = _physics_result(inputs)
        result["P"] = np.ones(inputs["wic_corrected"].shape)
        return result

    with pytest.raises(ValueError, match="must not return P"):
        PrecipitationImage(
            wic, si12, "zhang_paxton", invalid_physics,
            kp_series=_kp_series(),
        )


#%% Product-2 NetCDF

def test_precipitation_netcdf_is_method_specific_and_self_describing(tmp_path):
    wic, si12, _ = _binned_inputs()
    precipitation = PrecipitationImage(
        wic, si12, "zhang_paxton", _zhang_paxton_stub,
        kp_series=_kp_series(),
        proton_energy=5.0,
        proton_energy_uncertainty=0.5,
        source_products={"wic": "binned/wic/or_0001.nc", "si12": "binned/si12/or_0001.nc"},
        physics_provenance={"version": "test"},
    )
    output = tmp_path / "precipitation.nc"

    precipitation.to_nc(output)

    assert ORBIT_SCRIPT.precipitation_file_status(
        output, "zhang_paxton", "SI12", 5.0, 0.5
    ) == "complete"
    assert ORBIT_SCRIPT.precipitation_file_status(
        output, "zhang_paxton", "SI12", 2.0, 0.0
    ) == "mismatch"

    with Dataset(output) as nc:
        assert nc.product_type == "precipitation"
        assert nc.schema_version == 1
        assert nc.method == "zhang_paxton"
        assert nc.proton_method == "SI12"
        assert nc.proton_energy == 5.0
        assert nc.proton_energy_uncertainty == 0.5
        assert nc.time_match_tolerance_seconds == 2
        assert "independent source-cell errors" in nc.regrid_uncertainty
        assert nc.physics_function == "_zhang_paxton_stub"
        assert nc.physics_version == "test"
        assert nc.source_wic == "binned/wic/or_0001.nc"

        expected = {
            "time", "wic_source_index", "si12_source_index", "si13_source_index", "Kp",
            "Kp_interval_start", "ssalon", "wic", "dwic", "si12",
            "dsi12", "si13", "dsi13", "wic_weight", "si12_weight",
            "si13_weight", "w", "wic_corrected", "dwic_corrected",
            "si13_corrected", "dsi13_corrected", "E0", "dE0", "Fe",
            "dFe", "varE0Fe",
        }
        assert set(nc.variables) == expected
        for forbidden in ("counts", "P", "H", "dP", "dH"):
            assert forbidden not in nc.variables

        assert set(nc.groups["grid"].variables) == {"xi", "eta", "mlat", "mlt"}
        np.testing.assert_allclose(nc.groups["grid"]["xi"][:], TARGET_GRID.xi)
        np.testing.assert_allclose(nc.groups["grid"]["mlt"][:], TARGET_GRID.lon / 15)


def test_restart_rejects_an_old_regridding_rule(tmp_path):
    wic, si12, _ = _binned_inputs()
    precipitation = PrecipitationImage(
        wic, si12, "zhang_paxton", _zhang_paxton_stub,
        kp_series=_kp_series(),
    )
    output = tmp_path / "precipitation.nc"
    precipitation.to_nc(output)

    with Dataset(output, "r+") as nc:
        nc.regrid_method = "old centre-only interpolation"

    assert ORBIT_SCRIPT.precipitation_file_status(
        output, "zhang_paxton", "SI12", 2.0, 0.0
    ) == "mismatch"

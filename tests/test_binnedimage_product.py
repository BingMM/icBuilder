"""Product-1 API and NetCDF tests for ``PreImage`` and ``BinnedImage``."""

import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset, num2date

from icbuilder.binnedimage import BinnedImage
from icbuilder.preimage import PreImage


SCRIPT = Path(__file__).parents[1] / "scripts" / "pipeline" / "make_binned_orbit_files.py"
SPEC = importlib.util.spec_from_file_location("make_binned_orbit_files", SCRIPT)
ORBIT_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORBIT_SCRIPT)

PRECIPITATION_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "pipeline"
    / "make_precipitation_orbit_files.py"
)
PRECIPITATION_SPEC = importlib.util.spec_from_file_location(
    "make_precipitation_orbit_files", PRECIPITATION_SCRIPT
)
PRECIPITATION_ORBIT_SCRIPT = importlib.util.module_from_spec(PRECIPITATION_SPEC)
PRECIPITATION_SPEC.loader.exec_module(PRECIPITATION_ORBIT_SCRIPT)


class _OneCellGrid:
    """Small native sensor grid with the metadata written to Product 1."""

    shape = (1, 1)
    size = 1
    xi = np.array([[0.0]])
    eta = np.array([[0.0]])
    lat = np.array([[72.0]])
    lon = np.array([[-15.0]])
    projection = SimpleNamespace(
        position=np.array([18.0, 90.0]), orientation=0.0
    )
    L = 225000.0
    W = 225000.0
    Lres = 225000.0
    Wres = 225000.0
    R = 6481.2e3

    def bin_index(self, lon, lat):
        return np.zeros(lon.size, dtype=int), np.zeros(lat.size, dtype=int)


class _PreImage:
    """One WIC frame with all fields needed by ``BinnedImage``."""

    sensor = "WIC"
    shape = (1, 1, 3)
    ssalon = np.array([12.5])
    image = np.array([[[10.0, 20.0, 30.0]]])
    weight = np.ones_like(image)
    sza = np.array([[[70.0, 80.0, 90.0]]])
    dza = np.array([[[0.0, 30.0, 60.0]]])

    def get_mcoords(self, index):
        shape = self.image[index].shape
        return (
            np.full(shape, 70.0), np.zeros(shape), np.zeros(shape),
            self.ssalon[index],
        )

    def get_shimg(self, index):
        return self.image[index]

    def get_shimg_los(self, index):
        return self.image[index] * np.cos(np.radians(self.dza[index]))

    def get_dgw(self, index):
        return self.weight[index]

    def get_shw(self, index):
        return self.weight[index]

    def get_SZA(self, index):
        return self.sza[index]

    def get_DZA(self, index):
        return self.dza[index]


def _write_preimage_input(path):
    """Write the smallest input file accepted by ``PreImage``."""
    names = [
        "mlat", "mlon", "mlt", "glat", "glon", "img", "dgimg",
        "shimg", "dgmodel", "shweight", "dgweight", "sza", "dza",
    ]
    with Dataset(path, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("y", 1)
        nc.createDimension("x", 1)
        for name in names:
            variable = nc.createVariable(name, "f8", ("time", "y", "x"))
            variable[:] = 1.0


def test_preimage_validates_and_canonicalizes_sensor(tmp_path):
    input_path = tmp_path / "input.nc"
    _write_preimage_input(input_path)

    with Dataset(input_path) as nc:
        assert PreImage("wic", nc).sensor == "WIC"
        with pytest.raises(ValueError, match="WIC.*SI12.*SI13"):
            PreImage("unknown", nc)


def test_binnedimage_requires_one_datetime_per_frame():
    with pytest.raises(ValueError, match="one timestamp per image frame"):
        BinnedImage(_PreImage(), _OneCellGrid(), [])


def test_binnedimage_product_round_trip_schema(tmp_path):
    time = datetime(2001, 2, 3, 4, 5, 6)
    binned = BinnedImage(
        _PreImage(), _OneCellGrid(), [time], correction="SH",
        los_correction=False, binning_method="centre",
    )
    output = tmp_path / "wic.nc"
    binned.to_nc(output)

    with Dataset(output) as nc:
        assert nc.product_type == "binned_fuv"
        assert nc.schema_version == 1
        assert nc.sensor == "WIC"
        assert nc.image_correction == "SH"
        assert nc.los_correction == 0
        assert nc.binning_method == "centre"
        assert set(nc.variables) == {
            "counts", "mu", "sigma", "w", "sza", "dza", "los_factor",
            "coverage", "time", "ssalon",
        }
        assert "Kp" not in nc.variables
        for name in ("E0", "Fe", "P", "H", "Ep", "dEp"):
            assert name not in nc.variables
            assert name not in nc.ncattrs()

        np.testing.assert_array_equal(nc.variables["counts"][:], binned.counts)
        assert nc.variables["counts"].dtype == np.dtype("int32")
        np.testing.assert_allclose(nc.variables["mu"][:], binned.mu)
        np.testing.assert_allclose(nc.variables["sigma"][:], binned.sigma)
        np.testing.assert_allclose(nc.variables["w"][:], binned.w)
        assert np.all(np.isnan(nc.variables["coverage"][:]))
        np.testing.assert_allclose(nc.variables["ssalon"][:], binned.ssalon)

        time_variable = nc.variables["time"]
        decoded = num2date(
            time_variable[:], time_variable.units, time_variable.calendar,
            only_use_cftime_datetimes=False,
        )[0]
        assert decoded == time
        assert time_variable.time_zone == "UTC"

        grid = nc.groups["grid"]
        np.testing.assert_allclose(grid.position, [18.0, 90.0])
        assert grid.orientation == 0.0
        assert grid.Lres == 225000.0
        assert grid.Wres == 225000.0
        assert set(grid.variables) == {"xi", "eta", "mlat", "mlt"}
        for variable in grid.variables.values():
            assert variable.shape == binned.shape[1:]
        np.testing.assert_allclose(grid.variables["xi"][:], binned.grid.xi)
        np.testing.assert_allclose(grid.variables["eta"][:], binned.grid.eta)
        np.testing.assert_allclose(grid.variables["mlat"][:], binned.grid.lat)
        np.testing.assert_allclose(grid.variables["mlt"][:], [[23.0]])
        assert grid.variables["xi"].units == "radians"
        assert grid.variables["eta"].units == "radians"
        assert grid.variables["mlat"].units == "degrees"
        assert grid.variables["mlt"].units == "hours"

    assert binned.counts.dtype == np.int32
    for name in (
        "counts", "mu", "sigma", "w", "sza", "dza", "los_factor",
        "coverage",
    ):
        assert getattr(binned, name).shape == binned.shape


def test_atomic_save_publishes_a_complete_product(tmp_path):
    binned = BinnedImage(
        _PreImage(), _OneCellGrid(), [datetime(2001, 1, 1)],
        correction="SH", los_correction=False, binning_method="centre",
    )
    output = tmp_path / "or_0001.nc"

    ORBIT_SCRIPT.save_binned_file(binned, output)

    with Dataset(output) as nc:
        assert nc.product_type == "binned_fuv"
        assert nc.sensor == "WIC"
    assert ORBIT_SCRIPT.binned_file_status(
        output, "WIC", "SH", False, "centre"
    ) == "complete"
    assert ORBIT_SCRIPT.binned_file_status(
        output, "WIC", "DG", False, "centre"
    ) == "mismatch"
    assert not Path(str(output) + ".partial").exists()


@pytest.mark.parametrize(
    "script", [ORBIT_SCRIPT, PRECIPITATION_ORBIT_SCRIPT]
)
def test_orbits_are_discovered_directly_from_netcdf_files(tmp_path, script):
    for name in ("wic_or0002.nc", "or_0010.nc", "not_an_orbit.nc"):
        (tmp_path / name).touch()

    np.testing.assert_array_equal(script.get_orbits(tmp_path), [2, 10])


def test_precipitation_script_defaults_follow_the_product_layout():
    args = PRECIPITATION_ORBIT_SCRIPT.parse_args([])

    assert args.wic_folder == "binned/wic"
    assert args.s12_folder == "binned/si12"
    assert args.s13_folder == "binned/si13"
    assert args.output_folder == "precipitation"
    assert args.precipitation_method == "zhang_paxton"


def test_binned_script_defaults_to_footprint_binning():
    args = ORBIT_SCRIPT.parse_args([])

    assert args.binning_method == "footprint"


def test_precipitation_script_exposes_proton_configuration():
    args = PRECIPITATION_ORBIT_SCRIPT.parse_args([
        "--precipitation-method", "image_ratio",
        "--proton-energy-model", "constant",
        "--proton-energy", "5",
        "--proton-energy-uncertainty", "0.5",
    ])

    assert args.precipitation_method == "image_ratio"
    assert args.proton_energy_model == "constant"
    assert args.proton_energy == 5.0
    assert args.proton_energy_uncertainty == 0.5


def test_precipitation_script_uses_method_specific_sensor_support(
        tmp_path, monkeypatch):
    for sensor in ("wic", "si12", "si13"):
        (tmp_path / "binned" / sensor).mkdir(parents=True)

    for orbit in (1, 2, 3):
        (tmp_path / "binned" / "wic" / f"or_{orbit:04d}.nc").touch()
    for orbit in (2, 3):
        (tmp_path / "binned" / "si12" / f"or_{orbit:04d}.nc").touch()
    (tmp_path / "binned" / "si13" / "or_0003.nc").touch()

    calls = []
    kp_series = {"source": "test"}
    monkeypatch.setattr(
        PRECIPITATION_ORBIT_SCRIPT, "load_gfz_kp", lambda: kp_series
    )

    def record_orbit(
        orbit, input_paths, output_dir, kp_series, method, **settings
    ):
        calls.append((orbit, output_dir, kp_series, method))
        return orbit

    monkeypatch.setattr(
        PRECIPITATION_ORBIT_SCRIPT, "process_orbit", record_orbit
    )

    PRECIPITATION_ORBIT_SCRIPT.main(["--base", str(tmp_path)])
    assert calls == [
        (2, tmp_path / "precipitation", kp_series, "zhang_paxton"),
        (3, tmp_path / "precipitation", kp_series, "zhang_paxton"),
    ]

    calls.clear()
    PRECIPITATION_ORBIT_SCRIPT.main([
        "--base", str(tmp_path),
        "--precipitation_method", "image_ratio",
        "--output-folder", "ratio_P2",
    ])
    assert calls == [(
        3, tmp_path / "ratio_P2",
        kp_series, "image_ratio",
    )]


def test_precipitation_overwrite_recomputes_existing_orbits(tmp_path, monkeypatch):
    for sensor in ("wic", "si12", "si13"):
        sensor_dir = tmp_path / "binned" / sensor
        sensor_dir.mkdir(parents=True)
        (sensor_dir / "or_0001.nc").touch()

    output_dir = tmp_path / "ratio_P2"
    output_dir.mkdir(parents=True)
    (output_dir / "or_0001.nc").touch()

    calls = []
    monkeypatch.setattr(
        PRECIPITATION_ORBIT_SCRIPT, "load_gfz_kp", lambda: {}
    )
    monkeypatch.setattr(
        PRECIPITATION_ORBIT_SCRIPT,
        "process_orbit",
        lambda orbit, **kwargs: calls.append(orbit),
    )
    monkeypatch.setattr(
        PRECIPITATION_ORBIT_SCRIPT,
        "precipitation_file_status",
        lambda *args: "complete",
    )

    result = PRECIPITATION_ORBIT_SCRIPT.main([
        "--base", str(tmp_path),
        "--precipitation_method", "image_ratio",
        "--output-folder", "ratio_P2",
    ])
    assert result == []
    assert calls == []

    PRECIPITATION_ORBIT_SCRIPT.main([
        "--base", str(tmp_path),
        "--precipitation_method", "image_ratio",
        "--output-folder", "ratio_P2",
        "--overwrite",
    ])
    assert calls == [1]

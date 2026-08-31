"""Restart checks for the modular conductance-orbit script."""

import importlib.util
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "pipeline"
    / "make_conductance_orbit_files.py"
)
SPEC = importlib.util.spec_from_file_location("make_conductance_orbit_files", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_valid_file(path, omit=None):
    with Dataset(path, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("dim1", 1)
        nc.createDimension("dim2", 1)
        nc.product_type = "conductance"
        nc.schema_version = 2
        nc.precipitation_method = "zhang_paxton"
        nc.proton_flux_source = "SI12"
        nc.proton_energy_model = "constant"
        nc.proton_energy_constant = 2.0
        nc.proton_energy_uncertainty_constant = 0.0
        nc.conductance_model = "robinson"
        for name in MODULE.REQUIRED_FIELDS:
            if name != omit:
                nc.createVariable(name, "f4", ("time", "dim1", "dim2"))
        for name in ("time", "Kp", "Kp_interval_start", "ssalon"):
            if name != omit:
                nc.createVariable(name, "f4", ("time",))
        nc.createGroup("grid")


def test_completion_check_and_atomic_save(tmp_path):
    valid = tmp_path / "valid.nc"
    invalid = tmp_path / "invalid.nc"
    write_valid_file(valid)
    write_valid_file(invalid, omit="dH")

    assert MODULE.conductance_file_is_complete(valid)
    assert not MODULE.conductance_file_is_complete(invalid)

    final = tmp_path / "or_0001.nc"

    class Conductance:
        shape = (1, 1, 1)

        def to_nc(self, filename):
            write_valid_file(filename)

    MODULE.save_conductance_file(Conductance(), final)
    assert MODULE.conductance_file_is_complete(final)
    assert not Path(str(final) + ".partial").exists()


def test_orbits_are_discovered_from_precipitation_directory(tmp_path):
    for orbit in (8, 3):
        (tmp_path / f"or_{orbit:04d}.nc").touch()
    (tmp_path / "notes.nc").touch()

    assert MODULE.get_orbits(tmp_path) == [3, 8]


def test_existing_conductance_must_match_precipitation_configuration(tmp_path):
    conductance = tmp_path / "conductance.nc"
    precipitation = tmp_path / "precipitation.nc"
    write_valid_file(conductance)

    with Dataset(precipitation, "w") as nc:
        nc.method = "zhang_paxton"
        nc.proton_flux_source = "SI12"
        nc.proton_energy_model = "constant"
        nc.proton_energy_constant = 2.0
        nc.proton_energy_uncertainty_constant = 0.0

    assert MODULE.conductance_matches_precipitation(
        conductance, precipitation
    )

    with Dataset(precipitation, "a") as nc:
        nc.proton_energy_constant = 5.0

    assert not MODULE.conductance_matches_precipitation(
        conductance, precipitation
    )

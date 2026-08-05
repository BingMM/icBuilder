import importlib.util
from pathlib import Path

from netCDF4 import Dataset
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_conductance_orbit_files.py"
SPEC = importlib.util.spec_from_file_location("make_conductance_orbit_files", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

orbit_file_is_complete = MODULE.orbit_file_is_complete
save_orbit_file = MODULE.save_orbit_file
select_orbits_to_process = MODULE.select_orbits_to_process


IMAGE_VARIABLES = [
    "wic_avg", "s12_avg", "s13_avg", "wic_std", "s12_std", "s13_std",
    "E0", "dE0", "Fe", "dFe", "varE0Fe", "R", "dR",
    "P", "H", "dP", "dH", "w",
    "wic_sza", "wic_dza", "wic_los_factor",
    "s12_sza", "s12_dza", "s12_los_factor",
    "s13_sza", "s13_dza", "s13_los_factor",
]


def write_valid_orbit(path, energy_method="zhang_paxton", omit=None):
    """Write the smallest NetCDF that has the current orbit-file structure."""
    with Dataset(path, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("dim1", 36)
        nc.createDimension("dim2", 36)
        nc.electron_energy_method = energy_method

        for name in IMAGE_VARIABLES:
            if name != omit:
                nc.createVariable(name, "f4", ("time", "dim1", "dim2"))

        for name in ["time", "Kp", "Kp_lookup", "Kp_interval_start", "ssalon"]:
            if name != omit:
                nc.createVariable(name, "f4", ("time",))

        grid = nc.createGroup("grid")
        for name in ["position", "orientation", "L", "W", "Lres", "Wres", "R"]:
            grid.setncattr(name, 1)


def test_complete_orbit_file_validation(tmp_path):
    valid = tmp_path / "valid.nc"
    missing_variable = tmp_path / "missing.nc"
    wrong_method = tmp_path / "wrong_method.nc"
    corrupt = tmp_path / "corrupt.nc"

    write_valid_orbit(valid)
    write_valid_orbit(missing_variable, omit="Kp")
    write_valid_orbit(wrong_method, energy_method="image_ratio")
    corrupt.write_text("not a NetCDF file")

    assert orbit_file_is_complete(valid)
    assert not orbit_file_is_complete(missing_variable)
    assert not orbit_file_is_complete(wrong_method)
    assert not orbit_file_is_complete(corrupt)


def test_resume_selects_missing_and_invalid_orbits(tmp_path):
    write_valid_orbit(tmp_path / "or_0001.nc")
    (tmp_path / "or_0002.nc").write_text("partial output")

    assert select_orbits_to_process([3, 1, 2], tmp_path) == [2, 3]
    assert select_orbits_to_process([3, 1, 2], tmp_path, overwrite=True) == [1, 2, 3]


def test_atomic_save_exposes_only_valid_final(tmp_path):
    final_path = tmp_path / "or_0001.nc"

    class Conductance:
        def to_nc(self, path):
            write_valid_orbit(path)

    save_orbit_file(Conductance(), final_path)

    assert orbit_file_is_complete(final_path)
    assert not Path(str(final_path) + ".partial").exists()


def test_failed_save_preserves_existing_final(tmp_path):
    final_path = tmp_path / "or_0001.nc"
    original = b"existing complete output"
    final_path.write_bytes(original)

    class Conductance:
        def to_nc(self, path):
            Path(path).write_bytes(b"partial replacement")
            raise ValueError("simulated write failure")

    with pytest.raises(ValueError, match="simulated write failure"):
        save_orbit_file(Conductance(), final_path)

    assert final_path.read_bytes() == original
    assert not Path(str(final_path) + ".partial").exists()

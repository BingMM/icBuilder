"""Tests for viewing geometry preserved by ``ConductanceImage``."""

import numpy as np
from netCDF4 import Dataset

from icbuilder.conductanceimage import ConductanceImage


class _Projection:
    position = np.array([0.0, 90.0])
    orientation = np.array([0.0, 1.0])


class _Grid:
    projection = _Projection()
    L = 1.0
    W = 1.0
    Lres = 1.0
    Wres = 1.0
    R = 1.0


class _BinnedImage:
    """Small sensor-specific binned image for transfer and NetCDF tests."""

    shape = (1, 1, 1)
    ssalon = np.array([12.0])
    grid = _Grid()

    def __init__(self, value, correction, los_correction):
        self.mu = np.full(self.shape, value)
        self.sigma = np.full(self.shape, 0.1)
        self.w = np.ones(self.shape)
        self.sza = np.full(self.shape, value + 10.0)
        self.dza = np.full(self.shape, value + 20.0)
        self.los_factor = np.full(self.shape, value / 10.0)
        self.correction = correction
        self.los_correction = los_correction


def _skip_conductance_calculation(conductance_image):
    """Avoid unrelated physical inversion in this metadata-focused test."""

    conductance_image.w = np.ones(conductance_image.shape)


def test_geometry_survives_conductance_image_and_netcdf(
    monkeypatch,
    tmp_path,
):
    """All three sensor geometries and processing flags reach the file."""

    monkeypatch.setattr(
        ConductanceImage,
        "_compute_conductance",
        _skip_conductance_calculation,
    )

    wic = _BinnedImage(1.0, "SH", True)
    s12 = _BinnedImage(2.0, "DG", True)
    s13 = _BinnedImage(3.0, "DG", False)
    conductance = ConductanceImage(wic, s12, s13)

    assert conductance.wic_sza[0, 0, 0] == 11.0
    assert conductance.s12_dza[0, 0, 0] == 22.0
    assert conductance.s13_los_factor[0, 0, 0] == 0.3
    assert conductance.wic_los_correction is True
    assert conductance.s13_los_correction is False
    assert conductance.wic_image_correction == "SH"

    # ConductanceImage owns a copy rather than an alias to the BinnedImage.
    wic.sza[0, 0, 0] = -1.0
    assert conductance.wic_sza[0, 0, 0] == 11.0

    output = tmp_path / "conductance_geometry.nc"
    conductance.to_nc(output)

    with Dataset(output) as nc:
        for sensor in ("wic", "s12", "s13"):
            assert f"{sensor}_sza" in nc.variables
            assert f"{sensor}_dza" in nc.variables
            assert f"{sensor}_los_factor" in nc.variables
            assert nc.variables[f"{sensor}_sza"].units == "degrees"
            assert nc.variables[f"{sensor}_dza"].units == "degrees"
            assert nc.variables[f"{sensor}_los_factor"].units == "1"

        assert nc.variables["wic_sza"][0, 0, 0] == 11.0
        assert nc.variables["s12_dza"][0, 0, 0] == 22.0
        assert np.isclose(
            nc.variables["s13_los_factor"][0, 0, 0],
            0.3,
        )
        assert nc.wic_los_correction == 1
        assert nc.s13_los_correction == 0
        assert nc.wic_image_correction == "SH"
        assert nc.s12_image_correction == "DG"

"""Small end-to-end tests for the ConductanceImage E0 override."""

from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

import icbuilder.conductanceimage as conductance_module
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
    xi = np.array([[0.0]])
    eta = np.array([[0.0]])


class _BinnedImage:
    shape = (1, 1, 1)
    ssalon = np.array([12.0])
    grid = _Grid()

    def __init__(self, value):
        self.mu = np.full(self.shape, value)
        self.sigma = np.full(self.shape, 0.1)
        self.w = np.ones(self.shape)
        self.sza = np.full(self.shape, 70.0)
        self.dza = np.full(self.shape, 20.0)
        self.los_factor = np.full(self.shape, np.cos(np.radians(20.0)))
        self.correction = "test"
        self.los_correction = False


def _kp_provenance():
    return {
        "source": "GFZ Potsdam",
        "status": "def",
        "doi": "10.5880/Kp.0001",
        "licence": "CC BY 4.0",
        "query": "test query",
        "acquired": "2026-07-30",
        "sha256": "test checksum",
    }


def _lookup(kp):
    assert np.asarray(kp).shape == (1,)
    return {
        "kp": np.array([1.52]),
        "E0": np.full((1, 1, 1), 2.5),
        "dE0": np.full((1, 1, 1), 0.4),
        "E0_median": np.full((1, 1, 1), 2.4),
        "mlt": np.zeros((1, 1)),
        "xi": np.zeros((1, 1)),
        "eta": np.zeros((1, 1)),
        "provenance": {
            "description": "test lookup",
            "threshold_energy_flux_mW_m2": 0.05,
            "lower_mlat_degrees": 50.0,
            "upper_mlat_degrees": 90.0,
            "latitude_step_degrees": 0.01,
            "zhang_paxton_package_version": "0.1.0",
        },
    }


def _make_conductance(monkeypatch):
    monkeypatch.setattr(
        conductance_module,
        "load_zhang_paxton_lookup",
        _lookup,
    )
    return ConductanceImage(
        _BinnedImage(500.0),
        _BinnedImage(10.0),
        _BinnedImage(8.0),
        time=[datetime(2000, 1, 1, 1)],
        kp=[1.519],
        kp_interval_start=[datetime(2000, 1, 1)],
        kp_provenance=_kp_provenance(),
    )


def test_kp_is_required_by_the_production_energy_path():
    with pytest.raises(ValueError, match="requires Kp"):
        ConductanceImage(
            _BinnedImage(500.0),
            _BinnedImage(10.0),
            _BinnedImage(8.0),
        )


def test_frame_must_be_inside_its_stated_kp_interval(monkeypatch):
    monkeypatch.setattr(
        conductance_module,
        "load_zhang_paxton_lookup",
        _lookup,
    )

    with pytest.raises(ValueError, match="inside"):
        ConductanceImage(
            _BinnedImage(500.0),
            _BinnedImage(10.0),
            _BinnedImage(8.0),
            time=[datetime(2000, 1, 1, 3)],
            kp=[1.519],
            kp_interval_start=[datetime(2000, 1, 1)],
            kp_provenance=_kp_provenance(),
        )


def test_high_energy_zero_flux_keeps_finite_uncertainties(monkeypatch):
    def high_energy_lookup(kp):
        lookup = _lookup(kp)
        lookup["E0"][:] = 4.1
        return lookup

    monkeypatch.setattr(
        conductance_module,
        "load_zhang_paxton_lookup",
        high_energy_lookup,
    )
    conductance = ConductanceImage(
        _BinnedImage(1.0),
        _BinnedImage(10.0),
        _BinnedImage(8.0),
        time=[datetime(2000, 1, 1, 1)],
        kp=[9.0],
        kp_interval_start=[datetime(2000, 1, 1)],
        kp_provenance=_kp_provenance(),
    )

    assert conductance.Fe[0, 0, 0] == 0
    assert np.isfinite(conductance.dP[0, 0, 0])
    assert np.isfinite(conductance.dH[0, 0, 0])


def test_lookup_values_and_induced_covariance_reach_conductance(
    monkeypatch,
):
    covariance_used = []

    def uncertainty(E0, Fe, dE0, dFe, covariance):
        covariance_used.append(covariance)
        return 1.0

    monkeypatch.setattr(conductance_module, "peduncertainty", uncertainty)
    monkeypatch.setattr(conductance_module, "halluncertainty", uncertainty)
    conductance = _make_conductance(monkeypatch)

    assert conductance.kp[0] == 1.519
    assert conductance.kp_lookup[0] == 1.52
    assert conductance.E0[0, 0, 0] == 2.5
    assert conductance.dE0[0, 0, 0] == 0.4
    assert np.isfinite(conductance.varE0Fe[0, 0, 0])
    assert conductance.varE0Fe[0, 0, 0] != 0
    assert covariance_used == [
        conductance.varE0Fe[0, 0, 0],
        conductance.varE0Fe[0, 0, 0],
    ]


def test_kp_and_electron_energy_provenance_are_serialized(
    monkeypatch,
    tmp_path,
):
    conductance = _make_conductance(monkeypatch)
    output = tmp_path / "conductance.nc"
    conductance.to_nc(output)

    with Dataset(output) as nc:
        assert nc.variables["Kp"][0] == np.float32(1.519)
        assert nc.variables["Kp_lookup"][0] == np.float32(1.52)
        assert nc.variables["Kp_interval_start"][0] == 0
        assert nc.kp_source == "GFZ Potsdam"
        assert nc.kp_status == "def"
        assert nc.kp_doi == "10.5880/Kp.0001"
        assert nc.kp_licence == "CC BY 4.0"
        assert nc.kp_sha256 == "test checksum"
        assert nc.electron_energy_method == "zhang_paxton"
        assert "not formal model" in nc.dE0_interpretation
        assert "one-sided" in nc.zero_flux_uncertainty_definition
        assert "varE0Fe" in nc.variables
        assert np.isfinite(nc.variables["varE0Fe"][0, 0, 0])

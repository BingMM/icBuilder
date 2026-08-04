"""Small end-to-end tests for the ConductanceImage E0 override."""

from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

import icbuilder.conductanceimage as conductance_module
from icbuilder.conductanceimage import ConductanceImage
from icbuilder.imagesat_e0_eflux_estimates import E0_eflux_propagated
from icbuilder.imagesat_e0_eflux_estimates import e0_fe_covariance
from icbuilder.imagesat_e0_eflux_estimates import proton_response
from icbuilder.robinson import hall, halluncertainty, ped, peduncertainty


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


class _ArrayBinnedImage:
    """Binned-image stand-in retaining nontrivial masks and zero counts."""

    def __init__(self, mu, sigma, grid):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.shape = self.mu.shape
        self.w = np.full(self.shape, 0.75)
        self.sza = np.full(self.shape, 70.0)
        self.dza = np.full(self.shape, 20.0)
        self.los_factor = np.full(self.shape, np.cos(np.radians(20.0)))
        self.ssalon = np.arange(self.shape[0], dtype=float)
        self.grid = grid
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
        covariance_used.append(np.copy(covariance))
        return np.ones_like(E0), np.ones_like(E0)

    monkeypatch.setattr(
        ConductanceImage,
        "_robinson_uncertainty",
        staticmethod(uncertainty),
    )
    conductance = _make_conductance(monkeypatch)

    assert conductance.kp[0] == 1.519
    assert conductance.kp_lookup[0] == 1.52
    assert conductance.E0[0, 0, 0] == 2.5
    assert conductance.dE0[0, 0, 0] == 0.4
    assert np.isfinite(conductance.varE0Fe[0, 0, 0])
    assert conductance.varE0Fe[0, 0, 0] != 0
    assert len(covariance_used) == 1
    assert covariance_used[0][0] == conductance.varE0Fe[0, 0, 0]


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


def test_vectorized_zhang_paxton_matches_the_scalar_equations(monkeypatch):
    """Production vectorization preserves masks and the scalar calculation."""

    shape = (2, 2, 3)
    xi, eta = np.meshgrid(np.arange(3.0), np.arange(2.0))
    grid = type(
        "Grid",
        (),
        {
            "xi": xi,
            "eta": eta,
            "projection": _Projection(),
            "L": 1.0,
            "W": 1.0,
            "Lres": 1.0,
            "Wres": 1.0,
            "R": 1.0,
        },
    )()

    W = np.array([
        [[500.0, 1.0, np.nan], [100.0, 700.0, 20.0]],
        [[250.0, 50.0, 900.0], [2.0, 300.0, 450.0]],
    ])
    T = np.array([
        [[10.0, 10.0, 5.0], [0.0, 30.0, 2.0]],
        [[20.0, 1.0, 15.0], [8.0, np.nan, 4.0]],
    ])
    S = np.array([
        [[8.0, 8.0, 4.0], [1.0, 40.0, 0.1]],
        [[15.0, 2.0, 20.0], [5.0, 12.0, 6.0]],
    ])
    dW = np.full(shape, 0.3)
    dT = np.full(shape, 0.2)
    dS = np.full(shape, 0.4)
    dS[1, 0, 1] = np.nan

    lookup_E0 = np.linspace(1.5, 4.5, np.prod(shape)).reshape(shape)
    lookup_dE0 = np.linspace(0.2, 0.7, np.prod(shape)).reshape(shape)
    lookup_E0[1, 1, 2] = np.nan

    def array_lookup(kp):
        return {
            "kp": np.asarray(kp),
            "E0": lookup_E0,
            "dE0": lookup_dE0,
            "E0_median": lookup_E0,
            "mlt": np.zeros((2, 3)),
            "xi": xi,
            "eta": eta,
            "provenance": _lookup([1.52])["provenance"],
        }

    monkeypatch.setattr(
        conductance_module,
        "load_zhang_paxton_lookup",
        array_lookup,
    )
    times = [datetime(2000, 1, 1, 1), datetime(2000, 1, 1, 2)]
    conductance = ConductanceImage(
        _ArrayBinnedImage(W, dW, grid),
        _ArrayBinnedImage(T, dT, grid),
        _ArrayBinnedImage(S, dS, grid),
        time=times,
        kp=[1.52, 1.52],
        kp_interval_start=[datetime(2000, 1, 1)] * 2,
        kp_provenance=_kp_provenance(),
    )

    expected = {
        name: np.full(shape, np.nan)
        for name in (
            "E0", "dE0", "Fe", "dFe", "R", "dR",
            "varE0Fe", "P", "H", "dP", "dH",
        )
    }
    response = proton_response(2.0, 0.0)
    for index in np.ndindex(shape):
        values = [W[index], T[index], S[index], dW[index], dT[index], dS[index]]
        if np.any(np.isnan(values)):
            continue
        E0, Fe, dE0, dFe, R, dR = E0_eflux_propagated(
            [W[index], T[index], S[index]],
            [0, 0, 0],
            [dW[index], dT[index], dS[index]],
            2.0,
            0.0,
            E0=lookup_E0[index],
            dE0=lookup_dE0[index],
            proton_response_values=response,
        )
        covariance = e0_fe_covariance(E0, Fe, dE0)
        expected["E0"][index] = E0
        expected["dE0"][index] = dE0
        expected["Fe"][index] = Fe
        expected["dFe"][index] = dFe
        expected["R"][index] = R
        expected["dR"][index] = dR
        expected["varE0Fe"][index] = covariance
        expected["P"][index] = ped(E0, Fe)
        expected["H"][index] = hall(E0, Fe)
        expected["dP"][index] = peduncertainty(E0, Fe, dE0, dFe, covariance)
        expected["dH"][index] = halluncertainty(E0, Fe, dE0, dFe, covariance)

    exact_fields = ("E0", "dE0", "Fe", "R", "varE0Fe", "P", "H")
    for name in exact_fields:
        assert np.array_equal(
            getattr(conductance, name),
            expected[name],
            equal_nan=True,
        ), name

    # Vector ufunc ordering changes only last-place rounding in propagated
    # uncertainties; their support and scientific values remain unchanged.
    for name in ("dFe", "dR", "dP", "dH"):
        actual = getattr(conductance, name)
        assert np.array_equal(np.isnan(actual), np.isnan(expected[name])), name
        np.testing.assert_allclose(actual, expected[name], rtol=2e-15, atol=1e-13)

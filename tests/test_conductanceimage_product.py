"""Focused tests for the modular precipitation-to-conductance boundary."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from icbuilder import ConductanceImage


class _Grid:
    shape = (1, 1)
    xi = np.array([[0.0]])
    eta = np.array([[0.0]])
    lat = np.array([[70.0]])
    lon = np.array([[0.0]])
    projection = SimpleNamespace(
        position=np.array([0.0, 90.0]),
        orientation=np.array([0.0, 1.0]),
    )
    L = W = Lres = Wres = R = 1.0


class _Precipitation:
    product_type = "precipitation"
    method = "zhang_paxton"
    proton_flux_source = "SI12"
    proton_energy_model = "hardy"
    proton_energy_uncertainty_method = "not modelled by Hardy et al. (1991)"
    proton_energy_coordinate_note = "test coordinates"
    proton_response_energy_min = 0.47
    proton_response_energy_max = 46.7
    time = np.array([datetime(2000, 1, 1, 1)], dtype=object)
    ssalon = np.array([15.0])
    grid = _Grid()
    shape = (1, 1, 1)
    kp = np.array([1.5])
    kp_interval_start = np.array(["2000-01-01T00:00:00"], dtype="datetime64[s]")
    kp_provenance = {"source": "test"}
    physics_provenance = {"function": "precipitation_from_zhang_paxton"}
    Ep_model = np.full(shape, 0.3)
    Ep = np.full(shape, 0.47)
    dEp = np.zeros(shape)
    Ep_clipping_flag = np.ones(shape, dtype=bool)
    Fp = np.full(shape, 0.8)
    dFp = np.full(shape, 0.1)
    E0 = np.full(shape, 2.5)
    dE0 = np.full(shape, 0.4)
    Fe = np.full(shape, 3.0)
    dFe = np.full(shape, 0.5)
    varE0Fe = np.full(shape, -0.1)
    w = np.full(shape, 0.7)


def test_conductance_preserves_precipitation_choices_and_weight(tmp_path):
    conductance = ConductanceImage(_Precipitation())

    assert conductance.precipitation_method == "zhang_paxton"
    assert conductance.proton_flux_source == "SI12"
    assert conductance.proton_energy_model == "hardy"
    assert conductance.Ep_clipping_flag.all()
    np.testing.assert_allclose(conductance.w, 0.7)
    np.testing.assert_allclose(conductance.ssalon, 15.0)
    assert np.isfinite(conductance.P).all()
    assert np.isfinite(conductance.dH).all()

    output = tmp_path / "conductance.nc"
    conductance.to_nc(output)

    with Dataset(output) as nc:
        assert nc.product_type == "conductance"
        assert nc.precipitation_method == "zhang_paxton"
        assert nc.proton_flux_source == "SI12"
        assert nc.proton_energy_model == "hardy"
        assert nc.conductance_model == "robinson"
        assert set(nc.variables) == {
            "time", "Kp", "Kp_interval_start", "ssalon", "Ep_model", "Ep",
            "dEp", "Ep_clipping_flag", "Fp", "dFp", "E0", "dE0", "Fe",
            "dFe", "varE0Fe", "P", "H", "dP", "dH", "w",
        }
        np.testing.assert_allclose(nc.variables["w"][:], 0.7)

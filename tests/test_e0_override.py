"""Tests for supplying Zhang--Paxton E0 to the IMAGE count conversion."""

import numpy as np
import pytest

from icbuilder.imagesat_e0_eflux_estimates import E0_eflux_propagated
from icbuilder.imagesat_e0_eflux_estimates import e0_fe_covariance, fWm
from icbuilder.robinson import hall, halluncertainty
from icbuilder.robinson import ped, peduncertainty


def test_e0_and_de0_must_be_supplied_together():
    args = ([500.0, 10.0, 8.0], [0, 0, 0], [1, 1, 1], 2.0, 0.0)

    with pytest.raises(ValueError, match="together"):
        E0_eflux_propagated(*args, E0=2.0)
    with pytest.raises(ValueError, match="together"):
        E0_eflux_propagated(*args, dE0=0.5)


def test_override_is_unchanged_and_fe_does_not_depend_on_si13():
    common = {
        "dayglowcounts_list": [0, 0, 0],
        "dayglowcounts_unc": [1, 1, 1],
        "Ep": 2.0,
        "dEp": 0.0,
        "E0": 2.345,
        "dE0": 0.456,
    }
    low_si13 = E0_eflux_propagated(
        [500.0, 10.0, 8.0],
        **common,
    )
    high_si13 = E0_eflux_propagated(
        [500.0, 10.0, 80.0],
        **common,
    )

    assert low_si13[0] == 2.345
    assert low_si13[2] == 0.456
    assert high_si13[0] == 2.345
    assert high_si13[2] == 0.456
    assert low_si13[1] == high_si13[1]
    assert low_si13[3] == high_si13[3]
    assert low_si13[4] != high_si13[4]


def test_covariance_matches_a_finite_difference_at_fixed_wic_counts():
    E0 = 3.0
    dE0 = 0.4
    Wprime = 200.0
    Fe = Wprime / fWm(E0)
    step = 1e-5
    derivative = (
        Wprime / fWm(E0 + step) - Wprime / fWm(E0 - step)
    ) / (2 * step)

    expected = derivative * dE0**2
    np.testing.assert_allclose(
        e0_fe_covariance(E0, Fe, dE0),
        expected,
        rtol=1e-8,
    )


@pytest.mark.parametrize("E0", [3.9, 4.0, 4.1])
def test_zero_flux_uncertainty_is_a_one_sided_flux_excursion(E0):
    """The Fe=0 policy must remain finite across the old 4-keV singularity."""

    Fe = 0.0
    dE0 = 0.5
    dFe = 0.7
    covariance = 0.0

    dP = peduncertainty(E0, Fe, dE0, dFe, covariance)
    dH = halluncertainty(E0, Fe, dE0, dFe, covariance)

    assert np.isfinite(dP)
    assert np.isfinite(dH)
    assert dP == ped(E0, dFe)
    assert dH == hall(E0, dFe)

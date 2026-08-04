"""Tests for the fixed-grid Zhang--Paxton lookup."""

import numpy as np
import pytest

from icbuilder.grids import grid_mlt, make_wic_grid
from icbuilder.zhang_paxton_collapse import collapse_zhang_paxton
from icbuilder.zhang_paxton_lookup import (
    DEFAULT_LOOKUP_PATH,
    KP_VALUES,
    kp_to_index,
    load_zhang_paxton_lookup,
)


def test_kp_axis_has_every_hundredth_from_zero_through_nine():
    assert KP_VALUES.shape == (901,)
    assert KP_VALUES[0] == 0
    assert KP_VALUES[-1] == 9
    np.testing.assert_allclose(np.diff(KP_VALUES), 0.01, atol=2e-15)


def test_kp_rounds_to_nearest_hundredth():
    kp = [0, 0.0049, 0.005, 1.5149, 1.515, 1.519, 8.999, 9]
    expected = [0, 0, 1, 151, 152, 152, 900, 900]
    np.testing.assert_array_equal(kp_to_index(kp), expected)


@pytest.mark.parametrize("kp", [-0.001, 9.001, np.nan, np.inf])
def test_invalid_kp_is_rejected(kp):
    with pytest.raises(ValueError):
        kp_to_index(kp)


def test_lookup_matches_grid_and_direct_collapse():
    grid = make_wic_grid()
    lookup = load_zhang_paxton_lookup(1.519)

    assert lookup["kp"] == 1.52
    assert lookup["E0"].shape == (36, 36)
    assert lookup["dE0"].shape == (36, 36)
    assert lookup["E0_median"].shape == (36, 36)
    assert lookup["mlt"].shape == (36, 36)
    assert lookup["provenance"]["threshold_energy_flux_mW_m2"] == 0.05
    np.testing.assert_allclose(lookup["mlt"], grid_mlt(grid))

    cells = ([0, 17, 35], [0, 19, 35])
    direct = collapse_zhang_paxton(
        np.full(3, 1.52),
        lookup["mlt"][cells],
    )
    np.testing.assert_allclose(
        lookup["E0"][cells],
        direct["representative_energy"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        lookup["dE0"][cells],
        direct["weighted_spread"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        lookup["E0_median"][cells],
        direct["area_weighted_median_energy"],
        atol=2e-6,
    )


def test_lookup_accepts_multiple_frame_kp_values():
    lookup = load_zhang_paxton_lookup(np.array([1.519, 5.007]))

    np.testing.assert_allclose(lookup["kp"], [1.52, 5.01])
    assert lookup["E0"].shape == (2, 36, 36)
    assert lookup["dE0"].shape == (2, 36, 36)
    assert lookup["E0_median"].shape == (2, 36, 36)

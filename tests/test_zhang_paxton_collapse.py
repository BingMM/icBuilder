"""Focused tests for the exploratory Zhang--Paxton latitude collapse."""

import numpy as np
import pytest

import icbuilder.zhang_paxton_collapse as collapse_module


def test_area_weights_integrate_exact_latitude_band() -> None:
    """Cell areas must telescope to the analytic spherical-zone area."""

    centres, edges = collapse_module.regular_latitude_grid()
    weights = collapse_module.latitude_cell_area_weights(edges)
    expected = np.sin(np.deg2rad(90.0)) - np.sin(np.deg2rad(50.0))
    assert centres.size == 4000
    np.testing.assert_allclose(np.diff(centres), 0.01, rtol=0, atol=3e-14)
    np.testing.assert_allclose(weights.sum(), expected, rtol=0, atol=1e-14)


def test_only_component_containing_principal_flux_peak_is_selected() -> None:
    """Disconnected above-threshold background must not enter the mean."""

    centres, edges = collapse_module.regular_latitude_grid(50.0, 60.0, 1.0)
    energy = np.arange(centres.size, dtype=float)
    flux = np.array([0.0, 0.2, 0.2, 0.0, 0.0, 0.3, 1.0, 0.3, 0.0, 0.0])

    result = collapse_module.collapse_latitude_slice(
        energy,
        flux,
        edges,
        0.1,
    )

    np.testing.assert_array_equal(
        np.flatnonzero(result["selected"]),
        [5, 6, 7],
    )
    assert result["selected_lower_mlat"] == 55.0
    assert result["selected_upper_mlat"] == 58.0
    assert not result["empty"]


def test_area_weighted_mean_is_not_an_equal_cell_mean() -> None:
    """Mean and dE0 must use spherical rather than equal-cell weights."""

    energy = np.array([1.0, 9.0])
    flux = np.ones(2)
    edges = np.array([60.0, 75.0, 90.0])

    result = collapse_module.collapse_latitude_slice(
        energy,
        flux,
        edges,
        0.05,
    )
    weights = collapse_module.latitude_cell_area_weights(edges)
    expected = np.average(energy, weights=weights)
    expected_spread = np.sqrt(np.average((energy - expected) ** 2, weights=weights))

    assert result["representative_energy"] == expected
    assert result["weighted_spread"] == expected_spread
    assert result["representative_energy"] < np.mean(energy)
    assert result["area_weighted_median_energy"] == 1.0
    assert result["touches_equatorward_sampling_limit"]
    assert result["reaches_physical_pole"]
    assert not result["touches_poleward_sampling_limit"]


def test_empty_interval_returns_nan_and_explicit_flag() -> None:
    """A threshold miss must not silently turn into a background estimate."""

    energy = np.array([1.0, 2.0, 3.0])
    flux = np.array([0.01, 0.02, 0.01])
    result = collapse_module.collapse_latitude_slice(
        energy,
        flux,
        np.array([50.0, 51.0, 52.0, 53.0]),
        0.05,
    )

    assert result["empty"]
    assert np.isnan(result["representative_energy"])
    assert np.isnan(result["area_weighted_median_energy"])
    assert np.isnan(result["selected_lower_mlat"])
    assert not result["selected"].any()


def test_threshold_is_strict_and_domain_flags_are_distinct() -> None:
    """Cells exactly on the threshold stay out; a non-polar edge is flagged."""

    edges = np.array([50.0, 51.0, 52.0])
    empty = collapse_module.collapse_latitude_slice(
        [1.0, 2.0],
        [0.05, 0.05],
        edges,
    )
    assert empty["empty"]

    selected = collapse_module.collapse_latitude_slice(
        [1.0, 2.0],
        [0.06, 0.07],
        edges,
    )
    assert selected["touches_equatorward_sampling_limit"]
    assert selected["touches_poleward_sampling_limit"]
    assert not selected["reaches_physical_pole"]


def test_invalid_threshold_is_rejected() -> None:
    """The scientific cutoff must be finite and non-negative."""

    for threshold in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            collapse_module.collapse_latitude_slice(
                [1.0],
                [1.0],
                [50.0, 51.0],
                threshold,
            )


def test_model_wrapper_broadcasts_kp_and_mlt() -> None:
    """A Kp column and MLT row should produce the expected result grid."""

    result = collapse_module.collapse_zhang_paxton(
        np.array([[2.0], [5.0]]),
        np.array([[0.0, 6.0, 12.0, 18.0]]),
    )

    assert result["representative_energy"].shape == (2, 4)
    assert result["area_weighted_median_energy"].shape == (2, 4)
    assert result["weighted_spread"].shape == (2, 4)
    assert result["selected_lower_mlat"].shape == (2, 4)
    assert not result["empty"].any()
    assert np.isfinite(result["representative_energy"]).all()
    assert np.isfinite(result["area_weighted_median_energy"]).all()
    assert np.isfinite(result["weighted_spread"]).all()

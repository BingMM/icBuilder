"""Focused tests for the exploratory Zhang--Paxton latitude collapse."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "ZhangPaxton2008_collapse.py"
)
SPEC = importlib.util.spec_from_file_location("zhang_paxton_collapse", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
collapse_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collapse_module
SPEC.loader.exec_module(collapse_module)


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
        collapse_module.OvalThreshold(0.1),
    )

    np.testing.assert_array_equal(np.flatnonzero(result.selected), [5, 6, 7])
    assert result.selected_lower_mlat == 55.0
    assert result.selected_upper_mlat == 58.0
    assert not result.empty


def test_area_weighted_mean_is_not_an_equal_cell_mean() -> None:
    """Mean and dE0 must use spherical rather than equal-cell weights."""

    energy = np.array([1.0, 9.0])
    flux = np.ones(2)
    edges = np.array([60.0, 75.0, 90.0])

    result = collapse_module.collapse_latitude_slice(
        energy, flux, edges, collapse_module.OvalThreshold(0.05)
    )
    weights = collapse_module.latitude_cell_area_weights(edges)
    expected = np.average(energy, weights=weights)
    expected_spread = np.sqrt(np.average((energy - expected) ** 2, weights=weights))

    assert result.representative_energy == expected
    assert result.weighted_spread == expected_spread
    assert result.representative_energy < np.mean(energy)
    assert result.area_weighted_median_energy == 1.0
    assert result.touches_equatorward_sampling_limit
    assert result.reaches_physical_pole
    assert not result.touches_poleward_sampling_limit


def test_empty_interval_returns_nan_and_explicit_flag() -> None:
    """A threshold miss must not silently turn into a background estimate."""

    energy = np.array([1.0, 2.0, 3.0])
    flux = np.array([0.01, 0.02, 0.01])
    result = collapse_module.collapse_latitude_slice(
        energy,
        flux,
        np.array([50.0, 51.0, 52.0, 53.0]),
        collapse_module.OvalThreshold(0.05),
    )

    assert result.empty
    assert np.isnan(result.representative_energy)
    assert np.isnan(result.area_weighted_median_energy)
    assert np.isnan(result.selected_lower_mlat)
    assert not result.selected.any()


def test_diagnostic_mlt_grid_uses_three_minute_spacing() -> None:
    """The figures should sample the continuous model every 0.05 MLT hour."""

    centres, edges = collapse_module.regular_mlt_grid()
    assert centres.size == 480
    assert edges.size == 481
    np.testing.assert_allclose(np.diff(centres), 0.05, rtol=0, atol=1e-14)
    assert edges[0] == 0.0
    assert edges[-1] == 24.0


def test_model_wrapper_broadcasts_kp_and_mlt() -> None:
    """A Kp column and MLT row should produce the expected result grid."""

    result = collapse_module.collapse_zhang_paxton(
        np.array([[2.0], [5.0]]),
        np.array([[0.0, 6.0, 12.0, 18.0]]),
    )

    assert result.representative_energy.shape == (2, 4)
    assert result.area_weighted_median_energy.shape == (2, 4)
    assert result.weighted_spread.shape == (2, 4)
    assert result.selected_lower_mlat.shape == (2, 4)
    assert not result.empty.any()
    assert np.isfinite(result.representative_energy).all()
    assert np.isfinite(result.area_weighted_median_energy).all()
    assert np.isfinite(result.weighted_spread).all()


def test_polar_axis_places_pole_at_centre() -> None:
    """The radius=90-MLAT mapping must not be reversed by the axis limits."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(subplot_kw={"projection": "polar"})
    collapse_module._polar_axis(axis)

    assert axis.get_ylim() == (0.0, 40.0)
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "80°",
        "70°",
        "60°",
        "50°",
    ]
    plt.close(figure)

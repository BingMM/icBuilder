"""Small deterministic tests for footprint-overlap binning."""

import numpy as np

from icbuilder.footprints import overlap_mapping, overlap_statistics


class _IdentityProjection:
    """Treat longitude and latitude as projected coordinates for this test."""

    def geo2cube(self, longitude, latitude, set_points_off_cube_to_nan=True):
        return np.asarray(longitude), np.asarray(latitude)


class _TwoByTwoGrid:
    """Two regular cells in each direction, each with unit area."""

    shape = (2, 2)
    size = 4
    projection = _IdentityProjection()
    xi_mesh = np.array([[0.0, 1.0, 2.0]] * 3)
    eta_mesh = np.array([[0.0] * 3, [1.0] * 3, [2.0] * 3])


def test_matching_footprints_preserve_values_and_cover_each_cell():
    """A source grid identical to the target grid should map one-to-one."""

    mlat = np.array([[0.5, 0.5], [1.5, 1.5]])
    mlt = np.array([[0.5, 1.5], [0.5, 1.5]]) / 15
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    mapping, cell_area = overlap_mapping(mlat, mlt, _TwoByTwoGrid())
    mean, spread, count, coverage = overlap_statistics(
        values, mapping, cell_area, _TwoByTwoGrid.shape
    )

    np.testing.assert_allclose(mean, values)
    np.testing.assert_allclose(spread, 0)
    np.testing.assert_array_equal(count, 1)
    np.testing.assert_allclose(coverage, 1)


def test_nan_pixel_leaves_its_target_cell_uncovered():
    """Coverage and counts describe only detector pixels with valid data."""

    mlat = np.array([[0.5, 0.5], [1.5, 1.5]])
    mlt = np.array([[0.5, 1.5], [0.5, 1.5]]) / 15
    values = np.array([[1.0, np.nan], [3.0, 4.0]])

    mapping, cell_area = overlap_mapping(mlat, mlt, _TwoByTwoGrid())
    mean, _, count, coverage = overlap_statistics(
        values, mapping, cell_area, _TwoByTwoGrid.shape
    )

    assert np.isnan(mean[0, 1])
    assert count[0, 1] == 0
    assert coverage[0, 1] == 0

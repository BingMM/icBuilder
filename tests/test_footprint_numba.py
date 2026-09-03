"""Check the compiled footprint kernel against the readable calculation."""

import numpy as np

from icbuilder.footprints import (
    infer_footprints,
    overlap_mapping,
    rectangle_overlap,
)


class _IdentityProjection:
    def geo2cube(self, longitude, latitude, set_points_off_cube_to_nan=True):
        return np.asarray(longitude), np.asarray(latitude)


class _TwoByTwoGrid:
    shape = (2, 2)
    size = 4
    projection = _IdentityProjection()
    xi_mesh = np.array([[0.0, 1.0, 2.0]] * 3)
    eta_mesh = np.array([[0.0] * 3, [1.0] * 3, [2.0] * 3])


def test_compiled_mapping_matches_readable_polygon_calculation():
    grid = _TwoByTwoGrid()
    mlat = np.array([[0.4, 0.6], [1.4, 1.6]])
    mlt = np.array([[0.4, 1.4], [0.6, 1.6]]) / 15
    mapping, _ = overlap_mapping(mlat, mlt, grid)

    corners, valid = infer_footprints(mlat, mlt, grid)
    expected = np.zeros(mapping.shape)
    xi_edges = grid.xi_mesh[0]
    eta_edges = grid.eta_mesh[:, 0]

    for source_index in np.flatnonzero(valid):
        polygon = corners.reshape(-1, 4, 2)[source_index]
        for j in range(grid.shape[0]):
            for i in range(grid.shape[1]):
                expected[j * grid.shape[1] + i, source_index] = rectangle_overlap(
                    polygon,
                    xi_edges[i],
                    xi_edges[i + 1],
                    eta_edges[j],
                    eta_edges[j + 1],
                )

    np.testing.assert_allclose(mapping.toarray(), expected, atol=1e-14)

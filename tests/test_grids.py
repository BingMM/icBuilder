"""Tests for the canonical IMAGE Cubed-Sphere grid definitions."""

import numpy as np

from icbuilder.grids import grid_mlt, make_image_grids, make_wic_grid


def test_wic_grid_is_the_expected_36_by_36_grid():
    grid = make_wic_grid()

    assert grid.shape == (36, 36)
    assert grid.xi.shape == (36, 36)
    assert grid.eta.shape == (36, 36)

    mlt = grid_mlt(grid)
    assert mlt.shape == (36, 36)
    assert np.all((mlt >= 0.0) & (mlt < 24.0))
    assert not np.allclose(mlt, mlt[0, :][None, :])
    assert not np.allclose(mlt, mlt[:, 0][:, None])


def test_si_grid_edges_are_exactly_nested_in_wic_grid():
    grid_w, grid_s = make_image_grids()

    assert grid_w.shape == (36, 36)
    assert grid_s.shape == (18, 18)
    np.testing.assert_allclose(
        grid_s.xi_mesh[0, :],
        grid_w.xi_mesh[0, ::2],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        grid_s.eta_mesh[:, 0],
        grid_w.eta_mesh[::2, 0],
        rtol=0,
        atol=1e-14,
    )

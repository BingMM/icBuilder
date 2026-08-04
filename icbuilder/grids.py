"""Canonical Cubed-Sphere grids used by the IMAGE processing pipeline.

The Zhang--Paxton lookup table is indexed by the same ``(eta, xi)`` cells as
the binned WIC images. Keeping grid construction here prevents the table
builder and orbit-processing script from drifting apart.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from secsy import CSgrid, CSprojection


IMAGE_GRID_POSITION = (0, 90)
IMAGE_GRID_ORIENTATION = (0, 1)
IMAGE_GRID_LENGTH_METRES = 20_000_000.0
IMAGE_WIC_RESOLUTION_METRES = 225_000.0
IMAGE_SI_TARGET_RESOLUTION_METRES = 450_000.0
IMAGE_GRID_RADIUS_METRES = 6_481_200.0


def make_wic_grid() -> CSgrid:
    """Return the canonical 36-by-36 WIC Cubed-Sphere grid."""

    projection = CSprojection(
        IMAGE_GRID_POSITION,
        IMAGE_GRID_ORIENTATION,
    )
    grid = CSgrid(
        projection,
        IMAGE_GRID_LENGTH_METRES,
        IMAGE_GRID_LENGTH_METRES,
        IMAGE_WIC_RESOLUTION_METRES,
        IMAGE_WIC_RESOLUTION_METRES,
        R=IMAGE_GRID_RADIUS_METRES,
    )
    if grid.shape != (36, 36):
        raise RuntimeError(
            "canonical WIC grid must have shape (36, 36), "
            f"but secsy produced {grid.shape}"
        )
    return grid


def make_image_grids() -> tuple[CSgrid, CSgrid]:
    """Return the canonical fine WIC and exactly nested coarse SI grids."""

    grid_w = make_wic_grid()
    distance = grid_w.Lres * grid_w.shape[0]
    steps = int(round(distance / IMAGE_SI_TARGET_RESOLUTION_METRES))
    xi_edges = np.linspace(
        grid_w.xi_mesh[0, 0],
        grid_w.xi_mesh[0, -1],
        steps + 1,
    )
    eta_edges = np.linspace(
        grid_w.eta_mesh[0, 0],
        grid_w.eta_mesh[-1, 0],
        steps + 1,
    )
    coarse_resolution = distance / steps
    grid_s = CSgrid(
        CSprojection(IMAGE_GRID_POSITION, IMAGE_GRID_ORIENTATION),
        IMAGE_GRID_LENGTH_METRES,
        IMAGE_GRID_LENGTH_METRES,
        coarse_resolution,
        coarse_resolution,
        edges=(xi_edges, eta_edges),
        R=IMAGE_GRID_RADIUS_METRES,
    )
    return grid_w, grid_s


def grid_mlt(grid: CSgrid) -> NDArray[np.float64]:
    """Return the two-dimensional cell-centre MLT field for a CS grid.

    ``BinnedImage`` supplies magnetic local time to ``CSgrid`` as longitude
    in degrees (``MLT * 15``). The inverse mapping for a grid cell is therefore
    its Cubed-Sphere longitude divided by 15, wrapped onto ``[0, 24)``.
    """

    return np.mod(np.asarray(grid.lon, dtype=float) / 15.0, 24.0)

"""Map detector-pixel footprints onto a Cubed-Sphere grid.

Each detector pixel is approximated as a uniform quadrilateral. Its corners
are inferred from the local row and column spacing of the geolocated pixel
centres. The quadrilateral is then split among every CS cell that it overlaps.
"""

#%% Imports

import numpy as np
from scipy.sparse import csr_matrix


#%% Infer detector-pixel corners

def neighbour_step(values, axis):
    """Estimate one detector-pixel step from adjacent pixel centres."""

    values = np.asarray(values, dtype=float)
    previous = np.full_like(values, np.nan)
    following = np.full_like(values, np.nan)

    if axis == 0:
        previous[1:] = values[:-1]
        following[:-1] = values[1:]
    else:
        previous[:, 1:] = values[:, :-1]
        following[:, :-1] = values[:, 1:]

    step = np.full_like(values, np.nan)
    centre_ok = np.isfinite(values)
    previous_ok = np.isfinite(previous)
    following_ok = np.isfinite(following)

    both = centre_ok & previous_ok & following_ok
    step[both] = (following[both] - previous[both]) / 2

    forward = centre_ok & ~previous_ok & following_ok
    step[forward] = following[forward] - values[forward]

    backward = centre_ok & previous_ok & ~following_ok
    step[backward] = values[backward] - previous[backward]

    return step


def infer_footprints(mlat, mlt, grid):
    """Return four projected corners for each geolocated detector pixel."""

    longitude = np.mod(np.asarray(mlt) * 15, 360)
    xi, eta = grid.projection.geo2cube(
        longitude,
        mlat,
        set_points_off_cube_to_nan=True,
    )

    row = np.stack(
        [neighbour_step(xi, 0), neighbour_step(eta, 0)], axis=-1
    )
    column = np.stack(
        [neighbour_step(xi, 1), neighbour_step(eta, 1)], axis=-1
    )
    centre = np.stack([xi, eta], axis=-1)

    corners = np.stack(
        [
            centre - row / 2 - column / 2,
            centre - row / 2 + column / 2,
            centre + row / 2 + column / 2,
            centre + row / 2 - column / 2,
        ],
        axis=-2,
    )

    valid = np.all(np.isfinite(corners), axis=(-2, -1))
    return corners, valid


#%% Polygon clipping

def polygon_area(polygon):
    """Return unsigned polygon area in projected xi/eta coordinates."""

    if len(polygon) < 3:
        return 0.0

    polygon = np.asarray(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_at_boundary(polygon, coordinate, boundary, keep_above):
    """Clip a polygon against one vertical or horizontal boundary."""

    if len(polygon) == 0:
        return []

    clipped = []
    previous = np.asarray(polygon[-1], dtype=float)

    def inside(point):
        if keep_above:
            return point[coordinate] >= boundary
        return point[coordinate] <= boundary

    previous_inside = inside(previous)

    for current in polygon:
        current = np.asarray(current, dtype=float)
        current_inside = inside(current)

        if current_inside != previous_inside:
            difference = current[coordinate] - previous[coordinate]
            if difference != 0:
                fraction = (boundary - previous[coordinate]) / difference
                clipped.append(previous + fraction * (current - previous))

        if current_inside:
            clipped.append(current)

        previous = current
        previous_inside = current_inside

    return clipped


def rectangle_overlap(polygon, xmin, xmax, ymin, ymax):
    """Return polygon overlap area with one rectangular CS cell."""

    clipped = list(np.asarray(polygon, dtype=float))
    clipped = clip_at_boundary(clipped, 0, xmin, keep_above=True)
    clipped = clip_at_boundary(clipped, 0, xmax, keep_above=False)
    clipped = clip_at_boundary(clipped, 1, ymin, keep_above=True)
    clipped = clip_at_boundary(clipped, 1, ymax, keep_above=False)
    return polygon_area(clipped)


#%% Sparse footprint-to-grid mapping

def overlap_mapping(mlat, mlt, grid):
    """Build one sparse source-pixel to target-cell overlap mapping."""

    corners, valid = infer_footprints(mlat, mlt, grid)
    xi_edges = np.asarray(grid.xi_mesh[0], dtype=float)
    eta_edges = np.asarray(grid.eta_mesh[:, 0], dtype=float)
    ny, nx = grid.shape

    target_indices = []
    source_indices = []
    overlap_areas = []

    for source_index in np.flatnonzero(valid):
        polygon = corners.reshape(-1, 4, 2)[source_index]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        i_start = max(np.searchsorted(xi_edges, xmin, side="right") - 1, 0)
        i_stop = min(np.searchsorted(xi_edges, xmax, side="left"), nx)
        j_start = max(np.searchsorted(eta_edges, ymin, side="right") - 1, 0)
        j_stop = min(np.searchsorted(eta_edges, ymax, side="left"), ny)

        for j in range(j_start, j_stop):
            for i in range(i_start, i_stop):
                area = rectangle_overlap(
                    polygon,
                    xi_edges[i],
                    xi_edges[i + 1],
                    eta_edges[j],
                    eta_edges[j + 1],
                )
                if area > 0:
                    target_indices.append(j * nx + i)
                    source_indices.append(source_index)
                    overlap_areas.append(area)

    mapping = csr_matrix(
        (overlap_areas, (target_indices, source_indices)),
        shape=(grid.size, np.asarray(mlat).size),
    )
    cell_area = np.diff(eta_edges)[:, None] * np.diff(xi_edges)[None, :]
    return mapping, cell_area


#%% Apply one mapping to source fields

def overlap_mean(values, mapping, output_shape):
    """Return an overlap-weighted mean and contributing overlap area."""

    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0)

    overlap = np.asarray(mapping @ finite.astype(float)).ravel()
    numerator = np.asarray(mapping @ filled).ravel()

    mean = np.full(mapping.shape[0], np.nan)
    covered = overlap > 0
    mean[covered] = numerator[covered] / overlap[covered]
    return mean.reshape(output_shape), overlap.reshape(output_shape)


def overlap_statistics(values, mapping, cell_area, output_shape):
    """Return mean, provisional spread, contributors, and coverage."""

    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0)

    overlap = np.asarray(mapping @ finite.astype(float)).ravel()
    numerator = np.asarray(mapping @ filled).ravel()
    numerator_squared = np.asarray(mapping @ (filled**2)).ravel()

    mean = np.full(mapping.shape[0], np.nan)
    spread = np.full(mapping.shape[0], np.nan)
    covered = overlap > 0
    mean[covered] = numerator[covered] / overlap[covered]
    variance = numerator_squared[covered] / overlap[covered] - mean[covered] ** 2
    spread[covered] = np.sqrt(np.maximum(variance, 0))

    contributors = mapping.copy()
    contributors.data[:] = 1
    count = np.asarray(contributors @ finite.astype(np.int32)).ravel()

    coverage = overlap.reshape(output_shape) / cell_area
    coverage = np.minimum(coverage, 1)

    return (
        mean.reshape(output_shape),
        spread.reshape(output_shape),
        count.reshape(output_shape).astype(np.int32),
        coverage,
    )

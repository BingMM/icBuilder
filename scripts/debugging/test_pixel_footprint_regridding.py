"""Test footprint-aware regridding of IMAGE detector pixels.

The current BinnedImage assigns each detector pixel to the CS cell containing
its centre. This diagnostic instead estimates a four-corner footprint from the
local detector-pixel centre lattice and distributes the pixel over every CS
cell that it overlaps.

The production footprint option now uses the same local-parallelogram and
uniform-response assumptions. This script remains the visual diagnostic for
those assumptions and for centre-versus-footprint support.
"""

#%% Imports

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from netCDF4 import Dataset
import numpy as np

from icbuilder.grids import IMAGE_GRID_RADIUS_METRES, make_wic_grid


#%% Paths and plotting choices

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE_DATA = REPOSITORY / "example_data"
DEFAULT_OUTPUT = REPOSITORY / "figures" / "debugging" / "pixel_footprints"

ORBITS = ("0085", "0968")
SENSORS = ("wic", "s12", "s13")
DISPLAY_NAMES = {"wic": "WIC", "s12": "SI12", "s13": "SI13"}
IMAGE_FIELDS = {"wic": "shimg", "s12": "dgimg", "s13": "dgimg"}


#%% Read the original detector lattice

def orbit_file(base, sensor, orbit):
    """Return the processed fuvpy orbit file for one sensor."""

    return base / sensor / f"{sensor}_or{orbit}.nc"


def read_frame(base, sensor, orbit, frame):
    """Read one detector frame before CS-grid binning."""

    filename = orbit_file(base, sensor, orbit)
    with Dataset(filename) as nc:
        data = {
            name: np.asarray(
                np.ma.filled(nc.variables[name][frame], np.nan), dtype=float
            )
            for name in ("mlat", "mlt", "glat", "glon", "dza")
        }
        data["image"] = np.asarray(
            np.ma.filled(nc.variables[IMAGE_FIELDS[sensor]][frame], np.nan),
            dtype=float,
        )

    data["filename"] = filename
    return data


#%% Local pixel geometry

def neighbour_step(values, axis):
    """Estimate one detector-pixel step using adjacent pixel centres.

    Central differences are used where both neighbours exist. A one-sided
    difference is used beside a missing value or detector edge.
    """

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


def infer_projected_footprints(frame, grid):
    """Infer pixel-centre and four-corner coordinates in CS xi/eta."""

    magnetic_longitude = np.mod(frame["mlt"] * 15, 360)
    xi, eta = grid.projection.geo2cube(
        magnetic_longitude,
        frame["mlat"],
        set_points_off_cube_to_nan=True,
    )

    row_xi = neighbour_step(xi, axis=0)
    row_eta = neighbour_step(eta, axis=0)
    column_xi = neighbour_step(xi, axis=1)
    column_eta = neighbour_step(eta, axis=1)

    row = np.stack([row_xi, row_eta], axis=-1)
    column = np.stack([column_xi, column_eta], axis=-1)
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

    valid = (
        np.all(np.isfinite(corners), axis=(-2, -1))
        & np.isfinite(frame["image"])
        & np.isfinite(frame["dza"])
    )

    return centre, corners, valid


def geographic_pixel_area(frame):
    """Estimate footprint area from the geographic centre lattice [km2]."""

    latitude = np.radians(frame["glat"])
    longitude = np.radians(frame["glon"])
    radius_km = IMAGE_GRID_RADIUS_METRES / 1000

    xyz = radius_km * np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        axis=-1,
    )

    row = neighbour_step(xyz, axis=0)
    column = neighbour_step(xyz, axis=1)
    return np.linalg.norm(np.cross(row, column), axis=-1)


#%% Polygon overlap with rectangular CS cells

def polygon_area(polygon):
    """Return the unsigned area of a polygon in projected coordinates."""

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
                crossing = previous + fraction * (current - previous)
                clipped.append(crossing)

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


def check_polygon_clipping():
    """Small numerical check before processing the IMAGE polygons."""

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert np.isclose(rectangle_overlap(square, 0, 1, 0, 1), 1)
    assert np.isclose(rectangle_overlap(square, 0.5, 1.5, 0, 1), 0.5)
    assert rectangle_overlap(square, 2, 3, 2, 3) == 0


#%% Centre-count and footprint-overlap remapping

def remap_footprints(frame, centre, corners, valid, grid):
    """Compare point binning with fractional footprint overlap."""

    xi_edges = np.asarray(grid.xi_mesh[0], dtype=float)
    eta_edges = np.asarray(grid.eta_mesh[:, 0], dtype=float)
    ny, nx = grid.shape

    centre_count = np.zeros((ny, nx), dtype=int)
    overlap = np.zeros((ny, nx), dtype=float)
    weighted_image = np.zeros((ny, nx), dtype=float)
    contributing_pixels = np.zeros((ny, nx), dtype=int)

    # Existing point-centre assignment.
    j, i = grid.bin_index(
        np.mod(frame["mlt"] * 15, 360),
        frame["mlat"],
    )
    j = np.asarray(j).reshape(frame["mlat"].shape)
    i = np.asarray(i).reshape(frame["mlat"].shape)
    valid_centre = (
        valid
        & (i >= 0)
        & (i < nx)
        & (j >= 0)
        & (j < ny)
    )
    np.add.at(centre_count, (j[valid_centre], i[valid_centre]), 1)

    # Distribute every source footprint over intersected target cells.
    for source_index in zip(*np.where(valid)):
        polygon = corners[source_index]
        image_value = frame["image"][source_index]

        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        i_start = max(np.searchsorted(xi_edges, xmin, side="right") - 1, 0)
        i_stop = min(np.searchsorted(xi_edges, xmax, side="left"), nx)
        j_start = max(np.searchsorted(eta_edges, ymin, side="right") - 1, 0)
        j_stop = min(np.searchsorted(eta_edges, ymax, side="left"), ny)

        for jj in range(j_start, j_stop):
            for ii in range(i_start, i_stop):
                area = rectangle_overlap(
                    polygon,
                    xi_edges[ii],
                    xi_edges[ii + 1],
                    eta_edges[jj],
                    eta_edges[jj + 1],
                )
                if area <= 0:
                    continue

                overlap[jj, ii] += area
                weighted_image[jj, ii] += area * image_value
                contributing_pixels[jj, ii] += 1

    cell_area = np.diff(eta_edges)[:, None] * np.diff(xi_edges)[None, :]
    coverage = overlap / cell_area

    image = np.full((ny, nx), np.nan)
    covered = overlap > 0
    image[covered] = weighted_image[covered] / overlap[covered]

    return {
        "centre_count": centre_count,
        "coverage": coverage,
        "image": image,
        "contributing_pixels": contributing_pixels,
    }


#%% Plot the full-grid comparison

def plot_coverage(results, grid, output):
    """Compare centre counting and inferred footprint coverage."""

    xi_edges = grid.xi_mesh[0]
    eta_edges = grid.eta_mesh[:, 0]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.5), constrained_layout=True)

    columns = [
        ("wic", "centre_count", "WIC centre count", "viridis"),
        ("wic", "coverage", "WIC footprint coverage", "magma"),
        ("s13", "centre_count", "SI13 centre count", "viridis"),
        ("s13", "coverage", "SI13 footprint coverage", "magma"),
    ]

    for row, orbit in enumerate(ORBITS):
        for column, (sensor, field, title, colormap) in enumerate(columns):
            values = results[(orbit, sensor)][field]
            if field == "coverage":
                upper = min(max(np.nanpercentile(values, 99), 1), 1.5)
                label = "covered area / cell area"
            else:
                upper = max(np.nanpercentile(values, 99), 1)
                label = "source-pixel centres"

            mesh = axes[row, column].pcolormesh(
                xi_edges,
                eta_edges,
                values,
                shading="flat",
                cmap=colormap,
                vmin=0,
                vmax=upper,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_title(f"Orbit {orbit}: {title}")
            axes[row, column].set_xlabel("CS xi")
            axes[row, column].set_ylabel("CS eta")
            figure.colorbar(mesh, ax=axes[row, column], label=label, shrink=0.82)

    figure.suptitle(
        "Point-centre binning versus inferred detector-footprint overlap\n"
        f"Frame 000; target CS grid {grid.shape[1]} x {grid.shape[0]}"
    )
    save_figure(figure, output / "footprint_coverage_comparison")


#%% Plot detector polygons over individual CS cells

def plot_grid_detail(results, grid, output):
    """Show inferred pixel polygons and target cells in a local region."""

    xi_edges = np.asarray(grid.xi_mesh[0])
    eta_edges = np.asarray(grid.eta_mesh[:, 0])
    dx = np.median(np.diff(xi_edges))
    dy = np.median(np.diff(eta_edges))

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)

    for row, orbit in enumerate(ORBITS):
        for column, sensor in enumerate(("wic", "s13")):
            result = results[(orbit, sensor)]
            centre = result["centre"]
            corners = result["corners"]
            valid = result["valid"]
            dza = result["frame"]["dza"]

            # Focus on a valid footprint close to 45 degrees DZA. This usually
            # shows both projection stretching and several neighboring cells.
            candidates = valid & (np.abs(centre[..., 0]) < 0.6) & (np.abs(centre[..., 1]) < 0.6)
            score = np.where(candidates, np.abs(dza - 45), np.inf)
            focus_index = np.unravel_index(np.argmin(score), score.shape)
            focus = centre[focus_index]

            xmin, xmax = focus[0] - 2.5 * dx, focus[0] + 2.5 * dx
            ymin, ymax = focus[1] - 2.5 * dy, focus[1] + 2.5 * dy
            nearby = (
                valid
                & (centre[..., 0] > xmin - dx)
                & (centre[..., 0] < xmax + dx)
                & (centre[..., 1] > ymin - dy)
                & (centre[..., 1] < ymax + dy)
            )

            polygons = corners[nearby]
            collection = LineCollection(polygons, colors="tab:blue", linewidths=0.8, alpha=0.7)
            axes[row, column].add_collection(collection)
            points = axes[row, column].scatter(
                centre[..., 0][nearby],
                centre[..., 1][nearby],
                c=dza[nearby],
                s=12,
                cmap="viridis",
                vmin=0,
                vmax=75,
                zorder=3,
            )

            for edge in xi_edges[(xi_edges >= xmin - dx) & (xi_edges <= xmax + dx)]:
                axes[row, column].axvline(edge, color="black", lw=0.7, alpha=0.55)
            for edge in eta_edges[(eta_edges >= ymin - dy) & (eta_edges <= ymax + dy)]:
                axes[row, column].axhline(edge, color="black", lw=0.7, alpha=0.55)

            axes[row, column].set_xlim(xmin, xmax)
            axes[row, column].set_ylim(ymin, ymax)
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("CS xi")
            axes[row, column].set_ylabel("CS eta")
            axes[row, column].set_title(
                f"Orbit {orbit}, {DISPLAY_NAMES[sensor]}\n"
                "blue: inferred footprints; black: target cells"
            )
            figure.colorbar(points, ax=axes[row, column], label="DZA [degrees]", shrink=0.82)

    figure.suptitle("Local detector-pixel footprints inferred from neighbouring centres")
    save_figure(figure, output / "footprint_grid_detail")


#%% Plot the geometric validation against DZA

def binned_area_statistics(dza, area, edges):
    """Return median and 10--90% footprint-area ranges in DZA bins."""

    centres = (edges[:-1] + edges[1:]) / 2
    median = np.full(centres.shape, np.nan)
    lower = np.full(centres.shape, np.nan)
    upper = np.full(centres.shape, np.nan)

    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        selected = area[(dza >= start) & (dza < stop)]
        if selected.size < 10:
            continue
        lower[index], median[index], upper[index] = np.percentile(selected, [10, 50, 90])

    return centres, median, lower, upper


def plot_area_vs_dza(results, output):
    """Show whether inferred footprint area grows with viewing angle."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    colours = {"0085": "tab:blue", "0968": "tab:orange"}
    edges = np.arange(0, 80, 5)

    for axis, sensor in zip(axes, SENSORS):
        for orbit in ORBITS:
            result = results[(orbit, sensor)]
            dza = result["frame"]["dza"]
            area = result["geographic_area"]
            valid = result["valid"] & np.isfinite(area) & (area > 0) & (dza >= 0) & (dza < 75)

            centres, median, lower, upper = binned_area_statistics(dza[valid], area[valid], edges)
            axis.plot(centres, median, color=colours[orbit], label=f"orbit {orbit}")
            axis.fill_between(centres, lower, upper, color=colours[orbit], alpha=0.18)

        axis.set_yscale("log")
        axis.set_xlabel("DZA [degrees]")
        axis.set_ylabel("inferred footprint area [km$^2$]")
        axis.set_title(DISPLAY_NAMES[sensor])
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Pixel-centre geometry captures both range and DZA footprint changes")
    save_figure(figure, output / "footprint_area_vs_dza")


#%% Output and command line

def save_figure(figure, path):
    """Save a diagnostic as PNG and publication-quality PDF."""

    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def print_summary(orbit, sensor, result):
    """Print the main sampling and coverage diagnostics."""

    area = result["geographic_area"]
    valid = result["valid"] & np.isfinite(area) & (area > 0)
    remapped = result["remapped"]

    occupied_centres = np.mean(remapped["centre_count"] > 0)
    covered_cells = np.mean(remapped["coverage"] > 0)
    substantially_covered = np.mean(remapped["coverage"] >= 0.5)
    median_area = np.nanmedian(area[valid])

    print(
        f"orbit {orbit} {DISPLAY_NAMES[sensor]:4s}: "
        f"{np.sum(valid):5d} footprints, "
        f"median area {median_area:8.0f} km2 "
        f"(square width {np.sqrt(median_area):5.1f} km), "
        f"centre cells {occupied_centres:5.1%}, "
        f"any overlap {covered_cells:5.1%}, "
        f">=50% covered {substantially_covered:5.1%}"
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=EXAMPLE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_polygon_clipping()

    grid = make_wic_grid()
    results = {}

    for orbit in ORBITS:
        for sensor in SENSORS:
            frame = read_frame(args.base, sensor, orbit, args.frame)
            centre, corners, valid = infer_projected_footprints(frame, grid)
            remapped = remap_footprints(frame, centre, corners, valid, grid)

            results[(orbit, sensor)] = {
                "frame": frame,
                "centre": centre,
                "corners": corners,
                "valid": valid,
                "geographic_area": geographic_pixel_area(frame),
                "remapped": remapped,
                **remapped,
            }
            print_summary(orbit, sensor, results[(orbit, sensor)])

    plot_coverage(results, grid, args.output)
    plot_grid_detail(results, grid, args.output)
    plot_area_vs_dza(results, args.output)
    print(f"\nFigures written to {args.output}")


if __name__ == "__main__":
    main()

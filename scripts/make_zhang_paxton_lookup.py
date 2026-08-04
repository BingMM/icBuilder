"""Build the Zhang--Paxton lookup on the fixed IMAGE Cubed-Sphere grid."""

from argparse import ArgumentParser
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from icbuilder.grids import grid_mlt, make_wic_grid
from icbuilder.zhang_paxton_collapse import (
    DEFAULT_MLAT_STEP_DEGREES,
    FIGURE8_THRESHOLD,
    collapse_zhang_paxton,
)
from icbuilder.zhang_paxton_lookup import DEFAULT_LOOKUP_PATH, KP_VALUES


def collapse_one_kp(kp, mlt):
    """Collapse all 36 by 36 grid cells for one Kp value."""

    result = collapse_zhang_paxton(
        np.full(mlt.shape, kp),
        mlt,
        threshold=FIGURE8_THRESHOLD,
        lower_mlat=50,
        upper_mlat=90,
        latitude_step=DEFAULT_MLAT_STEP_DEGREES,
    )
    return (
        result["representative_energy"].astype("float32"),
        result["weighted_spread"].astype("float32"),
        result["area_weighted_median_energy"].astype("float32"),
    )


def calculate_lookup(workers=1):
    """Calculate E0, dE0, and median E0 for every lookup Kp."""

    if workers < 1:
        raise ValueError("workers must be at least 1")

    mlt = grid_mlt(make_wic_grid())
    calculate_layer = partial(collapse_one_kp, mlt=mlt)
    if workers == 1:
        layers = [
            calculate_layer(kp) for kp in tqdm(KP_VALUES, desc="Kp")
        ]
    else:
        layers = process_map(
            calculate_layer,
            KP_VALUES,
            max_workers=workers,
            chunksize=1,
            desc="Kp",
        )

    E0 = np.stack([layer[0] for layer in layers])
    dE0 = np.stack([layer[1] for layer in layers])
    E0_median = np.stack([layer[2] for layer in layers])
    return E0, dE0, E0_median


def write_lookup(path, E0, dE0, E0_median):
    """Write the three scientific lookup fields and their coordinates."""

    grid = make_wic_grid()
    expected_shape = (len(KP_VALUES), *grid.shape)
    for name, values in {
        "E0": E0,
        "dE0": dE0,
        "E0_median": E0_median,
    }.items():
        if values.shape != expected_shape or np.any(~np.isfinite(values)):
            raise ValueError(
                f"{name} must be finite and have shape {expected_shape}"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(path, "w") as nc:
        nc.createDimension("kp", len(KP_VALUES))
        nc.createDimension("eta", grid.shape[0])
        nc.createDimension("xi", grid.shape[1])

        coordinates = {
            "kp": (KP_VALUES, ("kp",), "1"),
            "xi": (grid.xi, ("eta", "xi"), "radians"),
            "eta": (grid.eta, ("eta", "xi"), "radians"),
            "mlt": (grid_mlt(grid), ("eta", "xi"), "hours"),
        }
        for name, (values, dimensions, units) in coordinates.items():
            variable = nc.createVariable(name, "f8", dimensions)
            variable[:] = values
            variable.units = units

        fields = {
            "E0": (
                E0,
                "Spherical-area-weighted mean electron energy",
            ),
            "dE0": (
                dE0,
                "Spherical-area-weighted latitude-profile spread",
            ),
            "E0_median": (
                E0_median,
                "Spherical-area-weighted median electron energy",
            ),
        }
        for name, (values, description) in fields.items():
            variable = nc.createVariable(
                name,
                "f4",
                ("kp", "eta", "xi"),
                zlib=True,
                complevel=4,
            )
            variable[:] = values
            variable.units = "keV"
            variable.description = description

        nc.description = (
            "Zhang and Paxton (2008) electron energy collapsed over latitude "
            "onto the fixed IMAGE Cubed-Sphere grid"
        )
        nc.threshold_energy_flux_mW_m2 = FIGURE8_THRESHOLD
        nc.lower_mlat_degrees = 50.0
        nc.upper_mlat_degrees = 90.0
        nc.latitude_step_degrees = DEFAULT_MLAT_STEP_DEGREES
        nc.zhang_paxton_package_version = version("ZhangPaxton2008")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    E0, dE0, E0_median = calculate_lookup(args.workers)
    write_lookup(args.output, E0, dE0, E0_median)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

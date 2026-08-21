"""Plot modular Product-3 conductance orbit files."""

#%% Imports

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from icreader import load as icload
from polplot import pp
from tqdm import tqdm


#%% Plot settings

REPOSITORY = Path(__file__).resolve().parents[1]


def positive_scale(values, quantile=0.999):
    """Return a useful zero-based colour scale for one orbit."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, 1

    upper = np.nanquantile(finite, quantile)
    if upper <= 0:
        upper = 1
    return 0, upper


def signed_scale(values, quantile=0.999):
    """Return a symmetric colour scale for a signed diagnostic."""

    values = np.asarray(values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return -1, 1

    limit = np.nanquantile(finite, quantile)
    if limit == 0:
        limit = 1
    return -limit, limit


def plot_fields(image, precipitation, wic_dza):
    """Define the Product-3 fields and derived diagnostics to display."""

    ratio_support = (
        np.isfinite(precipitation.wic_corrected)
        & np.isfinite(precipitation.si13_corrected)
    )

    wic_si13_ratio = np.full(image.shape, np.nan)
    np.divide(
        precipitation.wic_corrected,
        precipitation.si13_corrected,
        out=wic_si13_ratio,
        where=precipitation.si13_corrected > 0,
    )

    # Show viewing geometry only where the plotted ratio has both cameras.
    ratio_dza = np.where(ratio_support, wic_dza, np.nan)

    hall_pedersen_ratio = np.full(image.shape, np.nan)
    np.divide(image.H, image.P, out=hall_pedersen_ratio, where=image.P > 0)

    return [
        ("WIC", precipitation.wic_corrected, "Proton-corrected WIC [counts]", positive_scale(precipitation.wic_corrected), "viridis"),
        ("SI13", precipitation.si13_corrected, "Proton-corrected SI13 [counts]", positive_scale(precipitation.si13_corrected), "viridis"),
        ("R", wic_si13_ratio, "Corrected WIC / SI13", (0, 150), "viridis"),
        ("DZA", ratio_dza, "WIC DZA on WIC/SI13 support [degrees]", (0, 75), "viridis"),
        ("E0", image.E0, "Electron mean energy [keV]", positive_scale(image.E0), "viridis"),
        ("dE0", image.dE0, "Electron energy uncertainty [keV]", positive_scale(image.dE0), "magma"),
        ("Fe", image.Fe, "Electron energy flux [mW m$^{-2}$]", positive_scale(image.Fe), "viridis"),
        ("dFe", image.dFe, "Energy-flux uncertainty [mW m$^{-2}$]", positive_scale(image.dFe), "magma"),
        ("P", image.P, "Pedersen conductance [S]", positive_scale(image.P), "viridis"),
        ("dP", image.dP, "Pedersen uncertainty [S]", positive_scale(image.dP), "magma"),
        ("H", image.H, "Hall conductance [S]", positive_scale(image.H), "viridis"),
        ("dH", image.dH, "Hall uncertainty [S]", positive_scale(image.dH), "magma"),
        ("H/P", hall_pedersen_ratio, "Hall / Pedersen", positive_scale(hall_pedersen_ratio), "viridis"),
        ("cov", image.varE0Fe, "Covariance $E_0$, $F_e$", signed_scale(image.varE0Fe), "coolwarm"),
        ("w", image.w, "Combined observation weight", (0, 1), "viridis"),
    ]


#%% One frame

def plot_frame(image, frame, fields):
    """Plot one frame from a modular conductance product."""

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.flatten()

    latitude = image.mlat
    mlt = image.mlt

    for axis, (_, values, title, colour_range, colour_map) in zip(axes, fields):
        polar_axis = pp(axis)
        plotted = polar_axis.plotimg(
            latitude,
            mlt,
            values[frame],
            crange=colour_range,
            cmap=colour_map,
        )
        axis.set_title(title)
        fig.colorbar(plotted, ax=axis, shrink=0.72, pad=0.02)

    # Use the final panel for coordinates and frame metadata.
    axes[-1].set_axis_off()
    method = image.precipitation_method.replace("_", " ")
    metadata = (
        f"Time: {image.time[frame]}\n"
        f"Kp: {image.kp[frame]:.3g}\n"
        f"Precipitation: {method}\n"
        f"Proton correction: {image.proton_method}\n"
        f"Conductance: {image.conductance_model}"
    )
    axes[-1].text(0.05, 0.9, metadata, va="top", fontsize=14)

    # One polar panel carries the shared MLT and latitude labels.
    pp(axes[12]).writeLTlabels(fontsize=11)
    axes[12].text(
        0.84, 0.08, "50$^{\circ}$", transform=axes[12].transAxes,
        ha="center", va="center", fontsize=11,
    )

    fig.suptitle(f"Modular conductance product — frame {frame:03d}", fontsize=18)
    fig.subplots_adjust(wspace=0.28, hspace=0.2, top=0.92)
    return fig


#%% Orbit processing

def orbit_number(filename):
    """Read the four-digit orbit number from ``or_XXXX.nc``."""

    return int(filename.stem[-4:])


def find_files(input_dir, requested_orbits=None):
    """Return the available conductance files selected by the user."""

    files = sorted(input_dir.glob("or_*.nc"))
    if requested_orbits:
        requested_orbits = set(requested_orbits)
        files = [file for file in files if orbit_number(file) in requested_orbits]
    return files


def plot_orbit(filename, output_dir, requested_frame=None):
    """Load and plot one modular conductance orbit."""

    image = icload(filename)
    if image.product_type != "conductance":
        raise ValueError(f"not a conductance product: {filename}")

    if image.source_precipitation is None:
        raise ValueError(f"conductance product has no precipitation source: {filename}")

    precipitation_file = Path(image.source_precipitation).expanduser()
    precipitation = icload(precipitation_file)
    if precipitation.product_type != "precipitation":
        raise ValueError(f"not a precipitation product: {precipitation_file}")
    if precipitation.shape != image.shape:
        raise ValueError("precipitation and conductance dimensions do not match")

    if "wic" not in precipitation.source_products:
        raise ValueError(f"precipitation product has no WIC source: {precipitation_file}")
    wic_file = Path(precipitation.source_products["wic"]).expanduser()
    wic = icload(wic_file)
    wic_dza = np.asarray(wic.dza)[precipitation.wic_source_index]

    orbit = orbit_number(filename)
    orbit_output = output_dir / f"or_{orbit:04d}"
    orbit_output.mkdir(parents=True, exist_ok=True)

    if requested_frame is None:
        frames = range(image.shape[0])
    else:
        if requested_frame < 0 or requested_frame >= image.shape[0]:
            raise ValueError(
                f"frame {requested_frame} is outside orbit {orbit:04d} "
                f"with {image.shape[0]} frames"
            )
        frames = [requested_frame]

    fields = plot_fields(image, precipitation, wic_dza)
    for frame in frames:
        figure = plot_frame(image, frame, fields)
        figure.savefig(orbit_output / f"{frame:03d}.png", dpi=150)
        plt.close(figure)


#%% Command line

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot modular Product-3 conductance orbit files."
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=REPOSITORY / "example_data" / "conductance",
        help="Directory containing or_XXXX.nc conductance files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY / "example_data" / "figures" / "conductance",
        help="Directory in which orbit figure folders are created",
    )
    parser.add_argument(
        "--orbit", type=int, nargs="+",
        help="One or more orbit numbers; omit to plot every available orbit",
    )
    parser.add_argument(
        "--frame", type=int,
        help="Plot only this frame index from each selected orbit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    files = find_files(input_dir, args.orbit)
    if not files:
        raise FileNotFoundError(f"no selected conductance files in {input_dir}")

    plt.ioff()
    for filename in tqdm(files, desc="Plot conductance orbits"):
        plot_orbit(filename, output_dir, args.frame)


if __name__ == "__main__":
    main()

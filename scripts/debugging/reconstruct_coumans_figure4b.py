"""Reconstruct the IMAGE/DMSP comparison in Coumans et al. (2004), Fig. 4b.

An IMAGE timestamp is the centre of the short camera exposure. Following
Coumans et al. (2002), each frame is compared with DMSP samples from one
minute before to one minute after that central snapshot time.
"""

#%% Imports

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from icphysics.image import fE0m, wic_to_s13
from icreader import load as icload


#%% Inputs

REPOSITORY = Path(__file__).resolve().parents[2]
IMAGE_FILE = REPOSITORY / "example_data/precipitation/or_0968.nc"
DMSP_FILE = REPOSITORY / "example_data/dmsp/dmsp-f15_ssj_20011021_or_0968.nc"
OUTPUT = REPOSITORY / "figures/debugging/coumans_2004"

START = np.datetime64("2001-10-21T23:36:40")
#END = np.datetime64("2001-10-21T23:40:50")
END = np.datetime64("2001-10-21T23:59:59")
FRAME_HALF_SUPPORT_SECONDS = 60
DMSP_SMOOTHING_SECONDS = 10
MAX_ENERGY_KEV = 15


#%% Small calculation helpers

def moving_average(values, width):
    """Return a centred, NaN-aware moving average."""

    valid = np.isfinite(values)
    kernel = np.ones(width)
    total = np.convolve(np.where(valid, values, 0), kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")

    mean = np.full(values.shape, np.nan)
    enough_data = count >= width // 2 + 1
    mean[enough_data] = total[enough_data] / count[enough_data]
    return mean


def nearest_grid_cells(xi, eta, track_xi, track_eta):
    """Find the nearest regular Cubed-Sphere cell for every track point."""

    xi_index = np.argmin(np.abs(xi[0, :, None] - track_xi[None, :]), axis=0)
    eta_index = np.argmin(np.abs(eta[:, 0, None] - track_eta[None, :]), axis=0)
    return eta_index, xi_index


#%% Read and combine IMAGE and DMSP

def read_comparison_data(footprint_shift_seconds=0):
    """Read the two products and sample IMAGE along the centred DMSP track."""

    image = icload(IMAGE_FILE)
    image_times = np.asarray(image.time, dtype="datetime64[ns]")

    with xr.open_dataset(DMSP_FILE) as data:
        all_times = np.asarray(data.time.values)
        in_interval = (all_times >= START) & (all_times <= END)

        dmsp_times = all_times[in_interval]
        dmsp_energy = np.asarray(data.ELE_AVG_ENERGY.values)[in_interval] / 1000

        if footprint_shift_seconds == 0:
            footprint_index = np.flatnonzero(in_interval)
        else:
            footprint_times = (
                dmsp_times
                + np.timedelta64(footprint_shift_seconds, "s")
            )
            distance = np.abs(all_times[:, None] - footprint_times[None, :])
            footprint_index = np.argmin(distance, axis=0)

        dmsp_mlat = np.asarray(data.footprint_qd_lat.values)[footprint_index]
        dmsp_mlt = np.asarray(data.footprint_mlt.values)[footprint_index]

    # Each IMAGE frame represents the track within +/- 60 s of its centre time.
    time_difference = dmsp_times[:, None] - image_times[None, :]
    frame_index = np.argmin(np.abs(time_difference), axis=1)
    nearest_difference = time_difference[np.arange(dmsp_times.size), frame_index]
    keep = (np.abs(nearest_difference) <= np.timedelta64(FRAME_HALF_SUPPORT_SECONDS, "s"))

    dmsp_times = dmsp_times[keep]
    dmsp_mlat = dmsp_mlat[keep]
    dmsp_mlt = dmsp_mlt[keep]
    dmsp_energy = dmsp_energy[keep]
    frame_index = frame_index[keep]
    time_offset = nearest_difference[keep] / np.timedelta64(1, "s")

    # Sample corrected IMAGE counts at the mapped DMSP footprints.
    track_xi, track_eta = image.grid.projection.geo2cube(
        dmsp_mlt * 15,
        dmsp_mlat,
    )
    eta_index, xi_index = nearest_grid_cells(
        image.grid.xi,
        image.grid.eta,
        track_xi,
        track_eta,
    )

    wic = image.wic_corrected[frame_index, eta_index, xi_index]
    si13 = image.si13_corrected[frame_index, eta_index, xi_index]

    ratio = np.full(wic.shape, np.nan)
    valid = np.isfinite(wic) & np.isfinite(si13) & (si13 != 0)
    np.divide(wic, si13, out=ratio, where=valid)

    mean_energy = np.asarray(fE0m(ratio), dtype=float)
    mean_energy = np.minimum(mean_energy, MAX_ENERGY_KEV)

    return {
        "image_times": image_times,
        "dmsp_times": dmsp_times,
        "dmsp_energy": dmsp_energy,
        "dmsp_energy_smooth": moving_average(
            dmsp_energy, DMSP_SMOOTHING_SECONDS
        ),
        "time_offset": time_offset,
        "ratio": ratio,
        "mean_energy": mean_energy,
        "footprint_shift_seconds": footprint_shift_seconds,
    }


#%% Figure

def make_figure(comparison):
    """Plot the centred IMAGE/DMSP comparison."""

    figure, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)

    axes[0].plot(
        comparison["dmsp_times"],
        comparison["dmsp_energy"],
        ".",
        color="0.75",
        markersize=3,
        label="DMSP SSJ, 1 s",
    )
    axes[0].plot(
        comparison["dmsp_times"],
        comparison["dmsp_energy_smooth"],
        color="black",
        linewidth=2,
        label=f"DMSP, {DMSP_SMOOTHING_SECONDS}-s mean",
    )
    axes[0].plot(
        comparison["dmsp_times"],
        comparison["mean_energy"],
        color="crimson",
        linewidth=1.8,
        label="IMAGE WIC/SI13, capped at 15 keV",
    )
    axes[0].set_ylim(0, MAX_ENERGY_KEV)
    axes[0].set_ylabel("Mean electron energy [keV]")
    axes[0].legend(frameon=False)

    axes[1].plot(
        comparison["dmsp_times"],
        comparison["ratio"],
        color="tab:purple",
        linewidth=1.5,
    )
    axes[1].axhline(
        np.max(wic_to_s13),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Frey table maximum",
    )
    axes[1].set_ylabel("WIC/SI13")
    axes[1].set_xlabel("UT on 2001-10-21")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(START.astype(object), END.astype(object))
        for time in comparison["image_times"]:
            axis.axvline(time, color="0.6", linestyle=":", linewidth=0.8)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    title = (
        "Coumans et al. (2004), Figure 4b reconstruction\n"
        "IMAGE frames use centred +/- 60-s DMSP support"
    )
    if comparison["footprint_shift_seconds"]:
        title += (
            "; DMSP footprint shift "
            f"{comparison['footprint_shift_seconds']:+d} s"
        )
    figure.suptitle(title)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT / "coumans_2004_figure4b_reconstruction"
    figure.savefig(filename.with_suffix(".png"), dpi=200)
    figure.savefig(filename.with_suffix(".pdf"))
    plt.close(figure)


def report(comparison):
    """Print a short summary of one comparison."""

    finite = np.isfinite(comparison["ratio"])
    peak = np.nanargmax(comparison["ratio"])

    print(f"Centred matches: {comparison['dmsp_times'].size}")
    print(f"Finite IMAGE ratios: {finite.sum()}")
    print(
        "IMAGE-ratio peak: "
        f"{np.datetime_as_string(comparison['dmsp_times'][peak], unit='s')}"
    )
    print(
        "DMSP - IMAGE time offset: "
        f"{comparison['time_offset'].min():.0f} to "
        f"{comparison['time_offset'].max():.0f} s"
    )


#%% Run

def main():
    comparison = read_comparison_data()
    make_figure(comparison)
    report(comparison)


if __name__ == "__main__":
    main()

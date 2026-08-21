"""Plot the DMSP F15 track directly on a native WIC IDL image.

This is a Figure 4a-style geometry check for Coumans et al. (2004).  It does
not use the fuvpy orbit file, a binned IMAGE product, or the Cubed-Sphere
grid.  Each DMSP footprint is matched directly to the nearest native WIC
detector pixel using the geographic coordinates stored in the IDL record.

The paper used a WIC frame at 23:35:29 UTC. That frame is not available in
the current example data, so the earliest available frame (about 23:37:32
UTC) is used here. The complete pass is shown for context, while the segment
within one minute of the central IMAGE snapshot time is highlighted.
"""

#%% Imports

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.io import readsav
from scipy.spatial import cKDTree
import xarray as xr


#%% Files and comparison interval

REPOSITORY = Path(__file__).resolve().parents[2]

WIC_FILE = REPOSITORY / "example_data/wic_data/wic20012942337.idl"
DMSP_FILE = (
    REPOSITORY
    / "example_data/dmsp/dmsp-f15_ssj_20011021_or_0968.nc"
)
OUTPUT = REPOSITORY / "figures/debugging/coumans_2004"

START = np.datetime64("2001-10-21T23:36:40")
END = np.datetime64("2001-10-21T23:40:50")
FRAME_HALF_SUPPORT_SECONDS = 60


#%% Read the native WIC record and mapped DMSP track

def idl_time(record):
    """Convert the two-part IDL time field to a Python datetime."""

    year_and_day = int(record["TIME"][0])
    milliseconds = int(record["TIME"][1])

    start_of_day = datetime.strptime(str(year_and_day), "%Y%j")
    return start_of_day + timedelta(milliseconds=milliseconds)


def read_wic(filename):
    """Read the native detector image and its per-pixel geolocation."""

    record = readsav(filename, python_dict=True)["imageinfo"][0]

    return {
        "time": idl_time(record),
        "image": np.asarray(record["IMAGE"], dtype=float),
        "glat": np.asarray(record["GLAT"], dtype=float),
        "glon": np.asarray(record["GLON"], dtype=float),
        "emission_height": float(record["EMIS_HGT"]),
    }


def read_dmsp(filename):
    """Read the DMSP footprints already mapped to the IMAGE emission shell."""

    with xr.open_dataset(filename) as data:
        time = np.asarray(data.time.values)
        keep = (time >= START) & (time <= END)

        return {
            "time": time[keep],
            "glat": np.asarray(data.footprint_geodetic_lat.values)[keep],
            "glon": np.asarray(data.footprint_geodetic_lon.values)[keep],
            "height": float(data.attrs["footprint_height_km"]),
        }


#%% Match geographic footprints to native detector pixels

def unit_vectors(latitude, longitude):
    """Convert geographic latitude and longitude to Cartesian unit vectors."""

    latitude = np.radians(latitude)
    longitude = np.radians(longitude)

    return np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )


def detector_track(wic, dmsp):
    """Find the nearest geolocated WIC pixel for every DMSP footprint."""

    valid = (
        np.isfinite(wic["glat"])
        & np.isfinite(wic["glon"])
        & (wic["glat"] >= -90)
        & (wic["glat"] <= 90)
    )

    valid_pixels = np.flatnonzero(valid)
    wic_position = unit_vectors(wic["glat"][valid], wic["glon"][valid])
    dmsp_position = unit_vectors(dmsp["glat"], dmsp["glon"])

    distance, nearest = cKDTree(wic_position).query(dmsp_position)
    detector_index = valid_pixels[nearest]
    row, column = np.unravel_index(detector_index, wic["image"].shape)

    # Chord length on a unit sphere converted to an angular separation.
    separation = np.degrees(2 * np.arcsin(np.clip(distance / 2, 0, 1)))
    return row, column, separation


#%% Figure

def make_figure(wic, dmsp, row, column, separation):
    """Save the native detector-space WIC image and DMSP track."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    positive = wic["image"][wic["image"] > 0]
    upper = np.percentile(positive, 99.7)

    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    colours = axis.imshow(
        wic["image"],
        origin="lower",
        cmap="inferno",
        norm=PowerNorm(gamma=0.45, vmin=0, vmax=upper),
    )

    image_time = np.datetime64(wic["time"])
    time_difference = dmsp["time"] - image_time
    frame_support = (
        np.abs(time_difference)
        <= np.timedelta64(FRAME_HALF_SUPPORT_SECONDS, "s")
    )

    full_track = axis.plot(
        column,
        row,
        color="0.65",
        linestyle="--",
        linewidth=1.2,
        label="Complete DMSP F15 pass",
    )[0]
    full_track.set_path_effects(
        [path_effects.Stroke(linewidth=2.6, foreground="black"), path_effects.Normal()]
    )

    track = axis.plot(
        column[frame_support],
        row[frame_support],
        color="white",
        linewidth=1.7,
        label="DMSP track within IMAGE time +/- 60 s",
    )[0]
    track.set_path_effects(
        [path_effects.Stroke(linewidth=3.2, foreground="black"), path_effects.Normal()]
    )

    axis.scatter(
        column[frame_support][0], row[frame_support][0],
        color="limegreen", edgecolor="black", s=45, zorder=4,
        label=np.datetime_as_string(dmsp["time"][frame_support][0], unit="s")[-8:],
    )
    axis.scatter(
        column[frame_support][-1], row[frame_support][-1],
        color="red", edgecolor="black", s=45, zorder=4,
        label=np.datetime_as_string(dmsp["time"][frame_support][-1], unit="s")[-8:],
    )

    for minute in range(37, 41):
        mark_time = np.datetime64(f"2001-10-21T23:{minute:02d}:00")
        index = int(np.argmin(np.abs(dmsp["time"] - mark_time)))
        axis.scatter(
            column[index], row[index],
            color="white", edgecolor="black", s=20, zorder=4,
        )

    image_time_text = wic["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    axis.set_title(
        "DMSP F15 track on the native WIC IDL image\n"
        f"Central IMAGE snapshot time: {image_time_text} UTC; support +/- 60 s\n"
        "Coumans Figure 4a used 23:35:29 UTC"
    )
    axis.set_xlabel("WIC detector column")
    axis.set_ylabel("WIC detector row")
    axis.set_xlim(0, wic["image"].shape[1] - 1)
    axis.set_ylim(0, wic["image"].shape[0] - 1)
    axis.set_aspect("equal")
    axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="none",
        loc="lower left",
    )

    figure.colorbar(colours, ax=axis, label="IDL IMAGE [counts]", shrink=0.88)

    filename = OUTPUT / "coumans_2004_figure4a_idl_wic_track"
    figure.savefig(filename.with_suffix(".png"), dpi=220)
    figure.savefig(filename.with_suffix(".pdf"))
    plt.close(figure)

    print(f"WIC central snapshot time: {image_time_text} UTC")
    print(
        "Centered DMSP support: "
        f"{np.datetime_as_string(dmsp['time'][frame_support][0], unit='s')} to "
        f"{np.datetime_as_string(dmsp['time'][frame_support][-1], unit='s')} UTC"
    )
    print(f"IMAGE emission height: {wic['emission_height']:.0f} km")
    print(f"DMSP footprint height: {dmsp['height']:.0f} km")
    print(f"Track uses {len(set(zip(row, column)))} native detector pixels")
    print(
        "Nearest-pixel angular separation: "
        f"median {np.median(separation):.3f} deg, "
        f"maximum {np.max(separation):.3f} deg"
    )


def main():
    wic = read_wic(WIC_FILE)
    dmsp = read_dmsp(DMSP_FILE)
    row, column, separation = detector_track(wic, dmsp)
    make_figure(wic, dmsp, row, column, separation)


if __name__ == "__main__":
    main()

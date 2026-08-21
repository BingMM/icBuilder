"""Compare one processed IMAGE orbit with all locally available DMSP passes."""

#%% Imports

import argparse
import csv
import re
from pathlib import Path

from apexpy import Apex
from astropy.coordinates import EarthLocation
import astropy.units as u
from cdflib import CDF, cdfepoch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
from secsy import CSprojection

from icphysics.image import fE0m, wic_to_s13


#%% Read the IMAGE product

def read_image(filename):
    """Read corrected channels, times, and the grid stored with one orbit."""

    with Dataset(filename) as data:
        time_variable = data.variables["time"]
        times = num2date(
            time_variable[:],
            time_variable.units,
            only_use_cftime_datetimes=False,
        )

        grid = data.groups["grid"]
        radius = float(getattr(grid, "R", 6_501_200))

        return {
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "wic": np.asarray(data.variables["wic_corrected"][:], dtype=float),
            "si13": np.asarray(data.variables["si13_corrected"][:], dtype=float),
            "xi": np.asarray(grid.variables["xi"][:], dtype=float),
            "eta": np.asarray(grid.variables["eta"][:], dtype=float),
            "projection": CSprojection(grid.position, grid.orientation),
            "height": radius / 1000 - 6371.2,
            "method": str(getattr(data, "method", "unknown")),
        }


#%% Find and read local DMSP files

def dates_between(start, end):
    """Return compact UTC dates touched by one IMAGE orbit."""

    first = np.datetime64(start, "D")
    last = np.datetime64(end, "D")
    days = np.arange(first, last + np.timedelta64(1, "D"))
    return [np.datetime_as_string(day).replace("-", "") for day in days]


def find_dmsp_files(root, start, end):
    """Group available daily SSJ CDFs by satellite."""

    files = []
    for date in dates_between(start, end):
        files.extend(root.rglob(f"dmsp-f*_ssj_*_{date}_v*.cdf"))

    newest = {}
    for filename in sorted(set(files)):
        match = re.search(
            r"dmsp-(f\d+)_ssj_.*_(\d{8})_v.*\.cdf",
            filename.name.lower(),
        )
        if match:
            key = (match.group(1), match.group(2))
            if key not in newest or filename.name > newest[key].name:
                newest[key] = filename

    grouped = {}
    for (satellite, _), filename in sorted(newest.items()):
        grouped.setdefault(satellite, []).append(filename)

    return grouped


def read_dmsp(files, start, end):
    """Combine the part of several daily files overlapping the IMAGE orbit."""

    values = {
        "time": [],
        "geocentric_lat": [],
        "geocentric_lon": [],
        "radius": [],
        "energy": [],
    }

    for filename in files:
        cdf = CDF(filename)
        epoch = np.asarray(cdf.varget("Epoch"))
        times = np.asarray(cdfepoch.to_datetime(epoch), dtype="datetime64[ns]")
        keep = (times >= start) & (times <= end)
        if not np.any(keep):
            continue

        values["time"].append(times[keep])
        values["geocentric_lat"].append(np.asarray(cdf.varget("SC_GEOCENTRIC_LAT"))[keep])
        values["geocentric_lon"].append(np.asarray(cdf.varget("SC_GEOCENTRIC_LON"))[keep])
        values["radius"].append(np.asarray(cdf.varget("SC_GEOCENTRIC_R"))[keep])
        values["energy"].append(np.asarray(cdf.varget("ELE_AVG_ENERGY"))[keep] / 1000)

    if not values["time"]:
        return None

    combined = {name: np.concatenate(parts) for name, parts in values.items()}
    order = np.argsort(combined["time"])
    return {name: array[order] for name, array in combined.items()}


#%% Map DMSP into the IMAGE coordinate system

def map_dmsp(dmsp, image):
    """Map spacecraft positions to modified Apex at the IMAGE grid height."""

    latitude = np.radians(dmsp["geocentric_lat"])
    longitude = np.radians(dmsp["geocentric_lon"])
    radius = dmsp["radius"]

    x = radius * np.cos(latitude) * np.cos(longitude)
    y = radius * np.cos(latitude) * np.sin(longitude)
    z = radius * np.sin(latitude)

    spacecraft = EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)
    height = spacecraft.height.to_value(u.km)

    python_times = dmsp["time"].astype("datetime64[us]").astype(object)
    apex = Apex(date=python_times[0], refh=image["height"])
    mlat, mlon = apex.geo2apex(
        spacecraft.lat.deg,
        spacecraft.lon.deg,
        height,
    )
    mlt = np.array([
        apex.mlon2mlt(lon, time)
        for lon, time in zip(mlon, python_times)
    ])

    dmsp["mlat"] = mlat
    dmsp["mlt"] = mlt


def nearest_indices(values, targets):
    """Find the nearest sorted value for every target."""

    after = np.searchsorted(values, targets)
    before = np.clip(after - 1, 0, values.size - 1)
    after = np.clip(after, 0, values.size - 1)

    use_after = np.abs(values[after] - targets) < np.abs(values[before] - targets)
    return np.where(use_after, after, before)


def grid_edges(centres):
    """Return physical edges surrounding regular cell centres."""

    edges = np.empty(centres.size + 1)
    edges[1:-1] = (centres[:-1] + centres[1:]) / 2
    edges[0] = centres[0] - (centres[1] - centres[0]) / 2
    edges[-1] = centres[-1] + (centres[-1] - centres[-2]) / 2
    return edges


#%% Sample IMAGE and separate passes

def sample_image(dmsp, image, maximum_time_gap):
    """Sample corrected WIC and SI13 along the part of the track on the grid."""

    track_xi, track_eta = image["projection"].geo2cube(
        dmsp["mlt"] * 15,
        dmsp["mlat"],
        set_points_off_cube_to_nan=True,
    )

    xi_centres = image["xi"][0]
    eta_centres = image["eta"][:, 0]
    xi_edges = grid_edges(xi_centres)
    eta_edges = grid_edges(eta_centres)

    inside = (
        np.isfinite(track_xi)
        & np.isfinite(track_eta)
        & (track_xi >= xi_edges[0])
        & (track_xi <= xi_edges[-1])
        & (track_eta >= eta_edges[0])
        & (track_eta <= eta_edges[-1])
    )

    frame = nearest_indices(image["time"], dmsp["time"])
    time_gap = np.abs(dmsp["time"] - image["time"][frame])
    inside &= time_gap <= np.timedelta64(maximum_time_gap, "s")

    xi_index = nearest_indices(xi_centres, track_xi)
    eta_index = nearest_indices(eta_centres, track_eta)
    wic = image["wic"][frame, eta_index, xi_index]
    si13 = image["si13"][frame, eta_index, xi_index]

    ratio = np.full(dmsp["time"].shape, np.nan)
    valid_ratio = inside & np.isfinite(wic) & np.isfinite(si13) & (wic > 0) & (si13 > 0)
    ratio[valid_ratio] = wic[valid_ratio] / si13[valid_ratio]

    energy = np.full(ratio.shape, np.nan)
    energy[valid_ratio] = fE0m(ratio[valid_ratio])

    dmsp["inside"] = inside
    dmsp["ratio"] = ratio
    dmsp["image_energy"] = energy


def pass_indices(dmsp, minimum_samples):
    """Split the on-grid track into continuous one-second passes."""

    indices = np.flatnonzero(dmsp["inside"])
    if indices.size == 0:
        return []

    gaps = np.diff(dmsp["time"][indices]) > np.timedelta64(5, "s")
    sections = np.split(indices, np.flatnonzero(gaps) + 1)
    return [section for section in sections if section.size >= minimum_samples]


#%% Plot each useful pass

def moving_average(values, width):
    """Return a centred NaN-aware moving average."""

    finite = np.isfinite(values)
    total = np.convolve(np.where(finite, values, 0), np.ones(width), mode="same")
    count = np.convolve(finite.astype(float), np.ones(width), mode="same")
    output = np.full(values.shape, np.nan)
    use = count >= width // 2 + 1
    output[use] = total[use] / count[use]
    return output


def plot_pass(orbit, satellite, dmsp, indices, output_dir, smoothing):
    """Save one compact Coumans-style pass comparison."""

    time = dmsp["time"][indices]
    dmsp_energy = dmsp["energy"][indices]
    image_energy = np.minimum(dmsp["image_energy"][indices], 15)
    ratio = dmsp["ratio"][indices]
    mlat = dmsp["mlat"][indices]

    figure, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
        constrained_layout=True,
    )

    axes[0].plot(time, dmsp_energy, color="0.75", linewidth=0.7, label="DMSP SSJ, 1 s")
    axes[0].plot(time, moving_average(dmsp_energy, smoothing), color="black", linewidth=1.8, label=f"DMSP, {smoothing}-s mean")
    axes[0].plot(time, image_energy, color="crimson", linewidth=1.5, label="IMAGE WIC/SI13, capped at 15 keV")
    axes[0].set_ylabel("Mean energy [keV]")
    axes[0].set_ylim(0, 15)
    axes[0].legend(frameon=False)

    axes[1].plot(time, ratio, color="tab:purple", linewidth=1.3)
    axes[1].axhline(np.max(wic_to_s13), color="black", linestyle="--", linewidth=1, label="Frey table maximum")
    axes[1].set_ylabel("WIC/SI13")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)

    axes[2].plot(time, mlat, color="tab:blue", linewidth=1.5)
    axes[2].set_ylabel("MLAT [deg]")
    axes[2].set_xlabel("UTC")

    for axis in axes:
        axis.grid(alpha=0.2)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    start = np.datetime_as_string(time[0], unit="s")
    end = np.datetime_as_string(time[-1], unit="s")
    figure.suptitle(f"IMAGE orbit {orbit:04d} and DMSP {satellite.upper()}\n{start} to {end} UTC")

    label = start.replace("-", "").replace(":", "")
    output = output_dir / f"{satellite}_{label}.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


#%% Command line

def parse_args():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orbit", type=int)
    parser.add_argument("--base", type=Path, default=Path("/home/bing/dtu_server/IMAGE_FUV"))
    parser.add_argument("--image-folder", default="=precipitation_IR_P2")
    parser.add_argument("--dmsp-root", type=Path, default=Path("/media/bing/LaCie/dmsp_ssj"))
    parser.add_argument("--output-dir", type=Path, default=repository / "figures/debugging/dmsp")
    parser.add_argument("--maximum-time-gap-seconds", type=int, default=75)
    parser.add_argument("--minimum-pass-samples", type=int, default=60)
    parser.add_argument("--smoothing-seconds", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    image_file = args.base / args.image_folder / f"or_{args.orbit:04d}.nc"
    image = read_image(image_file)
    dmsp_files = find_dmsp_files(args.dmsp_root, image["time"][0], image["time"][-1])

    output_dir = args.output_dir / f"or_{args.orbit:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for satellite, files in sorted(dmsp_files.items()):
        dmsp = read_dmsp(files, image["time"][0], image["time"][-1])
        if dmsp is None:
            continue

        map_dmsp(dmsp, image)
        sample_image(dmsp, image, args.maximum_time_gap_seconds)

        for indices in pass_indices(dmsp, args.minimum_pass_samples):
            output = plot_pass(
                args.orbit, satellite, dmsp, indices,
                output_dir, args.smoothing_seconds,
            )
            finite = np.isfinite(dmsp["ratio"][indices])
            summary.append({
                "satellite": satellite.upper(),
                "start": np.datetime_as_string(dmsp["time"][indices[0]], unit="s"),
                "end": np.datetime_as_string(dmsp["time"][indices[-1]], unit="s"),
                "dmsp_samples": indices.size,
                "image_samples": int(finite.sum()),
                "figure": output.name,
            })

    summary_file = output_dir / "pass_summary.csv"
    with summary_file.open("w", newline="") as stream:
        fields = ["satellite", "start", "end", "dmsp_samples", "image_samples", "figure"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    print(f"IMAGE file: {image_file}")
    print(f"IMAGE method: {image['method']}; grid height: {image['height']:.1f} km")
    print(f"Available satellites: {', '.join(name.upper() for name in sorted(dmsp_files)) or 'none'}")
    print(f"Useful passes: {len(summary)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

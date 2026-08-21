"""Compare fuvpy background choices without changing downstream physics.

The script reprocesses the small example orbits with the current icBuilder
parameters and the camera-specific parameters from Ohma et al. (2024).  The
temporary fuvpy NetCDF files live under /tmp by default.  Only figures and a
CSV summary are written inside the repository.
"""

#%% Imports

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import apexpy
import fuvpy as fuv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apexpy.helpers import subsol
from netCDF4 import Dataset

from icbuilder import BinnedImage, PreImage, PrecipitationImage
from icbuilder.grids import make_image_grids
from icbuilder.kp import load_gfz_kp
from icphysics.image import wic_to_s13


#%% Processing configurations

SENSORS = {
    "WIC": {"folder": "wic", "data_folder": "wic_data", "prefix": "wic", "reflat": True},
    "SI12": {"folder": "s12", "data_folder": "s12_data", "prefix": "s12", "reflat": False},
    "SI13": {"folder": "s13", "data_folder": "s13_data", "prefix": "s13", "reflat": False},
}

PARAMETERS = {
    "current": {
        "WIC": (1e-3, 1e-4),
        "SI12": (1e-3, 1e-4),
        "SI13": (1e-3, 1e-4),
    },
    "publication": {
        "WIC": (1e-2, 1e-4),
        "SI12": (1e-1, 10.0),
        "SI13": (1e-1, 10.0),
    },
}

CORRECTIONS = ("DG", "SH")


#%% fuvpy processing

def orbit_files(base, sensor, orbit):
    """Read the historical file table and return one orbit's IDL files."""

    settings = SENSORS[sensor]
    table = pd.read_hdf(base / f"{settings['prefix']}files.h5", key="data")
    filenames = table.loc[table["orbit"] == orbit, "filename"]
    return [base / settings["data_folder"] / filename for filename in filenames]


def process_fuvpy_orbit(base, scratch, configuration, sensor, orbit, overwrite=False):
    """Run the published two-stage background model for one camera orbit."""

    settings = SENSORS[sensor]
    output = scratch / configuration / settings["folder"] / f"{settings['prefix']}_or{orbit:04d}.nc"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and not overwrite:
        return output

    files = orbit_files(base, sensor, orbit)
    if not files:
        raise FileNotFoundError(f"no {sensor} files found for orbit {orbit:04d}")

    bs_damping, sh_damping = PARAMETERS[configuration][sensor]
    print(
        f"{configuration:11s} {sensor:4s} orbit {orbit:04d}: "
        f"BS={bs_damping:g}, SH={sh_damping:g}"
    )

    images = fuv.read_idl(
        [str(filename) for filename in files],
        dzalim=75,
        reflat=settings["reflat"],
    )
    images = images.sel(date=images.hemisphere.date[images.hemisphere == "north"])
    images = images.assign(t_start=np.datetime_as_string(images["date"][0], unit="s"))

    images = fuv.backgroundmodel_BS(
        images,
        sKnots=[-3.5, -0.25, 0, 0.25, 1.5, 3.5],
        stop=0.01,
        n_tKnots=5,
        tukeyVal=5,
        dampingVal=bs_damping,
    )
    images = fuv.backgroundmodel_SH(
        images,
        4,
        4,
        n_tKnots=5,
        stop=0.01,
        tukeyVal=5,
        dampingVal=sh_damping,
    )

    encoding = {name: {"zlib": True, "complevel": 4} for name in images.data_vars}
    images.to_netcdf(output, format="NETCDF4", encoding=encoding)
    return output


#%% Native-grid binning

def safe_apex_convert(apex, time, glat, glon, height=110):
    """Convert only finite geographic pixels in one frame."""

    valid = np.isfinite(glat) & np.isfinite(glon)
    mlat = np.full_like(glat, np.nan)
    mlon = np.full_like(glon, np.nan)
    mlt = np.full_like(glon, np.nan)
    ssalon = np.nan

    if np.any(valid):
        mlat_valid, mlon_valid = apex.convert(
            glat[valid], glon[valid], "geo", "apex", height=height
        )
        mlat[valid] = mlat_valid
        mlon[valid] = mlon_valid

        subsolar_lat, subsolar_lon = subsol(time)
        _, ssalon = apex.geo2apex(subsolar_lat, subsolar_lon, 318550)
        mlt[valid] = (180 + mlon_valid - ssalon) / 15 % 24

    return mlat, mlon, mlt, ssalon


def bin_fuvpy_file(filename, sensor, grid):
    """Bin both the DG and SH fields using identical pixels and geometry."""

    with Dataset(filename) as nc:
        keep = ~np.all(np.isnan(nc.variables["mlat"][:]), axis=(1, 2))
        indices = np.flatnonzero(keep)

        start = datetime.strptime(nc.variables["t_start"][:], "%Y-%m-%dT%H:%M:%S")
        seconds = np.asarray(nc.variables["date"][:], dtype=int)
        time = np.array([start + timedelta(seconds=int(value)) for value in seconds], dtype=object)[keep]
        preimage = PreImage(sensor, nc, indices)

    for frame, frame_time in enumerate(time):
        apex = apexpy.Apex(frame_time)
        converted = safe_apex_convert(
            apex, frame_time, preimage.glat[frame], preimage.glon[frame]
        )
        preimage.mlat[frame], preimage.mlon[frame], preimage.mlt[frame], preimage.ssalon[frame] = converted

    keep = preimage.percent_full(grid) >= 0.1
    time = time[keep]
    preimage.discard(keep)

    binned = {}
    for correction in CORRECTIONS:
        binned[correction] = BinnedImage(
            preimage,
            grid,
            time,
            inflate_uncertainty=True,
            correction=correction,
            los_correction=False,
        )
    return binned


#%% Ratio summaries

def ratio_statistics(wic, si13):
    """Summarize the part of WIC/SI13 that enters the Frey lookup."""

    finite = np.isfinite(wic) & np.isfinite(si13)
    strong = finite & (wic > 0) & (si13 >= 3)

    ratio = np.full(wic.shape, np.nan)
    ratio[strong] = wic[strong] / si13[strong]
    inside = strong & (ratio >= wic_to_s13.min()) & (ratio <= wic_to_s13.max())

    return {
        "finite_pixels": int(finite.sum()),
        "strong_pixels": int(strong.sum()),
        "median_ratio": float(np.nanmedian(ratio[strong])),
        "inside_table_percent": 100 * inside.sum() / max(strong.sum(), 1),
        "ratio": ratio,
        "strong": strong,
    }


def build_products(binned, kp_series, proton_energy):
    """Run the unchanged Product-2 ratio path for all correction choices."""

    products = {}
    for wic_correction in CORRECTIONS:
        for si12_correction in CORRECTIONS:
            for si13_correction in CORRECTIONS:
                key = (wic_correction, si12_correction, si13_correction)
                products[key] = PrecipitationImage(
                    binned["WIC"][wic_correction],
                    binned["SI12"][si12_correction],
                    "image_ratio",
                    si13=binned["SI13"][si13_correction],
                    kp_series=kp_series,
                    proton_energy=proton_energy,
                )
    return products


def summarize_products(configuration, orbit, products):
    """Calculate native-support and identical-pixel ratio comparisons."""

    stage_values = {"before": {}, "after": {}}
    for key, product in products.items():
        stage_values["before"][key] = (product.wic, product.si13)
        stage_values["after"][key] = (product.wic_corrected, product.si13_corrected)

    rows = []
    for stage, values in stage_values.items():
        statistics = {key: ratio_statistics(*arrays) for key, arrays in values.items()}
        common = np.logical_and.reduce([item["strong"] for item in statistics.values()])

        for key, item in statistics.items():
            common_ratio = item["ratio"][common]
            common_inside = (
                (common_ratio >= wic_to_s13.min())
                & (common_ratio <= wic_to_s13.max())
            )
            rows.append({
                "configuration": configuration,
                "orbit": orbit,
                "stage": stage,
                "wic_correction": key[0],
                "si12_correction": key[1],
                "si13_correction": key[2],
                "finite_pixels": item["finite_pixels"],
                "strong_pixels": item["strong_pixels"],
                "median_ratio": item["median_ratio"],
                "inside_table_percent": item["inside_table_percent"],
                "common_strong_pixels": int(common.sum()),
                "common_median_ratio": float(np.nanmedian(common_ratio)),
                "common_inside_table_percent": 100 * common_inside.sum() / max(common.sum(), 1),
            })
    return rows


def summarize_dza(configuration, orbit, product, wic_dza):
    """Summarize the production-choice ratio in fixed WIC-DZA intervals."""

    before = ratio_statistics(product.wic, product.si13)
    after = ratio_statistics(product.wic_corrected, product.si13_corrected)

    # Use the same physical cells before and after proton correction. This
    # prevents a changing SI13 threshold from masquerading as a DZA effect.
    common = before["strong"] & after["strong"] & np.isfinite(wic_dza)
    dza_edges = [0, 30, 45, 60, 75]
    rows = []

    for lower, upper in zip(dza_edges[:-1], dza_edges[1:]):
        selected = common & (wic_dza >= lower) & (wic_dza < upper)
        for stage, statistics in (("before", before), ("after", after)):
            ratio = statistics["ratio"][selected]
            inside = (
                (ratio >= wic_to_s13.min())
                & (ratio <= wic_to_s13.max())
            )
            rows.append({
                "configuration": configuration,
                "orbit": orbit,
                "stage": stage,
                "dza_min": lower,
                "dza_max": upper,
                "pixels": int(selected.sum()),
                "median_ratio": float(np.nanmedian(ratio)) if ratio.size else np.nan,
                "inside_table_percent": 100 * inside.sum() / max(ratio.size, 1),
            })
    return rows


#%% Figures

def select_row(rows, configuration, orbit, stage, wic, si12, si13):
    """Return one row from the small diagnostic table."""

    for row in rows:
        if (
            row["configuration"] == configuration
            and row["orbit"] == orbit
            and row["stage"] == stage
            and row["wic_correction"] == wic
            and row["si12_correction"] == si12
            and row["si13_correction"] == si13
        ):
            return row
    raise KeyError("summary row not found")


def plot_pre_proton(rows, orbits, output):
    """Compare WIC/SI13 background choices before proton correction."""

    combinations = [("DG", "DG"), ("DG", "SH"), ("SH", "DG"), ("SH", "SH")]
    labels = [f"WIC {wic}\nSI13 {si13}" for wic, si13 in combinations]
    figure, axes = plt.subplots(2, len(orbits), figsize=(10, 7), sharey="row", constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(orbits))

    for column, orbit in enumerate(orbits):
        for configuration, marker in (("current", "o"), ("publication", "s")):
            selected = [
                select_row(rows, configuration, orbit, "before", wic, "DG", si13)
                for wic, si13 in combinations
            ]
            axes[0, column].plot(
                labels, [row["common_median_ratio"] for row in selected],
                marker=marker, label=configuration,
            )
            axes[1, column].plot(
                labels, [row["common_inside_table_percent"] for row in selected],
                marker=marker, label=configuration,
            )

        axes[0, column].axhspan(wic_to_s13.min(), wic_to_s13.max(), color="tab:green", alpha=0.12)
        axes[0, column].set_title(f"Orbit {orbit:04d}")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].tick_params(axis="x", rotation=20)

    axes[0, 0].set_ylabel("median WIC/SI13\nidentical strong pixels")
    axes[1, 0].set_ylabel("inside Frey table [%]\nidentical strong pixels")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Background correction before proton subtraction")
    figure.savefig(output / "fuvpy_pre_proton_comparison.png", dpi=180)
    plt.close(figure)


def plot_si_corrections(rows, orbits, output):
    """Test the user's DG choice for SI12 and SI13 after proton correction."""

    combinations = [("DG", "DG"), ("DG", "SH"), ("SH", "DG"), ("SH", "SH")]
    labels = [f"SI12 {si12}\nSI13 {si13}" for si12, si13 in combinations]
    figure, axes = plt.subplots(2, len(orbits), figsize=(10, 7), sharey="row", constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(orbits))

    for column, orbit in enumerate(orbits):
        for configuration, marker in (("current", "o"), ("publication", "s")):
            selected = [
                select_row(rows, configuration, orbit, "after", "SH", si12, si13)
                for si12, si13 in combinations
            ]
            axes[0, column].plot(
                labels, [row["common_median_ratio"] for row in selected],
                marker=marker, label=configuration,
            )
            axes[1, column].plot(
                labels, [row["common_inside_table_percent"] for row in selected],
                marker=marker, label=configuration,
            )

        axes[0, column].axhspan(wic_to_s13.min(), wic_to_s13.max(), color="tab:green", alpha=0.12)
        axes[0, column].set_title(f"Orbit {orbit:04d}, WIC SH")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].tick_params(axis="x", rotation=20)

    axes[0, 0].set_ylabel("median corrected ratio\nidentical strong pixels")
    axes[1, 0].set_ylabel("inside Frey table [%]\nidentical strong pixels")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Effect of SI12 and SI13 SH corrections after proton subtraction")
    figure.savefig(output / "fuvpy_si_correction_comparison.png", dpi=180)
    plt.close(figure)


def plot_background_maps(binned, configuration, orbit, output):
    """Show what SH changes relative to DG in each binned sensor image."""

    frame = 8
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for row, sensor in enumerate(("WIC", "SI12", "SI13")):
        dg = binned[sensor]["DG"].mu[frame]
        sh = binned[sensor]["SH"].mu[frame]
        difference = sh - dg

        vmax = max(np.nanpercentile(dg, 99), np.nanpercentile(sh, 99), 1)
        diffmax = max(np.nanpercentile(np.abs(difference), 99), 1)
        fields = [
            (dg, "DG", "viridis", 0, vmax),
            (sh, "SH", "viridis", 0, vmax),
            (difference, "SH minus DG", "RdBu_r", -diffmax, diffmax),
        ]
        for column, (values, title, cmap, vmin, vmax_field) in enumerate(fields):
            image = axes[row, column].imshow(
                values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_field
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
        axes[row, 0].set_ylabel(sensor)

    figure.suptitle(f"{configuration.capitalize()} parameters, orbit {orbit:04d}, frame {frame}")
    figure.savefig(output / f"fuvpy_{configuration}_or_{orbit:04d}_background_maps.png", dpi=180)
    plt.close(figure)


def plot_dza(dza_rows, orbits, output):
    """Show ratio dependence on WIC viewing angle for fixed physical cells."""

    figure, axes = plt.subplots(2, len(orbits), figsize=(10, 7), sharex="col", sharey="row", constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(orbits))

    for column, orbit in enumerate(orbits):
        for configuration, color in (("current", "tab:blue"), ("publication", "tab:orange")):
            for stage, linestyle in (("before", "-"), ("after", "--")):
                selected = [
                    row for row in dza_rows
                    if row["configuration"] == configuration
                    and row["orbit"] == orbit
                    and row["stage"] == stage
                ]
                x = [(row["dza_min"] + row["dza_max"]) / 2 for row in selected]
                median_ratio = [
                    row["median_ratio"] if row["pixels"] >= 30 else np.nan
                    for row in selected
                ]
                inside_table = [
                    row["inside_table_percent"] if row["pixels"] >= 30 else np.nan
                    for row in selected
                ]
                label = f"{configuration}, {stage} proton"
                axes[0, column].plot(x, median_ratio, "o", color=color, linestyle=linestyle, label=label)
                axes[1, column].plot(x, inside_table, "o", color=color, linestyle=linestyle)

        axes[0, column].axhspan(wic_to_s13.min(), wic_to_s13.max(), color="tab:green", alpha=0.12)
        axes[0, column].set_title(f"Orbit {orbit:04d}")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("binned WIC DZA [degrees]")

    axes[0, 0].set_ylabel("median WIC/SI13")
    axes[1, 0].set_ylabel("inside Frey table [%]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Production background choice on the same cells before and after proton correction")
    figure.savefig(output / "fuvpy_ratio_by_wic_dza.png", dpi=180)
    plt.close(figure)


#%% Outputs and main program

def write_csv(rows, filename):
    """Write all values used in the summary figures."""

    with filename.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=repository / "example_data")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/icbuilder_fuvpy_debug"))
    parser.add_argument("--output", type=Path, default=repository / "figures" / "debugging")
    parser.add_argument("--orbits", type=int, nargs="+", default=[85, 86])
    parser.add_argument("--proton-energy", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grid_wic, grid_si = make_image_grids()
    sensor_grids = {"WIC": grid_wic, "SI12": grid_si, "SI13": grid_si}
    kp_series = load_gfz_kp()
    rows = []
    dza_rows = []

    for configuration in PARAMETERS:
        for orbit in args.orbits:
            binned = {}
            for sensor in SENSORS:
                filename = process_fuvpy_orbit(
                    args.base,
                    args.scratch,
                    configuration,
                    sensor,
                    orbit,
                    overwrite=args.overwrite,
                )
                binned[sensor] = bin_fuvpy_file(filename, sensor, sensor_grids[sensor])

            products = build_products(binned, kp_series, args.proton_energy)
            rows.extend(summarize_products(configuration, orbit, products))

            production_product = products[("SH", "DG", "DG")]
            wic_indices = production_product.source_indices["wic"]
            wic_dza = binned["WIC"]["SH"].dza[wic_indices]
            dza_rows.extend(
                summarize_dza(
                    configuration,
                    orbit,
                    production_product,
                    wic_dza,
                )
            )
            plot_background_maps(binned, configuration, orbit, args.output)

    write_csv(rows, args.output / "fuvpy_background_summary.csv")
    write_csv(dza_rows, args.output / "fuvpy_dza_summary.csv")
    plot_pre_proton(rows, args.orbits, args.output)
    plot_si_corrections(rows, args.orbits, args.output)
    plot_dza(dza_rows, args.orbits, args.output)

    print("\nCurrent production choice: WIC SH, SI12 DG, SI13 DG")
    for configuration in PARAMETERS:
        for orbit in args.orbits:
            row = select_row(rows, configuration, orbit, "after", "SH", "DG", "DG")
            print(
                f"{configuration:11s} orbit {orbit:04d}: "
                f"median={row['median_ratio']:.1f}, "
                f"inside={row['inside_table_percent']:.1f}%"
            )
    print(f"\nSaved figures and summary in {args.output}")


if __name__ == "__main__":
    main()

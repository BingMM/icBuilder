"""Trace where the IMAGE WIC/SI13 electron-energy retrieval fails.

This script uses the two small example orbits already stored in the
repository.  It does not rewrite any binned, precipitation, or conductance
products.  All output is diagnostic and is written to ``figures/debugging``.
"""

#%% Imports

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from scipy.io import readsav

from icbuilder import PrecipitationImage
from icphysics.image import EE, wic_to_s13


#%% Retrieval categories

CATEGORY_NAMES = [
    "weak WIC and SI13\n$E_0=0.2$",
    "weak SI13\n$E_0=1$",
    "ratio below table\n$E_0=0.2$",
    "inside table\nretrieved $E_0$",
    "ratio above table\n$E_0=25$",
    "no finite ratio",
]


def ratio_and_categories(wic, si13):
    """Calculate WIC/SI13 and reproduce the active retrieval branches."""

    wic = np.asarray(wic, dtype=float)
    si13 = np.asarray(si13, dtype=float)

    finite = np.isfinite(wic) & np.isfinite(si13)
    ratio = np.full(wic.shape, np.nan)
    divide = finite & (wic != 0) & (si13 != 0)
    ratio[divide] = wic[divide] / si13[divide]

    weak_both = finite & (si13 < 3) & (wic < 50)
    weak_si13 = finite & (si13 < 3) & (wic >= 50)
    use_ratio = finite & (si13 >= 3)
    below_table = use_ratio & (ratio < wic_to_s13.min())
    inside_table = use_ratio & (ratio >= wic_to_s13.min()) & (ratio <= wic_to_s13.max())
    above_table = use_ratio & (ratio > wic_to_s13.max())

    classified = weak_both | weak_si13 | below_table | inside_table | above_table
    no_ratio = finite & ~classified

    categories = [
        weak_both,
        weak_si13,
        below_table,
        inside_table,
        above_table,
        no_ratio,
    ]
    return ratio, categories, finite


def summarize_stage(orbit, stage, wic, si13):
    """Return one compact numerical summary of a ratio stage."""

    ratio, categories, finite = ratio_and_categories(wic, si13)
    strong = finite & (wic > 0) & (si13 >= 3)
    inside = strong & (ratio >= wic_to_s13.min()) & (ratio <= wic_to_s13.max())

    row = {
        "orbit": orbit,
        "stage": stage,
        "finite_pixels": int(finite.sum()),
        "strong_pixels": int(strong.sum()),
        "median_strong_ratio": float(np.nanmedian(ratio[strong])),
        "strong_pixels_inside_table_percent": 100 * inside.sum() / max(strong.sum(), 1),
    }

    for name, mask in zip(CATEGORY_NAMES, categories):
        column = name.split("\n")[0].replace(" ", "_") + "_percent"
        row[column] = 100 * mask.sum() / max(finite.sum(), 1)

    return row, ratio, categories


#%% Load the modular calculation without saving a new product

def load_ratio_product(base, orbit, proton_energy):
    """Run Product 2 in memory from the tracked Product-1 example files."""

    binned = base / "binned"
    return PrecipitationImage(
        binned / "wic" / f"or_{orbit:04d}.nc",
        binned / "si12" / f"or_{orbit:04d}.nc",
        "image_ratio",
        si13=binned / "si13" / f"or_{orbit:04d}.nc",
        proton_energy=proton_energy,
    )


#%% Summary figures

def plot_response_domain(products, output_dir):
    """Show the active response curve and the ratios delivered to it."""

    before = []
    after = []
    for product in products.values():
        ratio_before, _, finite_before = ratio_and_categories(product.wic, product.si13)
        ratio_after, _, finite_after = ratio_and_categories(
            product.wic_corrected, product.si13_corrected
        )

        before.append(ratio_before[finite_before & (product.wic > 0) & (product.si13 >= 3)])
        after.append(
            ratio_after[
                finite_after
                & (product.wic_corrected > 0)
                & (product.si13_corrected >= 3)
            ]
        )

    before = np.concatenate(before)
    after = np.concatenate(after)

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), constrained_layout=True)

    axes[0].plot(wic_to_s13, EE, "o-", color="black", label="active Frey-table interpolation")
    axes[0].axhline(15, color="tab:red", linestyle="--", label="15 keV cap used by Coumans et al. (2004)")
    axes[0].axvspan(wic_to_s13.min(), wic_to_s13.max(), color="tab:green", alpha=0.12)
    axes[0].set(xlabel="WIC/SI13 ratio", ylabel="$E_0$ [keV]", xlim=(20, 155), ylim=(0, 27))
    axes[0].legend(frameon=False)
    axes[0].set_title("Response-table domain")

    upper = max(300, np.nanpercentile(np.concatenate([before, after]), 95))
    bins = np.geomspace(max(1, np.nanmin(np.concatenate([before, after]))), upper, 55)
    axes[1].hist(before, bins=bins, density=True, histtype="step", linewidth=2, label="before proton correction")
    axes[1].hist(after, bins=bins, density=True, histtype="step", linewidth=2, label="after proton correction")
    axes[1].axvspan(wic_to_s13.min(), wic_to_s13.max(), color="tab:green", alpha=0.12, label="response-table support")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="WIC/SI13 ratio for SI13 $\geq 3$", ylabel="probability density", xlim=(bins[0], bins[-1]))
    axes[1].legend(frameon=False)
    axes[1].set_title("Ratios in example orbits 0085 and 0086")

    figure.savefig(output_dir / "image_ratio_response_domain.png", dpi=180)
    plt.close(figure)


def plot_category_fractions(summaries, output_dir):
    """Show how much of each orbit is a retrieval, fallback, or saturation."""

    columns = [name.split("\n")[0].replace(" ", "_") + "_percent" for name in CATEGORY_NAMES]
    x = np.arange(len(columns))
    width = 0.2

    figure, axis = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    for i, row in enumerate(summaries):
        values = [row[column] for column in columns]
        label = f"orbit {row['orbit']:04d}, {row['stage']}"
        axis.bar(x + (i - 1.5) * width, values, width, label=label)

    axis.set_xticks(x, CATEGORY_NAMES)
    axis.set_ylabel("fraction of finite pixels [%]")
    axis.set_title("Most pixels do not use an interior response-table retrieval")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_dir / "image_ratio_retrieval_categories.png", dpi=180)
    plt.close(figure)


#%% Spatial trace

def count_limit(*arrays):
    """Use one robust colour limit for a before/after sensor pair."""

    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return max(np.nanpercentile(finite, 99), 1)


def plot_signal_trace(product, orbit, output_dir):
    """Show the signal, correction, ratio, and retrieval for one frame."""

    ratio, categories, _ = ratio_and_categories(
        product.wic_corrected, product.si13_corrected
    )
    strong_per_frame = (
        np.isfinite(ratio) & (product.si13_corrected >= 3)
    ).sum(axis=(1, 2))
    frame = int(np.argmax(strong_per_frame))

    wic_max = count_limit(product.wic[frame], product.wic_corrected[frame])
    si13_max = count_limit(product.si13[frame], product.si13_corrected[frame])
    ratio_max = max(300, np.nanpercentile(ratio[frame], 95))

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    fields = [
        (product.wic[frame], "binned WIC", 0, wic_max, None),
        (product.si13[frame], "regridded SI13", 0, si13_max, None),
        (ratio[frame], "corrected WIC/SI13", None, None, LogNorm(vmin=10, vmax=ratio_max)),
        (product.wic_corrected[frame], "proton-corrected WIC", 0, wic_max, None),
        (product.si13_corrected[frame], "proton-corrected SI13", 0, si13_max, None),
        (product.E0[frame], "retrieved $E_0$ [keV]", 0, 25, None),
    ]

    for axis, (values, title, vmin, vmax, norm) in zip(axes.flat, fields):
        image = axis.pcolormesh(
            product.grid.xi,
            product.grid.eta,
            values,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.78)

    total = sum(mask[frame].sum() for mask in categories)
    inside = categories[3][frame].sum()
    figure.suptitle(
        f"Orbit {orbit:04d}, {product.time[frame]:%Y-%m-%d %H:%M:%S} UTC — "
        f"{100 * inside / max(total, 1):.1f}% interior retrieval"
    )
    figure.savefig(output_dir / f"or_{orbit:04d}_image_ratio_signal_trace.png", dpi=180)
    plt.close(figure)


#%% IDL ingestion and background stages

def plain_array(variable):
    """Convert a NetCDF variable to an ndarray with missing values as NaN."""

    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float)


def load_detector_stages(base, sensor, orbit, source_index, time):
    """Load IDL, fuvpy input, and background-corrected detector images."""

    settings = {
        "WIC": ("wic", "wic", "SH"),
        "SI13": ("s13", "s13", "DG"),
    }
    folder, prefix, correction = settings[sensor]

    idl_file = base / f"{folder}_data" / f"{prefix}{time:%Y%j%H%M}.idl"
    record = readsav(idl_file, python_dict=True)["imageinfo"][0]
    idl_image = np.asarray(record["IMAGE"], dtype=float)

    nc_file = base / folder / f"{prefix}_or{orbit:04d}.nc"
    with Dataset(nc_file) as nc:
        fuvpy_image = plain_array(nc.variables["img"])[source_index]
        corrected = plain_array(nc.variables[f"{correction.lower()}img"])[source_index]

    return idl_image, fuvpy_image, corrected, int(record["CALIBRATION_FLAG"]), correction


def plot_detector_stages(base, product, orbit, output_dir):
    """Show raw ingestion and background subtraction before spatial binning."""

    ratio = np.divide(
        product.wic_corrected,
        product.si13_corrected,
        out=np.full(product.shape, np.nan),
        where=product.si13_corrected > 0,
    )
    frame = int(np.argmax((np.isfinite(ratio) & (product.si13_corrected >= 3)).sum(axis=(1, 2))))

    sensors = ["WIC", "SI13"]
    figure, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)

    for row, sensor in enumerate(sensors):
        source_index = int(product.source_indices[sensor.lower()][frame])
        idl, fuvpy, corrected, calibration_flag, correction = load_detector_stages(
            base, sensor, orbit, source_index, product.time[frame]
        )
        difference = fuvpy - idl

        raw_max = max(np.nanpercentile(idl, 99), np.nanpercentile(fuvpy, 99), 1)
        difference_max = max(np.nanpercentile(np.abs(difference), 99), 1)
        corrected_max = max(np.nanpercentile(np.abs(corrected), 99), 1)

        fields = [
            (idl, "IDL IMAGE", "viridis", 0, raw_max),
            (fuvpy, "saved fuvpy img", "viridis", 0, raw_max),
            (difference, "fuvpy minus IDL", "RdBu_r", -difference_max, difference_max),
            (corrected, f"{correction} background corrected", "RdBu_r", -corrected_max, corrected_max),
        ]

        for column, (values, title, cmap, vmin, vmax) in enumerate(fields):
            axis = axes[row, column]
            image = axis.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, shrink=0.72)

        axes[row, 0].set_ylabel(f"{sensor}\ncalibration flag {calibration_flag}")

    figure.suptitle(
        f"Orbit {orbit:04d}, detector-space input and background stages\n"
        "WIC fuvpy input includes reflat=True; SI13 fuvpy img reproduces the IDL IMAGE"
    )
    figure.savefig(output_dir / f"or_{orbit:04d}_detector_background_stages.png", dpi=180)
    plt.close(figure)


#%% Output table and main program

def write_summary(rows, filename):
    """Save the values behind the diagnostic summary plots."""

    with filename.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=repository / "example_data")
    parser.add_argument("--output", type=Path, default=repository / "figures" / "debugging")
    parser.add_argument("--orbits", type=int, nargs="+", default=[85, 86])
    parser.add_argument("--proton-energy", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    products = {}
    summaries = []

    for orbit in args.orbits:
        product = load_ratio_product(args.base, orbit, args.proton_energy)
        products[orbit] = product

        before, _, _ = summarize_stage(orbit, "before proton correction", product.wic, product.si13)
        after, _, _ = summarize_stage(
            orbit,
            "after proton correction",
            product.wic_corrected,
            product.si13_corrected,
        )
        summaries.extend([before, after])

        plot_signal_trace(product, orbit, args.output)
        plot_detector_stages(args.base, product, orbit, args.output)

    plot_response_domain(products, args.output)
    plot_category_fractions(summaries, args.output)
    write_summary(summaries, args.output / "image_ratio_summary.csv")

    for row in summaries:
        print(
            f"orbit {row['orbit']:04d}, {row['stage']}: "
            f"median strong ratio {row['median_strong_ratio']:.1f}; "
            f"{row['strong_pixels_inside_table_percent']:.1f}% inside table"
        )
    print(f"\nSaved diagnostics in {args.output}")


if __name__ == "__main__":
    main()
